# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.typed_torch import apply_module
from megatron.core.utils import make_viewless_tensor


class Gemma4TransformerBlock(TransformerBlock):
    """Gemma 4 decoder stack: a :class:`TransformerBlock` that owns the KV bus.

    Overrides the eager layer loop so that, per forward, it (a) instantiates a fresh
    cross-layer KV bus (a plain dict keyed by layer type), (b) hands each layer its
    ``per_layer_inputs[:, :, i, :]`` slice, and (c) selects the per-layer-type rotary
    ``cos/sin`` and additive attention mask. This keeps all KV-share / PLE / mask
    plumbing out of the base block (R1: no base-block kwarg churn).

    Activation recompute (``recompute_granularity='full'``) is supported via a
    KV-bus-aware per-layer checkpointing path (see :meth:`_recompute_layers`).
    Inference / selective-recompute paths are out of scope for this training port.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        per_layer_inputs: Tensor = None,
        rotary_cos_sin_by_type: dict = None,
        attention_mask_by_type: dict = None,
        packed_seq_params=None,
        **kwargs,
    ):
        """Run the gemma4 decoder layers. ``hidden_states`` is [s, b, h]."""
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )

        # Full activation recompute: only in training with grad enabled (never for
        # eval/inference). 'selective' granularity is intentionally not honored here
        # (Gemma4SelfAttention bypasses core_attention, so there is no core_attn
        # module to wrap); it silently falls through to the plain eager loop.
        recompute = (
            self.config.recompute_granularity == "full"
            and self.training
            and torch.is_grad_enabled()
        )

        if recompute:
            hidden_states = self._recompute_layers(
                hidden_states,
                per_layer_inputs=per_layer_inputs,
                rotary_cos_sin_by_type=rotary_cos_sin_by_type,
                attention_mask_by_type=attention_mask_by_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            # Fresh KV bus per forward (plain dict; single device, no CP/TP interaction).
            kv_bus: dict = {}
            for i, layer in enumerate(self.layers):
                layer_type = layer.self_attention.layer_type
                hidden_states, _ = layer(
                    hidden_states,
                    attention_mask=attention_mask_by_type[layer_type],
                    per_layer_input=per_layer_inputs[:, :, i, :],
                    rotary_cos_sin=rotary_cos_sin_by_type[layer_type],
                    kv_bus=kv_bus,
                    packed_seq_params=packed_seq_params,
                )

        if self.final_layernorm is not None:
            hidden_states = apply_module(self.final_layernorm)(hidden_states)

        return hidden_states

    def _recompute_layers(
        self,
        hidden_states: Tensor,
        *,
        per_layer_inputs: Tensor,
        rotary_cos_sin_by_type: dict,
        attention_mask_by_type: dict,
        packed_seq_params,
    ) -> Tensor:
        """Per-layer activation checkpointing that threads the cross-layer KV bus.

        The cross-layer KV bus is the reason the base block's ``_checkpointed_forward``
        cannot be reused: a *producer* layer (``store_full_length_kv``; layer 22 sliding,
        layer 23 full in E4B) writes its post-norm/post-RoPE ``(k, v)`` into the bus, and
        *borrower* layers (``is_kv_shared_layer``; 24-41) read it. The borrower->producer
        KV-share gradients (producer 23 accumulates grad from itself + 3 full borrowers,
        producer 22 from itself + 15 sliding borrowers) must survive recompute.

        Two failure modes have to be avoided together:

        * If a producer were checkpointed the ordinary way, its ``(k, v)`` would be
          captured from the *no-grad* recompute forward -> ``requires_grad=False`` ->
          borrower->producer grads silently dropped.
        * If a borrower checkpoint captured its producer's LIVE ``(k, v)`` via closure
          (not as explicit inputs), each borrower's re-entrant recompute-backward would
          traverse into the single shared producer graph and free it -> the next
          borrower raises "backward through the graph a second time".

        Chosen design:
        * Producers run EAGERLY (not checkpointed) -> their ``(k, v)`` are live
          outer-graph tensors, so the many-borrowers->one-producer fan-in flows through
          the producer's normal backward exactly once, with standard grad accumulation.
          Cost is only the 2 producer layers' retained activations.
        * Every other layer is checkpointed per-layer. Borrowers take their type's
          ``(k, v)`` as EXPLICIT checkpoint inputs, so the checkpoint detaches them and
          the borrower's recompute-backward stops at those detached leaves (no traversal
          into / double-free of the producer graph); the returned input-grads are routed
          by the outer engine to the live producer tensors.

        (Returning ``(k, v)`` as checkpoint OUTPUTS of a checkpointed producer -- a
        multi-output re-entrant checkpoint -- was tried first and mis-handles the shared
        k/v graph in ``CheckpointFunction`` backward; running producers eagerly is both
        simpler and correct.)

        Recompute is per-layer (finest granularity). ``recompute_num_layers`` /
        ``recompute_method`` are not sub-divided further: per-layer is already the finest
        unit and the KV bus requires layer-aware checkpoint boundaries.
        """
        dsa = self.config.distribute_saved_activations
        # layer_type -> (k, v): LIVE outer-graph tensors written by eager producers.
        bus: dict = {}
        # Layer types that actually have a borrower this forward. A producer is only
        # treated as a *bus* producer (run eagerly, k/v published) if its type is
        # borrowed -- otherwise (e.g. num_kv_shared_layers==0) it is a plain own layer
        # whose k/v stay internal, exactly like the non-recompute path.
        borrowed_types = {
            lyr.self_attention.layer_type
            for lyr in self.layers
            if lyr.self_attention.is_kv_shared_layer
        }

        def make_custom(index: int, layer_type: str):
            """Checkpointed forward for an own OR borrower layer (single tensor output).

            Every grad-REQUIRING outer tensor the layer consumes is passed as an explicit
            checkpoint input so ``CheckpointFunction`` detaches it in recompute (and routes
            its grad back to the outer graph once): ``hs``, the per-layer PLE slice, and --
            for borrowers -- the producer's ``(k, v)``. The per-layer input is essential:
            ``per_layer_inputs`` is a single live grad-requiring tensor shared by all layers,
            so capturing its slice via closure would make each layer's recompute-backward
            traverse (and free) the shared PLE graph -> "backward a second time" on the next
            layer. ``attention_mask`` / ``rotary_cos_sin`` are ``requires_grad=False`` buffers,
            so they are safe to capture via closure.
            """
            layer = self.layers[index]

            def custom_forward(hs, ple_slice, *bus_tensors):
                local_bus: dict = {}
                if bus_tensors:
                    local_bus[layer_type] = (bus_tensors[0], bus_tensors[1])
                out, _ = layer(
                    hs,
                    attention_mask=attention_mask_by_type[layer_type],
                    per_layer_input=ple_slice,
                    rotary_cos_sin=rotary_cos_sin_by_type[layer_type],
                    kv_bus=local_bus,
                    packed_seq_params=packed_seq_params,
                )
                return out

            return custom_forward

        for index, layer in enumerate(self.layers):
            attn = layer.self_attention
            layer_type = attn.layer_type
            ple_slice = per_layer_inputs[:, :, index, :]

            if attn.is_kv_shared_layer:
                # Borrower: pass its producer's live (k, v) as explicit checkpoint inputs.
                k, v = bus[layer_type]
                hidden_states = tensor_parallel.checkpoint(
                    make_custom(index, layer_type), dsa, hidden_states, ple_slice, k, v
                )
            elif attn.store_full_length_kv and layer_type in borrowed_types:
                # Producer with borrowers: run EAGERLY so its (k, v) stay live in the
                # outer graph (single, correct producer backward under grad fan-in).
                local_bus: dict = {}
                hidden_states, _ = layer(
                    hidden_states,
                    attention_mask=attention_mask_by_type[layer_type],
                    per_layer_input=ple_slice,
                    rotary_cos_sin=rotary_cos_sin_by_type[layer_type],
                    kv_bus=local_bus,
                    packed_seq_params=packed_seq_params,
                )
                bus[layer_type] = local_bus[layer_type]
            else:
                # Own layer (or borrower-less producer): plain per-layer checkpoint.
                hidden_states = tensor_parallel.checkpoint(
                    make_custom(index, layer_type), dsa, hidden_states, ple_slice
                )

        return hidden_states
