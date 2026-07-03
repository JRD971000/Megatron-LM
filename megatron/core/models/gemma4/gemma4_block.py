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
        *borrower* layers (``is_kv_shared_layer``; 24-41) read it. If a producer were
        checkpointed the ordinary way, its ``(k, v)`` would be captured from the *no-grad*
        recompute-forward and land in the bus with ``requires_grad=False`` -- silently
        dropping every borrower->producer KV-share gradient (the exact grads the port
        relies on: producer 23 accumulates grad from itself + 3 full borrowers, producer
        22 from itself + 15 sliding borrowers).

        Fix: make the bus tensors first-class edges of the *outer* autograd graph. Each
        producer's ``(k, v)`` are RETURNED from its checkpoint (so they carry a real
        ``grad_fn``), and each borrower takes its layer-type's ``(k, v)`` as explicit
        checkpoint *inputs*. The standard autograd engine then handles the fan-in
        (many borrowers -> one producer, accumulated once) and graph freeing exactly as
        in the non-recompute path -- unlike closure-captured tensors, which would trigger
        one manual backward (and buffer-free) per borrower through a shared producer.

        Recompute is per-layer (finest granularity). ``recompute_num_layers`` /
        ``recompute_method`` are not sub-divided further: per-layer is already the
        finest unit and the KV bus requires layer-aware checkpoint boundaries.
        """
        dsa = self.config.distribute_saved_activations
        # layer_type -> (k, v): outer-graph tensors produced so far this forward.
        bus: dict = {}
        # Layer types that actually have a borrower this forward. A producer is only
        # surfaced as a *bus* producer (its k/v returned as checkpoint outputs) if its
        # type is borrowed -- otherwise those k/v would be dangling checkpoint outputs
        # with no consumer, and the checkpoint backward would raise on their None grads
        # (e.g. num_kv_shared_layers==0). Such a borrower-less producer falls through to
        # the plain own-layer path (its k/v stay internal, exactly like non-recompute).
        borrowed_types = {
            lyr.self_attention.layer_type
            for lyr in self.layers
            if lyr.self_attention.is_kv_shared_layer
        }

        def make_custom(index: int, layer_type: str, is_producer: bool):
            layer = self.layers[index]

            def custom_forward(hs, *bus_tensors):
                # Borrower: rebuild its layer-type's (k, v) from the explicit inputs so
                # the layer's forward reads them from a local (per-call) bus dict.
                local_bus: dict = {}
                if bus_tensors:
                    local_bus[layer_type] = (bus_tensors[0], bus_tensors[1])
                out, _ = layer(
                    hs,
                    attention_mask=attention_mask_by_type[layer_type],
                    per_layer_input=per_layer_inputs[:, :, index, :],
                    rotary_cos_sin=rotary_cos_sin_by_type[layer_type],
                    kv_bus=local_bus,
                    packed_seq_params=packed_seq_params,
                )
                if is_producer:
                    # Surface the freshly-stored (k, v) as checkpoint OUTPUTS so they
                    # become outer-graph edges the borrowers can depend on.
                    k, v = local_bus[layer_type]
                    return out, k, v
                return out

            return custom_forward

        for index, layer in enumerate(self.layers):
            attn = layer.self_attention
            layer_type = attn.layer_type
            is_bus_producer = attn.store_full_length_kv and layer_type in borrowed_types
            cf = make_custom(index, layer_type, is_bus_producer)

            if attn.is_kv_shared_layer:
                # Borrower: pass its producer's (k, v) as explicit checkpoint inputs.
                k, v = bus[layer_type]
                hidden_states = tensor_parallel.checkpoint(cf, dsa, hidden_states, k, v)
            elif is_bus_producer:
                # Producer with borrowers: capture its (k, v) checkpoint outputs.
                hidden_states, k, v = tensor_parallel.checkpoint(cf, dsa, hidden_states)
                bus[layer_type] = (k, v)
            else:
                # Own layer (or borrower-less producer): plain per-layer checkpoint.
                hidden_states = tensor_parallel.checkpoint(cf, dsa, hidden_states)

        return hidden_states
