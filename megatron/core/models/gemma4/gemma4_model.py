# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.models.gemma4.gemma4_block import Gemma4TransformerBlock
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.gemma4_mask import (
    build_causal_mask,
    build_sliding_window_causal_mask,
)
from megatron.core.transformer.gemma4_ple import Gemma4PLE
from megatron.core.transformer.gemma4_rope import Gemma4RotaryEmbedding
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module


def _reset_position_ids_from_cu_seqlens(cu_seqlens: Tensor) -> Tensor:
    """R1: build per-document *reset* RoPE positions from cu_seqlens.

    Under SFT packing a single ``[1, s]`` sequence concatenates several documents
    end-to-end. Gemma4 RoPE reads ``position_ids`` directly (gemma4_rope.py:107), so a
    flat ``arange(s)`` gives documents 2+ the WRONG positions even though the varlen
    kernel blocks cross-document attention (the kernel fixes attention leakage, not RoPE).

    Given the cumulative ``cu_seqlens = [0, l0, l0+l1, ...]`` this returns the
    concatenation of ``arange(len)`` per segment, e.g. ``[0, 3, 7] -> [0, 1, 2, 0, 1, 2, 3]``.
    Vectorized (no per-segment python / host sync) via a segment-id gather.
    """
    cu = cu_seqlens.reshape(-1).to(torch.long)
    seg_lens = (cu[1:] - cu[:-1]).clamp(min=0)
    # Which segment each packed token belongs to, e.g. lens [3, 4] -> [0,0,0,1,1,1,1].
    seg_ids = torch.repeat_interleave(
        torch.arange(seg_lens.numel(), device=cu.device), seg_lens
    )
    seg_starts = cu[:-1][seg_ids]
    return torch.arange(seg_ids.numel(), device=cu.device) - seg_starts


def _assert_position_ids_reset(position_ids: Tensor, cu_seqlens: Tensor) -> None:
    """R1: assert a caller-supplied ``position_ids`` resets per document.

    Single source of truth for packed positions is the dataloader (training) or the
    model (parity scripts). When the caller already passed ``position_ids`` we must NOT
    overwrite it, but we must guard against a flat ``arange`` silently corrupting RoPE.
    Cheap check: the first token of every packed segment must have position 0. A flat
    arange makes segment ``i`` start at ``cu_seqlens[i] != 0`` -> raise, which forces
    training to use ``reset_position_ids=True`` (the SFT dataloader builds per-conversation
    ``range(len)``; see megatron/training/datasets/sft_dataset.py).
    """
    cu = cu_seqlens.reshape(-1).to(torch.long)
    seg_starts = cu[:-1]
    starts_pos = position_ids.reshape(-1)[seg_starts]
    if not bool(torch.all(starts_pos == 0)):
        raise AssertionError(
            "Gemma4 packed forward (R1): supplied position_ids do not reset to 0 at the "
            "cu_seqlens document boundaries (they look like a flat arange). Per-document "
            "RoPE would be wrong. Set reset_position_ids=True so positions reset per "
            "document (the SFT dataloader already builds per-conversation range(len))."
        )


class Gemma4Model(GPTModel):
    """Gemma 4 E4B model (HF ``Gemma4ForCausalLM``, modeling_gemma4.py:1646-1905).

    Subclasses :class:`GPTModel` and adds the model-level Gemma4 specifics, all kept
    out of the shared base (R4):

    * **sqrt(H) embedding scaling** applied inside ``forward`` (bf16-rounded scalar, so
      sqrt(2560) rounds in bf16 before the multiply, matching HF
      ``Gemma4TextScaledWordEmbedding``).
    * **Per-Layer Embeddings (PLE)** precomputed into ``per_layer_inputs`` [B,S,L,P] and
      fed one slice per decoder layer via :class:`Gemma4TransformerBlock`.
    * **final logit softcap** ``30 * tanh(logits / 30)`` applied directly to the logits
      so it reaches the loss (``_scale_logits`` no-ops unless ``use_mup``).

    Uses :class:`Gemma4TransformerBlock` as its decoder, ``position_embedding_type=
    'none'`` (the base RoPE is skipped; a :class:`Gemma4RotaryEmbedding` supplies the
    per-layer-type cos/sin), and tied input/output embeddings.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        **kwargs,
    ):
        kwargs.setdefault("position_embedding_type", "none")
        kwargs.setdefault("share_embeddings_and_output_weights", True)
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )

        # Replace the base decoder with the gemma4 block (owns KV bus + PLE threading).
        self.decoder = Gemma4TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
            vp_stage=self.vp_stage,
        )

        self.rotary_emb = Gemma4RotaryEmbedding(
            sliding_head_dim=self.config.sliding_kv_channels,
            full_head_dim=self.config.full_kv_channels,
            sliding_base=self.config.sliding_rotary_base,
            full_base=self.config.full_rotary_base,
        )
        self.ple = Gemma4PLE(
            config=self.config,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            ple_dim=self.config.hidden_size_per_layer_input,
            vocab_size_per_layer_input=self.config.vocab_size_per_layer_input,
            eps=self.config.layernorm_epsilon,
            tp_group=self.pg_collection.tp,
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Gemma4 forward. Returns logits [b, s, V] (labels=None) or loss [b, s]."""
        b, s = input_ids.shape
        device = input_ids.device

        packed_seq_params = kwargs.get("packed_seq_params")

        position_ids_provided = position_ids is not None
        if position_ids is None:
            position_ids = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)

        # R1 (CRITICAL): under packing, RoPE positions MUST reset per document, otherwise
        # documents 2+ get the wrong RoPE (the varlen kernel blocks cross-doc attention but
        # does NOT fix positions). See TASK-001 R1. This runs BEFORE self.embedding (which
        # consumes position_ids) and is a no-op when unpacked.
        # Source of truth: the dataloader in training (position_ids provided), the model in
        # parity scripts (position_ids None). cu_seqlens_q may be the PADDED variant -- it is
        # the only one PackedSeqParams exposes (pretrain_gpt.py forces cu_seqlens==padded);
        # V2/V4 use NO intra-pack padding so padded==real there and the derivation is exact.
        if packed_seq_params is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q_padded
            if cu_seqlens is None:
                cu_seqlens = packed_seq_params.cu_seqlens_q
            if cu_seqlens is not None:
                if position_ids_provided:
                    # Do NOT overwrite the caller's positions; just verify they reset.
                    _assert_position_ids_reset(position_ids, cu_seqlens)
                else:
                    # Parity path: derive reset positions from cu_seqlens (b == 1 packing).
                    position_ids = _reset_position_ids_from_cu_seqlens(cu_seqlens).unsqueeze(0)

        # Embedding [s, b, h] (MLM seq-first) with bf16-rounded sqrt(H) scaling.
        decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        if self.config.apply_embedding_scaling:
            embed_dtype = self.embedding.word_embeddings.weight.dtype
            scale = torch.tensor(self.config.hidden_size**0.5).to(embed_dtype)
            decoder_input = decoder_input * scale

        # PLE precompute. The foundation Gemma4PLE consumes inputs_embeds in [B,S,H];
        # decoder_input is [s,b,h] so transpose. Result [B,S,L,P] -> [s,b,L,P] for the
        # seq-first decoder.
        #
        # Under sequence parallelism the embedding has reduce-scattered decoder_input to
        # [s/TP, b, h], but PLE needs the FULL-sequence embeddings (E_proj projects the
        # whole-sequence inputs_embeds). Gather the sequence back for the PLE precompute
        # only; the residual stream handed to the decoder stays sequence-sharded.
        # ``tensor_parallel_output_grad=False`` is REQUIRED: the gather's default backward
        # reduce-scatters+sums over TP, but the PLE branch's ple-dim scatter backward
        # already all-gathers the full (duplicated) PLE gradient onto every rank, so a
        # second TP reduction would overcount the PLE->decoder_input gradient by TP. The
        # split-only backward takes each rank's own sequence slice. No-op at TP=1.
        ple_embeds = decoder_input
        if self.config.sequence_parallel:
            ple_embeds = tensor_parallel.gather_from_sequence_parallel_region(
                decoder_input, tensor_parallel_output_grad=False
            )
        inputs_embeds = ple_embeds.transpose(0, 1).contiguous()  # [b, s, h]
        per_layer_inputs = self.ple(input_ids, inputs_embeds)  # [b, s, L, P]
        per_layer_inputs = per_layer_inputs.transpose(0, 1).contiguous()  # [s, b, L, P]
        # Shard the PLE dim (last) across TP so each rank's slice matches its gate output
        # [s, b, P/TP]. Identity at TP=1.
        per_layer_inputs = tensor_parallel.scatter_to_tensor_model_parallel_region(
            per_layer_inputs
        )  # [s, b, L, P/TP]

        # Per-layer-type rope cos/sin and additive masks (finfo.min fill).
        dtype = decoder_input.dtype
        rotary_cos_sin_by_type = {
            "sliding": self.rotary_emb(decoder_input, position_ids, "sliding"),
            "full": self.rotary_emb(decoder_input, position_ids, "full"),
        }
        attention_mask_by_type = {
            "full": build_causal_mask(s, dtype, device),
            "sliding": build_sliding_window_causal_mask(
                s, self.config.sliding_window, dtype, device
            ),
        }

        hidden_states = self.decoder(
            decoder_input,
            per_layer_inputs=per_layer_inputs,
            rotary_cos_sin_by_type=rotary_cos_sin_by_type,
            attention_mask_by_type=attention_mask_by_type,
            packed_seq_params=packed_seq_params,
        )

        # LM head (tied) -> final logit softcap -> loss/return.
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        softcap = self.config.final_logit_softcapping
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)

        if labels is None:
            return logits.transpose(0, 1).contiguous()  # [s, b, V] -> [b, s, V]
        return self.compute_language_model_loss(labels, logits)
