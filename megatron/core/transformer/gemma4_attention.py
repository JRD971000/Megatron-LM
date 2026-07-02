# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor

from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module

# Pure-python FFPA wheel install tree (installed --no-deps to avoid a torch shadow;
# see V_RESULTS V0 notes). The RUN env sets PYTHONPATH to this; _attn_dispatch also
# inserts it defensively before importing ffpa_attn.
_FFPA_INSTALL = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh"
    "/Gemma4_mlm/ffpa_install"
)


@dataclass
class Gemma4SelfAttentionSubmodules(SelfAttentionSubmodules):
    """:class:`SelfAttentionSubmodules` plus the scaleless per-head V norm."""

    v_layernorm: Union[ModuleSpec, type, None] = None


class Gemma4SelfAttention(SelfAttention):
    """Gemma 4 self-attention with scaleless v_norm and cross-layer KV sharing.

    Differences from the base :class:`SelfAttention` (all per HF
    ``Gemma4TextAttention``, modeling_gemma4.py:1229-1290):

    * ``softmax_scale = 1.0`` for both layer types (no ``1/sqrt(head_dim)``); set on
      the per-layer config.
    * q/k norm via the spec's ``q_layernorm``/``k_layernorm`` (``Gemma4RMSNorm`` over
      head_dim, applied pre-RoPE) and a NEW scaleless ``v_layernorm`` applied to V.
    * RoPE applied with externally-supplied per-layer-type cos/sin (full-width
      ``rotate_half``), POST q/k norm.
    * KV bus: producer layers write their post-norm/post-RoPE (k, v) into a shared
      dict; borrower layers read (k, v) from it and skip their own k/v processing.

    The producer/borrower role is derived from the config (``num_kv_shared_layers``)
    and this layer's 1-based ``layer_number``, matching HF's
    ``first_kv_shared_layer_idx`` / ``store_full_length_kv`` logic exactly.

    ``forward`` is overridden with a clean training-only eager path (no inference /
    flash-decode / packed-sequence / CP machinery; this port is DDP=1, all
    parallelism = 1) so the bitwise islands (fp32 softmax, ``finfo.min`` additive
    mask, scale=1.0, scaleless v_norm) are explicit.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Gemma4SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            **kwargs,
        )

        # Scaleless per-head V RMSNorm (no weight) over head_dim. Built only on
        # producer/own layers; borrowers reuse a producer's already-normed V.
        # The v_layernorm submodule is a norm BUILDER (a plain function, like
        # q/k_layernorm), so it is called directly rather than via build_module
        # (build_module returns a FunctionType unchanged instead of invoking it).
        self.v_layernorm = (
            submodules.v_layernorm(
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
            if submodules.v_layernorm is not None
            else None
        )

        # "full" layers drop the sliding window (get_config_for_layer sets it None).
        self.layer_type = "full" if self.config.window_size is None else "sliding"

        # Cross-layer KV-share role (HF modeling_gemma4.py:1199-1204). layer_number is
        # 1-based; HF layer_idx is 0-based.
        num_shared = getattr(self.config, "num_kv_shared_layers", 0)
        first_shared_idx = self.config.num_layers - num_shared
        layer_idx = self.layer_number - 1
        self.is_kv_shared_layer = num_shared > 0 and layer_idx >= first_shared_idx
        # Producer = last own-layer of this layer_type among layers [0, first_shared_idx).
        prev_types = self.config.layer_types[:first_shared_idx]
        self.store_full_length_kv = (
            not self.is_kv_shared_layer
            and layer_idx == len(prev_types) - 1 - prev_types[::-1].index(self.layer_type)
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        rotary_cos_sin: Optional[tuple] = None,
        kv_bus: Optional[dict] = None,
        packed_seq_params: Optional[object] = None,
        allow_eager_packed: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Clean eager forward. ``hidden_states`` is [s, b, h] (MLM seq-first).

        ``attention_mask`` is the additive (``finfo.min``) mask for this layer type,
        broadcastable to [b, np, sq, sk]. ``rotary_cos_sin`` is (cos, sin) of width
        head_dim for this layer type, shape [b, sq, head_dim]. ``kv_bus`` is the
        per-forward shared K/V dict keyed by layer_type.
        """
        # Fused QKV projection -> q [s, b, np, hd], k/v [s, b, ng, hd]. q_norm/k_norm
        # are applied inside get_query_key_value_tensors (per-head over head_dim).
        query, key, value = self.get_query_key_value_tensors(hidden_states)

        cos, sin = rotary_cos_sin
        # cos/sin are [b, s, hd]; MLM tensors are [s, b, h, hd] -> move cos/sin to
        # seq-first and broadcast over (b, heads). apply_rotary_pos_emb expects
        # [..., hd] with cos/sin unsqueezed on the head axis.
        cos = cos.transpose(0, 1).unsqueeze(2)  # [s, b, 1, hd]
        sin = sin.transpose(0, 1).unsqueeze(2)

        # Query is always projected, normed, and roped on this layer.
        query = query * cos + _rotate_half(query) * sin

        if self.is_kv_shared_layer:
            # Borrow producer's post-norm/post-RoPE K/V; discard own k/v projections.
            key, value = kv_bus[self.layer_type]
        else:
            key = key * cos + _rotate_half(key) * sin
            if self.v_layernorm is not None:
                value = apply_module(self.v_layernorm)(value)
            if self.store_full_length_kv:
                kv_bus[self.layer_type] = (key, value)

        context = self._attn_dispatch(
            query, key, value, attention_mask, packed_seq_params, allow_eager_packed
        )

        # [s, b, np, hd] -> [s, b, np*hd] -> o_proj.
        context = context.reshape(context.size(0), context.size(1), -1)
        output, bias = apply_module(self.linear_proj)(context)
        return output, bias

    def _attn_dispatch(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        packed_seq_params: Optional[object],
        allow_eager_packed: bool,
    ) -> Tensor:
        """Route the (already QKV/norm/RoPE/v-norm-processed) q/k/v to a backend.

        Backends are selected by ``self.config.gemma4_attention_backend``:

        * ``"eager"`` (default): byte-identical to the historical behavior via
          :meth:`_gemma4_core_attention`. Refuses packed input (R3) unless the
          keyword-only ``allow_eager_packed`` negative-control escape is set,
          because the eager additive-mask path cannot express leak-free packing.
        * ``"ffpa_flash"``: FFPA for full (head_dim=512) layers, FlashAttention
          for sliding (head_dim=256) layers, honoring ``cu_seqlens`` for packing.
          ``softmax_scale`` is passed EXPLICITLY (== 1.0; flash otherwise defaults
          to 1/sqrt(d) -- R2). GQA is handled natively by the kernels
          (``enable_gqa=True`` for FFPA, native for flash); K/V keep ``ng`` heads,
          NO manual repeat.

        q/k/v come in Megatron seq-first layout: query ``[s, b, np, hd]``,
        key/value ``[s, b, ng, hd]``. Returns context ``[s, b, np, hd]`` so the
        caller's reshape + o_proj is unchanged.
        """
        backend = self.config.gemma4_attention_backend

        if backend == "eager":
            if packed_seq_params is not None and not allow_eager_packed:
                # R3: the eager additive-mask path attends across the full [s, s]
                # grid and cannot block cross-document leakage. Mirror
                # dot_product_attention.py's packed-seq refusal.
                raise AssertionError(
                    "Gemma4 eager attention backend does not support packed sequences "
                    "(packed_seq_params was provided): the additive-mask path leaks "
                    "across documents. Use gemma4_attention_backend='ffpa_flash' for "
                    "packed SFT, or set the keyword-only allow_eager_packed=True "
                    "(negative control only)."
                )
            return self._gemma4_core_attention(query, key, value, attention_mask)

        if backend != "ffpa_flash":
            raise ValueError(
                f"Unknown Gemma4 gemma4_attention_backend {backend!r}; "
                "expected 'eager' or 'ffpa_flash'."
            )

        # ---- ffpa_flash: lazy, guarded imports -------------------------------
        try:
            import sys

            if _FFPA_INSTALL not in sys.path:
                # Defensive: the RUN env sets PYTHONPATH, but insert anyway so the
                # pure-python ffpa wheel is importable without relying on it.
                sys.path.insert(0, _FFPA_INSTALL)
            import ffpa_attn
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except ImportError as e:
            raise ImportError(
                "gemma4_attention_backend='ffpa_flash' requires the 'ffpa_attn' and "
                "'flash_attn' packages. Ensure PYTHONPATH includes the ffpa_install "
                f"tree ({_FFPA_INSTALL}) and that flash_attn is installed in the "
                f"container. Original error: {e}"
            ) from e

        scale = self.config.softmax_scale  # == 1.0, passed EXPLICITLY (R2).
        s_q, b, np_, hd = query.shape

        packed = packed_seq_params is not None

        if self.layer_type == "full":
            if packed:
                # b MUST be 1 for THD/varlen packing.
                assert b == 1, f"varlen (packed) path requires b==1, got b={b}"
                cu_q = packed_seq_params.cu_seqlens_q
                max_q = packed_seq_params.max_seqlen_q
                max_kv = packed_seq_params.max_seqlen_kv
                # [s, 1, n, hd] -> [T, n, hd] = [T, H, D].
                q = query.squeeze(1).contiguous()
                k = key.squeeze(1).contiguous()
                v = value.squeeze(1).contiguous()
                assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
                out = ffpa_attn.ffpa_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_q,
                    cu_q,  # self-attention: same cu_seqlens for q and kv.
                    max_q,
                    max_kv,
                    softmax_scale=scale,
                    causal=True,
                    enable_gqa=True,
                )
                out = out[0] if isinstance(out, tuple) else out
                # [T, n, hd] -> [s, 1, n, hd] = [s, b, np, hd].
                return out.reshape(s_q, b, np_, hd)
            # dense: [s, b, n, hd] -> [b, n, s, hd] = [B, H, S, D].
            q = query.permute(1, 2, 0, 3).contiguous()
            k = key.permute(1, 2, 0, 3).contiguous()
            v = value.permute(1, 2, 0, 3).contiguous()
            assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
            out = ffpa_attn.ffpa_attn_func(
                q, k, v, is_causal=True, scale=scale, enable_gqa=True
            )
            out = out[0] if isinstance(out, tuple) else out
            # [B, H, S, D] -> [s, b, np, hd].
            return out.permute(2, 0, 1, 3).contiguous()

        if self.layer_type == "sliding":
            window = self.config.sliding_window  # 512; flash left window = window - 1.
            if packed:
                assert b == 1, f"varlen (packed) path requires b==1, got b={b}"
                cu_q = packed_seq_params.cu_seqlens_q
                max_q = packed_seq_params.max_seqlen_q
                max_kv = packed_seq_params.max_seqlen_kv
                # [s, 1, n, hd] -> [T, n, hd] = [T, H, D].
                q = query.squeeze(1).contiguous()
                k = key.squeeze(1).contiguous()
                v = value.squeeze(1).contiguous()
                assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
                out = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_q,
                    cu_q,
                    max_q,
                    max_kv,
                    softmax_scale=scale,
                    causal=True,
                    window_size=(window - 1, 0),
                )
                out = out[0] if isinstance(out, tuple) else out
                # [T, n, hd] -> [s, 1, n, hd].
                return out.reshape(s_q, b, np_, hd)
            # dense: [s, b, n, hd] -> [b, s, n, hd] = [B, S, H, D].
            q = query.permute(1, 0, 2, 3).contiguous()
            k = key.permute(1, 0, 2, 3).contiguous()
            v = value.permute(1, 0, 2, 3).contiguous()
            assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
            out = flash_attn_func(
                q, k, v, causal=True, softmax_scale=scale, window_size=(window - 1, 0)
            )
            out = out[0] if isinstance(out, tuple) else out
            # [B, S, H, D] -> [s, b, np, hd] (permute(1,0,2,3) is self-inverse here).
            return out.permute(1, 0, 2, 3).contiguous()

        raise ValueError(f"Unknown Gemma4 layer_type {self.layer_type!r}.")

    def _gemma4_core_attention(
        self, query: Tensor, key: Tensor, value: Tensor, attention_mask: Optional[Tensor]
    ) -> Tensor:
        """Eager attention mirroring HF ``eager_attention_forward`` (scale=1.0, fp32 softmax).

        query [s, b, np, hd]; key/value [s, b, ng, hd]. Returns context [s, b, np, hd].
        """
        s_q, b, np_, hd = query.shape
        ng = key.size(2)
        s_k = key.size(0)
        groups = np_ // ng

        # [s, b, n, hd] -> [b, n, s, hd].
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)

        # GQA: repeat K/V groups to match query heads (HF repeat_kv).
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        # (q @ k^T) * 1.0 ; softmax_scale = 1.0 for both layer types (HF scaling=1.0).
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.config.softmax_scale
        if attention_mask is not None:
            scores = scores + attention_mask  # additive finfo.min mask

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        context = torch.matmul(probs, v)  # [b, np, s, hd]

        # [b, np, s, hd] -> [s, b, np, hd].
        return context.permute(2, 0, 1, 3).contiguous()


def _rotate_half(x: Tensor) -> Tensor:
    """Full-width rotate_half (split at hd/2), matching gemma4_rope.rotate_half."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
