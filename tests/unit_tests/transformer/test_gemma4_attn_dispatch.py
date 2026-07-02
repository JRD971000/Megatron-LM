# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Unit test for Gemma4SelfAttention._attn_dispatch (TASK-005 / I1).

Validates the two-backend kernel dispatch (FFPA full head_dim=512 + Flash sliding
head_dim=256, honoring cu_seqlens for packing) against the eager reference
``_gemma4_core_attention`` within the adopted V0 tolerance:

    max_abs <= 3e-2  OR  rel <= 2e-2   (bf16 attention noise; S-stable).

Cases:
  * dense full   (ffpa_attn_func)        ~ eager  (per layer type).
  * dense sliding (flash_attn_func)      ~ eager.
  * packed 2-doc (varlen) full+sliding   ~ eager-per-doc (block-diagonal).
  * R3: eager backend + packed_seq_params (without allow_eager_packed) raises;
        allow_eager_packed=True is accepted.

Requires a GPU + ffpa_attn (PYTHONPATH=ffpa_install) + flash_attn; run in the
container on the current slurm job (see TASK-005/TASK-003 recipe).

Import gemma4_common FIRST so the nvrx __version__ shim is applied before any
megatron import (bare `import megatron.core` asserts in the container).
"""
import os
import sys
import types

import pytest
import torch

# --- nvrx shim: import examples/gemma4/gemma4_common before megatron.core -----
_REPO = os.path.dirname(  # tests/unit_tests/transformer -> repo root
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(_REPO, "examples", "gemma4"))
import gemma4_common  # noqa: F401,E402  (applies the nvrx version shim)

from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402
from megatron.core.transformer.gemma4_attention import Gemma4SelfAttention  # noqa: E402

DEV = "cuda"
DT = torch.bfloat16
NP, NG = 8, 2  # query heads / kv groups (GQA 8/2)
ATOL, RTOL = 3e-2, 2e-2  # adopted V0 tolerance


def _mk_attn(layer_type, backend="ffpa_flash"):
    """A minimally-constructed Gemma4SelfAttention: only .config + .layer_type are
    read by _attn_dispatch / _gemma4_core_attention."""
    attn = Gemma4SelfAttention.__new__(Gemma4SelfAttention)
    attn.config = types.SimpleNamespace(
        attention_backend=backend, softmax_scale=1.0, sliding_window=512
    )
    attn.layer_type = layer_type
    return attn


def _rand_qkv(s, hd, b=1):
    """query [s, b, np, hd], key/value [s, b, ng, hd] in bf16 on device.

    scale=1.0 (Gemma4 pins softmax_scale=1.0, no 1/sqrt(d)). With raw N(0,1) q/k,
    q.k over head_dim has std ~sqrt(hd) (~22 for hd=512), which makes the softmax
    razor-peaked (near one-hot): a tiny bf16 score difference between the naive
    eager path (bf16 scores) and the fp32-internal kernels flips the argmax key and
    yields a full-magnitude output difference -- a test artifact, not a dispatch bug.
    The real model feeds RMSNorm'd (bounded) q/k, so scale q/k to std(scores) ~1
    (factor hd**-0.25 each) to exercise a representative, well-conditioned softmax
    where the eager oracle and the kernels agree within the S-stable V0 tolerance.
    """
    cond = hd ** -0.25
    q = torch.randn(s, b, NP, hd, device=DEV, dtype=DT) * cond
    k = torch.randn(s, b, NG, hd, device=DEV, dtype=DT) * cond
    v = torch.randn(s, b, NG, hd, device=DEV, dtype=DT)
    return q, k, v


def _compare(name, out, ref):
    out, ref = out.float(), ref.float()
    mad = (out - ref).abs().max().item()
    rel = mad / (ref.abs().max().item() + 1e-6)
    ok = mad <= ATOL or rel <= RTOL
    print(f"[{name}] max_abs={mad:.4e} rel={rel:.4e} -> {'PASS' if ok else 'FAIL'}")
    assert ok, f"{name}: max_abs={mad:.4e} rel={rel:.4e} exceeds tol"
    return mad, rel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("layer_type,hd", [("full", 512), ("sliding", 256)])
def test_dense_dispatch_matches_eager(layer_type, hd):
    """Dense (unpacked) ffpa_flash ~ eager _gemma4_core_attention, per layer type."""
    torch.manual_seed(0)
    s = 1024
    q, k, v = _rand_qkv(s, hd)

    eager = _mk_attn(layer_type, backend="eager")
    kern = _mk_attn(layer_type, backend="ffpa_flash")

    # eager reference must apply the SAME mask the kernel uses: causal for full
    # (ffpa is_causal=True), causal+window for sliding (flash window_size=(511,0)).
    ref = eager._gemma4_core_attention(q, k, v, _ref_mask(layer_type, s, DT))
    out = kern._attn_dispatch(q, k, v, None, None, False)

    assert out.shape == ref.shape == (s, 1, NP, hd)
    _compare(f"dense.{layer_type}", out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("layer_type,hd", [("full", 512), ("sliding", 256)])
def test_packed_dispatch_matches_per_doc_eager(layer_type, hd):
    """Packed 2-doc varlen ~ eager run per-doc (block-diagonal, leak-free)."""
    torch.manual_seed(1)
    lens = [400, 624]
    s = sum(lens)  # 1024
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=DEV, dtype=torch.int32)
    max_len = max(lens)
    psp = PackedSeqParams(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
        qkv_format="thd",
    )

    q, k, v = _rand_qkv(s, hd)  # [s, 1, np, hd]
    kern = _mk_attn(layer_type, backend="ffpa_flash")
    out = kern._attn_dispatch(q, k, v, None, psp, False)
    assert out.shape == (s, 1, NP, hd)

    # eager per-doc reference: run each document standalone (block-diagonal).
    eager = _mk_attn(layer_type, backend="eager")
    ref = torch.empty_like(out)
    for i in range(len(lens)):
        a, e = cu[i].item(), cu[i + 1].item()
        L = e - a
        qi, ki, vi = q[a:e], k[a:e], v[a:e]  # [L, 1, n, hd]
        ref[a:e] = eager._gemma4_core_attention(qi, ki, vi, _ref_mask(layer_type, L, DT))

    _compare(f"packed.{layer_type}", out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_eager_packed_raises_R3():
    """R3: eager + packed_seq_params raises unless allow_eager_packed is set."""
    torch.manual_seed(2)
    s = 32
    q, k, v = _rand_qkv(s, 256)
    cu = torch.tensor([0, 16, 32], device=DEV, dtype=torch.int32)
    psp = PackedSeqParams(
        cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=16, max_seqlen_kv=16, qkv_format="thd"
    )
    eager = _mk_attn("sliding", backend="eager")

    with pytest.raises((AssertionError, ValueError)):
        eager._attn_dispatch(q, k, v, None, psp, False)

    # escape hatch: allow_eager_packed=True must NOT raise (returns eager output).
    out = eager._attn_dispatch(q, k, v, None, psp, True)
    assert out.shape == (s, 1, NP, 256)
    print("[R3] eager+packed raises without escape; accepted with allow_eager_packed=True -> PASS")


def _ref_mask(layer_type, s, dtype):
    """Additive finfo.min mask [1, 1, s, s] matching the kernel for this layer type:
    plain causal for full (ffpa is_causal=True); causal + left-window=511 for sliding
    (window=512 allows kv > q-512, matching flash window_size=(511, 0))."""
    idx = torch.arange(s, device=DEV)
    keep = idx[None, :] <= idx[:, None]  # causal
    if layer_type == "sliding":
        keep = keep & (idx[:, None] - idx[None, :] < 512)
    m = torch.zeros(s, s, device=DEV, dtype=dtype)
    m.masked_fill_(~keep, torch.finfo(dtype).min)
    return m[None, None]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
