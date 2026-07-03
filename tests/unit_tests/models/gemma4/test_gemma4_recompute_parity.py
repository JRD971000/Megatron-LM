# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Fwd+bwd parity for Gemma4 activation recompute (recompute_granularity='full').

The Gemma4TransformerBlock owns a cross-layer KV bus (producer layers 22/23 write
post-norm/post-RoPE k/v; borrower layers 24-41 read it), so the base block's
``_checkpointed_forward`` cannot be reused. ``_recompute_layers`` adds per-layer
checkpointing that THREADS the bus through the checkpoint boundary (producer k/v as
checkpoint OUTPUTS, borrower k/v as explicit checkpoint INPUTS) so borrower->producer
gradients survive recompute. This test proves that adding recompute changes NOTHING
numerically: same weights + same inputs, recompute OFF vs ON must give the same logits
(forward) and the same parameter gradients (backward).

Gates:
  * a monkeypatched counter proves recompute actually ran and is PER-LAYER
    (tensor_parallel.checkpoint called exactly num_layers times);
  * eager (unpacked) is the deterministic PRIMARY gate: logits + every param grad
    match to ~bitwise (tight tol);
  * ffpa_flash packed (the real SFT varlen path) + ffpa_flash dense match within the
    V0 kernel-noise tolerance (flash/ffpa bwd use atomics -> not bitwise).

Import gemma4_common FIRST (nvrx __version__ shim + sys.path) before any megatron import.
Needs a GPU; the ffpa_flash cases additionally need ffpa_attn (PYTHONPATH=ffpa_install)
and flash_attn in the container.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "examples", "gemma4"))
import gemma4_common  # noqa: F401,E402  (nvrx shim + sys.path, before megatron)

from gemma4_common import (  # noqa: E402
    _apply_attention_backend,
    _ensure_ffpa_on_path,
    _init_distributed,
    _make_config,
)

SMALL_VOCAB = 512
NUM_LAYERS = 42  # keep 42 so producers (22 sliding / 23 full) + borrowers (24-41) exist
# eager: deterministic -> ~bitwise. ffpa/flash: atomic bwd -> V0 kernel-noise tol.
TOL_EAGER = dict(atol=5e-4, rtol=2e-3)
TOL_KERNEL = dict(atol=3e-2, rtol=2e-2)

_INIT_DONE = False


def _build(backend):
    global _INIT_DONE
    if backend == "ffpa_flash":
        _ensure_ffpa_on_path()
    if not _INIT_DONE:
        _init_distributed()  # TP=PP=CP=1 + model_parallel_cuda_manual_seed(123)
        _INIT_DONE = True

    from megatron.core.models.gemma4.gemma4_layer_specs import get_gemma4_layer_local_spec
    from megatron.core.models.gemma4.gemma4_model import Gemma4Model

    config = _make_config(num_layers=NUM_LAYERS, hidden=256, ffn=512)
    config, _ = _apply_attention_backend(config, backend)
    config.vocab_size_per_layer_input = SMALL_VOCAB  # PLE table matches the small vocab
    assert config.softmax_scale == 1.0

    spec = get_gemma4_layer_local_spec(config)
    model = Gemma4Model(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=SMALL_VOCAB,
        max_sequence_length=512,
    ).bfloat16().cuda()
    model.train()
    return model


class _CheckpointSpy:
    """Wrap tensor_parallel.checkpoint to count calls (proves recompute executed)."""

    def __init__(self):
        import megatron.core.tensor_parallel as tp

        self._tp = tp
        self._orig = tp.checkpoint
        self.count = 0

    def __enter__(self):
        def spy(fn, dsa, *args):
            self.count += 1
            return self._orig(fn, dsa, *args)

        self._tp.checkpoint = spy
        return self

    def __exit__(self, *a):
        self._tp.checkpoint = self._orig


def _run(model, input_ids, position_ids, packed_seq_params, recompute):
    cfg = model.decoder.config  # the Gemma4TransformerBlock's config
    if recompute:
        cfg.recompute_granularity = "full"
        cfg.recompute_method = "uniform"
        cfg.recompute_num_layers = 1
    else:
        cfg.recompute_granularity = None

    model.zero_grad(set_to_none=True)
    torch.manual_seed(1234)  # dropout is 0, but keep RNG identical across OFF/ON
    kw = {}
    if packed_seq_params is not None:
        kw["packed_seq_params"] = packed_seq_params
    with _CheckpointSpy() as spy:
        logits = model(input_ids, position_ids=position_ids, attention_mask=None, **kw)
        # Deterministic scalar loss straight from logits (avoids any CE-path variance).
        loss = logits.float().pow(2).mean()
        loss.backward()
    grads = {
        n: p.grad.detach().float().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return logits.detach().float().clone(), grads, spy.count


def _worst(a, b):
    d = (a - b).abs()
    mad = d.max().item()
    rel = mad / (b.abs().max().item() + 1e-12)
    return mad, rel


def _check(off, on, tol, tag):
    logits_off, grads_off, cnt_off = off
    logits_on, grads_on, cnt_on = on

    # recompute must have actually run, per-layer, only in the ON pass.
    assert cnt_off == 0, f"{tag}: recompute OFF still called checkpoint {cnt_off}x"
    assert cnt_on == NUM_LAYERS, (
        f"{tag}: expected {NUM_LAYERS} per-layer checkpoints, got {cnt_on}"
    )

    lm, lr = _worst(logits_on, logits_off)
    print(f"[{tag}] logits max_abs={lm:.3e} rel={lr:.3e}")
    assert lm <= tol["atol"] or lr <= tol["rtol"], f"{tag}: logits diverged {lm:.3e}/{lr:.3e}"

    assert grads_off.keys() == grads_on.keys()
    worst_name, worst_mad, worst_rel = None, 0.0, 0.0
    for n in grads_off:
        mad, rel = _worst(grads_on[n], grads_off[n])
        ok = mad <= tol["atol"] or rel <= tol["rtol"]
        assert ok, f"{tag}: grad '{n}' diverged max_abs={mad:.3e} rel={rel:.3e}"
        if rel > worst_rel:
            worst_name, worst_mad, worst_rel = n, mad, rel
    print(
        f"[{tag}] {len(grads_off)} grads OK; worst='{worst_name}' "
        f"max_abs={worst_mad:.3e} rel={worst_rel:.3e} -> PASS"
    )


def _unpacked_inputs(b=2, s=48):
    torch.manual_seed(0)
    ids = torch.randint(0, SMALL_VOCAB, (b, s), dtype=torch.long, device="cuda")
    pos = torch.arange(s, device="cuda").unsqueeze(0).expand(b, -1).contiguous()
    return ids, pos, None


def _packed_inputs(doc_lens=(20, 28)):
    from megatron.core.models.gemma4.gemma4_model import _reset_position_ids_from_cu_seqlens
    from megatron.core.packed_seq_params import PackedSeqParams

    torch.manual_seed(0)
    T = sum(doc_lens)
    ids = torch.randint(0, SMALL_VOCAB, (1, T), dtype=torch.long, device="cuda")
    cu = torch.tensor(
        [0] + torch.tensor(doc_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
    )
    m = int(max(doc_lens))
    psp = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=m,
        max_seqlen_kv=m,
    )
    pos = _reset_position_ids_from_cu_seqlens(cu).unsqueeze(0).cuda()
    return ids, pos, psp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_recompute_parity_eager_unpacked():
    """PRIMARY deterministic gate: eager unpacked, recompute OFF vs ON -> ~bitwise."""
    model = _build("eager")
    ids, pos, _ = _unpacked_inputs()
    off = _run(model, ids, pos, None, recompute=False)
    on = _run(model, ids, pos, None, recompute=True)
    _check(off, on, TOL_EAGER, "eager.unpacked")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_recompute_parity_ffpa_dense():
    """ffpa_flash dense (unpacked), recompute OFF vs ON within V0 kernel tol."""
    model = _build("ffpa_flash")
    ids, pos, _ = _unpacked_inputs(b=1, s=48)
    off = _run(model, ids, pos, None, recompute=False)
    on = _run(model, ids, pos, None, recompute=True)
    _check(off, on, TOL_KERNEL, "ffpa.dense")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_recompute_parity_ffpa_packed():
    """ffpa_flash packed varlen (the real SFT path), recompute OFF vs ON within V0 tol."""
    model = _build("ffpa_flash")
    ids, pos, psp = _packed_inputs()
    off = _run(model, ids, pos, psp, recompute=False)
    on = _run(model, ids, pos, psp, recompute=True)
    _check(off, on, TOL_KERNEL, "ffpa.packed")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
