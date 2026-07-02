"""Two-backend parity harness for Gemma 4 E4B (V1/V2/V3).

Compares the opt-in ``ffpa_flash`` attention backend (FFPA for full head_dim=512
layers + FlashAttention for sliding head_dim=256 layers) against the EAGER oracle
on the SAME converted checkpoint, matching the tolerance / greedy bars recorded
in V_RESULTS.md. The eager path is the STAGE-0 bitwise-verified oracle; the new
backend is tolerance-matched against it.

Modes:
  * ``v1`` forward-logit parity (unpacked): FIXED_TOKENS + >=3 chat texts, run under
    the test backend, compare max-abs logit diff vs eager + assert greedy identical.
    R2: the whole-model forward exercises BOTH the FFPA full layers and the Flash
    sliding layers, all pinned to config.softmax_scale=1.0 (asserted).
  * ``v2`` packed no-leak: pack N conversations with cu_seqlens (PackedSeqParams,
    qkv_format="thd", NO intra-pack padding), compare each doc's per-token logits vs
    that SAME doc run STANDALONE (unpacked). Also DEMONSTRATE that eager+packed leaks
    (via I1's ``allow_eager_packed`` escape kwarg): per-doc diff grows / greedy differs.
  * ``v3`` greedy generation: N-token greedy on >=3 prompts; assert ffpa_flash tokens
    == eager tokens EXACTLY (bf16-immune argmax control).

Import gemma4_common FIRST (nvrx shim + sys.path for THIS worktree's megatron).
The dispatch + packed plumbing live in megatron/core (tasks I1/I2). This harness
only SETS config.attention_backend. Where a mode needs the not-yet-merged dispatch,
it raises a clear "AWAITS I1/I2" message instead of silently mis-testing.

Run with the container python (nemo.26.06), e.g.:
    python3 examples/gemma4/parity_backend.py --mode v1 --backend ffpa_flash
    python3 examples/gemma4/parity_backend.py --mode v1 --backend eager   # dry-run: diff must be 0
    python3 examples/gemma4/parity_backend.py --mode v3 --backend ffpa_flash --gen-tokens 50
"""
import argparse
import os
import sys

# gemma4_common FIRST: nvrx version shim + inserts THIS worktree root on sys.path
# BEFORE any megatron import (skill gotcha #10).
import gemma4_common  # noqa: F401  (import-for-side-effects: must be first)
import torch
from gemma4_common import (
    _apply_attention_backend,
    _ensure_ffpa_on_path,
    _init_distributed,
    _make_config,
)

HF_WEIGHTS = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "gemma4-playground/weights/gemma-4-E4B-it"
)
# Flat distcp checkpoint used by the parity scripts (HYBRID-gemma4-mlm).
MLM_CKPT = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "Gemma4_mlm/code_dev/shared-state/implementations/HYBRID-gemma4-mlm/mlm_ckpt"
)

# STAGE-0 fixed ids (mlm_to_hf_convert.py). Leading <bos>=2, trailing <eos>=3.
FIXED_TOKENS = [2, 651, 1234, 99, 17, 8, 200, 9000, 42, 3]

# >=3 chat-templated texts (V1/V3). Short so greedy gen stays < max_sequence_length.
DEFAULT_TEXTS = [
    "What is the capital of France? Answer in one word.",
    "Explain gravity in one sentence.",
    "List three prime numbers.",
    "Translate 'good morning' to Spanish.",
]

# Adopted tolerance (see module docstring / V_RESULTS.md). max-abs OR rel gate.
TOL_LOGIT_MAXABS = 3e-2
TOL_LOGIT_REL = 2e-2


# --------------------------------------------------------------------------- #
# tokenization
# --------------------------------------------------------------------------- #
class _RawDecoder:
    """Minimal decode shim around a tokenizers.Tokenizer (matches text_parity)."""

    def __init__(self, t):
        self._t = t

    def decode(self, ids):
        return self._t.decode(list(ids))


def _tokenizer():
    """Return a tokenizer bundle for chat-templating.

    Preferred: HF AutoTokenizer + its real chat template. Fallback (some container
    transformers builds trip an ``extra_special_tokens``-as-list bug at construction,
    see text_parity.py): the raw ``tokenizers`` lib + a manual Gemma chat template.
    Parity is unaffected -- the SAME ids feed both backends either way.
    """
    try:
        from transformers import AutoTokenizer

        return ("hf", AutoTokenizer.from_pretrained(HF_WEIGHTS))
    except Exception as e:
        print(f"[tok] AutoTokenizer failed ({type(e).__name__}: {e}); "
              "falling back to raw tokenizers + manual Gemma template", flush=True)
        from tokenizers import Tokenizer

        return ("raw", Tokenizer.from_file(os.path.join(HF_WEIGHTS, "tokenizer.json")))


def _decoder(bundle):
    mode, tok = bundle
    return tok if mode == "hf" else _RawDecoder(tok)


def _extract_ids_1d(res):
    """Extract a 1-D python list of token ids from any apply_chat_template return type:
    torch.Tensor, dict/BatchEncoding (.input_ids), tokenizers.Encoding (.ids), or list."""
    import numpy as _np

    if isinstance(res, torch.Tensor):
        arr = res.tolist()
    elif isinstance(res, dict) or hasattr(res, "input_ids"):
        v = res["input_ids"]
        return _extract_ids_1d(v)
    elif hasattr(res, "ids"):  # tokenizers.Encoding
        arr = list(res.ids)
    else:
        arr = _np.asarray(res).tolist()
    # squeeze a leading batch dim of 1 if present ([[...]] -> [...])
    if len(arr) > 0 and isinstance(arr[0], (list, tuple)):
        arr = arr[0]
    return [int(x) for x in arr]


def _chat_ids(bundle, text):
    """Chat-template tokenize a single user turn -> LongTensor [1, S]."""
    mode, tok = bundle
    if mode == "hf":
        messages = [{"role": "user", "content": text}]
        # nemo.26.06's transformers returns a tokenizers.Encoding (not a tensor) from
        # apply_chat_template; extract ids robustly across all return types (tensor /
        # dict|BatchEncoding / tokenizers.Encoding / list), mirroring sft_tokenizer.
        res = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        seq = _extract_ids_1d(res)
        return torch.tensor([seq], dtype=torch.long)
    # raw fallback: standard Gemma chat format, ensure leading <bos>=2.
    prompt = f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
    seq = tok.encode(prompt).ids
    if not seq or seq[0] != 2:
        seq = [2] + seq
    return torch.tensor([seq], dtype=torch.long)


def _build_inputs(include_fixed=True):
    """Return list of (name, ids[1,S]) for V1/V3."""
    tok = _tokenizer()
    inputs = []
    if include_fixed:
        inputs.append(("FIXED_TOKENS", torch.tensor([FIXED_TOKENS], dtype=torch.long)))
    for t in DEFAULT_TEXTS:
        inputs.append((t[:40], _chat_ids(tok, t)))
    return tok, inputs


# --------------------------------------------------------------------------- #
# model build + checkpoint load
# --------------------------------------------------------------------------- #
def _distributed_ready():
    import megatron.core.parallel_state as ps

    if not (torch.distributed.is_initialized() and ps.model_parallel_is_initialized()):
        _init_distributed()


def _load_model(backend, ckpt=MLM_CKPT):
    """Build a bf16/cuda Gemma4Model on the given backend + load the converted ckpt.

    Returns (model, field_present). ``field_present`` is False when this tree does
    not yet carry config.attention_backend (I1/I2 unmerged) -> ffpa_flash will run
    the eager path until the merge, which the caller reports.
    """
    from megatron.core import dist_checkpointing
    from megatron.core.models.gemma4.gemma4_layer_specs import get_gemma4_layer_local_spec
    from megatron.core.models.gemma4.gemma4_model import Gemma4Model

    if backend == "ffpa_flash":
        if not _ensure_ffpa_on_path():
            print("[warn] ffpa_install dir not found; ffpa_attn import will fail", flush=True)

    _distributed_ready()
    config = _make_config()
    config, field_present = _apply_attention_backend(config, backend)

    spec = get_gemma4_layer_local_spec(config)
    model = Gemma4Model(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=262144,
        max_sequence_length=512,
    ).bfloat16().cuda()
    model.eval()

    print(f"[load] backend={backend} ckpt={ckpt}", flush=True)
    sharded_sd = model.sharded_state_dict()
    loaded = dist_checkpointing.load(sharded_sd, ckpt)
    model.load_state_dict(loaded, strict=False)
    return model, field_present


def _report_backend(backend, field_present):
    if backend == "ffpa_flash" and not field_present:
        print(
            "\n[AWAITS I1/I2] config.attention_backend is NOT a field in this tree yet.\n"
            "  The ffpa_flash dispatch is not merged, so this run exercises the EAGER path.\n"
            "  Plumbing (backend flag set, ckpt load, tokenization, comparison) is validated;\n"
            "  full ffpa_flash parity is gated on the I1+I2 merge (STAGE 5).\n",
            flush=True,
        )
    cfg = _make_config()
    n_full = len(cfg.full_attention_layers)
    n_total = cfg.num_layers
    print(
        f"[R2] softmax_scale must be 1.0 for the sliding(flash) + full(ffpa) kernels: "
        f"config.softmax_scale={cfg.softmax_scale}  "
        f"(sliding layers={n_total - n_full}, full layers={n_full})",
        flush=True,
    )
    assert cfg.softmax_scale == 1.0, "R2 violated: softmax_scale != 1.0"


# --------------------------------------------------------------------------- #
# comparison helpers
# --------------------------------------------------------------------------- #
def _logit_gate(diff_maxabs, rel):
    return (diff_maxabs <= TOL_LOGIT_MAXABS) or (rel <= TOL_LOGIT_REL)


def _compare(name, ref, test):
    """ref/test: logits [1,S,V] fp32 cpu. Returns dict of stats + pass flag."""
    diff = (ref - test).abs()
    maxabs = diff.max().item()
    denom = ref.abs().max().item() + 1e-12
    rel = maxabs / denom
    ref_g = ref.argmax(-1)
    test_g = test.argmax(-1)
    greedy_ok = torch.equal(ref_g, test_g)
    passed = _logit_gate(maxabs, rel) and greedy_ok
    print(
        f"  [{name}] max-abs={maxabs:.4e} rel={rel:.4e} "
        f"greedy={'MATCH' if greedy_ok else 'DIFFER'} -> {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return {"name": name, "maxabs": maxabs, "rel": rel, "greedy_ok": greedy_ok, "pass": passed}


# --------------------------------------------------------------------------- #
# V1 — forward-logit parity (unpacked)
# --------------------------------------------------------------------------- #
def run_v1(args):
    print("\n===== V1: forward-logit parity (test backend vs eager oracle) =====")
    _tok, inputs = _build_inputs(include_fixed=True)

    # Oracle first (eager), collect logits, then free before building the test model.
    eager_model, _ = _load_model("eager")
    ref_logits = {}
    with torch.no_grad():
        for name, ids in inputs:
            ref_logits[name] = eager_model(ids.cuda()).float().cpu()
    del eager_model
    torch.cuda.empty_cache()

    test_model, field_present = _load_model(args.backend)
    _report_backend(args.backend, field_present)
    results = []
    with torch.no_grad():
        for name, ids in inputs:
            test_logits = test_model(ids.cuda()).float().cpu()
            results.append(_compare(name, ref_logits[name], test_logits))
    del test_model
    torch.cuda.empty_cache()

    ok = all(r["pass"] for r in results)
    print(f"\nV1 verdict: {'PASS' if ok else 'FAIL'} "
          f"(TOL_LOGIT: max-abs<={TOL_LOGIT_MAXABS} OR rel<={TOL_LOGIT_REL}; greedy exact)")
    return ok


# --------------------------------------------------------------------------- #
# V2 — packed no-leak (per-doc vs standalone) + eager-leak demonstration
# --------------------------------------------------------------------------- #
def _build_pack(tok, texts):
    """Pack chat-templated conversations with NO intra-pack padding (R4).

    Returns:
      packed_ids [1, T], cu_seqlens int32 [n+1] on cuda (cumulative), max_seqlen,
      position_ids [1, T] RESET per doc (concat arange(len) — R1 source of truth for
      parity scripts), and per-doc (start, end) spans into T.
    """
    per_doc = [_chat_ids(tok, t)[0] for t in texts]  # list of 1-D LongTensors
    lens = [d.numel() for d in per_doc]
    packed = torch.cat(per_doc).unsqueeze(0)  # [1, T]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()),
                      dtype=torch.int32, device="cuda")
    pos = torch.cat([torch.arange(n, dtype=torch.long) for n in lens]).unsqueeze(0)
    spans, off = [], 0
    for n in lens:
        spans.append((off, off + n))
        off += n
    return packed, cu, max(lens), pos, spans


def _packed_seq_params(cu, max_seqlen):
    from megatron.core.packed_seq_params import PackedSeqParams

    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=int(max_seqlen),
        max_seqlen_kv=int(max_seqlen),
    )


def _forward_packed(model, packed_ids, pos, psp, allow_eager_packed=None):
    """Call the model on a b=1 THD pack. allow_eager_packed forwards I1's escape
    kwarg (only for the eager-leak negative control)."""
    kw = dict(position_ids=pos.cuda(), packed_seq_params=psp)
    if allow_eager_packed is not None:
        kw["allow_eager_packed"] = allow_eager_packed
    return model(packed_ids.cuda(), **kw).float().cpu()


def run_v2(args):
    print("\n===== V2: packed no-leak (per-doc vs standalone) =====")
    tok = _tokenizer()
    texts = DEFAULT_TEXTS[: max(2, args.num_packed)]
    packed_ids, cu, max_seqlen, pos, spans = _build_pack(tok, texts)
    print(f"  packed {len(texts)} docs, cu_seqlens={cu.tolist()}, T={packed_ids.shape[1]}")

    # Standalone per-doc logits under the TEST backend, run through the SAME varlen
    # kernel as the pack (each doc as its own single-segment pack). This isolates
    # cross-document LEAKAGE from varlen-vs-dense kernel precision: on this peaked
    # model (max_softmax_prob~1.0) a dense-vs-varlen rounding difference alone is ~5%
    # rel, which would swamp the leak signal. Multi-doc-pack doc-i vs single-doc-pack
    # doc-i uses the identical kernel/precision, so any residual diff is pure leakage.
    test_model, field_present = _load_model(args.backend)
    _report_backend(args.backend, field_present)
    standalone = []
    with torch.no_grad():
        for t in texts:
            ids1 = _chat_ids(tok, t)  # [1, n]
            n1 = ids1.shape[1]
            cu1 = torch.tensor([0, n1], dtype=torch.int32, device="cuda")
            pos1 = torch.arange(n1, dtype=torch.long).unsqueeze(0)
            psp1 = _packed_seq_params(cu1, n1)
            standalone.append(_forward_packed(test_model, ids1, pos1, psp1))

    # Packed forward under the TEST backend.
    try:
        with torch.no_grad():
            psp = _packed_seq_params(cu, max_seqlen)
            packed_logits = _forward_packed(test_model, packed_ids, pos, psp)
    except TypeError as e:
        print(f"\n[AWAITS I1/I2] Gemma4Model.forward does not accept packed_seq_params yet "
              f"({e}). V2 packed path is gated on the I2 plumbing + I1 varlen dispatch merge. "
              f"Packing/cu_seqlens/position-reset construction above is validated.", flush=True)
        del test_model
        torch.cuda.empty_cache()
        return None

    results = []
    for (s, e), sa in zip(spans, standalone):
        results.append(_compare(f"doc[{s}:{e}]", sa, packed_logits[:, s:e, :]))
    del test_model
    torch.cuda.empty_cache()
    ok = all(r["pass"] for r in results)
    print(f"\nV2 (packed no-leak) verdict: {'PASS' if ok else 'FAIL'} "
          f"(per-doc PER-TOKEN max-abs<={TOL_LOGIT_MAXABS} OR rel<={TOL_LOGIT_REL}; greedy exact)")

    # ------ negative control: the EAGER backend leaks across documents --------- #
    # The eager additive-mask path ignores cu_seqlens: on a concatenated pack it uses
    # a full causal mask over T, so doc-i (i>=1) attends to earlier documents. We show
    # this WITHIN the eager backend (no backend confound): doc-i run inside the pack vs
    # doc-i run ALONE. No packed_seq_params is passed (so R3 does not fire and eager
    # runs its normal full-mask path — exactly the naive-packing failure mode). Reset
    # positions are supplied so RoPE is per-doc-correct and ONLY the attention leak shows.
    print("\n----- V2 negative control: eager (full-mask) leaks across documents -----")
    eager_model, _ = _load_model("eager")
    with torch.no_grad():
        eager_alone = []
        for t in texts:
            ids1 = _chat_ids(tok, t)
            eager_alone.append(eager_model(ids1.cuda()).float().cpu())
        eager_packed = eager_model(packed_ids.cuda(), position_ids=pos.cuda()).float().cpu()
    leaks = []
    for (s, e), ea in zip(spans, eager_alone):
        d = (ea - eager_packed[:, s:e, :]).abs().max().item()
        g = torch.equal(ea.argmax(-1), eager_packed[:, s:e, :].argmax(-1))
        leaks.append((d, g))
        print(f"  eager doc[{s}:{e}] alone-vs-in-pack max-abs={d:.4e} greedy={'MATCH' if g else 'DIFFER'}")
    leaked = any((not g) or (d > 1.0) for d, g in leaks[1:])  # docs 2+ should diverge hugely
    print(f"  eager-leak demonstrated: {leaked} (docs 2+ diverge from their standalone run)")
    del eager_model
    torch.cuda.empty_cache()
    return ok


# --------------------------------------------------------------------------- #
# V3 — greedy generation parity
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _greedy(model, ids, n_new):
    ids = ids.cuda()
    for _ in range(n_new):
        logits = model(ids)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids[0].tolist()


def run_v3(args):
    print(f"\n===== V3: greedy generation parity ({args.gen_tokens} tokens) =====")
    tok, inputs = _build_inputs(include_fixed=False)  # chat prompts only
    prompts = inputs[:3] if len(inputs) >= 3 else inputs

    eager_model, _ = _load_model("eager")
    eager_seqs = {name: _greedy(eager_model, ids, args.gen_tokens) for name, ids in prompts}
    del eager_model
    torch.cuda.empty_cache()

    test_model, field_present = _load_model(args.backend)
    _report_backend(args.backend, field_present)
    all_ok = True
    for name, ids in prompts:
        test_seq = _greedy(test_model, ids, args.gen_tokens)
        ref_seq = eager_seqs[name]
        ok = ref_seq == test_seq
        all_ok = all_ok and ok
        first_div = next((i for i, (a, b) in enumerate(zip(ref_seq, test_seq)) if a != b), None)
        print(f"  [{name}] {'EXACT' if ok else f'DIVERGE@{first_div}'} "
              f"(len={len(test_seq)})")
    del test_model
    torch.cuda.empty_cache()
    print(f"\nV3 verdict: {'PASS' if all_ok else 'FAIL'} (exact greedy token match)")
    return all_ok


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["v1", "v2", "v3"], required=True)
    ap.add_argument("--backend", choices=["eager", "ffpa_flash"], default="ffpa_flash",
                    help="Test backend compared against the eager oracle. "
                         "eager -> dry-run (logit diff must be 0).")
    ap.add_argument("--mlm-src", default=None,
                    help="Megatron-LM source tree to import (default = THIS worktree "
                         "root, inserted by gemma4_common). Set to test another tree.")
    ap.add_argument("--ffpa-install", default=None,
                    help="Override the ffpa_attn install dir (default = gemma4_common.FFPA_INSTALL).")
    ap.add_argument("--gen-tokens", type=int, default=50, help="V3 greedy tokens to generate.")
    ap.add_argument("--num-packed", type=int, default=3, help="V2 conversations per pack.")
    args = ap.parse_args()

    if args.mlm_src:
        # Prepend so THIS overrides gemma4_common's default worktree-root insertion.
        if args.mlm_src not in sys.path:
            sys.path.insert(0, args.mlm_src)
        print(f"[mlm-src] importing megatron from {args.mlm_src}", flush=True)
    if args.ffpa_install:
        _ensure_ffpa_on_path(args.ffpa_install)

    print(f"MODE={args.mode}  BACKEND={args.backend}  "
          f"MLM_SRC={args.mlm_src or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(gemma4_common.__file__))))}")

    ok = {"v1": run_v1, "v2": run_v2, "v3": run_v3}[args.mode](args)
    if ok is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
