#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Offline SFT pre-packing: bin-pack conversations into fixed-length rows.

The Megatron SFT loader (megatron/training/datasets/sft_dataset.py) packs
conversations only *within* a single jsonl row: it splits a row's ``messages``
on system-role messages and concatenates the resulting conversations up to
``--seq-length``. With one conversation per row (the common case) every sample
degenerates to a single right-padded document and no real packing happens.

This script fixes that offline, dataset-agnostically:

1. Stream the input jsonl (never loads the whole file into memory).
2. Tokenize every conversation with the SAME tokenizer path training uses
   (``SFTTokenizer`` / ``tokenize_conversation`` with the chat template selected
   by ``--prompt-format``), so measured lengths are exact, not estimates.
3. First-fit-decreasing bin-pack conversations into bins of ``--seq-length``
   tokens (with optional ``--pad-granularity`` to mirror the loader's
   context-parallel per-conversation padding).
4. Write one jsonl row per bin: concatenated ``messages`` plus a
   ``tools_per_conversation`` list (element i = tools for conversation i),
   which the loader consumes via ``SFTLowLevelDataset.get_tools_per_conversation``.
5. Optionally (``--verify N``) re-read the OUTPUT through the same
   ``datasets``-library path the training loader uses and assert the tokens of
   every conversation are identical to the clean-JSON tokenization -- this
   catches Arrow schema-unification artifacts (null-injection / key reordering)
   that could otherwise shift token counts between packing time and train time.

Rows whose messages do not start with a system-role message cannot be safely
concatenated (the loader would merge them into the previous conversation); they
are passed through unmodified as single-row bins and counted in the stats.

cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-ffpa
python3 examples/gemma4/pack_sft_dataset.py \
  --input  /lustre/fsw/portfolios/llmservice/users/ameyasunilm/datasets/tool_calling_with_execution/tool_calling_past_octopus-05152026-reasoning_on-raw_messages_format.jsonl \
  --output ../code_dev/shared-state/sft_run/tool_calling_past_octopus-05152026-reasoning_on-raw_messages_format_packed_8192.jsonl \
  --tokenizer-model /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/gemma4-playground/weights/gemma-4-E4B-it \
  --prompt-format gemma --seq-length 8192 --num-workers 8 --verify 32

Example (Gemma 4 E4B, inside the training container):

    python3 examples/gemma4/pack_sft_dataset.py \
        --input  /path/to/sft.jsonl \
        --output /path/to/sft_packed_8192.jsonl \
        --tokenizer-model /path/to/gemma-4-E4B-it \
        --prompt-format gemma \
        --seq-length 8192 \
        --verify 32
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _add_repo_to_path() -> None:
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    _apply_nvrx_version_shim()


def _apply_nvrx_version_shim() -> None:
    """Container<->MLM compat: populate nvidia_resiliency_ext.__version__ BEFORE
    megatron imports (mirrors the shim in pretrain_gemma4.py; see there for why)."""
    try:
        import nvidia_resiliency_ext as _nvrx

        need = True
        try:
            from packaging.version import Version as _V

            cur = getattr(_nvrx, "__version__", None)
            need = cur is None or _V(str(cur)) < _V("0.6.0")
        except Exception:
            need = not hasattr(_nvrx, "__version__")
        if need:
            _nvrx.__version__ = "0.6.0"
    except Exception:
        pass


def build_tokenizer(tokenizer_model: str, prompt_format: str):
    """Build the SFT tokenizer exactly as megatron training does.

    Mirrors megatron/core/tokenizers/utils/build_tokenizer.py for
    ``--tokenizer-type SFTTokenizer``: library 'sft' + prompt_format.
    """
    _add_repo_to_path()
    from megatron.core.tokenizers import MegatronTokenizer

    return MegatronTokenizer.from_pretrained(
        tokenizer_path=tokenizer_model,
        metadata_path={"library": "sft"},
        prompt_format=prompt_format,
    )


def strip_none(obj):
    """Recursively drop dict keys whose value is None.

    MUST stay in sync with megatron/training/datasets/sft_dataset.py::_strip_none
    (replicated here so worker processes need no megatron import before the
    tokenizer is built). The training loader applies this to every row before
    tokenizing (datasets/Arrow null-injection cleanup), and real tool-calling
    data carries e.g. ``"content": null`` on tool-call-only assistant turns --
    tokenizing the RAW record instead crashes the chat template with
    'can only concatenate str (not "NoneType")'. All packer-side tokenization
    must therefore go through this stripped view to match train time.
    """
    if isinstance(obj, dict):
        # Preserve tool-call ``arguments`` subtrees verbatim: null values there
        # are SEMANTIC (the ground-truth template renders them as ``key:None``);
        # stripping would drop supervised argument keys from the target.
        return {
            k: (v if k == "arguments" else strip_none(v))
            for k, v in obj.items()
            if v is not None
        }
    if isinstance(obj, list):
        return [strip_none(v) for v in obj]
    return obj


def normalize_tool_call_arguments(record: Dict) -> int:
    """Parse JSON-string ``tool_calls[].function.arguments`` into dicts, in place.

    OpenAI-style data stores arguments as a JSON-ENCODED STRING. The Gemma4 chat
    template branches on the type (sft_tokenizer.py: ``arguments is mapping`` ->
    native ``call:name{key:value,...}`` rendering; ``is string`` -> VERBATIM
    passthrough), so string arguments train the model to emit raw JSON inside
    the call braces -- e.g. ``call:f{{"a": 1}}`` instead of ``call:f{a:1}`` --
    which the eval-side parser cannot decode (observed as ~0 AST scores on
    BFCL). Returns the number of argument strings converted. Strings that do
    not parse to a JSON object are left untouched.
    """
    n_ok = 0
    n_bad = 0
    for message in record.get("messages") or []:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                try:
                    value = json.loads(function["arguments"])
                except (ValueError, TypeError):
                    n_bad += 1  # unparseable -> would render verbatim (broken grammar)
                    continue
                if isinstance(value, dict):
                    function["arguments"] = value
                    n_ok += 1
                else:
                    # Parses to list/scalar/str (incl. double-encoded JSON): the
                    # template would dump it verbatim -- the exact failure mode
                    # this normalization removes. Flag as a record error so the
                    # --on-error policy (abort/drop) handles it loudly.
                    n_bad += 1
    return n_ok, n_bad


def split_conversations(messages: List[Dict]) -> List[List[Dict]]:
    """Split merged messages on system-role boundaries.

    Must stay in sync with SFTDataset._split_conversations
    (megatron/training/datasets/sft_dataset.py).
    """
    conversations: List[List[Dict]] = []
    current: List[Dict] = []
    for msg in messages:
        if msg["role"] == "system":
            if current:
                conversations.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        conversations.append(current)
    return conversations


def round_up(value: int, granularity: int) -> int:
    if granularity <= 1:
        return value
    return -(-value // granularity) * granularity


def _tools_for_conversations(record: Dict, n_conversations: int) -> Optional[List[Any]]:
    """Per-conversation tools for a record: honors an existing
    tools_per_conversation key (already-packed input; makes re-packing
    composable), else replicates the row-level tools. Returns None on a
    length mismatch (caller reports the record as failed)."""
    tools_per_conversation = record.get("tools_per_conversation")
    if tools_per_conversation is not None:
        if len(tools_per_conversation) != n_conversations:
            return None
        return list(tools_per_conversation)
    return [record.get("tools")] * n_conversations


class _Worker:
    """Per-process tokenizer holder for multiprocessing length measurement."""

    tokenizer = None
    args = None
    normalize_tool_args = True

    @classmethod
    def init(cls, tokenizer_model: str, prompt_format: str, normalize_tool_args: bool = True) -> None:
        cls.tokenizer = build_tokenizer(tokenizer_model, prompt_format)
        cls.normalize_tool_args = normalize_tool_args

    @classmethod
    def measure(cls, payload: Tuple[int, str]) -> Tuple[int, List[int], bool, str]:
        """Return (record_idx, per-conversation token lengths, packable, error)."""
        record_idx, line = payload
        try:
            # Tokenize the STRIPPED view -- identical to what the training loader
            # sees (see strip_none). The output file still gets the original line.
            record = json.loads(line)
            if cls.normalize_tool_args:
                _, n_bad_args = normalize_tool_call_arguments(record)
                if n_bad_args:
                    return (
                        record_idx,
                        [],
                        False,
                        f"{n_bad_args} tool-call argument string(s) do not parse to a "
                        "JSON object (list/scalar/double-encoded/unparseable) -- "
                        "verbatim template rendering would train the broken call grammar",
                    )
            record = strip_none(record)
            messages = record["messages"]
            conversations = split_conversations(messages)
            if not conversations:
                return record_idx, [], False, "no conversations"
            tools_list = _tools_for_conversations(record, len(conversations))
            if tools_list is None:
                return (
                    record_idx,
                    [],
                    False,
                    "tools_per_conversation length does not match conversation count",
                )
            # A row is only safely re-packable if its messages start with a
            # system message; otherwise concatenation merges it into the
            # previous conversation in the bin.
            packable = messages[0]["role"] == "system"
            lengths = []
            for conversation, conv_tools in zip(conversations, tools_list):
                tokens, _ = cls.tokenizer.tokenize_conversation(
                    conversation,
                    return_target=True,
                    add_generation_prompt=False,
                    tools=conv_tools,
                )
                lengths.append(len(tokens))
            return record_idx, lengths, packable, ""
        except Exception as exc:  # surface per-record failures without dying
            return record_idx, [], False, f"{type(exc).__name__}: {exc}"


def _index_lines(path: str) -> List[int]:
    """Byte offset of every line start (streaming; no content kept)."""
    offsets = []
    with open(path, "rb") as f:
        pos = 0
        for line in f:
            offsets.append(pos)
            pos += len(line)
    return offsets


def _read_line(path_handle, offset: int) -> str:
    path_handle.seek(offset)
    return path_handle.readline().decode("utf-8")


def measure_all(
    input_path: str,
    offsets: List[int],
    tokenizer_model: str,
    prompt_format: str,
    num_workers: int,
    normalize_tool_args: bool = True,
) -> List[Tuple[int, List[int], bool, str]]:
    """Tokenize every record and return per-conversation lengths."""

    def payloads():
        with open(input_path, "rb") as f:
            for idx in range(len(offsets)):
                yield idx, _read_line(f, offsets[idx])

    results: List[Optional[Tuple[int, List[int], bool, str]]] = [None] * len(offsets)
    if num_workers <= 1:
        _Worker.init(tokenizer_model, prompt_format, normalize_tool_args)
        for payload in payloads():
            res = _Worker.measure(payload)
            results[res[0]] = res
            _progress(res[0] + 1, len(offsets))
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")  # fork is unsafe with HF tokenizers' threads
        with ctx.Pool(
            processes=num_workers,
            initializer=_Worker.init,
            initargs=(tokenizer_model, prompt_format, normalize_tool_args),
        ) as pool:
            done = 0
            for res in pool.imap_unordered(_Worker.measure, payloads(), chunksize=8):
                results[res[0]] = res
                done += 1
                _progress(done, len(offsets))
    print()
    return results  # type: ignore[return-value]


def _progress(done: int, total: int) -> None:
    if done % 16 == 0 or done == total:
        print(f"\r[measure] {done}/{total} records tokenized", end="", flush=True)


def best_fit_decreasing(
    items: List[Tuple[int, int]],  # (record_idx, cost)
    capacity: int,
) -> List[List[int]]:
    """Best-fit-decreasing over whole records, O(N log N)-ish via residual buckets.

    A record's cost is the sum of ``round_up(conv_len, pad_granularity)`` over its
    conversations. This is EXACTLY the loader's accumulation: the running total is
    always a multiple of the granularity, so ``round_up(cum + len) == cum +
    round_up(len)`` -- costs are additive and bin feasibility is a plain sum.

    Residuals are bounded integers in [0, capacity], so open bins are kept in
    per-residual buckets with a small sorted index of nonempty residuals; each
    placement is a bisect + O(#distinct residuals) update instead of a scan over
    all bins (the naive first-fit scan is O(N * bins) -- hours at millions of
    records). Deterministic: items sorted by (cost desc, record_idx), LIFO buckets.
    """
    import bisect

    bins: List[List[int]] = []
    buckets: Dict[int, List[int]] = {}  # residual -> [bin_id, ...] (LIFO)
    avail: List[int] = []  # sorted distinct residuals with nonempty buckets

    for record_idx, cost in sorted(items, key=lambda x: (-x[1], x[0])):
        pos = bisect.bisect_left(avail, cost)
        if pos < len(avail):
            residual = avail[pos]  # tightest bin that fits (best fit)
            bucket = buckets[residual]
            bin_id = bucket.pop()
            if not bucket:
                del buckets[residual]
                avail.pop(pos)
            bins[bin_id].append(record_idx)
            new_residual = residual - cost
        else:
            bin_id = len(bins)
            bins.append([record_idx])
            new_residual = capacity - cost
        if new_residual > 0:
            bucket = buckets.get(new_residual)
            if bucket is None:
                buckets[new_residual] = [bin_id]
                bisect.insort(avail, new_residual)
            else:
                bucket.append(bin_id)
    return bins


def write_output(
    input_path: str,
    output_path: str,
    offsets: List[int],
    bins: List[List[int]],
    passthrough: List[int],
    normalize_tool_args: bool = True,
) -> int:
    """Write packed bins + passthrough rows.

    Unchanged rows keep their line content verbatim (modulo trailing-newline
    shape) UNLESS tool-call argument normalization
    changed them (see normalize_tool_call_arguments) -- the output file must
    contain the normalized form the lengths were measured on. Returns the
    number of normalized argument strings across all written rows.
    """
    n_normalized = 0

    def _single_row_line(fin, record_idx: int) -> str:
        nonlocal n_normalized
        line = _read_line(fin, offsets[record_idx])
        if normalize_tool_args:
            record = json.loads(line)
            n, _ = normalize_tool_call_arguments(record)
            if n:
                n_normalized += n
                return json.dumps(record, ensure_ascii=False) + "\n"
        return line.rstrip("\n") + "\n"

    with open(input_path, "rb") as fin, open(output_path, "w") as fout:
        for bin_records in bins:
            if len(bin_records) == 1:
                fout.write(_single_row_line(fin, bin_records[0]))
                continue
            messages: List[Dict] = []
            tools_per_conversation: List[Any] = []
            for record_idx in bin_records:
                record = json.loads(_read_line(fin, offsets[record_idx]))
                if normalize_tool_args:
                    n_normalized += normalize_tool_call_arguments(record)[0]
                # Derive conversation/tool metadata from the STRIPPED view (what
                # measure/train see); write the unstripped messages.
                meta_record = strip_none(record)
                convs = split_conversations(meta_record["messages"])
                messages.extend(record["messages"])
                tools_per_conversation.extend(
                    _tools_for_conversations(meta_record, len(convs))
                )
            # Always write the key for merged rows (even all-None) so that
            # downstream tooling can distinguish packer-merged rows from
            # passthrough rows.
            row: Dict[str, Any] = {
                "messages": messages,
                "tools_per_conversation": tools_per_conversation,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        for record_idx in passthrough:
            fout.write(_single_row_line(fin, record_idx))
    return n_normalized


def verify_output(
    output_path: str,
    tokenizer,
    seq_length: int,
    pad_granularity: int,
    num_rows: int,
    merged_cap: int = 512,
) -> None:
    """Re-read the OUTPUT through the training loader's datasets/Arrow path and
    assert token identity vs clean-JSON tokenization + capacity feasibility.

    Every packer-merged bin (the only rows where packer correctness is
    load-bearing) is ALWAYS checked; ``num_rows`` only controls how many of the
    remaining single-record/passthrough rows are sampled.
    """
    _add_repo_to_path()
    import numpy as np

    from megatron.training.datasets.sft_dataset import SFTLowLevelDataset

    low_level = SFTLowLevelDataset(output_path)
    total = len(low_level)
    offsets = []
    merged_rows = []
    other_rows = []
    with open(output_path, "rb") as f:
        pos = 0
        for row_idx, line in enumerate(f):
            offsets.append(pos)
            pos += len(line)
            # Cheap classification; a rare false positive (the literal string
            # inside message content) just verifies one extra row.
            if b'"tools_per_conversation"' in line:
                merged_rows.append(row_idx)
            else:
                other_rows.append(row_idx)
    step = max(1, len(other_rows) // num_rows) if num_rows > 0 else 1
    if merged_cap > 0 and len(merged_rows) > merged_cap:
        # Explicit cap for huge outputs (re-tokenizing every merged bin twice is
        # O(days) at millions of rows); evenly-spaced sample, never silent.
        mstep = max(1, len(merged_rows) // merged_cap)
        merged_selected = merged_rows[::mstep]
        print(f"[verify] CAPPED: checking {len(merged_selected)} of "
              f"{len(merged_rows)} merged bins (--verify-merged-cap {merged_cap})")
    else:
        merged_selected = merged_rows
    rows_to_check = sorted(set(merged_selected) | set(other_rows[::step]))
    print(f"[verify] checking {len(merged_selected)} merged bins + "
          f"{len(rows_to_check) - len(merged_selected)} sampled other rows "
          f"(of {total} total)")
    checked = 0
    with open(output_path, "rb") as f:
        for row_idx in rows_to_check:
            # Path A: training view (datasets/Arrow + _strip_none).
            arrow_messages = low_level[row_idx]
            arrow_tools_per_conv = low_level.get_tools_per_conversation(row_idx)
            arrow_tools = low_level.get_tools(row_idx)
            arrow_convs = split_conversations(arrow_messages)
            # Path B: clean JSON view (what the packer measured; stripped like
            # the loader strips -- see strip_none).
            clean = strip_none(json.loads(_read_line(f, offsets[row_idx])))
            clean_convs = split_conversations(clean["messages"])
            clean_tools_per_conv = clean.get("tools_per_conversation")
            clean_tools = clean.get("tools")
            assert len(arrow_convs) == len(clean_convs), (
                f"row {row_idx}: conversation count differs between Arrow "
                f"({len(arrow_convs)}) and clean JSON ({len(clean_convs)})"
            )
            cum = 0
            for conv_idx in range(len(clean_convs)):
                if arrow_tools_per_conv is not None:
                    a_tools = arrow_tools_per_conv[conv_idx]
                else:
                    a_tools = arrow_tools
                if clean_tools_per_conv is not None:
                    c_tools = clean_tools_per_conv[conv_idx]
                else:
                    c_tools = clean_tools
                a_tokens, a_target = tokenizer.tokenize_conversation(
                    arrow_convs[conv_idx],
                    return_target=True,
                    add_generation_prompt=False,
                    tools=a_tools,
                )
                c_tokens, _ = tokenizer.tokenize_conversation(
                    clean_convs[conv_idx],
                    return_target=True,
                    add_generation_prompt=False,
                    tools=c_tools,
                )
                assert list(a_tokens) == list(c_tokens), (
                    f"row {row_idx} conversation {conv_idx}: tokens differ between "
                    f"the Arrow (train-time) and clean-JSON (pack-time) paths -- "
                    f"Arrow schema unification changed the rendered prompt"
                )
                if len(a_tokens) > 0 and not np.any(np.asarray(a_target) != -100):
                    print(f"[verify] WARNING row {row_idx} conversation {conv_idx}: "
                          f"all-masked target (no supervised tokens)")
                cum = round_up(cum + len(a_tokens), pad_granularity)
            if clean_tools_per_conv is not None:
                # Packer-merged row: capacity is guaranteed by construction.
                assert cum <= seq_length, (
                    f"row {row_idx}: packed length {cum} exceeds --seq-length "
                    f"{seq_length}; the loader would truncate this bin"
                )
            elif cum > seq_length:
                # Passthrough row that was already oversize before packing;
                # the loader right-truncates it, exactly as without packing.
                print(f"[verify] note: passthrough row {row_idx} is oversize "
                      f"({cum} > {seq_length}); loader will right-truncate")
            checked += 1
    print(f"[verify] OK: {checked} rows re-checked through the datasets/Arrow path "
          f"(token identity + capacity + non-empty supervision)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="input jsonl (messages[/tools] rows)")
    parser.add_argument("--output", required=True, help="output packed jsonl")
    parser.add_argument("--tokenizer-model", required=True,
                        help="HF tokenizer dir (e.g. the gemma-4-E4B-it weights dir)")
    parser.add_argument("--prompt-format", default="gemma",
                        help="SFTTokenizer prompt format (== --sft-tokenizer-prompt-format)")
    parser.add_argument("--seq-length", type=int, required=True,
                        help="training --seq-length; bins never exceed this many tokens")
    parser.add_argument("--pad-granularity", type=int, default=1,
                        help="reserve per-conversation padding to this multiple "
                             "(set to 2*context_parallel_size when training with CP>1)")
    parser.add_argument("--oversize", choices=["keep", "drop"], default="keep",
                        help="records longer than --seq-length alone: keep as their own "
                             "row (loader right-truncates, as today) or drop")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="tokenization worker processes (1 = in-process)")
    parser.add_argument("--verify", type=int, default=32, metavar="N",
                        help="re-check ~N output rows through the datasets/Arrow "
                             "training path (0 = skip)")
    parser.add_argument("--verify-merged-cap", type=int, default=512,
                        help="max merged bins to re-verify (evenly sampled; 0 = all). "
                             "Re-tokenizing every merged bin is O(days) on huge outputs")
    parser.add_argument("--no-normalize-tool-args", action="store_true",
                        help="disable parsing JSON-string tool_calls[].function.arguments "
                             "into dicts (see normalize_tool_call_arguments: string "
                             "arguments render VERBATIM in the chat template and train "
                             "the model to emit raw JSON inside call braces -> ~0 AST "
                             "eval scores). Normalization is ON by default")
    parser.add_argument("--on-error", choices=["abort", "drop"], default="abort",
                        help="records that fail tokenization: abort (default) or drop "
                             "them from the output with a loud count. NOTE such records "
                             "would crash the training loader identically, so dropping "
                             "is the only trainable option for dirty data")
    args = parser.parse_args()

    if args.pad_granularity == 1:
        print("[pack] NOTE: --pad-granularity 1 assumes context_parallel_size=1 at "
              "train time. For CP>1, pack with --pad-granularity $((2*CP)) -- a "
              "mismatch makes merged rows overflow at train time, which the SFT "
              "loader now rejects with an assertion (no silent truncation).")

    print(f"[pack] indexing {args.input}")
    offsets = _index_lines(args.input)
    print(f"[pack] {len(offsets)} records")

    normalize_tool_args = not args.no_normalize_tool_args
    results = measure_all(
        args.input, offsets, args.tokenizer_model, args.prompt_format,
        args.num_workers, normalize_tool_args=normalize_tool_args,
    )

    errors = [(i, err) for i, _, _, err in results if err]
    failed_ids = set()
    if errors:
        for i, err in errors[:5]:
            print(f"[pack] ERROR record {i}: {err}", file=sys.stderr)
        if args.on_error == "abort":
            raise SystemExit(
                f"{len(errors)} records failed to tokenize; aborting "
                f"(use --on-error drop to exclude them -- they would crash the "
                f"training loader identically)"
            )
        failed_ids = {i for i, _ in errors}
        print(f"[pack] DROPPING {len(failed_ids)} records that failed tokenization "
              f"(--on-error drop); they would crash the training loader identically")

    row_lengths: Dict[int, List[int]] = {}
    packable_items: List[Tuple[int, int]] = []
    passthrough: List[int] = []
    dropped: List[int] = []
    for record_idx, lengths, packable, _ in results:
        if record_idx in failed_ids:
            continue
        row_lengths[record_idx] = lengths
        cum = 0
        for conv_len in lengths:
            cum = round_up(cum + conv_len, args.pad_granularity)
        if cum > args.seq_length:
            if args.oversize == "drop":
                dropped.append(record_idx)
            else:
                passthrough.append(record_idx)
        elif not packable:
            passthrough.append(record_idx)
        else:
            packable_items.append((record_idx, cum))

    bins = best_fit_decreasing(packable_items, args.seq_length)

    n_convs = sum(len(v) for v in row_lengths.values())
    total_tokens = sum(sum(v) for v in row_lengths.values())
    out_rows = len(bins) + len(passthrough)
    fill = [
        sum(sum(row_lengths[r]) for r in b) / args.seq_length for b in bins
    ] or [0.0]
    print(f"[pack] {len(offsets)} records / {n_convs} conversations -> "
          f"{len(bins)} packed rows + {len(passthrough)} passthrough rows "
          f"({len(dropped)} dropped oversize)")
    print(f"[pack] packed-bin fill: mean {sum(fill)/len(fill):.1%}, "
          f"min {min(fill):.1%}, max {max(fill):.1%}")
    print(f"[pack] token utilization vs unpacked: {out_rows} x {args.seq_length} slots "
          f"for {total_tokens} real tokens "
          f"({total_tokens / max(1, out_rows * args.seq_length):.1%} full) "
          f"vs {len(offsets)} rows before packing "
          f"({total_tokens / max(1, len(offsets) * args.seq_length):.1%} full)")

    n_normalized = write_output(
        args.input, args.output, offsets, bins, passthrough,
        normalize_tool_args=normalize_tool_args,
    )
    print(f"[pack] normalized {n_normalized} JSON-string tool-call arguments -> dicts "
          f"(native template rendering); disable with --no-normalize-tool-args")
    print(f"[pack] wrote {args.output}")

    if args.verify > 0:
        tokenizer = build_tokenizer(args.tokenizer_model, args.prompt_format)
        verify_output(
            args.output, tokenizer, args.seq_length, args.pad_granularity,
            args.verify, merged_cap=args.verify_merged_cap,
        )


if __name__ == "__main__":
    main()
