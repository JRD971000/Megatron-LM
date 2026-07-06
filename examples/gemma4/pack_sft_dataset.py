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
  --input  ../code_dev/shared-state/sft_run/octopus_subset_512.jsonl \
  --output ../code_dev/shared-state/sft_run/octopus_subset_512_packed_8192.jsonl \
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

    @classmethod
    def init(cls, tokenizer_model: str, prompt_format: str) -> None:
        cls.tokenizer = build_tokenizer(tokenizer_model, prompt_format)

    @classmethod
    def measure(cls, payload: Tuple[int, str]) -> Tuple[int, List[int], bool, str]:
        """Return (record_idx, per-conversation token lengths, packable, error)."""
        record_idx, line = payload
        try:
            record = json.loads(line)
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
) -> List[Tuple[int, List[int], bool, str]]:
    """Tokenize every record and return per-conversation lengths."""

    def payloads():
        with open(input_path, "rb") as f:
            for idx in range(len(offsets)):
                yield idx, _read_line(f, offsets[idx])

    results: List[Optional[Tuple[int, List[int], bool, str]]] = [None] * len(offsets)
    if num_workers <= 1:
        _Worker.init(tokenizer_model, prompt_format)
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
            initargs=(tokenizer_model, prompt_format),
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


def first_fit_decreasing(
    items: List[Tuple[int, int]],  # (record_idx, effective_row_length)
    capacity: int,
    pad_granularity: int,
    row_lengths: Dict[int, List[int]],
) -> List[List[int]]:
    """FFD over whole records (a record's conversations stay together).

    Bin feasibility replays the loader's accumulation: after each conversation
    the running total is rounded up to ``pad_granularity`` (the loader's
    context-parallel padding); the total must never exceed ``capacity``. The
    round-up accumulation is a left fold, so a bin's load extends incrementally
    from its current load without re-folding the whole bin.
    """

    def extend_load(load: int, record_idx: int) -> int:
        for conv_len in row_lengths[record_idx]:
            load = round_up(load + conv_len, pad_granularity)
        return load

    bins: List[List[int]] = []
    bin_loads: List[int] = []
    # Sort by length desc; ties by record index for determinism.
    for record_idx, _ in sorted(items, key=lambda x: (-x[1], x[0])):
        placed = False
        for bin_idx in range(len(bins)):
            new_load = extend_load(bin_loads[bin_idx], record_idx)
            if new_load <= capacity:
                bins[bin_idx].append(record_idx)
                bin_loads[bin_idx] = new_load
                placed = True
                break
        if not placed:
            bins.append([record_idx])
            bin_loads.append(extend_load(0, record_idx))
    return bins


def write_output(
    input_path: str,
    output_path: str,
    offsets: List[int],
    bins: List[List[int]],
    passthrough: List[int],
) -> None:
    """Write packed bins + passthrough rows (original line, unmodified)."""
    with open(input_path, "rb") as fin, open(output_path, "w") as fout:
        for bin_records in bins:
            if len(bin_records) == 1:
                # Single-record bin: keep the original row byte-identical
                # (avoids touching rows that gained nothing from packing).
                fout.write(_read_line(fin, offsets[bin_records[0]]).rstrip("\n") + "\n")
                continue
            messages: List[Dict] = []
            tools_per_conversation: List[Any] = []
            for record_idx in bin_records:
                record = json.loads(_read_line(fin, offsets[record_idx]))
                convs = split_conversations(record["messages"])
                messages.extend(record["messages"])
                tools_per_conversation.extend(
                    _tools_for_conversations(record, len(convs))
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
            fout.write(_read_line(fin, offsets[record_idx]).rstrip("\n") + "\n")


def verify_output(
    output_path: str,
    tokenizer,
    seq_length: int,
    pad_granularity: int,
    num_rows: int,
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
    rows_to_check = sorted(set(merged_rows) | set(other_rows[::step]))
    print(f"[verify] checking all {len(merged_rows)} merged bins + "
          f"{len(rows_to_check) - len(merged_rows)} sampled other rows "
          f"(of {total} total)")
    checked = 0
    with open(output_path, "rb") as f:
        for row_idx in rows_to_check:
            # Path A: training view (datasets/Arrow + _strip_none).
            arrow_messages = low_level[row_idx]
            arrow_tools_per_conv = low_level.get_tools_per_conversation(row_idx)
            arrow_tools = low_level.get_tools(row_idx)
            arrow_convs = split_conversations(arrow_messages)
            # Path B: clean JSON view (what the packer measured).
            clean = json.loads(_read_line(f, offsets[row_idx]))
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
    args = parser.parse_args()

    if args.pad_granularity == 1:
        print("[pack] NOTE: --pad-granularity 1 assumes context_parallel_size=1 at "
              "train time. For CP>1, pack with --pad-granularity $((2*CP)) -- a "
              "mismatch makes merged rows overflow at train time, which the SFT "
              "loader now rejects with an assertion (no silent truncation).")

    print(f"[pack] indexing {args.input}")
    offsets = _index_lines(args.input)
    print(f"[pack] {len(offsets)} records")

    results = measure_all(
        args.input, offsets, args.tokenizer_model, args.prompt_format, args.num_workers
    )

    errors = [(i, err) for i, _, _, err in results if err]
    if errors:
        for i, err in errors[:5]:
            print(f"[pack] ERROR record {i}: {err}", file=sys.stderr)
        raise SystemExit(f"{len(errors)} records failed to tokenize; aborting")

    row_lengths: Dict[int, List[int]] = {}
    packable_items: List[Tuple[int, int]] = []
    passthrough: List[int] = []
    dropped: List[int] = []
    for record_idx, lengths, packable, _ in results:
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

    bins = first_fit_decreasing(
        packable_items, args.seq_length, args.pad_granularity, row_lengths
    )

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

    write_output(args.input, args.output, offsets, bins, passthrough)
    print(f"[pack] wrote {args.output}")

    if args.verify > 0:
        tokenizer = build_tokenizer(args.tokenizer_model, args.prompt_format)
        verify_output(
            args.output, tokenizer, args.seq_length, args.pad_granularity, args.verify
        )


if __name__ == "__main__":
    main()
