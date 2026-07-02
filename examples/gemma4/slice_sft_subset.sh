#!/bin/bash
# ============================================================================
# Slice a small, FIXED subset of the ~32GB tool-calling SFT jsonl for V4/V5.
#
# The full dataset (34.8GB) must NEVER be loaded whole -- SFTLowLevelDataset does
# datasets.load_dataset("json", data_files=...) which would blow up memory. Each
# jsonl line is one packable record ("messages" list, optional "tools"), so
# head -N lines is a valid, self-contained subset. Deterministic (head, not shuffle)
# so both backend runs see byte-identical data.
#
# Usage:  bash examples/gemma4/slice_sft_subset.sh [N_LINES]
#   N_LINES defaults to 256. Writes ${OUT} (skips if already present + non-empty).
# ============================================================================
set -eu

N=${1:-${N_LINES:-256}}
SRC=${SFT_SRC:-/lustre/fsw/portfolios/llmservice/users/ameyasunilm/datasets/tool_calling_with_execution/tool_calling_past_octopus-05152026-reasoning_on-raw_messages_format.jsonl}
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm
OUT_DIR=${OUT_DIR:-${ROOT}/code_dev/shared-state/ffpa_sft_run}
OUT=${OUT:-${OUT_DIR}/sft_subset_${N}.jsonl}

mkdir -p "${OUT_DIR}"

if [ ! -f "${SRC}" ]; then
  echo "ERROR: source jsonl not found: ${SRC}" >&2
  exit 1
fi

if [ -s "${OUT}" ]; then
  echo "[slice] reuse existing subset: ${OUT} ($(wc -l < "${OUT}") lines, $(du -h "${OUT}" | cut -f1))"
  exit 0
fi

echo "[slice] head -n ${N} ${SRC} -> ${OUT}"
head -n "${N}" "${SRC}" > "${OUT}"
echo "[slice] done: $(wc -l < "${OUT}") lines, $(du -h "${OUT}" | cut -f1)"
echo "${OUT}"
