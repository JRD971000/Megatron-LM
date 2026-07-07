#!/bin/bash
# ============================================================================
# Gemma 4 E4B INTERACTIVE SFT run with the FFPA/Flash attention backend.
#
# This is the FFPA sibling of gemma4_e4b_sft_interactive.sh. It runs the SAME
# SFT setup (real tool-calling data + Gemma4 `gemma` masking, base MLM ckpt) but
# on the FFPA branch's two-backend attention (FFPA for full head_dim=512 layers,
# FlashAttention for sliding head_dim=256 layers), which computes attention in
# O(seq) memory -- so it fits long sequences (e.g. SEQLEN=8192) where the eager
# backend OOMs on the quadratic score matrix.
#
# Run this *inside* an already-allocated interactive container shell, e.g.:
#
#   srun -A coreai_dlalgo_genai -p interactive --nodes=1 --gpus-per-node=8 \
#        --container-image=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/containers/nemo.26.06.sqsh \
#        --container-mounts=/lustre:/lustre,/home/ataghibakhsh:/home/ataghibakhsh \
#        --no-container-mount-home --pty bash
#   # then, from the prompt inside the container:
#   bash code_dev/scripts/gemma4_e4b_sft_ffpa_interactive.sh
#
# THREE things differ from the eager script (everything else is identical):
#   1. MLM points at the Megatron-LM-ffpa WORKTREE (branch alit/gemma4-ffpa-packed-sft),
#      which has the two-backend attention dispatch + cu_seqlens plumbing.
#   2. The ffpa_attn CuTeDSL kernel install is put on PYTHONPATH (H100/SM90 only).
#   3. --gemma4-attention-backend ffpa_flash is passed (Gemma4-specific selector;
#      NOT the base --attention-backend, which is TE's local/flash/fused knob).
#
# WHY ffpa is REQUIRED here (not just faster): on the ffpa branch the EAGER backend
# hard-errors when --sft passes packed_seq_params (cu_seqlens), because the eager
# additive-mask path cannot express leak-free packing. So SFT on this branch => ffpa_flash.
# (With one conversation per jsonl line + MBS=1 each sequence is a single document,
# so cu_seqlens has one segment and there is no cross-document leak anyway; ffpa
# honors that single-segment cu_seqlens correctly.)
#
# SFT constraints honored (hard asserts in sft_dataset.py):
#   * --no-create-attention-mask-in-dataloader is REQUIRED.
#   * do NOT pass --reset-position-ids / --reset-attention-mask.
#   * --split 100,0,0 (all train; no valid/test loader with --eval-iters 0).
# ============================================================================
set -eu

# ---- env (copied from the proven gemma4 V7 inner bash + reference interactive) --
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SLURM_CPU_BIND SLURM_DISTRIBUTION || true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_INDUCTOR_DISABLE=1

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm
# *** ffpa branch code tree (worktree on alit/gemma4-ffpa-packed-sft) ***
MLM=${MLM:-${ROOT}/Megatron-LM-ffpa}
# *** ffpa_attn CuTeDSL kernel install (H100/SM90) ***
FFPA_INSTALL=${FFPA_INSTALL:-${ROOT}/ffpa_install}
RUN_DIR=${ROOT}/code_dev/shared-state/sft_ffpa_run
mkdir -p ${RUN_DIR}/triton_cache ${RUN_DIR}/checkpoints ${RUN_DIR}/data_cache
export TRITON_CACHE_DIR=${RUN_DIR}/triton_cache
export TRITON_HOME=${RUN_DIR}/triton_cache
# datasets/Arrow cache MUST be on lustre: the SFT loader does datasets.load_dataset
# on DATA_PATH, and the Arrow split for a 34.5GB packed jsonl would blow the 10G
# /home quota via the default ~/.cache location.
export HF_HOME=${RUN_DIR}/hf_cache
export HF_DATASETS_CACHE=${RUN_DIR}/hf_cache/datasets
mkdir -p ${HF_DATASETS_CACHE}

# ---- BASE E4B MLM checkpoint (../checkpoints/MLM/base). Flat dist-checkpoint, so
# ---- build a Megatron-layout VIEW (<ckpt>_mg: pointer -> release/ -> symlinks incl
# ---- the hidden .metadata index) that --load can find. torch_dist reshards on load.
LOAD_SRC=${LOAD_SRC:-${ROOT}/checkpoints/MLM/gemma-4-E4B-base-mlm}
if [ -f "${LOAD_SRC}/latest_checkpointed_iteration.txt" ]; then
  LOAD_CKPT="${LOAD_SRC}"
else
  LOAD_CKPT="${LOAD_SRC}_mg"
  mkdir -p "${LOAD_CKPT}/release"
  echo "release" > "${LOAD_CKPT}/latest_checkpointed_iteration.txt"
  shopt -s dotglob
  for f in "${LOAD_SRC}"/*; do
    bn=$(basename "$f")
    [ "${bn}" = "latest_checkpointed_iteration.txt" ] && continue
    ln -sfn "$f" "${LOAD_CKPT}/release/${bn}"
  done
  shopt -u dotglob
  echo "built Megatron-layout view: ${LOAD_CKPT} (release/ -> ${LOAD_SRC})"
fi

# ---- SFT data + tokenizer (stock Gemma4 chat template; `gemma` masking). --------
TOKENIZER=${TOKENIZER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/gemma4-playground/weights/gemma-4-E4B-it}
# jsonl with {"messages":[...], "tools":[...]} per line (one conversation per line).
# DATA_PATH=${DATA_PATH:-/lustre/fsw/portfolios/llmservice/users/ameyasunilm/datasets/tool_calling_with_execution/tool_calling_past_octopus-05152026-reasoning_on-raw_messages_format.jsonl}
# DATA_PATH=${DATA_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/code_dev/shared-state/sft_run/octopus_subset_512.jsonl}
DATA_PATH=${DATA_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/code_dev/shared-state/sft_run/octopus_subset_512_packed_8192.jsonl}
# DATA_PATH=${DATA_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/code_dev/shared-state/sft_run/synth_tool_1024.jsonl}
# ---- parallelism. *** TP must divide num_attention_heads=8: TP in {1,2,4,8}. ***
# ---- num_query_groups=2 < TP uses the base GQA KV-replication path (upstream #3627,
# ---- inherited by Gemma4SelfAttention); TP=4/8 verified vs TP=1 oracle. PP unsupported
# ---- (cross-layer kv_bus), CP unsupported. Scale TP and/or DP for memory: NPROC = TP * DP.
NPROC=${NPROC:-8}
TP=${TP:-8}
PP=${PP:-1}
CP=${CP:-1}
if [ $(( 8 % TP )) -ne 0 ]; then
  echo "ERROR: TP=${TP} unsupported -- TP must divide num_attention_heads=8 (use TP=1,2,4,8)." >&2
  exit 1
fi

# ---- REAL Gemma 4 E4B dims -- must match the checkpoint param shapes. ----------
# vocab NOT passed: SFTTokenizer reports len(tokenizer)=262144 -> pads to 262144.
NUM_LAYERS=${NUM_LAYERS:-42}
HIDDEN=${HIDDEN:-2560}
FFN=${FFN:-10240}
HEADS=${HEADS:-8}
KV_GROUPS=${KV_GROUPS:-2}

# ---- run knobs. SEQLEN 8192 is now feasible (ffpa attention is O(seq)). ---------
SEQLEN=${SEQLEN:-8192}
MBS=${MBS:-1}
GBS=${GBS:-8}
NUM_STEPS=${NUM_STEPS:-100}
TRAIN_SAMPLES=$((GBS * NUM_STEPS))

# NOTE: tied embeddings (real E4B) -> do NOT pass --untie-embeddings-and-output-weights.
options=" \
    --use-mcore-models \
    --transformer-impl local \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN} \
    --ffn-hidden-size ${FFN} \
    --num-attention-heads ${HEADS} \
    --group-query-attention \
    --num-query-groups ${KV_GROUPS} \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --qk-layernorm \
    --disable-bias-linear \
    --position-embedding-type none \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --init-method-std 0.02 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --seq-length ${SEQLEN} \
    --max-position-embeddings ${SEQLEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-samples ${TRAIN_SAMPLES} \
    --lr 1e-5 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-warmup-samples 0 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --use-distributed-optimizer \
    --load ${LOAD_CKPT} \
    --finetune \
    --ckpt-format torch_dist \
    --ckpt-fully-parallel-load \
    --no-load-optim \
    --no-load-rng \
    --sft \
    --sft-tokenizer-prompt-format gemma \
    --tokenizer-type SFTTokenizer \
    --tokenizer-model ${TOKENIZER} \
    --data-path ${DATA_PATH} \
    --split 100,0,0 \
    --no-create-attention-mask-in-dataloader \
    --distributed-timeout-minutes 20 \
    --disable-gloo-process-groups \
    --num-workers 1 \
    --log-interval 1 \
    --eval-iters 0 \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl native \
    --eval-interval 100000 \
    --save-interval 100000 \
    --logging-level 20 \
    --log-throughput \
    --data-cache-path ${RUN_DIR}/data_cache \
    "

PER_TOKEN_LOSS=${PER_TOKEN_LOSS:-1}
BACKEND=${BACKEND:-ffpa_flash}
USE_FREEZE_PLE=${USE_FREEZE_PLE:-0}
USE_SP=${USE_SP:-1}
RECOMPUTE=${RECOMPUTE:-1}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}

# ---- Weights & Biases logging (mirrors nm6 3B_hybrid_moe_flex.sh) ---------------
export WANDB_ENTITY="${WANDB_ENTITY:-nvidia}"
export WANDB_API_KEY="${WANDB_API_KEY:-9ff9778a6919bcd94302a0815b2136cc79e03751}"
WANDB_PROJECT="${WANDB_PROJECT:-Gemma4-E4B-SFT}"
RC_SUFFIX=""
if [ "${RECOMPUTE:-0}" = "1" ]; then RC_SUFFIX="-recompute"; fi
WANDB_NAME="${WANDB_NAME:-gemma4-e4b-sft-${BACKEND:-ffpa_flash}-seq${SEQLEN}-tp${TP}${RC_SUFFIX}}"
if [ -n "${WANDB_API_KEY}" ]; then
    # --tensorboard-dir is REQUIRED for scalar logging: in training_log() the whole
    # loss/lr/grad-norm/batch-size block (which also feeds wandb) is gated on the
    # tensorboard writer existing, i.e. on --tensorboard-dir being set.
    TENSORBOARD_DIR=${RUN_DIR}/tensorboard/${WANDB_NAME}
    mkdir -p ${RUN_DIR}/wandb ${TENSORBOARD_DIR}
    options="${options} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-exp-name ${WANDB_NAME} \
    --wandb-save-dir ${RUN_DIR}/wandb \
    --tensorboard-dir ${TENSORBOARD_DIR}"
fi


# ============================ Toggleable knobs (env-overridable) ============================
# Attention backend: BACKEND=ffpa_flash (default; FFPA full head_dim=512 + Flash sliding
# head_dim=256, honors cu_seqlens for packing) or BACKEND=eager (bitwise reference path).
# NOTE: eager REJECTS packed sequences (R3 guard) -> with --sft you MUST use ffpa_flash;
# BACKEND=eager is for unpacked parity/debug only. Always emitted.
options="${options} --gemma4-attention-backend ${BACKEND:-ffpa_flash}"
# The eager backend refuses packed sequences by default (R3 leak guard). This script packs
# ONE conversation per sequence (MBS=1 + one conversation per jsonl line), so every pack is a
# SINGLE document and eager's causal mask is leak-free -- enable the escape so BACKEND=eager
# runs SFT here, matching the tp-sp branch's eager SFT. Auto-on for eager only; harmless for
# ffpa_flash. DO NOT carry this to a config that packs MULTIPLE conversations per sequence.
if [ "${BACKEND:-ffpa_flash}" = "eager" ]; then
  options="${options} --gemma4-allow-eager-packed"
fi

# Freeze the Per-Layer-Embedding table (~2.8B params, replicated per rank): frees ~20GB/rank
# of grad+optimizer state. ON by default; set USE_FREEZE_PLE=0 to train the PLE table.
[ "${USE_FREEZE_PLE:-1}" = "1" ] && options="${options} --freeze-ple"

# Sequence parallelism (shards norm/activation along seq across TP; needs TP>1). ON by
# default; set USE_SP=0 to disable.
[ "${USE_SP:-1}" = "1" ] && options="${options} --sequence-parallel"

# Per-layer activation recompute (full recompute of every Gemma4 decoder layer via the
# KV-bus-aware Gemma4TransformerBlock._recompute_layers path). OFF by default; RECOMPUTE=1
# frees decoder activation memory so a longer SEQLEN fits. The loss curve MUST match
# RECOMPUTE=0 at a SEQLEN both fit (forward/backward are numerically unchanged). Optional
# RECOMPUTE_NUM_LAYERS (default 1 = per layer).
[ "${RECOMPUTE:-0}" = "1" ] && options="${options} --recompute-granularity full --recompute-method uniform --recompute-num-layers ${RECOMPUTE_NUM_LAYERS:-1}"
# ============================================================================================

# Per-token loss normalization (reported lm loss = global sum(loss)/sum(supervised
# tokens) instead of mean-of-microbatch-means; grads normalized by the global token
# count). ON by default so runs are directly comparable with the CP branch's CP>1 runs
# (which REQUIRE it). Set PER_TOKEN_LOSS=0 for the legacy normalization.
# NOTE: was `if ["..."]` (missing spaces) -- the test always failed under bash and the
# flag was silently never added; fixed.
if [ "${PER_TOKEN_LOSS:-1}" = "1" ]; then
  options="${options} --calculate-per-token-loss"
fi

# ffpa kernel FIRST on the path, then the ffpa-branch MLM tree.
export PYTHONPATH=${FFPA_INSTALL}:${MLM}:${PYTHONPATH:-}
cd ${MLM}
echo "########## gemma4 E4B SFT (FFPA) ckpt=${LOAD_CKPT} data=${DATA_PATH} MLM=${MLM} NPROC=${NPROC} TP=${TP} SEQLEN=${SEQLEN} wandb=${WANDB_PROJECT}/${WANDB_NAME} ##########"
torchrun --nproc-per-node=${NPROC} --master-port=${MASTER_PORT:-12399} ${MLM}/pretrain_gemma4.py ${options}
echo "EXIT_TRAIN=$?"
echo "########## DONE (ffpa_flash) ##########"
