#!/bin/bash
# ============================================================================
# Gemma 4 E4B BATCH (sbatch) SFT run with the FFPA/Flash attention backend.
#
# This is the sbatch sibling of gemma4_e4b_sft_ffpa_interactive.sh. The BODY
# (model config, SFT data, options, toggleable knobs, wandb) is byte-for-byte
# the same as the interactive script; only the LAUNCH is different:
#   * interactive: run by hand inside an already-allocated container shell via
#     `torchrun --nproc-per-node=... pretrain_gemma4.py`.
#   * batch (this): a self-contained sbatch job whose header + container srun
#     launch mirror nm6_megatron_lm/.../3B_hybrid_moe_flex.sh. Each node gets
#     ONE srun task that spawns `torchrun --nproc-per-node=8` (the proven
#     Gemma4 launcher, which sets RANK/WORLD_SIZE/LOCAL_RANK/MASTER_* that
#     Megatron reads in arguments.py). Scale by --nodes; TP*PP*CP must divide
#     the total rank count (nodes*8), remainder = DP.
#
# LAUNCH (edit knobs inline, then submit):
# for i in {1..1}; do TP=8 SEQLEN=8192 BACKEND=ffpa_flash sbatch -p batch --account=coreai_dlalgo_genai --job-name=gemma4_e4b_sft_ffpa --time=4:00:00 --nodes=1 gemma4_e4b_sft_ffpa_batch.sh ; done;
#
# THREE things differ from the eager script (everything else is identical):
#   1. MLM points at the Megatron-LM-ffpa WORKTREE (branch alit/gemma4-ffpa-packed-sft),
#      which has the two-backend attention dispatch + cu_seqlens plumbing.
#   2. The ffpa_attn CuTeDSL kernel install is put on PYTHONPATH (H100/SM90 only).
#   3. --gemma4-attention-backend ffpa_flash is passed (Gemma4-specific selector;
#      NOT the base --attention-backend, which is TE's local/flash/fused knob).
#
# SFT constraints honored (hard asserts in sft_dataset.py):
#   * --no-create-attention-mask-in-dataloader is REQUIRED.
#   * do NOT pass --reset-position-ids / --reset-attention-mask.
#   * --split ${SPLIT} (default 99.7,0.3,0.0): 0.3% of rows held out as the valid
#     set (~2263 rows on the full packed octopus file), evaluated EVAL_ITERS x GBS
#     = 4 x 512 = 2048 samples every --eval-interval iterations.
#   * checkpoints: --save ${CHECKPOINT_DIR} (stable name); resubmitted jobs RESUME
#     from it automatically (optimizer/rng/iteration); first run fine-tunes from
#     the base checkpoint.
# ============================================================================

# ---- sbatch header (mirrors 3B_hybrid_moe_flex.sh; -p/--account/--nodes/--time/
# ---- --job-name are overridden by the sbatch CLI above). ONE task per node:
# ---- torchrun fans out the 8 GPU ranks itself, so --ntasks-per-node=1.
#SBATCH -p batch
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=gemma4_e4b_sft_ffpa

set -eu

# ---- env (copied from the proven gemma4 V7 inner bash + reference interactive) --
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SLURM_CPU_BIND SLURM_DISTRIBUTION || true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_INDUCTOR_DISABLE=1
# ---- NCCL / misc (mirrors 3B_hybrid_moe_flex.sh) --------------------------------
export NCCL_DEBUG=WARN
export TORCHINDUCTOR_WORKER_START=fork

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm
# *** ffpa branch code tree (worktree on alit/gemma4-ffpa-packed-sft) ***
MLM=${MLM:-${ROOT}/Megatron-LM-ffpa}
# *** ffpa_attn CuTeDSL kernel install (H100/SM90) ***
FFPA_INSTALL=${FFPA_INSTALL:-${ROOT}/ffpa_install}
# *** container image (FFPA branch needs nemo.26.06) ***
IMAGE_PATH=${IMAGE_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/containers/nemo.26.06.sqsh}
RUN_DIR=${ROOT}/code_dev/shared-state/sft_ffpa_run
LOGS_DIR=${RUN_DIR}/logs
mkdir -p ${RUN_DIR}/triton_cache ${RUN_DIR}/checkpoints ${RUN_DIR}/data_cache ${LOGS_DIR}
export TRITON_CACHE_DIR=${RUN_DIR}/triton_cache
export TRITON_HOME=${RUN_DIR}/triton_cache

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
DATA_PATH=${DATA_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/code_dev/shared-state/sft_run/tool_calling_past_octopus-05152026-reasoning_on-raw_messages_format_packed_8192.jsonl}
# DATA_PATH=${DATA_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/code_dev/shared-state/sft_run/octopus_subset_512.jsonl}
# DATA_PATH=${DATA_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/code_dev/shared-state/sft_run/octopus_subset_512_packed_8192.jsonl}
# DATA_PATH=${DATA_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/code_dev/shared-state/sft_run/synth_tool_1024.jsonl}
# ---- parallelism. *** TP must divide num_attention_heads=8: TP in {1,2,4,8}. ***
# ---- num_query_groups=2 < TP uses the base GQA KV-replication path (upstream #3627,
# ---- inherited by Gemma4SelfAttention); TP=4/8 verified vs TP=1 oracle. PP unsupported
# ---- (cross-layer kv_bus), CP unsupported. Scale TP and/or DP for memory: total ranks
# ---- = NNODES * GPUS_PER_NODE = TP * PP * CP * DP.
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
GBS=${GBS:-512}
NUM_STEPS=${NUM_STEPS:-10000}
TRAIN_SAMPLES=$((GBS * NUM_STEPS))

# ---- eval: EVAL_ITERS x GBS samples per eval (4 x 512 = 2048). The valid split
# ---- is carved from DATA_PATH by --split: 0.3% of the 754310-row packed octopus
# ---- file = ~2263 held-out rows >= 2048, so one eval pass sees distinct rows.
EVAL_ITERS=${EVAL_ITERS:-4}
SPLIT=${SPLIT:-99.7,0.3,0.0}

# ---- checkpointing: stable dir (not job-id-keyed) so resubmitted jobs RESUME.
# ---- First run: no checkpoint in CHECKPOINT_DIR -> load the base ckpt with
# ---- --finetune (iteration 0, fresh optimizer). Later runs: a checkpoint exists
# ---- -> true resume from it (optimizer + rng + iteration restored).
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints/sft_ffpa_seq${SEQLEN}_tp${TP}_gbs${GBS}}
mkdir -p ${CHECKPOINT_DIR}
if [ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
  LAST_ITER=$(cat ${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt)
  # Graceful exit when the schedule is already complete (resubmitting a finished
  # run otherwise dies in the sampler: "no samples left to consume"). NOTE: the
  # LR-scheduler checkpoint pins the TOTAL schedule length -- keep NUM_STEPS
  # constant across resubmissions of the same CHECKPOINT_DIR.
  if [ "${LAST_ITER}" != "release" ] && [ "${LAST_ITER}" -ge "${NUM_STEPS}" ]; then
    echo "Training already complete (checkpoint at iteration ${LAST_ITER} >= NUM_STEPS=${NUM_STEPS}); nothing to do."
    exit 0
  fi
  LOAD_ARGS="--load ${CHECKPOINT_DIR}"
  echo "RESUMING from ${CHECKPOINT_DIR} (iteration ${LAST_ITER})"
else
  LOAD_ARGS="--load ${LOAD_CKPT} --finetune --no-load-optim --no-load-rng"
  echo "FRESH START from base ckpt ${LOAD_CKPT}; saving to ${CHECKPOINT_DIR}"
fi

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
    ${LOAD_ARGS} \
    --save ${CHECKPOINT_DIR} \
    --ckpt-format torch_dist \
    --ckpt-fully-parallel-load \
    --sft \
    --sft-tokenizer-prompt-format gemma \
    --tokenizer-type SFTTokenizer \
    --tokenizer-model ${TOKENIZER} \
    --data-path ${DATA_PATH} \
    --split ${SPLIT} \
    --no-create-attention-mask-in-dataloader \
    --distributed-timeout-minutes 20 \
    --disable-gloo-process-groups \
    --num-workers 1 \
    --log-interval 1 \
    --eval-iters ${EVAL_ITERS} \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl native \
    --eval-interval 50 \
    --save-retain-interval 200 \
    --save-interval 50 \
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
WANDB_NAME="${WANDB_NAME:-gemma4-e4b-sft-${BACKEND:-ffpa_flash}-seq${SEQLEN}-tp${TP}${RC_SUFFIX}-GBS${GBS}}"
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
if [ "${PER_TOKEN_LOSS:-1}" = "1" ]; then
  options="${options} --calculate-per-token-loss"
fi

# ---- multi-node rendezvous for torchrun (matches the interactive torchrun path) --
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=${MASTER_PORT:-12399}

echo "########## gemma4 E4B SFT (FFPA, BATCH) ckpt=${LOAD_CKPT} data=${DATA_PATH} MLM=${MLM} NODES=${SLURM_NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} TP=${TP} SEQLEN=${SEQLEN} wandb=${WANDB_PROJECT}/${WANDB_NAME} master=${MASTER_ADDR}:${MASTER_PORT} ##########"

# One srun task per node; each launches torchrun which spawns the 8 GPU ranks and
# sets RANK/WORLD_SIZE/LOCAL_RANK. ffpa kernel FIRST on the path, then the ffpa MLM tree.
srun -l \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    bash -c "
set -eu
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_INDUCTOR_DISABLE=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR}
export TRITON_HOME=${TRITON_HOME}
export WANDB_ENTITY='${WANDB_ENTITY}'
export WANDB_API_KEY='${WANDB_API_KEY}'
export PYTHONPATH=${FFPA_INSTALL}:${MLM}:\${PYTHONPATH:-}
cd ${MLM}
torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc-per-node=${GPUS_PER_NODE} \
    --node-rank=\${SLURM_NODEID} \
    --rdzv-id=${SLURM_JOB_ID} \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    ${MLM}/pretrain_gemma4.py ${options}
"
echo "EXIT_TRAIN=$?"
echo "########## DONE (ffpa_flash, BATCH) ##########"

# ---- more launch examples (edit knobs inline, then submit) ----------------------
# single node, TP=8, seq 8192, ffpa:
# for i in {1..1}; do TP=8 SEQLEN=8192 BACKEND=ffpa_flash sbatch -p batch --account=coreai_dlalgo_genai --job-name=gemma4_e4b_sft_ffpa --time=4:00:00 --nodes=1 gemma4_e4b_sft_ffpa_batch.sh ; done;
# 4 nodes (32 ranks: TP=8 -> DP=4), bigger GBS:

# for i in {1..1}; do TP=8 GBS=512 SEQLEN=8192 BACKEND=ffpa_flash sbatch -p batch --account=coreai_dlalgo_genai --job-name=gemma4_e4b_sft_ffpa_16n --time=4:00:00 --nodes=16 /lustre/fsw/portfolios/coreai/users/ataghibakhsh/Gemma4_mlm/code_dev/scripts/gemma4_e4b_sft_ffpa_batch.sh ; done;
# for i in {1..1}; do TP=8 GBS=128 SEQLEN=8192 BACKEND=ffpa_flash sbatch -p interactive --account=coreai_dlalgo_genai --job-name=gemma4_e4b_sft_ffpa_16n --time=4:00:00 --nodes=1 /lustre/fsw/portfolios/coreai/users/ataghibakhsh/Gemma4_mlm/code_dev/scripts/gemma4_e4b_sft_ffpa_batch.sh ; done;


# for i in {1..3}; do
#   TP=8 GBS=512 SEQLEN=8192 BACKEND=ffpa_flash \
#   sbatch -p batch --account=coreai_dlalgo_genai --job-name=gemma4_e4b_sft_ffpa_16n \
#     --time=4:00:00 --nodes=16 --dependency=singleton \
#     /lustre/fsw/portfolios/coreai/users/ataghibakhsh/Gemma4_mlm/code_dev/scripts/gemma4_e4b_sft_ffpa_batch.sh
# done