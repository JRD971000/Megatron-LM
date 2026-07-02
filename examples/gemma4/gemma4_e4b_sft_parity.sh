#!/bin/bash
# ============================================================================
# Gemma 4 E4B SFT training-curve parity harness (V4/V5) -- ONE backend per run.
#
# Runs a short, DETERMINISTIC packed SFT on the real E4B ckpt with EITHER the
# eager oracle OR the ffpa_flash backend, emitting a per-step loss trace. Run it
# TWICE (BACKEND=eager, then BACKEND=ffpa_flash) with the SAME config, then overlay
# the two traces with plot_loss_overlay.py -> V4 training-curve parity.
#
# Everything except the attention backend is byte-identical between the two runs:
# fixed --seed, mbs/gbs, LR schedule, TP<=2/PP=1, seq len, packing, and the SAME
# sliced subset. That isolates the backend as the only variable.
#
# Run inside an already-allocated interactive container shell (see
# gemma4_e4b_load_interactive.sh header for the srun/--container-image recipe):
#     BACKEND=eager      bash examples/gemma4/gemma4_e4b_sft_parity.sh
#     BACKEND=ffpa_flash bash examples/gemma4/gemma4_e4b_sft_parity.sh
#
# ---- backend selection ------------------------------------------------------
# The gemma4 string field config.gemma4_attention_backend ("eager"|"ffpa_flash") is
# selected by the --gemma4-attention-backend CLI arg on pretrain_gemma4.py (NOT the
# base --attention-backend enum, which is TE's local/flash/fused selector -- a
# DIFFERENT knob). By default (USE_BACKEND_FLAG=1) BACKEND is passed through as
# --gemma4-attention-backend ${BACKEND}. Set USE_BACKEND_FLAG=0 to fall back to the
# eager default and use BACKEND only to label the output file.
# ============================================================================
set -eu

# ---- env (from the proven gemma4 load-interactive smoke) ------------------------
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SLURM_CPU_BIND SLURM_DISTRIBUTION || true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_INDUCTOR_DISABLE=1

BACKEND=${BACKEND:-eager}                 # eager | ffpa_flash  (selects backend + labels trace)
USE_BACKEND_FLAG=${USE_BACKEND_FLAG:-1}   # 1 = pass --gemma4-attention-backend ${BACKEND}
BACKEND_FLAG=${BACKEND_FLAG:---gemma4-attention-backend}  # CLI flag name (see pretrain_gemma4.py)

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm
MLM=${MLM_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}   # THIS worktree root
IMPL=${ROOT}/code_dev/shared-state/implementations/HYBRID-gemma4-mlm
RUN_DIR=${RUN_DIR:-${ROOT}/code_dev/shared-state/ffpa_sft_run}
FFPA_INSTALL=${FFPA_INSTALL:-${ROOT}/ffpa_install}
HF_WEIGHTS=${HF_WEIGHTS:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/gemma4-playground/weights/gemma-4-E4B-it}
mkdir -p ${RUN_DIR}/triton_cache ${RUN_DIR}/checkpoints ${RUN_DIR}/data_cache
export TRITON_CACHE_DIR=${RUN_DIR}/triton_cache
export TRITON_HOME=${RUN_DIR}/triton_cache

# ---- training-layout checkpoint (Megatron --load view of the flat distcp). Tied
# ---- embeddings (real E4B) -> do NOT pass --untie-embeddings-and-output-weights.
LOAD_CKPT=${LOAD_CKPT:-${IMPL}/mlm_ckpt_mg}

# ---- data: slice a small FIXED subset of the 32GB jsonl (never load the whole file).
N_LINES=${N_LINES:-256}
SUBSET=${SUBSET:-${RUN_DIR}/sft_subset_${N_LINES}.jsonl}
N_LINES=${N_LINES} OUT=${SUBSET} bash "$(dirname "${BASH_SOURCE[0]}")/slice_sft_subset.sh" ${N_LINES}

# ---- parallelism. TP MUST be 1 or 2 (Gemma4SelfAttention is a parallelism<=2 port;
# ---- see gemma4_e4b_load_interactive.sh). PP unsupported (cross-layer kv_bus).
NPROC=${NPROC:-8}
TP=${TP:-2}
PP=${PP:-1}
CP=${CP:-1}
if [ "${TP}" -gt 2 ]; then
  echo "ERROR: TP=${TP} unsupported for Gemma4 (num_query_groups=2). Use TP=1 or TP=2." >&2
  exit 1
fi

# ---- REAL Gemma 4 E4B dims (must match the ckpt param shapes). --------------------
NUM_LAYERS=${NUM_LAYERS:-42}
HIDDEN=${HIDDEN:-2560}
FFN=${FFN:-10240}
HEADS=${HEADS:-8}
KV_GROUPS=${KV_GROUPS:-2}
VOCAB=${VOCAB:-262144}

# ---- DETERMINISTIC smoke knobs (identical across both backend runs). --------------
SEED=${SEED:-1234}
SEQLEN=${SEQLEN:-2048}
MBS=${MBS:-1}          # packing => micro-batch-size must be 1
GBS=${GBS:-4}
NUM_STEPS=${NUM_STEPS:-20}
TRAIN_SAMPLES=$((GBS * NUM_STEPS))

LOSS_LOG=${RUN_DIR}/loss_${BACKEND}.log

# NOTE on packing/eager: SFT always packs (cu_seqlens via PackedSeqParams). The SFT dataloader
# BUILDS per-conversation reset position_ids natively (range(len) per conv) and ASSERTS
# `not reset_position_ids` (sft_dataset.py:134) -- so we must NOT pass --reset-position-ids;
# those reset positions flow to the model and satisfy R1's per-document RoPE assert directly.
# SFT also asserts create_attention_mask==False (sft_dataset.py:184), so we DO pass
# --no-create-attention-mask-in-dataloader. Gemma4 builds its own masks anyway: the eager path
# builds attention_mask_by_type inside Gemma4Model.forward, and the ffpa_flash path uses cu_seqlens
# within each doc; ffpa_flash ignores the mask and uses cu_seqlens (leak-free by construction).
# Whether eager+packed additionally requires I1's allow_eager_packed / R3 escape is an I1/I2
# decision resolved in STAGE 5.
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
    --seed ${SEED} \
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
    --no-create-attention-mask-in-dataloader \
    --tokenizer-type SFTTokenizer \
    --tokenizer-model ${HF_WEIGHTS} \
    --data-path ${SUBSET} \
    --split 100,0,0 \
    --distributed-timeout-minutes 20 \
    --disable-gloo-process-groups \
    --num-workers 1 \
    --log-interval 1 \
    --eval-iters 0 \
    --eval-interval 100000 \
    --save-interval 100000 \
    --logging-level 20 \
    --log-throughput \
    --data-cache-path ${RUN_DIR}/data_cache \
    "

# ---- gemma4 backend flag (see backend selection note in the header) ---------
if [ "${USE_BACKEND_FLAG}" = "1" ]; then
  options="${options} ${BACKEND_FLAG} ${BACKEND}"
  if [ "${BACKEND}" = "ffpa_flash" ]; then
    export PYTHONPATH=${FFPA_INSTALL}:${MLM}:${PYTHONPATH:-}
  fi
fi
export PYTHONPATH=${MLM}:${PYTHONPATH:-}
cd ${MLM}

echo "########## gemma4 E4B SFT parity (BACKEND=${BACKEND}, USE_BACKEND_FLAG=${USE_BACKEND_FLAG}, TP=${TP}) ##########"
echo "########## loss trace -> ${LOSS_LOG} ##########"
torchrun --nproc-per-node=${NPROC} --master-port=12399 ${MLM}/pretrain_gemma4.py ${options} 2>&1 | tee ${LOSS_LOG}
echo "EXIT_TRAIN=${PIPESTATUS[0]}"
echo "########## DONE (${BACKEND}) -- plot with: python examples/gemma4/plot_loss_overlay.py \\
      --eager ${RUN_DIR}/loss_eager.log --ffpa ${RUN_DIR}/loss_ffpa_flash.log ##########"
