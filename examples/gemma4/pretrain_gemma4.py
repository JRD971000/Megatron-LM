# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

r"""Pretrain (or distill) Gemma 4 (E4B).

Thin entrypoint mirroring ``pretrain_gpt.py``: it reuses the GPT data pipeline,
batch generation, loss, and forward step verbatim and only swaps in a Gemma4
model builder. The builder constructs a :class:`Gemma4TransformerConfig` (which
carries the heterogeneous per-layer specs, softcap, sqrt(H) embedding scaling
and PLE knobs) and a :class:`Gemma4Model`, which applies the final-logit softcap
and sqrt(H) embedding scaling internally so they reach the training loss path.

When ``--export-kd-teacher-load`` is set, the Gemma4 student is wrapped in a ModelOpt
``DistillationModel`` alongside a Gemma4 teacher and trained with a Knowledge-Distillation loss
(logit KL by default) -- e.g. to recover accuracy after Minitron pruning (see prune_gemma4_e4b.py).

Distillation in the NeMo container (nvcr.io/nvidia/nemo:26.06)
-------------------------------------------------------------
Mount the same three things as the pruning example (Model-Optimizer overlays, the Gemma4 Megatron-LM
fork *over* the container's bundled MLM, and a workspace); data is standard pre-tokenized
``.bin``/``.idx`` via ``--data-path`` (tokenize with ``modelopt.torch.utils.plugins.megatron_preprocess_data``):

    export MODELOPT_DIR=/path/to/Model-Optimizer         # branch: kmorabia/prune-gemma4-e4b
    export MEGATRON_LM_DIR=/path/to/Megatron-LM          # branch: alit/gemma4-e4b-tp-sp
    export WORKSPACE=/path/to/workspace                  # teacher ckpt + tokenized data + output

    docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --net=host --rm -it \
        -v ${MODELOPT_DIR}/modelopt:/opt/venv/lib/python3.12/site-packages/modelopt \
        -v ${MODELOPT_DIR}/modelopt_recipes:/opt/venv/lib/python3.12/site-packages/modelopt_recipes \
        -v ${MEGATRON_LM_DIR}:/opt/Megatron-Bridge/3rdparty/Megatron-LM \
        -v ${WORKSPACE}:/workspace \
        -w /opt/Megatron-Bridge/3rdparty/Megatron-LM \
        nvcr.io/nvidia/nemo:26.06 bash

    hf auth login --token <your token>   # if the tokenizer / dataset are gated

Prerequisites:
  * teacher = a Gemma4 Megatron dist-checkpoint (e.g. the unpruned model from
    examples/gemma4/hf_to_mlm_convert.py) + a NeMo-style ``teacher_model_config.yaml`` giving at
    least ``num_layers`` / ``hidden_size`` / ``ffn_hidden_size`` / ``num_attention_heads`` (the
    fields where the teacher differs from the student), in the ckpt dir or via
    ``--export-kd-teacher-model-config``;
  * student = built from the CLI dims; optionally ``--load`` a pruned Gemma4 checkpoint;
  * data = ``.bin``/``.idx`` prefix for ``--data-path``.

Run (single GPU, or multi-GPU with TP + sequence-parallel; PP is NOT supported by the Gemma4 model.
``--cross-entropy-fusion-impl native`` is required -- ModelOpt KD is incompatible with the TE
cross-entropy fusion):

    torchrun --nproc-per-node 2 examples/gemma4/pretrain_gemma4.py \
        --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 \
        --export-kd-teacher-load /workspace/teacher_ckpt \
        --export-kd-teacher-model-config /workspace/teacher_model_config.yaml \
        --cross-entropy-fusion-impl native \
        --data-path /workspace/tokenized/<prefix> \
        --tokenizer-type HuggingFaceTokenizer --tokenizer-model google/gemma-4-E4B \
        --num-layers 42 --hidden-size 2560 --ffn-hidden-size 10240 \
        --num-attention-heads 8 --group-query-attention --num-query-groups 2 \
        --normalization RMSNorm --qk-layernorm --disable-bias-linear \
        --position-embedding-type none --transformer-impl local \
        --seq-length 4096 --max-position-embeddings 4096 --bf16 \
        --micro-batch-size 1 --global-batch-size 128 --train-iters 1000 \
        --lr 1e-4 --min-lr 1e-5 --lr-decay-style cosine \
        --load /workspace/pruned_student_ckpt --save /workspace/distilled_out \
        --ckpt-format torch_dist --data-cache-path /workspace/dcache

Optional ``--export-kd-cfg <distill.yaml>`` customizes the KD loss (logit layers, intermediate-layer
pairs, ``kd_loss_scale``, ``logit_kl_temperature``, ``skip_lm_loss``); the default is logit-only KL.
"""

# Capture the true program start time BEFORE any heavy imports.
import time

_PROGRAM_START_TIME = time.time()

import os
import warnings

rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def _apply_nvrx_version_shim():
    """Work around a container<->MLM incompatibility BEFORE megatron is imported.

    The container's ``nvidia_resiliency_ext`` ships the async-ckpt modules but
    exposes no ``__version__``, so MLM's ``dist_checkpointing/strategies/nvrx.py``
    ``is_nvrx_min_version()`` crashes at import. Populate ``__version__`` first.
    We never use async checkpointing here (single device, DDP=1), so the value
    only needs to satisfy the ``>=0.6.0`` guard. Mirrors code_dev/scripts/mlm_env.py.
    """
    try:
        import nvidia_resiliency_ext as _nvrx

        if not hasattr(_nvrx, "__version__"):
            v = None
            try:
                from importlib.metadata import version

                v = version("nvidia_resiliency_ext")
            except Exception:
                v = None
            try:
                from packaging.version import Version as _V

                if v is None or _V(v) < _V("0.6.0"):
                    v = "0.6.0"
            except Exception:
                v = v or "0.6.0"
            _nvrx.__version__ = v
    except Exception:
        # nvrx absent entirely -> MLM's HAVE_NVRX=False path handles it fine.
        pass


_apply_nvrx_version_shim()

# This script imports repo-root modules (``pretrain_gpt``) that are NOT part of the installed
# ``megatron`` package. Running examples/gemma4/pretrain_gemma4.py only puts the script's own dir on
# sys.path, so add the Megatron-LM repo root (three parents up) here -- then it runs from any cwd
# with no PYTHONPATH.
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from functools import partial

import modelopt.torch.distill as mtd
import modelopt.torch.distill.plugins.megatron as mtd_mcore
import modelopt.torch.opt as mto
import torch

from megatron.core.enums import ModelType
from megatron.core.models.gemma4.gemma4_layer_specs import (
    get_gemma4_layer_local_spec,
    get_gemma4_layer_with_transformer_engine_spec,
)
from megatron.core.models.gemma4.gemma4_model import Gemma4Model
from megatron.core.transformer.gemma4_config import Gemma4TransformerConfig
from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.model_builder import _load_teacher_model_config
from megatron.training import get_args, pretrain, print_rank_0, set_startup_timestamps
from megatron.training.argument_utils import (
    core_transformer_config_from_args,
    gpt_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import parse_and_validate_args

# Reuse the GPT data pipeline, batch logic, loss, and forward step unchanged.
from pretrain_gpt import forward_step, get_embedding_ranks, train_valid_test_datasets_provider


def _apply_gemma4_config_overrides(config, args):
    """Set the HF-Gemma4 activation / PLE knobs that have no dedicated Megatron CLI flag."""
    # Gemma4 MLP is a GeGLU with tanh-approx GELU. There is no CLI flag for this exact
    # combination (--swiglu is SiLU, --quick-geglu is quick_gelu), so set it here to
    # match the HF model regardless of the activation flags passed on the command line.
    config.gated_linear_unit = True
    config.activation_func = partial(torch.nn.functional.gelu, approximate="tanh")
    config.bias_activation_fusion = False

    # The PLE per-layer token table is indexed by the SAME vocab as embed_tokens
    # (HF: vocab_size_per_layer_input == vocab_size). Keep them consistent with the
    # model vocab so a smaller --vocab-size (e.g. for a smoke run) shrinks the PLE
    # table too instead of leaving it pinned at the 262144 default.
    config.vocab_size_per_layer_input = args.padded_vocab_size


def _build_gemma4_model(args, config, pre_process, post_process, vp_stage, pg_collection):
    """Construct a :class:`Gemma4Model` from a populated :class:`Gemma4TransformerConfig`."""
    use_te = args.transformer_impl == "transformer_engine"
    transformer_layer_spec = (
        get_gemma4_layer_with_transformer_engine_spec(config)
        if use_te
        else get_gemma4_layer_local_spec(config)
    )
    return Gemma4Model(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        # Gemma4 ties input/output embeddings; default share unless explicitly untied.
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )


def gemma4_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Build a :class:`Gemma4Model` (mirrors ``gpt_builder``), with optional ModelOpt distillation.

    When ``--export-kd-teacher-load`` is set, the student is wrapped in a
    ``modelopt.torch.distill.DistillationModel`` alongside a **Gemma4** teacher -- mirroring
    ``modelopt_gpt_hybrid_builder``'s KD path, but building a Gemma4 teacher (via
    ``Gemma4TransformerConfig`` + the gemma4 layer spec) instead of a GPT/Hybrid one.
    """
    print_rank_0('building Gemma4 model ...')
    if config is None:
        # Build the Gemma4 subclass so its heterogeneous per-layer specs, softcap,
        # sqrt(H) embedding scaling, and PLE dims are populated from the defaults.
        config = core_transformer_config_from_args(args, config_class=Gemma4TransformerConfig)
    _apply_gemma4_config_overrides(config, args)

    model = _build_gemma4_model(args, config, pre_process, post_process, vp_stage, pg_collection)

    # ModelOpt Knowledge Distillation: wrap the student with a Gemma4 teacher.
    if getattr(args, "export_kd_teacher_load", None):
        print_rank_0("Distillation: Enabled (Gemma4 teacher).")
        assert not args.manual_gc, "ModelOpt Distillation is incompatible with --manual-gc."
        assert not args.tp_comm_overlap, (
            "ModelOpt Distillation is incompatible with --tp-comm-overlap."
        )
        assert args.cross_entropy_fusion_impl != "te", (
            "ModelOpt Distillation is incompatible with the TE cross-entropy fusion "
            "(use --cross-entropy-fusion-impl native)."
        )

        # The teacher arch differs from the (pruned) student only in the fields overridden by its
        # NeMo-style config (in the teacher ckpt dir or via --export-kd-teacher-model-config).
        teacher_config_raw = _load_teacher_model_config(args.export_kd_teacher_load)
        teacher_config = core_transformer_config_from_args(
            teacher_config_raw, config_class=Gemma4TransformerConfig
        )
        _apply_gemma4_config_overrides(teacher_config, args)
        teacher_model = _build_gemma4_model(
            args, teacher_config, pre_process, post_process, vp_stage, pg_collection
        )

        distill_cfg = mtd_mcore.setup_distillation_config(
            args.export_kd_cfg, student_cfg=config, teacher_cfg=teacher_config
        )
        kd_config = {
            "teacher_model": teacher_model,
            "criterion": distill_cfg.criterion,
            "loss_balancer": distill_cfg.loss_balancer,
        }
        model = mtd.convert(model, mode=[("kd_loss", kd_config)])
        # MCore-specific tweaks (sharded state, pipeline parallel, optional skip-lm-loss).
        mtd_mcore.adjust_distillation_model_for_mcore(model, distill_cfg)
        # Remove KD mode state to prevent re-conversion issues after restore.
        mto.ModeloptStateManager(model).state_dict().pop()

    return model


def gemma4_model_provider(
    pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None
):
    """Always build via ``gemma4_builder`` (which handles KD with a **Gemma4** teacher).

    The shared ``model_provider`` swaps the builder for ``modelopt_gpt_hybrid_builder`` whenever
    ``args.modelopt_enabled`` is set (which distillation turns on), which would build a GPT
    student/teacher instead of Gemma4. Call ``gemma4_builder`` directly to bypass that swap;
    ModelOpt KD infra (teacher checkpoint load, KD loss) keys off ``--export-kd-teacher-load`` and
    the ``DistillationModel`` type, not ``modelopt_enabled``.
    """
    return gemma4_builder(
        get_args(),
        pre_process,
        post_process,
        vp_stage=vp_stage,
        config=config,
        pg_collection=pg_collection,
    )


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup).
    _MAIN_ENTRY_TIME = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets.
    setattr(train_valid_test_datasets_provider, "is_distributed", True)

    # Register ModelOpt args (--export-kd-teacher-load, --export-kd-cfg, ...) so distillation can
    # be enabled from the CLI; harmless (defaults off) for plain pretraining.
    args = parse_and_validate_args(
        extra_args_provider=add_modelopt_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    # Use the Gemma4 transformer config in the pretrain config container too so the
    # container is consistent with the model the builder constructs.
    transformer_cfg = core_transformer_config_from_args(args, config_class=Gemma4TransformerConfig)
    model_cfg = gpt_config_from_args(args, config=transformer_cfg)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        gemma4_model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        get_embedding_ranks=get_embedding_ranks,
    )
