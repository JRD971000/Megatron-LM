# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Pretrain Gemma 4 (E4B).

Thin entrypoint mirroring ``pretrain_gpt.py``: it reuses the GPT data pipeline,
batch generation, loss, and forward step verbatim and only swaps in a Gemma4
model builder. The builder constructs a :class:`Gemma4TransformerConfig` (which
carries the heterogeneous per-layer specs, softcap, sqrt(H) embedding scaling
and PLE knobs) and a :class:`Gemma4Model`, which applies the final-logit softcap
and sqrt(H) embedding scaling internally so they reach the training loss path.
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

        # Fire when __version__ is missing OR a dev pre-release below 0.6.0 (e.g.
        # "0.6.0.dev69+..."), which PEP 440 orders BELOW the final "0.6.0" and so
        # fails the >=0.6.0 guard in dist_checkpointing/strategies/nvrx.py. We never
        # use async checkpointing here, so a satisfying value is safe. Mirrors the
        # robust shim in examples/gemma4/gemma4_common.py.
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
        # nvrx absent entirely -> MLM's HAVE_NVRX=False path handles it fine.
        pass


_apply_nvrx_version_shim()


def _apply_te_fused_adam_int32_shim():
    """Make TE FusedAdam correct for params with numel > 2**31 - 1.

    TE's multi-tensor kernels store each tensor's numel in a 32-bit int
    (TensorListMetadata.sizes), so a tensor above 2**31 - 1 elements wraps
    negative and the kernel launches zero blocks: FusedAdam SILENTLY SKIPS the
    update (verified standalone: a 2.82B-element param sees no change after
    step()). Gemma4 hits this when training the PLE table
    (embed_tokens_per_layer, 262144 x 10752 = 2.82B params, replicated across
    TP) with DP=1 (e.g. TP=8 on one node) -- the distributed optimizer then
    cannot shard the main grad/param below 2**31. The l2-grad-norm variant of
    the same bug crashes with an illegal memory access in get_grad_norm_fp32
    (fixed in megatron/core/optimizer/clip_grads.py).

    Wrap fused_adam's multi_tensor_applier to split every oversized tensor into
    <2**31-element flat views, applied consistently across all tensor lists.
    All fused_adam multi-tensor ops are elementwise across aligned lists, so
    the split is mathematically exact and in-place semantics are preserved
    (views share storage).
    """
    limit = 2**31 - 1
    chunk = 2**30
    try:
        import transformer_engine.pytorch.optimizers.fused_adam as _fa
    except Exception:
        return
    inner = _fa.multi_tensor_applier
    if getattr(inner, "_int32_split_shim", False):
        return

    def _split_applier(op, noop_flag, tensor_lists, *args):
        if not any(t.numel() > limit for t in tensor_lists[0]):
            return inner(op, noop_flag, tensor_lists, *args)
        out_lists = [[] for _ in tensor_lists]
        for i, first in enumerate(tensor_lists[0]):
            numel = first.numel()
            assert all(lst[i].numel() == numel for lst in tensor_lists), (
                "multi-tensor lists are not elementwise-aligned; cannot split "
                "an oversized tensor safely"
            )
            if numel <= limit:
                for li, lst in enumerate(tensor_lists):
                    out_lists[li].append(lst[i])
            else:
                for li, lst in enumerate(tensor_lists):
                    assert lst[i].is_contiguous()
                    out_lists[li].extend(lst[i].view(-1).split(chunk))
        return inner(op, noop_flag, out_lists, *args)

    _split_applier._int32_split_shim = True
    _fa.multi_tensor_applier = _split_applier


_apply_te_fused_adam_int32_shim()

from functools import partial

import torch

from megatron.core.enums import ModelType
from megatron.core.models.gemma4.gemma4_layer_specs import (
    get_gemma4_layer_local_spec,
    get_gemma4_layer_with_transformer_engine_spec,
)
from megatron.core.models.gemma4.gemma4_model import Gemma4Model
from megatron.core.transformer.gemma4_config import Gemma4TransformerConfig
from megatron.training import pretrain, print_rank_0, set_startup_timestamps
from megatron.training.argument_utils import (
    core_transformer_config_from_args,
    gpt_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import parse_and_validate_args
from model_provider import model_provider

# Reuse the GPT data pipeline, batch logic, loss, and forward step unchanged.
from pretrain_gpt import forward_step, get_embedding_ranks, train_valid_test_datasets_provider


def add_gemma4_args(parser):
    """Add Gemma4-specific CLI args (mirrors ``add_modelopt_args``)."""
    group = parser.add_argument_group(title="gemma4")
    group.add_argument(
        "--gemma4-attention-backend",
        type=str,
        default="eager",
        choices=["eager", "ffpa_flash"],
        help="Gemma4-only attention backend selector. 'eager' (default) is the bitwise "
        "eager path; 'ffpa_flash' is the opt-in two-backend kernel path (FFPA full + "
        "Flash sliding). Distinct from the base --attention-backend (TE's AttnBackend "
        "enum), which Gemma4 does not use.",
    )
    group.add_argument(
        "--gemma4-allow-eager-packed",
        action="store_true",
        help="Allow the EAGER attention backend to run on packed sequences (cu_seqlens), "
        "which it otherwise refuses (its additive-mask path leaks across documents). Safe "
        "ONLY when every packed sequence is a SINGLE document (e.g. one conversation per "
        "line + MBS=1) -- then the causal mask is leak-free and eager matches the tp-sp "
        "branch's eager SFT. No effect with --gemma4-attention-backend ffpa_flash.",
    )
    group.add_argument(
        "--freeze-ple",
        action="store_true",
        help="Freeze the Per-Layer-Embedding table (Gemma4PLE.embed_tokens_per_layer, "
        "~2.8B params). It is a plain nn.Embedding REPLICATED on every TP rank (not "
        "VocabParallel), so training it costs a full fp32 main-grad buffer (~11GB) + "
        "Adam state (~8.5GB) PER RANK. Freezing it (requires_grad=False before the "
        "distributed optimizer is built) frees ~20GB static -- the difference between "
        "OOM and fitting a long SEQLEN. Recommended for SFT (PLE is a base lookup table).",
    )
    return parser


def gemma4_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Build a :class:`Gemma4Model` (mirrors ``gpt_builder`` for the GPT model)."""
    print_rank_0('building Gemma4 model ...')
    if config is None:
        # Build the Gemma4 subclass so its heterogeneous per-layer specs, softcap,
        # sqrt(H) embedding scaling, and PLE dims are populated from the defaults.
        config = core_transformer_config_from_args(args, config_class=Gemma4TransformerConfig)
        # core_transformer_config_from_args does NOT set gemma4-only fields; wire the
        # attention backend from the CLI arg here (default "eager"). This is a str, not
        # the base AttnBackend enum -> get_config_for_layer propagates it per-layer.
        config.gemma4_attention_backend = getattr(args, "gemma4_attention_backend", "eager")
        config.gemma4_allow_eager_packed = getattr(args, "gemma4_allow_eager_packed", False)

    # Gemma4 MLP is a GeGLU with tanh-approx GELU. There is no CLI flag for this exact
    # combination (--swiglu is SiLU, --quick-geglu is quick_gelu), so set it here to
    # match the HF model regardless of the activation flags passed on the command line.
    config.gated_linear_unit = True
    config.activation_func = partial(torch.nn.functional.gelu, approximate="tanh")
    config.bias_activation_fusion = False

    # Gemma4 architectural flags that have NO dedicated CLI flag (or whose CLI default is
    # wrong for this model) and are mandatory for the layer spec to build correctly. Force
    # them here so a training command can't silently mis-build (esp. under TP>1/SP):
    #   * qk_layernorm: the spec wires q/k RMSNorm builders; base attention rejects them
    #     unless qk_layernorm=True (attention.py), and the CLI default is False.
    #   * add_bias_linear / add_qkv_bias: Gemma4 has no linear biases; CLI default is True.
    config.qk_layernorm = True
    config.add_bias_linear = False
    config.add_qkv_bias = False
    # num_query_groups is a real dimension (E4B = 2 kv groups vs 8 q heads). It must be set
    # to a GQA value; the CLI defaults it to num_attention_heads, which silently breaks
    # GQA head-sharding at TP>num_query_groups (e.g. TP=8). Fail loudly if mis-set.
    assert (
        config.num_query_groups is not None
        and config.num_query_groups < config.num_attention_heads
    ), (
        f"Gemma4 requires GQA: num_query_groups ({config.num_query_groups}) must be set "
        f"< num_attention_heads ({config.num_attention_heads}); pass "
        f"--group-query-attention --num-query-groups 2 (E4B)."
    )

    # The PLE per-layer token table is indexed by the SAME vocab as embed_tokens
    # (HF: vocab_size_per_layer_input == vocab_size). Keep them consistent with the
    # model vocab so a smaller --vocab-size (e.g. for a smoke run) shrinks the PLE
    # table too instead of leaving it pinned at the 262144 default.
    config.vocab_size_per_layer_input = args.padded_vocab_size

    use_te = args.transformer_impl == "transformer_engine"
    if use_te:
        transformer_layer_spec = get_gemma4_layer_with_transformer_engine_spec(config)
    else:
        transformer_layer_spec = get_gemma4_layer_local_spec(config)

    model = Gemma4Model(
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

    # Optionally freeze the Per-Layer-Embedding table. Done HERE (before the optimizer /
    # DDP grad buffers are built in pretrain()) so the distributed optimizer skips these
    # params entirely -- no grad buffer, no Adam state -> frees ~20GB/rank. The PLE table
    # is a plain nn.Embedding replicated across TP, so this is the single biggest static win.
    if getattr(args, "freeze_ple", False) and hasattr(model, "ple"):
        n_frozen = 0
        for p in model.ple.parameters():
            p.requires_grad_(False)
            n_frozen += p.numel()
        print_rank_0(f"  froze PLE (model.ple): {n_frozen/1e9:.2f}B params requires_grad=False")

    return model


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup).
    _MAIN_ENTRY_TIME = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets.
    setattr(train_valid_test_datasets_provider, "is_distributed", True)

    args = parse_and_validate_args(
        extra_args_provider=add_gemma4_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    # Use the Gemma4 transformer config in the pretrain config container too so the
    # container is consistent with the model the builder constructs.
    transformer_cfg = core_transformer_config_from_args(args, config_class=Gemma4TransformerConfig)
    # Keep the container config consistent with the builder: set the gemma4-only
    # attention backend from the CLI arg (core_transformer_config_from_args does not).
    transformer_cfg.gemma4_attention_backend = getattr(args, "gemma4_attention_backend", "eager")
    transformer_cfg.gemma4_allow_eager_packed = getattr(args, "gemma4_allow_eager_packed", False)
    model_cfg = gpt_config_from_args(args, config=transformer_cfg)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        partial(model_provider, gemma4_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        get_embedding_ranks=get_embedding_ranks,
    )
