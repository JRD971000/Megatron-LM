"""Shared setup + model builders for the Gemma 4 E4B example scripts.

Import THIS FIRST in any container-side Gemma4 script (it sets up sys.path for the
Megatron-LM source tree and works around a container<->MLM incompatibility before
megatron is imported). It also provides the three helpers the conversion / parity
scripts share: ``_init_distributed``, ``_make_config`` and ``_build_model``.

Container<->MLM workaround: the container's ``nvidia_resiliency_ext`` ships the
async-ckpt modules but exposes no ``__version__``, so MLM's
``dist_checkpointing/strategies/nvrx.py:is_nvrx_min_version()`` crashes at import
(AttributeError). We populate ``__version__`` before megatron is imported. These
single-device scripts never use async checkpointing, so the value only needs to
satisfy the ``>=0.6.0`` guard.

Run scripts with the CONTAINER python (this module first).
"""
import functools
import os
import sys

# Megatron-LM root is three levels up from this file: examples/gemma4/ -> examples/ -> <repo>.
MLM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if MLM not in sys.path:
    sys.path.insert(0, MLM)

try:
    import nvidia_resiliency_ext as _nvrx  # noqa: E402

    # MLM's is_nvrx_min_version() reads _nvrx.__version__ and asserts it is >= 0.6.0.
    # Two failure modes in containers: (a) __version__ missing entirely (AttributeError),
    # or (b) __version__ is a dev pre-release of 0.6.0 (e.g. "0.6.0.dev69+..."), which
    # PEP 440 orders BELOW the final "0.6.0" and so fails the guard. Force a satisfying
    # value in both cases. These single-device scripts never use async checkpointing.
    _need = True
    try:
        from packaging.version import Version as _V
        _cur = getattr(_nvrx, "__version__", None)
        _need = _cur is None or _V(str(_cur)) < _V("0.6.0")
    except Exception:
        _need = not hasattr(_nvrx, "__version__")
    if _need:
        _nvrx.__version__ = "0.6.0"
except Exception:
    # nvrx absent entirely -> MLM's HAVE_NVRX=False path handles it fine.
    pass

import torch  # noqa: E402

# Gemma4 MLP/PLE use gelu_pytorch_tanh == F.gelu(approximate="tanh"). Use the full
# sqrt(2/pi) constant (not fast_gelu's truncated 0.7978845608) for bitwise fidelity.
GELU_TANH = functools.partial(torch.nn.functional.gelu, approximate="tanh")

# Pure-python FFPA wheel (installed --no-deps; see V_RESULTS.md V0). Used via
# PYTHONPATH / sys.path so ``import ffpa_attn`` resolves for the ffpa_flash backend.
FFPA_INSTALL = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "Gemma4_mlm/ffpa_install"
)


def _ensure_ffpa_on_path(install=FFPA_INSTALL):
    """Make ``import ffpa_attn`` resolvable (sys.path + PYTHONPATH for subprocesses).

    Returns True if the install dir exists and is now on the path. Idempotent.
    Only needed for the ``ffpa_flash`` backend; the eager path never imports ffpa.
    """
    if not install or not os.path.isdir(install):
        return False
    if install not in sys.path:
        sys.path.insert(0, install)
    pp = os.environ.get("PYTHONPATH", "")
    if install not in pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = install + (os.pathsep + pp if pp else "")
    return True


def _apply_attention_backend(config, backend):
    """Set ``config.attention_backend`` on a Gemma4TransformerConfig.

    ``attention_backend`` is the opt-in two-backend flag added by I1/I2 (default
    ``"eager"``). This harness only SETS it; the dispatch lives in megatron/core.
    If the field is absent (I1/I2 not yet merged into this tree) the instance
    attribute is still set and ``field_present=False`` is returned so callers can
    warn that the ffpa_flash path awaits the merge. ``backend`` None or ``"eager"``
    is a behavioural no-op (eager is the default).
    """
    field_present = hasattr(config, "attention_backend")
    if backend is not None:
        # Gemma4TransformerConfig is a non-frozen dataclass -> plain setattr. When
        # the field exists, get_config_for_layer's re-attach loop propagates it
        # per-layer automatically (gemma4_config.py); when it does not, this is an
        # inert instance attribute until I1/I2 land.
        setattr(config, "attention_backend", backend)
    return config, field_present


def _init_distributed():
    """Single-process (DDP=TP=PP=CP=SP=1) init for the example scripts."""
    import megatron.core.parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "12399")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    ps.initialize_model_parallel(1, 1)
    # Required before building VocabParallelEmbedding / parallel linears (adds the
    # 'model-parallel-rng' cuda rng state used by _initialize_affine_weight_gpu).
    model_parallel_cuda_manual_seed(123)


def _make_config(num_layers=42, hidden=2560, ffn=10240, attention_backend=None):
    """Real Gemma 4 E4B Megatron config (local-spec bitwise target).

    ``attention_backend`` (``"eager"`` | ``"ffpa_flash"`` | None) selects the
    opt-in two-backend attention path (I1/I2). None/``"eager"`` = default eager
    behaviour. See ``_apply_attention_backend``.
    """
    from megatron.core.transformer.gemma4_config import Gemma4TransformerConfig

    config = Gemma4TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        ffn_hidden_size=ffn,
        num_attention_heads=8,
        num_query_groups=2,
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=GELU_TANH,
        add_bias_linear=False,
        qk_layernorm=True,
        bias_activation_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_softmax_in_fp32=True,
        masked_softmax_fusion=False,
        pipeline_dtype=torch.bfloat16,
    )
    config, _ = _apply_attention_backend(config, attention_backend)
    return config


def _build_model(spec_fn, config, vocab=262144):
    """Build a bf16/cuda Gemma4Model from a layer-spec factory."""
    from megatron.core.models.gemma4.gemma4_model import Gemma4Model

    spec = spec_fn(config)
    model = Gemma4Model(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=vocab,
        max_sequence_length=512,
    )
    return model.bfloat16().cuda()
