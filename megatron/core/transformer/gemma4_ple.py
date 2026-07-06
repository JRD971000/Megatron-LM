# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.gemma4_norm import Gemma4RMSNorm
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class Gemma4PLE(MegatronModule):
    """Per-Layer Embeddings for Gemma 4 (modeling_gemma4.py:1615-1630, 1737-1814).

    Computes ``per_layer_inputs`` [B, S, L, P] fed one slice per decoder layer:

        E_tok  = embed_tokens_per_layer(ids) * sqrt(P)        # bf16-rounded scale
        E_proj = per_layer_projection_norm(
                     per_layer_model_projection(inputs_embeds) * (1 / sqrt(H)))
        per_layer_inputs = (E_proj + E_tok) * (1 / sqrt(2))

    ``inputs_embeds`` are the already-√H-scaled token embeddings. Embedding scales
    are cast to the embedding weight dtype before the multiply, matching HF's
    ``Gemma4TextScaledWordEmbedding`` bf16 rounding.

    ``embed_tokens_per_layer`` is a :class:`VocabParallelEmbedding` sharded over
    the vocab dimension across TP (262144 x 10752 = 2.82B params full; the
    replicated original cost ~5.6GB bf16 weights + ~45GB grad/Adam state PER
    RANK when trained, and its unsharded main grad exceeded the int32 numel
    limit of the TE multi-tensor kernels at DP=1). The masked local lookup +
    all-reduce reconstructs values exactly (each token id is owned by exactly
    one rank, so the sum adds zeros); at TP=1 it degenerates to a plain lookup,
    preserving the bitwise-parity path. This class must be a MegatronModule so
    dist-checkpointing recurses into the embedding's own ``sharded_state_dict``
    (vocab-dim ShardedTensor) instead of flattening it as replicated.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        num_layers: int,
        ple_dim: int = 256,
        vocab_size_per_layer_input: int = 262144,
        eps: float = 1e-6,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)
        self.num_layers = num_layers
        self.ple_dim = ple_dim
        self.embed_scale = ple_dim**0.5
        self.projection_scale = hidden_size**-0.5
        self.input_scale = 2.0**-0.5

        # NOTE: fresh (random) initialization uses config.init_method, unlike the
        # original plain nn.Embedding's N(0,1) reset. Gemma4 PLE weights are always
        # populated from a converted HF checkpoint, so only random-init smokes see
        # this; documented rather than matched.
        self.embed_tokens_per_layer = tensor_parallel.VocabParallelEmbedding(
            vocab_size_per_layer_input,
            num_layers * ple_dim,
            init_method=config.init_method,
            reduce_scatter_embeddings=False,  # PLE precompute needs the full sequence
            config=config,
            tp_group=tp_group,
        )
        self.per_layer_model_projection = torch.nn.Linear(hidden_size, num_layers * ple_dim, bias=False)
        self.per_layer_projection_norm = Gemma4RMSNorm(ple_dim, eps=eps)

    def forward(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # Token-identity component (E_tok).
        scale = torch.tensor(self.embed_scale).to(self.embed_tokens_per_layer.weight.dtype)
        e_tok = self.embed_tokens_per_layer(input_ids) * scale
        e_tok = e_tok.reshape(*input_ids.shape, self.num_layers, self.ple_dim)

        # Context component (E_proj).
        e_proj = self.per_layer_model_projection(inputs_embeds) * self.projection_scale
        e_proj = e_proj.reshape(*inputs_embeds.shape[:-1], self.num_layers, self.ple_dim)
        e_proj = self.per_layer_projection_norm(e_proj)

        return (e_proj + e_tok) * self.input_scale
