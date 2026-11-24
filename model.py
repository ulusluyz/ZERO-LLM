import math
import torch
import torch.nn as nn
from config import ModelConfig
from embeddings import TokenEmbedding
from rope import RoPE
from rmsnorm import RMSNorm
from transformer_block import TransformerBlock
from lm_head import LMHead


class ZEROModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.tok_embeddings = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.rope = RoPE(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(config, self.rope) for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size)

        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.embedding.weight

        # Initialize weights with LLaMA/GPT style scheme
        self.apply(self._init_weights)

        # Scale residual projections once, after ordinary initialization.
        scaled_std = 0.02 / math.sqrt(2 * config.num_layers)
        for layer in self.layers:
            torch.nn.init.normal_(layer.attention.o_proj.weight, mean=0.0, std=scaled_std)
            torch.nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=scaled_std)

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, LMHead):
            if not self.config.tie_embeddings:
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0,
                kv_cache=None, use_cache: bool = False):
        """
        input_ids shape: (batch_size, seq_len)
        Returns logits (batch, sequence, vocabulary). With use_cache=True,
        returns (logits, per-layer KV cache); caching requires eval mode.
        """
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must be a nonempty (batch, sequence) tensor")
        if use_cache and kv_cache is None and start_pos != 0:
            raise ValueError("A new KV cache must start at position zero")
        if kv_cache is not None:
            if not use_cache or len(kv_cache) != len(self.layers):
                raise ValueError("kv_cache requires use_cache and one entry per layer")
            start_pos = kv_cache[0][0].size(2)
            if any(entry[0].size(2) != start_pos for entry in kv_cache):
                raise ValueError("Inconsistent cache lengths")
        x = self.tok_embeddings(input_ids)
        new_cache = []

        for i, layer in enumerate(self.layers):
            result = layer(x, start_pos=start_pos,
                           kv_cache=kv_cache[i] if kv_cache is not None else None,
                           use_cache=use_cache)
            if use_cache:
                x, cache = result
                new_cache.append(cache)
            else:
                x = result

        x = self.norm(x)
        logits = self.lm_head(x)
        return (logits, new_cache) if use_cache else logits

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Returns total or non-embedding trainable parameter count.
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.tok_embeddings.embedding.weight.numel()
        return n_params
