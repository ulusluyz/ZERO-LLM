import torch
import torch.nn as nn
from config import ModelConfig
from rmsnorm import RMSNorm
from attention import CausalSelfAttention
from mlp import SwiGLUMLP
from rope import RoPE


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, rope: RoPE):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = CausalSelfAttention(config, rope)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # Pre-Norm Causal Attention with Residual Connection
        h = x + self.attention(self.input_layernorm(x), start_pos=start_pos)
        # Pre-Norm Feed Forward Network with Residual Connection
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out
