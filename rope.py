import torch
import torch.nn as nn
from typing import Tuple


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute freqs_cis
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        x shape: (batch, seq_len, num_heads, head_dim)
        """
        seq_len = x.size(1)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        # Reshape x to complex numbers
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

        # Broadcast freqs_cis to (1, seq_len, 1, head_dim // 2)
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

        # Apply rotation
        x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
        return x_out.type_as(x)
