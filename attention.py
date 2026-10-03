import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig
from rope import RoPE


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, rope: RoPE):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        assert config.hidden_size == self.num_heads * self.head_dim, "hidden_size must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rope = rope
        self.dropout_p = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        self.resid_dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, seq_len, num_heads/num_kv_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q = self.rope(q, start_pos=start_pos)
        k = self.rope(k, start_pos=start_pos)

        # Repeat KV heads for Grouped-Query Attention (GQA) if num_kv_groups > 1
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        # Transpose to (batch, num_heads, seq_len, head_dim) for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Explicit Causal Attention Calculation: (Q @ K^T) / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask: tril mask with -inf for future tokens
        if seq_len > 1:
            causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=scores.dtype)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)

        # Output = Probs @ V -> shape: (batch, num_heads, seq_len, head_dim)
        output = torch.matmul(probs, v)

        # Reshape back to (batch, seq_len, hidden_size)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        return self.resid_dropout(output)
