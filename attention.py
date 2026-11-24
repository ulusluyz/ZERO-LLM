import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig
from rope import RoPE


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, rope: RoPE):
        super().__init__()
        config.validate()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rope = rope
        self.dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, start_pos: int = 0, kv_cache=None, use_cache=False):
        if use_cache and self.training:
            raise ValueError("KV caching is only supported in eval mode")
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

        # Transpose to (batch, num_heads, seq_len, head_dim) for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        past_len = 0
        if kv_cache is not None:
            if not use_cache:
                raise ValueError("kv_cache requires use_cache=True")
            past_k, past_v = kv_cache
            if past_k.shape != past_v.shape or past_k.shape[:2] != (batch_size, self.num_kv_heads) or past_k.size(-1) != self.head_dim:
                raise ValueError("Invalid KV cache shape")
            past_len = past_k.size(2)
            if start_pos != past_len:
                raise ValueError("Cache length and start_pos disagree")
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        cache = (k, v)
        # Cached chunks need a lower-right aligned mask; single-token decode
        # can attend to the entire cache. SDPA selects optimized backends.
        mask = None
        if past_len and seq_len > 1:
            mask = torch.arange(k.size(2), device=x.device)[None, :] <= (
                past_len + torch.arange(seq_len, device=x.device)[:, None])
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=past_len == 0,
            dropout_p=self.dropout_p if self.training else 0.0,
            enable_gqa=self.num_kv_groups > 1,
        )

        # Reshape back to (batch, seq_len, hidden_size)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        output = self.resid_dropout(output)
        return (output, cache) if use_cache else output
