import math
import torch
import pytest
from config import get_tiny_config
from rope import RoPE
from attention import CausalSelfAttention


def test_rope_dimensions_and_rotation():
    dim = 32
    seq_len = 10
    batch_size = 2
    num_heads = 4

    rope = RoPE(dim=dim, max_seq_len=64)
    x = torch.randn(batch_size, seq_len, num_heads, dim)
    out = rope(x)

    assert out.shape == x.shape
    # Norm of embeddings should be preserved after rotation
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-4)


def test_causal_self_attention_shape():
    config = get_tiny_config()
    rope = RoPE(dim=config.head_dim, max_seq_len=config.max_seq_len)
    attn = CausalSelfAttention(config, rope)

    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    out = attn(x)

    assert out.shape == (batch_size, seq_len, config.hidden_size)


def test_causal_mask_prevents_future_leakage():
    config = get_tiny_config()
    rope = RoPE(dim=config.head_dim, max_seq_len=config.max_seq_len)
    attn = CausalSelfAttention(config, rope)
    attn.eval()

    seq_len = 5
    x1 = torch.randn(1, seq_len, config.hidden_size)
    x2 = x1.clone()
    # Change future tokens in x2 (at index 3 and 4)
    x2[:, 3:, :] = torch.randn(1, 2, config.hidden_size)

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    # Outputs at index 0, 1, 2 should be identical because future tokens (3,4) must be masked
    assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-5)
    # Output at index 3 and 4 should differ
    assert not torch.allclose(out1[:, 3:, :], out2[:, 3:, :], atol=1e-5)


def test_grouped_query_attention():
    config = get_tiny_config()
    config.num_attention_heads = 8
    config.num_kv_heads = 2
    rope = RoPE(dim=config.head_dim, max_seq_len=config.max_seq_len)
    attn = CausalSelfAttention(config, rope)

    x = torch.randn(2, 8, config.hidden_size)
    out = attn(x)
    assert out.shape == (2, 8, config.hidden_size)
