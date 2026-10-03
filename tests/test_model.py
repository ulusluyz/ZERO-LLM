import torch
import pytest
from config import (
    get_tiny_config,
    get_100m_config,
    get_300m_config,
    get_500m_config,
    get_1b_config,
)
from model import ZEROModel
from rmsnorm import RMSNorm
from mlp import SwiGLUMLP


def test_preset_configs():
    tiny = get_tiny_config()
    m100 = get_100m_config()
    m300 = get_300m_config()
    m500 = get_500m_config()
    b1 = get_1b_config()

    assert tiny.hidden_size == 128
    assert m100.hidden_size == 768
    assert m300.hidden_size == 1024
    assert m500.hidden_size == 1536
    assert b1.hidden_size == 2048


def test_model_initialization_and_forward():
    config = get_tiny_config()
    model = ZEROModel(config)

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_parameter_count_computation():
    config = get_tiny_config()
    model = ZEROModel(config)

    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)

    assert total_params > 0
    assert non_emb_params == total_params - (config.vocab_size * config.hidden_size)


def test_weight_tying():
    config = get_tiny_config()
    config.tie_embeddings = True
    model = ZEROModel(config)

    assert torch.equal(model.lm_head.weight, model.tok_embeddings.embedding.weight)


def test_rmsnorm_forward():
    norm = RMSNorm(dim=64)
    x = torch.randn(2, 10, 64)
    out = norm(x)

    assert out.shape == x.shape
    # Check mean squared value is close to 1
    rms = torch.rsqrt(out.pow(2).mean(-1) + norm.eps)
    assert torch.allclose(out.pow(2).mean(-1), torch.ones_like(out.pow(2).mean(-1)), atol=1e-2)


def test_swiglu_mlp_forward():
    mlp = SwiGLUMLP(hidden_size=128, intermediate_size=344)
    x = torch.randn(2, 10, 128)
    out = mlp(x)

    assert out.shape == (2, 10, 128)
