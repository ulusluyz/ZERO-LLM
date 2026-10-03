import torch
import pytest
from config import get_tiny_config
from model import ZEROModel
from inference import generate_tokens, sample_top_k_top_p


def test_top_k_top_p_sampling_masking():
    logits = torch.tensor([[1.0, 2.0, 10.0, 3.0, 0.5]])

    # Top-K = 2: Only 10.0 and 3.0 should remain, others set to -inf
    filtered_k = sample_top_k_top_p(logits, top_k=2)
    assert torch.isinf(filtered_k[0, 0])
    assert torch.isinf(filtered_k[0, 1])
    assert not torch.isinf(filtered_k[0, 2])
    assert not torch.isinf(filtered_k[0, 3])

    # Top-P = 0.5: Only the highest logit 10.0 should remain
    filtered_p = sample_top_k_top_p(logits, top_p=0.5)
    assert not torch.isinf(filtered_p[0, 2])


def test_greedy_token_generation():
    config = get_tiny_config()
    model = ZEROModel(config)
    prompt_ids = [10, 20, 30]

    out_ids = generate_tokens(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=10,
        temperature=0.0,
    )

    assert len(out_ids) == len(prompt_ids) + 10
    assert out_ids[: len(prompt_ids)] == prompt_ids


def test_temperature_sampling_generation():
    config = get_tiny_config()
    model = ZEROModel(config)
    prompt_ids = [5, 15, 25]

    out_ids = generate_tokens(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=10,
        temperature=0.8,
        top_k=10,
        top_p=0.9,
    )

    assert len(out_ids) == len(prompt_ids) + 10
    assert out_ids[: len(prompt_ids)] == prompt_ids
