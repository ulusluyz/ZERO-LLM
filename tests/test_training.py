import os
import tempfile
import torch
import torch.nn as nn
import pytest

from config import get_tiny_config
from model import ZEROModel
from dataset import TextDataset
from checkpoint import save_checkpoint, load_checkpoint


def test_forward_backward_loss_and_optimizer_step():
    config = get_tiny_config()
    model = ZEROModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Initial loss
    logits = model(input_ids)
    loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))
    initial_loss = loss.item()

    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    # Optimization steps to check loss reduction
    for _ in range(5):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"


def test_checkpoint_save_and_reload():
    config = get_tiny_config()
    model = ZEROModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    out_before = model(input_ids)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_checkpoint(tmp_path, model, optimizer, step=42, config_dict=config.__dict__)

        # New model instance
        new_model = ZEROModel(config)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)

        ckpt_info = load_checkpoint(tmp_path, new_model, new_optimizer)

        assert ckpt_info["step"] == 42
        out_after = new_model(input_ids)

        assert torch.allclose(out_before, out_after, atol=1e-6)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_text_dataset_creation():
    token_ids = list(range(100))
    seq_len = 16
    dataset = TextDataset(token_ids, seq_len=seq_len)

    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)
    assert torch.equal(x[1:], y[:-1])
