import copy
import math
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from attention import CausalSelfAttention
from checkpoint import save_checkpoint, load_checkpoint, read_checkpoint, find_latest_checkpoint
from config import ModelConfig
from dataset import TextDataset, IGNORE_INDEX
from inference import generate_tokens, sample_top_k_top_p
from model import ZEROModel
from rope import RoPE
from tokenizer import Tokenizer
from train import train, train_epoch, make_scheduler, data_sources, DEMO_TRAIN, tokenize_source


def small_config(**changes):
    values = dict(vocab_size=260, hidden_size=16, num_layers=2,
                  num_attention_heads=4, num_kv_heads=2, intermediate_size=32, max_seq_len=16)
    values.update(changes)
    return ModelConfig(**values)


def test_padding_and_tail_targets_are_not_training_labels():
    dataset = TextDataset([2, 8, 0, 3, 10, 11], seq_len=4, pad_id=1)
    assert len(dataset) == 2
    x, y = dataset[1]
    assert x.tolist() == [10, 1, 1, 1]
    assert y.tolist() == [11, -100, -100, -100]
    targets = torch.cat([dataset[i][1] for i in range(len(dataset))])
    assert targets[targets != IGNORE_INDEX].tolist() == [8, 0, 3, 10, 11]
    logits = torch.randn(4, 260, requires_grad=True)
    loss = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX)
    loss.backward()
    assert torch.equal(logits.grad[1:], torch.zeros_like(logits.grad[1:]))


@pytest.mark.parametrize('tokens,length', [([], 8), ([1], 8), ([1, 2], 0)])
def test_invalid_dataset_rejected(tokens, length):
    with pytest.raises(ValueError):
        TextDataset(tokens, length)


def test_default_tokenizer_does_not_collapse_corpus_and_handles_specials():
    text = '\n'.join(DEMO_TRAIN)
    tok = Tokenizer()
    tok.train(text, 1000)
    assert len(tok.encode(text)) > 100
    assert tok.decode(tok.encode(text)) == text
    ids = tok.encode('merhaba</s><s>dünya')
    assert tok.eos_id in ids and tok.bos_id in ids
    assert tok.decode(ids) == 'merhaba</s><s>dünya'
    with pytest.raises(ValueError, match='fresh tokenizer'):
        tok.train(text, 1001)


def test_validation_split_and_duplicate_detection(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('birinci belge\nikinci belge\nüçüncü belge\n')
    training, validation = data_sources(corpus)
    assert list(training()) == ['birinci belge', 'ikinci belge']
    assert list(validation()) == ['üçüncü belge']
    tok = Tokenizer()
    _, _, hashes = tokenize_source(lambda: iter(['aynı belge']), tok, tmp_path / 'train.bin')
    with pytest.raises(ValueError, match='Identical document'):
        tokenize_source(lambda: iter(['aynı belge']), tok, tmp_path / 'val.bin', hashes)
    corpus.write_text('tek belge')
    with pytest.raises(ValueError, match='two nonempty'):
        data_sources(corpus)


def test_accumulation_matches_large_batch_with_partial_group_and_padding():
    torch.manual_seed(11)
    a = ZEROModel(small_config())
    b = copy.deepcopy(a)
    dataset = TextDataset(list(range(10)), 4)
    assert len(dataset) == 3
    def run(model, batch, accumulation):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = make_scheduler(optimizer, 2)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        return train_epoch(model, DataLoader(dataset, batch_size=batch), optimizer,
                           scheduler, scaler, torch.device('cpu'), accumulation, 1e9)
    first = run(a, 1, 2)
    second = run(b, 2, 1)
    assert first[1:] == second[1:] == (9, 2)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)


def test_cosine_schedule_stops_at_zero():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1)
    scheduler = make_scheduler(optimizer, 2)
    rates = []
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]['lr'])
    assert rates == [0.5, 0, 0, 0, 0]


def test_amp_skipped_update_does_not_advance_schedule():
    class SkippingScaler:
        scale_value = 8
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): pass
        def update(self): self.scale_value /= 2
        def get_scale(self): return self.scale_value
    model = ZEROModel(small_config())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = make_scheduler(optimizer, 1)
    before = copy.deepcopy(model.state_dict())
    _, _, updates = train_epoch(model, DataLoader(TextDataset([2, 8, 3], 4)), optimizer,
                                scheduler, SkippingScaler(), torch.device('cpu'))
    assert updates == scheduler.last_epoch == 0
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[key])


@pytest.mark.parametrize('kv_heads', [1, 2, 4])
def test_gqa_matches_explicit_attention_reference(kv_heads):
    config = small_config(num_kv_heads=kv_heads)
    rope = RoPE(config.head_dim, config.max_seq_len)
    attn = CausalSelfAttention(config, rope).eval()
    x = torch.randn(2, 5, config.hidden_size)
    q = rope(attn.q_proj(x).view(2, 5, 4, 4)).transpose(1, 2)
    k = rope(attn.k_proj(x).view(2, 5, kv_heads, 4)).transpose(1, 2)
    v = attn.v_proj(x).view(2, 5, kv_heads, 4).transpose(1, 2)
    k = k.repeat_interleave(4 // kv_heads, dim=1)
    v = v.repeat_interleave(4 // kv_heads, dim=1)
    scores = q @ k.transpose(-1, -2) / 2
    scores.masked_fill_(torch.ones(5, 5, dtype=torch.bool).triu(1), -float('inf'))
    reference = (scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 5, 16)
    torch.testing.assert_close(attn(x), attn.o_proj(reference), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('kv_heads', [1, 2, 4])
def test_cache_matches_full_forward_for_token_and_chunk(kv_heads):
    model = ZEROModel(small_config(num_kv_heads=kv_heads)).eval()
    ids = torch.randint(0, 260, (2, 8))
    with torch.no_grad():
        reference = model(ids)
        prefix, cache = model(ids[:, :3], use_cache=True)
        chunk, cache = model(ids[:, 3:6], kv_cache=cache, use_cache=True)
        pieces = [prefix, chunk]
        for i in range(6, 8):
            out, cache = model(ids[:, i:i + 1], kv_cache=cache, use_cache=True)
            pieces.append(out)
    torch.testing.assert_close(torch.cat(pieces, dim=1), reference, atol=1e-6, rtol=1e-5)
    assert cache[0][0].shape == (2, kv_heads, 8, 4)


def test_cached_generation_matches_uncached_at_window_rollover():
    model = ZEROModel(small_config(max_seq_len=4)).eval()
    cached = generate_tokens(model, [2, 10, 11, 12, 13], 8, temperature=0, use_cache=True)
    plain = generate_tokens(model, [2, 10, 11, 12, 13], 8, temperature=0, use_cache=False)
    assert cached == plain


def test_eos_stops_generation(monkeypatch):
    model = ZEROModel(small_config())
    def force_eos(x):
        logits = x.new_zeros(*x.shape[:-1], 260)
        logits[..., 3] = 10
        return logits
    monkeypatch.setattr(model.lm_head, 'forward', force_eos)
    assert generate_tokens(model, [2, 10], 10, temperature=0, eos_id=3) == [2, 10, 3]


@pytest.mark.parametrize('kwargs', [dict(prompt_ids=[]), dict(prompt_ids=[2], temperature=-1),
    dict(prompt_ids=[2], top_p=0), dict(prompt_ids=[260]), dict(prompt_ids=[2], max_new_tokens=-1)])
def test_invalid_generation_parameters(kwargs):
    with pytest.raises(ValueError):
        generate_tokens(ZEROModel(small_config()), **kwargs)


@pytest.mark.parametrize('values', [dict(hidden_size=15), dict(num_kv_heads=3),
    dict(num_layers=0), dict(num_attention_heads=0), dict(max_seq_len=0), dict(dropout=1),
    dict(rope_theta=float("inf")), dict(rms_norm_eps=float("inf"))])
def test_invalid_configs(values):
    with pytest.raises(ValueError):
        small_config(**values)


def test_rope_angles_relative_positions_and_bounds():
    rope = RoPE(4, max_seq_len=8)
    vector = torch.tensor([[[[1., 0., 1., 0.]]]])
    out = rope(vector, start_pos=1).flatten()
    torch.testing.assert_close(out, torch.tensor([math.cos(1), math.sin(1), math.cos(.01), math.sin(.01)]))
    q, k = torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4)
    torch.testing.assert_close((rope(q, 2) * rope(k, 5)).sum(), (q * rope(k, 3)).sum())
    with pytest.raises(ValueError): rope(vector, 8)
    with pytest.raises(ValueError): rope(vector, -1)
    with pytest.raises(ValueError): RoPE(3)


def test_checkpoint_rng_optimizer_and_tokenizer_round_trip(tmp_path):
    tok = Tokenizer()
    model = ZEROModel(small_config())
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.tensor([[2, 7, 3]])).sum().backward()
    optimizer.step()
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    generator = torch.Generator().manual_seed(7)
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model, optimizer, tokenizer=tok, scaler=scaler, epoch=1, data_generator=generator)
    expected = (random.random(), np.random.random(), torch.rand(3), torch.randperm(8, generator=generator))
    restored = ZEROModel(small_config())
    opt = torch.optim.AdamW(restored.parameters())
    load_checkpoint(path, restored, opt, tokenizer=tok, scaler=scaler,
                    data_generator=generator, restore_random=True)
    assert expected[:2] == (random.random(), np.random.random())
    assert torch.equal(expected[2], torch.rand(3))
    assert torch.equal(expected[3], torch.randperm(8, generator=generator))
    assert opt.state_dict()['state']
    for key in model.state_dict():
        assert torch.equal(model.state_dict()[key], restored.state_dict()[key])
    other = Tokenizer(special_tokens=['<pad>', '<unk>', '<s>', '</s>'])
    with pytest.raises(ValueError, match='tokenizer'):
        load_checkpoint(path, restored, tokenizer=other)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_nested_equal(x, y)
    else:
        assert a == b


def test_training_resume_equivalence_and_checkpoint_only_generation(tmp_path, monkeypatch):
    import train as training
    monkeypatch.setattr(training, 'get_device', lambda: torch.device('cpu'))
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    common = ['--epochs', '2', '--seq_len', '32', '--batch_size', '3', '--grad_accum_steps', '3']
    train(common + ['--checkpoint_dir', str(full)])
    train(common + ['--checkpoint_dir', str(resumed), '--stop_after_epoch', '1'])
    first = find_latest_checkpoint(resumed)
    train(common + ['--checkpoint_dir', str(resumed), '--resume_from', str(first)])
    full_final = read_checkpoint(find_latest_checkpoint(full))
    final_path = find_latest_checkpoint(resumed)
    final = read_checkpoint(final_path)
    assert full_final['epoch'] == final['epoch'] == 2
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
                'scaler_state_dict', 'rng_state', 'data_generator_state', 'tokenizer'):
        assert_nested_equal(full_final[key], final[key])
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / 'generate.py'), '--checkpoint', str(final_path),
                             '--prompt', '', '--max_new_tokens', '2'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Loaded checkpoint' in result.stdout
    # Same vocabulary size, different token-ID semantics must still be rejected.
    other = Tokenizer.from_dict(final['tokenizer'])
    other.special_tokens[:2] = list(reversed(other.special_tokens[:2]))
    other.vocab[0], other.vocab[1] = other.vocab[1], other.vocab[0]
    wrong = tmp_path / 'wrong.json'
    other.save(wrong)
    result = subprocess.run([sys.executable, str(repo / 'generate.py'), '--checkpoint', str(final_path),
                             '--tokenizer_file', str(wrong)], capture_output=True, text=True)
    assert result.returncode != 0 and 'does not match checkpoint' in result.stderr
    with pytest.raises(ValueError, match='same data'):
        train(common + ['--checkpoint_dir', str(resumed), '--resume_from', str(first), '--lr', '0.02'])


@pytest.mark.parametrize('argument', ['--corpus_file', '--tokenizer_file', '--resume_from'])
def test_missing_training_file_fails(tmp_path, argument):
    with pytest.raises(FileNotFoundError):
        train([argument, str(tmp_path / 'missing'), '--checkpoint_dir', str(tmp_path / 'out')])


def test_missing_generation_checkpoint_fails(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / 'generate.py'), '--checkpoint',
                             str(tmp_path / 'missing')], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'randomly initialized' not in result.stdout
