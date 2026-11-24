import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path
import time
from array import array
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ModelConfig, get_tiny_config, get_100m_config, get_300m_config, get_500m_config, get_1b_config
from model import ZEROModel
from tokenizer import Tokenizer
from dataset import TextDataset, IGNORE_INDEX
from checkpoint import (save_checkpoint, load_checkpoint, read_checkpoint,
                        find_latest_checkpoint, prune_checkpoints, checkpoint_candidates)
from recovery import ResumableBatchSampler, StopRequest, TrainingLock
from utils import set_seed, get_device, get_gpu_memory_info, get_model_summary


DEMO_TRAIN = [
    f"Örnek {i}: ZERO LLM sıfırdan öğrenir. Türkçe harfler ç Ç ğ Ğ ı İ ö Ö ş Ş ü Ü."
    for i in range(40)
]
DEMO_VAL = [
    "Doğrulama: Bugün hava açık, çocuklar bahçede oynuyor.",
    "Bağımsız örnek: Öğrenci kitabını masanın üzerine bıraktı.",
]


def documents(path):
    # One nonempty line is a document. Corpus files are scanned, not read at once.
    with open(path, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield line.rstrip("\r\n")


def data_sources(corpus_file=None, val_file=None):
    if corpus_file is None:
        if val_file is not None:
            raise ValueError("--val_file requires --corpus_file")
        return lambda: iter(DEMO_TRAIN), lambda: iter(DEMO_VAL)
    if not Path(corpus_file).is_file():
        raise FileNotFoundError(corpus_file)
    if val_file:
        if not Path(val_file).is_file():
            raise FileNotFoundError(val_file)
        if Path(corpus_file).resolve() == Path(val_file).resolve():
            raise ValueError("Training and validation files must be different")
        return lambda: documents(corpus_file), lambda: documents(val_file)
    count = sum(1 for _ in documents(corpus_file))
    if count < 2:
        raise ValueError("Provide at least two nonempty document lines or a separate --val_file")
    split = count - max(1, count // 10)
    return (lambda: itertools.islice(documents(corpus_file), split),
            lambda: itertools.islice(documents(corpus_file), split, None))


def sample_text(source, limit):
    chunks = []
    for doc in source():
        text = doc[:limit]
        chunks.append(text)
        limit -= len(text) + 1
        if limit <= 0:
            break
    return "\n".join(chunks)


def tokenize_source(source, tokenizer, path, forbidden_hashes=None):
    """Write an int64 token cache incrementally and return its mapped tensor."""
    digest = hashlib.sha256()
    doc_hashes = set()
    count = 0
    # Temporary cache: interrupted preprocessing must not overwrite a complete file.
    temporary = path.with_suffix('.tmp')
    try:
        with open(temporary, 'wb') as stream:
            for doc in source():
                doc_hash = hashlib.sha256(doc.strip().encode('utf-8')).hexdigest()
                if forbidden_hashes is not None and doc_hash in forbidden_hashes:
                    raise ValueError("Identical document appears in training and validation")
                doc_hashes.add(doc_hash)
                ids = [tokenizer.bos_id] + tokenizer.encode(doc) + [tokenizer.eos_id]
                payload = array('q', ids).tobytes()
                stream.write(payload)
                digest.update(payload)
                count += len(ids)
        if count < 3:
            raise ValueError("Training and validation each need nonempty text")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    tokens = torch.from_file(str(path), shared=False, size=count, dtype=torch.long)
    return tokens, digest.hexdigest(), doc_hashes


def make_scheduler(optimizer, total_steps):
    # Clamp after the planned horizon instead of starting another cosine cycle.
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)))


def train_epoch(model, loader, optimizer, scheduler, scaler, device,
                grad_accum_steps=1, max_grad_norm=1.0, mixed_precision=False, *,
                start_batch=0, initial_metrics=None, on_boundary=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    initial_metrics = initial_metrics or {}
    total_loss = initial_metrics.get('loss_sum', 0.0)
    total_tokens = initial_metrics.get('tokens', 0)
    group_tokens = updates = 0
    end_batch = start_batch + len(loader)
    for index, (x, y) in enumerate(loader, start=start_batch):
        x, y = x.to(device), y.to(device)
        count = int((y != IGNORE_INDEX).sum())
        with torch.amp.autocast(device_type=device.type, enabled=mixed_precision):
            logits = model(x)
            # Accumulate sums, then divide gradients by the actual valid-token count.
            loss_sum = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       y.reshape(-1), ignore_index=IGNORE_INDEX, reduction='sum')
        scaler.scale(loss_sum).backward()
        total_loss += loss_sum.item()
        total_tokens += count
        group_tokens += count
        if (index + 1) % grad_accum_steps == 0 or index + 1 == end_batch:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_tokens)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # AMP can skip an overflowing update; don't advance its schedule/step.
            if scaler.get_scale() >= old_scale:
                scheduler.step()
                updates += 1
            optimizer.zero_grad(set_to_none=True)
            group_tokens = 0
            status = {'next_batch': index + 1, 'loss_sum': total_loss, 'tokens': total_tokens,
                      'updates': initial_metrics.get('updates', 0) + updates}
            if on_boundary is not None and on_boundary(status):
                break
    return total_loss / total_tokens, total_tokens, updates


@torch.no_grad()
def evaluate(model, loader, device, mixed_precision=False):
    model.eval()
    loss_sum = 0.0
    tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=mixed_precision):
            logits = model(x)
            loss_sum += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                                        ignore_index=IGNORE_INDEX, reduction='sum').item()
        tokens += int((y != IGNORE_INDEX).sum())
    return loss_sum / tokens


def train(argv=None):
    parser = argparse.ArgumentParser(description="ZERO LLM - Training Engine")
    parser.add_argument('--preset', default=None, choices=['tiny', '100m', '300m', '500m', '1b'])
    parser.add_argument('--corpus_file')
    parser.add_argument('--val_file')
    parser.add_argument('--tokenizer_file')
    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    parser.add_argument('--resume_from', '--resume', help='Checkpoint path or auto for latest valid saved state')
    parser.add_argument('--epochs', type=int, default=None, help='Total planned epochs, including completed ones')
    parser.add_argument('--stop_after_epoch', type=int, help='Save and exit early without changing the planned schedule')
    parser.add_argument('--seq_len', type=int, help='Training context; defaults to preset maximum')
    parser.add_argument('--tokenizer_sample_chars', type=int, default=1_000_000)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--grad_accum_steps', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--mixed_precision', action='store_true', default=None)
    parser.add_argument('--no_mixed_precision', dest='mixed_precision', action='store_false', default=None)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--save_every_steps', type=int, default=100)
    parser.add_argument('--save_every_seconds', type=float, default=300.0)
    parser.add_argument('--keep_checkpoints', type=int, default=3)
    parser.add_argument('--stop_after_step', type=int, help='Save and exit at this global optimizer step')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args(argv)
    with TrainingLock(Path(args.checkpoint_dir)):
        return run_training(args)


def run_training(args):
    directory = Path(args.checkpoint_dir)
    resume_path = find_latest_checkpoint(directory) if args.resume_from == 'auto' else args.resume_from
    if not args.resume_from and checkpoint_candidates(directory):
        raise ValueError('Checkpoint directory is not empty; use --resume_from auto or a new directory')
    checkpoint = read_checkpoint(resume_path) if resume_path else None
    saved_settings = checkpoint.get('extra_info', {}).get('training_settings', {}) if checkpoint else {}
    defaults = dict(epochs=1, batch_size=4, grad_accum_steps=1, lr=3e-4,
                    weight_decay=0.01, max_grad_norm=1.0, mixed_precision=False, seed=42)
    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, saved_settings.get(key, default))
    for name in ('epochs', 'batch_size', 'grad_accum_steps', 'tokenizer_sample_chars',
                 'save_every_steps', 'keep_checkpoints'):
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    if (not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay)
            or args.weight_decay < 0 or not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0
            or not math.isfinite(args.save_every_seconds) or args.save_every_seconds <= 0):
        raise ValueError('Invalid learning rate, weight decay, gradient norm or save interval')
    if args.keep_checkpoints < 2:
        raise ValueError('keep_checkpoints must be at least 2 to retain a recovery backup')
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        raise ValueError('stop_after_epoch must lie within the planned epochs')
    if args.stop_after_step is not None and args.stop_after_step <= 0:
        raise ValueError('stop_after_step must be positive')
    set_seed(args.seed)
    device = get_device()
    amp = args.mixed_precision and device.type == 'cuda'
    print(f'Using device: {device}')
    presets = {'tiny': get_tiny_config, '100m': get_100m_config, '300m': get_300m_config,
               '500m': get_500m_config, '1b': get_1b_config}
    if checkpoint is not None:
        if checkpoint.get('format_version') != 3 or checkpoint.get('tokenizer') is None:
            raise ValueError('Exact resume requires a version 3 training bundle; v2 bundles are inference-only')
        config = ModelConfig(**checkpoint['config'])
        tokenizer = Tokenizer.from_dict(checkpoint['tokenizer'])
        if args.preset and args.preset != checkpoint['extra_info'].get('preset'):
            raise ValueError('Resume preset differs from checkpoint')
        if args.dropout is not None and args.dropout != config.dropout:
            raise ValueError('Resume dropout differs from checkpoint')
        if args.seq_len is not None and args.seq_len != config.max_seq_len:
            raise ValueError('Resume sequence length differs from checkpoint')
        if args.tokenizer_file and Tokenizer.load(args.tokenizer_file).fingerprint != tokenizer.fingerprint:
            raise ValueError('Resume tokenizer differs from checkpoint')
    else:
        config = presets[args.preset or 'tiny']()
        if args.seq_len is not None:
            config.max_seq_len = args.seq_len
        config.dropout = args.dropout if args.dropout is not None else config.dropout
        config.validate()
        tokenizer = Tokenizer.load(args.tokenizer_file) if args.tokenizer_file else Tokenizer()
    if checkpoint is not None:
        saved_files = checkpoint['extra_info'].get('data_files', {})
        if args.corpus_file is None:
            args.corpus_file = saved_files.get('corpus_file')
        if args.val_file is None:
            args.val_file = saved_files.get('val_file')
    train_source, val_source = data_sources(args.corpus_file, args.val_file)
    if checkpoint is None and not args.tokenizer_file:
        tokenizer.train(sample_text(train_source, args.tokenizer_sample_chars), config.vocab_size)
    config.vocab_size = tokenizer.vocab_size
    # Per-process token caches avoid corrupting another run's mapped files.
    cache_dir = directory / f'.tokens-{os.getpid()}'
    cache_dir.mkdir()
    try:
        train_tokens, train_hash, hashes = tokenize_source(train_source, tokenizer, cache_dir / 'train.bin')
        val_tokens, val_hash, _ = tokenize_source(val_source, tokenizer, cache_dir / 'val.bin', hashes)
        train_ds = TextDataset(train_tokens, config.max_seq_len, tokenizer.pad_id)
        val_ds = TextDataset(val_tokens, config.max_seq_len, tokenizer.pad_id)
        total_batches = math.ceil(len(train_ds) / args.batch_size)
        total_steps = math.ceil(total_batches / args.grad_accum_steps) * args.epochs
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                generator=torch.Generator().manual_seed(args.seed))
        settings = {key: getattr(args, key) for key in ('epochs', 'batch_size', 'grad_accum_steps',
                    'lr', 'weight_decay', 'max_grad_norm', 'mixed_precision', 'seed')}
        settings.update(train_hash=train_hash, val_hash=val_hash, device_type=device.type,
                        total_steps=total_steps, torch_version=str(torch.__version__),
                        python_version=sys.version.split()[0])
        if checkpoint is not None and checkpoint['extra_info'].get('training_settings') != settings:
            raise ValueError('Resume requires the same data, device and planned training settings')
        tokenizer.save(str(directory / 'tokenizer.json'))
        (directory / 'config.json').write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
        model = ZEROModel(config).to(device)
        print(get_model_summary(model))
        print(f'Train tokens: {len(train_tokens)} | Validation tokens: {len(val_tokens)} | Context: {config.max_seq_len}')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = make_scheduler(optimizer, total_steps)
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
        global_step = 0
        progress = {'epoch': 0, 'next_batch': 0, 'metrics': {}}
        if checkpoint is not None:
            meta = load_checkpoint(resume_path, model, optimizer, scheduler, device=str(device),
                                   tokenizer=tokenizer, scaler=scaler, restore_random=True)
            global_step, progress = meta['step'], meta['progress']
            if not isinstance(progress, dict) or not 0 <= progress.get('epoch', -1) <= args.epochs:
                raise ValueError('Invalid saved training progress')
            batch = progress.get('next_batch', -1)
            if not 0 <= batch <= total_batches or (batch != total_batches and batch % args.grad_accum_steps):
                raise ValueError('Saved position is not an accumulation boundary')
            print(f'Resumed at epoch {progress["epoch"]}, next batch {batch}, step {global_step}', flush=True)
        last_saved_time = time.monotonic()
        last_saved_step = global_step
        preset = args.preset or (checkpoint['extra_info']['preset'] if checkpoint else 'tiny')

        def persist(current, *, val_loss=None, train_loss=None):
            nonlocal last_saved_time, last_saved_step
            stamp = time.time_ns()
            name = (f'checkpoint_e{current["epoch"]:06d}_b{current["next_batch"]:08d}_'
                    f's{global_step:012d}_{stamp}.pt')
            path = directory / name
            metrics = current['metrics']
            info = {'preset': preset, 'training_settings': settings,
                    'data_files': {key: str(Path(getattr(args, key)).resolve()) if getattr(args, key) else None
                                   for key in ('corpus_file', 'val_file')},
                    'train_loss': (train_loss if train_loss is not None else
                                   metrics.get('loss_sum', 0) / max(metrics.get('tokens', 0), 1)),
                    'val_loss': val_loss}
            save_checkpoint(str(path), model, optimizer, scheduler, step=global_step, extra_info=info,
                            tokenizer=tokenizer, scaler=scaler, epoch=current['epoch'], progress=current)
            # Publication succeeded before any previous snapshot is pruned.
            prune_checkpoints(directory, args.keep_checkpoints, path)
            last_saved_time, last_saved_step = time.monotonic(), global_step
            print(f'Checkpoint saved to {path}', flush=True)

        if checkpoint is None:
            persist(progress)
        if progress['epoch'] == args.epochs:
            print('Training plan is already complete; no optimizer steps were repeated.', flush=True)
            return
        if args.stop_after_step is not None and global_step >= args.stop_after_step:
            print('Requested stop step has already been reached.', flush=True)
            return
        with StopRequest() as stop:
            for epoch in range(progress['epoch'], args.epochs):
                start_batch = progress['next_batch'] if epoch == progress['epoch'] else 0
                metrics = progress['metrics'] if start_batch else {}
                sampler = ResumableBatchSampler(len(train_ds), args.batch_size, args.seed, epoch, start_batch)
                # DataLoader consumes a private RNG only; reconstructing it on resume
                # must not consume the model/dropout RNG restored from the checkpoint.
                train_loader = DataLoader(train_ds, batch_sampler=sampler,
                                          generator=torch.Generator().manual_seed(args.seed + epoch))
                base_step = global_step
                previous_updates = metrics.get('updates', 0)

                def boundary(status):
                    nonlocal global_step, progress
                    global_step = base_step + status['updates'] - previous_updates
                    progress = {'epoch': epoch, 'next_batch': status['next_batch'],
                                'metrics': {k: status[k] for k in ('loss_sum', 'tokens', 'updates')}}
                    requested = stop.requested or (args.stop_after_step is not None
                                                  and global_step >= args.stop_after_step)
                    due = (global_step - last_saved_step >= args.save_every_steps
                           or time.monotonic() - last_saved_time >= args.save_every_seconds)
                    if due or requested:
                        persist(progress)
                    return requested

                start = time.perf_counter()
                loss, count, updates = train_epoch(model, train_loader, optimizer, scheduler, scaler, device,
                    args.grad_accum_steps, args.max_grad_norm, amp, start_batch=start_batch,
                    initial_metrics=metrics, on_boundary=boundary)
                if stop.requested or (args.stop_after_step is not None and global_step >= args.stop_after_step):
                    print('Stopped safely after a completed accumulation group; resume with --resume_from auto.', flush=True)
                    return
                elapsed = time.perf_counter() - start
                val_loss = evaluate(model, val_loader, device, amp)
                rate = (count - metrics.get('tokens', 0)) / elapsed
                print(f'Epoch {epoch + 1}/{args.epochs} | Step {global_step} | Train Loss: {loss:.4f} | '
                      f'Val Loss: {val_loss:.4f} | Tokens/sec: {rate:.2f} | '
                      f'GPU Mem: {get_gpu_memory_info()["allocated_mb"]:.1f}MB', flush=True)
                progress = {'epoch': epoch + 1, 'next_batch': 0, 'metrics': {}}
                persist(progress, val_loss=val_loss, train_loss=loss)
                if stop.requested or args.stop_after_epoch == epoch + 1:
                    break
    finally:
        for path in cache_dir.iterdir():
            path.unlink()
        cache_dir.rmdir()


if __name__ == '__main__':
    train()
