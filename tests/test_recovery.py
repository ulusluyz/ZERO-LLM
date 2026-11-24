"""Recovery tests use real process termination; no physical power-loss claim."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch

import checkpoint as checkpoints
from checkpoint import (save_checkpoint, read_checkpoint, find_latest_checkpoint,
                        prune_checkpoints, verify_checksum)
from model import ZEROModel
from recovery import ResumableBatchSampler, TrainingLock
from tokenizer import Tokenizer
from train import train
from test_regressions import small_config, assert_nested_equal


def snapshot(directory, number, model=None):
    model = model or ZEROModel(small_config())
    path = directory / f'checkpoint_e000000_b{number:08d}_s{number:012d}_1.pt'
    save_checkpoint(path, model, tokenizer=Tokenizer(), step=number,
                    progress={'epoch': 0, 'next_batch': number, 'metrics': {}})
    return path


def test_sampler_restores_position_without_consuming_global_rng():
    before = torch.get_rng_state().clone()
    complete = list(ResumableBatchSampler(23, 4, 42, 2))
    assert list(ResumableBatchSampler(23, 4, 42, 2, 3)) == complete[3:]
    assert sorted(i for batch in complete for i in batch) == list(range(23))
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize('damage', ['truncate', 'flip', 'missing_checksum'])
def test_latest_checkpoint_falls_back_on_corruption(tmp_path, damage):
    previous = snapshot(tmp_path, 1)
    latest = snapshot(tmp_path, 2)
    if damage == 'truncate':
        latest.write_bytes(b'interrupted file')
    elif damage == 'flip':
        data = bytearray(latest.read_bytes())
        data[len(data) // 2] ^= 1
        latest.write_bytes(data)
    else:
        Path(str(latest) + '.sha256').unlink()
    (tmp_path / '.checkpoint-incomplete.tmp').write_bytes(b'unfinished serialization')
    with pytest.warns(RuntimeWarning, match='Skipping unusable'):
        assert find_latest_checkpoint(tmp_path) == previous
    with pytest.raises(ValueError):
        read_checkpoint(latest)


def test_auto_resume_refuses_to_restart_when_all_snapshots_are_broken(tmp_path):
    bad = snapshot(tmp_path, 1)
    bad.write_bytes(b'broken')
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match='refusing to restart silently'):
        train(['--checkpoint_dir', str(tmp_path), '--resume_from', 'auto'])
    assert len(list(tmp_path.glob('*.pt'))) == 1


def test_interrupted_publication_preserves_previous_snapshot(tmp_path, monkeypatch):
    previous = snapshot(tmp_path, 1)
    original = checkpoints.os.replace
    def interrupted(source, destination):
        if str(destination).endswith('.sha256'):
            raise OSError('simulated interruption between publication renames')
        return original(source, destination)
    monkeypatch.setattr(checkpoints.os, 'replace', interrupted)
    with pytest.raises(OSError):
        snapshot(tmp_path, 2)
    with pytest.warns(RuntimeWarning):
        assert find_latest_checkpoint(tmp_path) == previous
    verify_checksum(previous)


def test_fsync_failure_does_not_replace_previous_snapshot(tmp_path, monkeypatch):
    previous = snapshot(tmp_path, 1)
    def fail(_):
        raise OSError('simulated disk full or IO failure')
    monkeypatch.setattr(checkpoints.os, 'fsync', fail)
    with pytest.raises(OSError):
        snapshot(tmp_path, 2)
    assert find_latest_checkpoint(tmp_path) == previous
    assert not list(tmp_path.glob('*.tmp'))


def test_checkpoint_files_and_directory_are_fsynced(tmp_path, monkeypatch):
    calls = []
    original = checkpoints.os.fsync
    def record(fd):
        calls.append(fd)
        return original(fd)
    monkeypatch.setattr(checkpoints.os, 'fsync', record)
    path = snapshot(tmp_path, 1)
    assert len(calls) >= 3  # checkpoint file, checksum file, directory entries
    verify_checksum(path)


def test_retention_keeps_backups_and_ignores_incomplete_files(tmp_path):
    paths = [snapshot(tmp_path, i) for i in range(1, 5)]
    corrupt = paths[-1]
    corrupt.write_bytes(b'bad latest')
    published = snapshot(tmp_path, 5)
    prune_checkpoints(tmp_path, keep=2, published_path=published)
    assert paths[2].exists() and published.exists()
    assert not paths[0].exists() and not paths[1].exists()
    assert corrupt.exists()  # leave forensic evidence; it does not consume a valid backup slot
    assert find_latest_checkpoint(tmp_path) == published


def test_concurrent_training_in_same_directory_is_rejected(tmp_path):
    with TrainingLock(tmp_path):
        with pytest.raises(RuntimeError, match='Another training process'):
            with TrainingLock(tmp_path):
                pass
    with TrainingLock(tmp_path):
        pass


def compare_training_states(a, b):
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
                'scaler_state_dict', 'rng_state', 'tokenizer', 'step', 'progress'):
        assert_nested_equal(a[key], b[key])
    assert a['extra_info']['train_loss'] == b['extra_info']['train_loss']
    assert a['extra_info']['val_loss'] == b['extra_info']['val_loss']


def test_mid_epoch_auto_resume_matches_uninterrupted_training_with_dropout(tmp_path, monkeypatch):
    import train as training
    monkeypatch.setattr(training, 'get_device', lambda: torch.device('cpu'))
    full, split = tmp_path / 'full', tmp_path / 'split'
    options = ['--epochs', '2', '--seq_len', '32', '--batch_size', '3',
               '--grad_accum_steps', '3', '--dropout', '0.2', '--save_every_steps', '2']
    train(options + ['--checkpoint_dir', str(full)])
    train(options + ['--checkpoint_dir', str(split), '--stop_after_step', '2'])
    partial = read_checkpoint(find_latest_checkpoint(split))
    assert partial['progress']['epoch'] == 0 and partial['progress']['next_batch'] == 6
    # All mathematical settings are restored when omitted.
    train(['--checkpoint_dir', str(split), '--resume_from', 'auto'])
    final_path = find_latest_checkpoint(split)
    compare_training_states(read_checkpoint(find_latest_checkpoint(full)), read_checkpoint(final_path))
    # Restarting an already completed run is a no-op.
    train(['--checkpoint_dir', str(split), '--resume_from', 'auto'])
    assert find_latest_checkpoint(split) == final_path


def test_time_based_checkpoint_interval(tmp_path, monkeypatch):
    import train as training
    monkeypatch.setattr(training, 'get_device', lambda: torch.device('cpu'))
    train(['--epochs', '1', '--seq_len', '32', '--save_every_seconds', '0.000001',
           '--save_every_steps', '99999', '--keep_checkpoints', '100', '--checkpoint_dir', str(tmp_path)])
    partial = [read_checkpoint(path)['progress'] for path in tmp_path.glob('*.pt')]
    assert any(state['epoch'] == 0 and state['next_batch'] > 0 for state in partial)


def kill_after_periodic_save(tmp_path, options, sig):
    repo = Path(__file__).resolve().parents[1]
    log = tmp_path / 'process.log'
    directory = tmp_path / 'interrupted'
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    with open(log, 'w') as stream:
        process = subprocess.Popen([sys.executable, '-u', str(repo / 'train.py'), *options,
                                    '--checkpoint_dir', str(directory)], stdout=stream,
                                   stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if log.read_text().count('Checkpoint saved to') >= 2:
                    process.send_signal(sig)
                    break
                if process.poll() is not None:
                    pytest.fail(f'Child exited too early: {log.read_text()}')
                time.sleep(.01)
            else:
                pytest.fail(f'Timed out waiting for periodic save: {log.read_text()}')
            code = process.wait(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    return directory, code, log.read_text()


@pytest.mark.skipif(sys.platform != 'linux', reason='Real POSIX process interruption test')
@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM, signal.SIGKILL])
def test_process_interruption_and_recovery_matches_full_run(tmp_path, monkeypatch, sig):
    import train as training
    monkeypatch.setattr(training, 'get_device', lambda: torch.device('cpu'))
    options = ['--epochs', '1', '--seq_len', '16', '--batch_size', '3',
               '--grad_accum_steps', '3', '--dropout', '0.2', '--save_every_steps', '1']
    directory, code, log = kill_after_periodic_save(tmp_path, options, sig)
    if sig == signal.SIGKILL:
        assert code == -signal.SIGKILL
    else:
        assert code == 0 and 'Stopped safely' in log
    partial = read_checkpoint(find_latest_checkpoint(directory))
    assert partial['step'] > 0
    assert partial['progress']['epoch'] == 0
    assert partial['progress']['next_batch'] > 0
    # Kernel lock ownership must have disappeared even after SIGKILL.
    train(['--checkpoint_dir', str(directory), '--resume_from', 'auto'])
    full = tmp_path / 'full'
    train(options + ['--checkpoint_dir', str(full)])
    compare_training_states(read_checkpoint(find_latest_checkpoint(full)),
                            read_checkpoint(find_latest_checkpoint(directory)))


def test_auto_resume_restores_custom_data_paths(tmp_path, monkeypatch):
    import train as training
    monkeypatch.setattr(training, 'get_device', lambda: torch.device('cpu'))
    corpus, validation = tmp_path / 'train.txt', tmp_path / 'val.txt'
    corpus.write_text('\n'.join(f'Belge {i}: Türkçe örnek eğitim cümlesi.' for i in range(20)))
    validation.write_text('Bağımsız doğrulama belgesi; farklı içerik.')
    directory = tmp_path / 'checkpoints'
    train(['--corpus_file', str(corpus), '--val_file', str(validation), '--seq_len', '16',
           '--epochs', '2', '--checkpoint_dir', str(directory), '--stop_after_step', '1'])
    train(['--checkpoint_dir', str(directory), '--resume_from', 'auto'])
    saved = read_checkpoint(find_latest_checkpoint(directory))
    assert saved['epoch'] == 2
    assert saved['extra_info']['data_files']['corpus_file'] == str(corpus.resolve())


def test_at_least_two_backups_required(tmp_path):
    with pytest.raises(ValueError, match='at least 2'):
        train(['--checkpoint_dir', str(tmp_path), '--keep_checkpoints', '1'])
