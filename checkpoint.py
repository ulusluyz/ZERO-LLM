import os
import hashlib
from pathlib import Path
import warnings
import pickle
import random
import tempfile
from dataclasses import asdict

import numpy as np
import torch
from tokenizer import Tokenizer


def capture_rng():
    state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(filepath, model, optimizer=None, scheduler=None, step=0,
                    config_dict=None, extra_info=None, *, tokenizer=None,
                    scaler=None, epoch=0, data_generator=None, progress=None):
    """Durably publish a checkpoint plus checksum; training uses unique filenames.

    Interrupted publication may leave an unpaired file, which recovery rejects.
    Previous, uniquely named checkpoints remain available.
    """
    filepath = os.path.abspath(filepath)
    ensure_directory(os.path.dirname(filepath))
    state = {
        "format_version": 3,
        "progress": progress,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "step": step, "epoch": epoch,
        "config": config_dict if config_dict is not None else asdict(model.config),
        "extra_info": extra_info or {},
        "tokenizer": tokenizer.to_dict() if tokenizer is not None else None,
        "tokenizer_hash": tokenizer.fingerprint if tokenizer is not None else None,
        "rng_state": capture_rng(),
        "data_generator_state": data_generator.get_state() if data_generator is not None else None,
    }
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(filepath), prefix=".checkpoint-", suffix=".tmp")
    checksum_tmp = None
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = file_digest(temporary)
        checksum_fd, checksum_tmp = tempfile.mkstemp(dir=os.path.dirname(filepath), prefix=".checkpoint-", suffix=".tmp")
        with os.fdopen(checksum_fd, "w", encoding="ascii") as stream:
            stream.write(digest + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, filepath)
        os.replace(checksum_tmp, filepath + ".sha256")
        sync_directory(os.path.dirname(filepath))
    finally:
        for path in (temporary, checksum_tmp):
            if path is not None and os.path.exists(path):
                os.unlink(path)


def ensure_directory(directory):
    path = Path(directory).absolute()
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        sync_directory(created)
        sync_directory(created.parent)


def sync_directory(directory):
    # POSIX directory fsync persists the renames, not only the file contents.
    # Windows has no O_DIRECTORY: file fsync + atomic replace remain in use.
    if hasattr(os, "O_DIRECTORY"):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksum(path):
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"Missing checkpoint checksum: {sidecar}")
    if sidecar.read_text(encoding="ascii").strip() != file_digest(path):
        raise ValueError(f"Checkpoint checksum mismatch: {path}")


def checkpoint_candidates(directory):
    # Names encode epoch/batch/step before the timestamp, avoiding reliance on mtime.
    return sorted(Path(directory).glob("checkpoint_*.pt"), reverse=True)


def find_latest_checkpoint(directory):
    candidates = checkpoint_candidates(directory)
    for path in candidates:
        try:
            checkpoint = read_checkpoint(path)
            if checkpoint.get("format_version") != 3 or checkpoint.get("progress") is None:
                raise ValueError("Not a version 3 resumable training checkpoint")
            return path
        except (OSError, ValueError, RuntimeError, EOFError, KeyError, TypeError, pickle.UnpicklingError) as error:
            warnings.warn(f"Skipping unusable checkpoint {path.name}: {error}", RuntimeWarning)
    if candidates:
        raise ValueError("No valid resumable checkpoint found; refusing to restart silently")
    return None


def prune_checkpoints(directory, keep, published_path):
    if keep < 1:
        raise ValueError("keep must be positive")
    # Verify the snapshot just published, not a potentially corrupt newer filename.
    verify_checksum(published_path)
    valid = []
    for path in checkpoint_candidates(directory):
        if not path.name.startswith("checkpoint_e"):
            continue
        try:
            verify_checksum(path)
        except (OSError, ValueError):
            continue
        valid.append(path)
    for path in valid[keep:]:
        path.unlink()
        Path(str(path) + ".sha256").unlink(missing_ok=True)
    sync_directory(directory)


def read_checkpoint(filepath, device="cpu"):
    filepath = str(filepath)
    has_checksum = Path(filepath + ".sha256").exists()
    if has_checksum:
        verify_checksum(filepath)
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Invalid ZERO checkpoint")
    if checkpoint.get("format_version", 1) >= 3 and not has_checksum:
        raise ValueError("Version 3 checkpoint is incomplete: missing checksum")
    if checkpoint.get("tokenizer") is not None:
        tok = Tokenizer.from_dict(checkpoint["tokenizer"])
        if tok.fingerprint != checkpoint.get("tokenizer_hash"):
            raise ValueError("Checkpoint tokenizer fingerprint mismatch")
        if checkpoint.get("config", {}).get("vocab_size") != tok.vocab_size:
            raise ValueError("Checkpoint config and tokenizer vocabulary disagree")
    return checkpoint


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu", *,
                    tokenizer=None, scaler=None, data_generator=None, restore_random=False):
    checkpoint = read_checkpoint(filepath, device)
    if checkpoint.get("config") is not None and checkpoint["config"] != asdict(model.config):
        raise ValueError("Checkpoint configuration does not match model")
    if tokenizer is not None and checkpoint.get("tokenizer_hash") != tokenizer.fingerprint:
        raise ValueError("Checkpoint tokenizer does not match supplied tokenizer")
    model.load_state_dict(checkpoint["model_state_dict"])
    for obj, key in ((optimizer, "optimizer_state_dict"), (scheduler, "scheduler_state_dict"),
                     (scaler, "scaler_state_dict")):
        if obj is not None and checkpoint.get(key) is not None:
            obj.load_state_dict(checkpoint[key])
    if data_generator is not None and checkpoint.get("data_generator_state") is not None:
        data_generator.set_state(checkpoint["data_generator_state"].cpu())
    if restore_random:
        if checkpoint.get("rng_state") is None:
            raise ValueError("Checkpoint lacks RNG state required for exact resume")
        restore_rng(checkpoint["rng_state"])
    return {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
