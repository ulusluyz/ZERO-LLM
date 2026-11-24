"""Deterministic data position and cooperative stop handling for single-process training."""
import math
import signal
import threading

import torch
from torch.utils.data import Sampler


class ResumableBatchSampler(Sampler):
    """Recreate an epoch permutation without consuming the model's RNG stream."""

    def __init__(self, size, batch_size, seed, epoch, start_batch=0):
        self.size, self.batch_size = size, batch_size
        self.seed, self.epoch, self.start_batch = seed, epoch, start_batch
        self.total_batches = math.ceil(size / batch_size)
        if not 0 <= start_batch <= self.total_batches:
            raise ValueError("Saved batch position lies outside the dataset")

    def __len__(self):
        return self.total_batches - self.start_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator)
        for batch in range(self.start_batch, self.total_batches):
            yield indices[batch * self.batch_size:(batch + 1) * self.batch_size].tolist()


class StopRequest:
    """Signals request a save at the next accumulation boundary; never do IO in a handler."""

    def __init__(self):
        self.requested = False
        self.signal_number = None
        self.previous = {}

    def _handle(self, number, frame):
        self.requested = True
        self.signal_number = number

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGINT, signal.SIGTERM):
                self.previous[number] = signal.signal(number, self._handle)
        return self

    def __exit__(self, *args):
        for number, handler in self.previous.items():
            signal.signal(number, handler)


class TrainingLock:
    """OS-released lock: a killed process cannot leave a stale ownership marker."""

    def __init__(self, directory):
        self.directory = directory
        self.stream = None

    def __enter__(self):
        import fcntl
        from checkpoint import ensure_directory
        ensure_directory(self.directory)
        self.stream = open(self.directory / '.training.lock', 'a+b')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise RuntimeError('Another training process is using this checkpoint directory') from error
        return self

    def __exit__(self, *args):
        self.stream.close()
