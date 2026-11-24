import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class TextDataset(Dataset):
    """Lazy next-token windows; retain tails and mask padded targets."""

    def __init__(self, token_ids, seq_len: int, pad_id: int = 1):
        if seq_len <= 0 or len(token_ids) < 2:
            raise ValueError("Dataset needs positive seq_len and at least two tokens")
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.tokens = torch.as_tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return (len(self.tokens) - 1 + self.seq_len - 1) // self.seq_len

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        start = idx * self.seq_len
        count = min(self.seq_len, len(self.tokens) - start - 1)
        x = torch.full((self.seq_len,), self.pad_id, dtype=torch.long)
        y = torch.full((self.seq_len,), IGNORE_INDEX, dtype=torch.long)
        x[:count] = self.tokens[start:start + count]
        y[:count] = self.tokens[start + 1:start + count + 1]
        return x, y
