import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class TextDataset(Dataset):
    """
    Dataset for Next-Token Prediction Causal Language Modeling.
    Converts token IDs into overlapping chunk sequences of length seq_len.
    Input (x): token_1, token_2, ..., token_N
    Target (y): token_2, token_3, ..., token_N+1
    """

    def __init__(self, token_ids: List[int], seq_len: int):
        self.seq_len = seq_len
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []

        if len(token_ids) <= seq_len:
            # Pad sequence if smaller than seq_len + 1
            padded = token_ids + [0] * (seq_len + 1 - len(token_ids))
            x = torch.tensor(padded[:seq_len], dtype=torch.long)
            y = torch.tensor(padded[1 : seq_len + 1], dtype=torch.long)
            self.data.append((x, y))
        else:
            for i in range(0, len(token_ids) - seq_len, seq_len):
                chunk = token_ids[i : i + seq_len + 1]
                if len(chunk) < seq_len + 1:
                    break
                x = torch.tensor(chunk[:seq_len], dtype=torch.long)
                y = torch.tensor(chunk[1 : seq_len + 1], dtype=torch.long)
                self.data.append((x, y))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]
