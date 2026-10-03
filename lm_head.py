import torch
import torch.nn as nn


class LMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Projection to logits: x @ weight.T
        return torch.matmul(x, self.weight.t())
