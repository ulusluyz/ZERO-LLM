import torch
import torch.nn.functional as F
from typing import List, Optional
from model import ZEROModel
from tokenizer import Tokenizer


def sample_top_k_top_p(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0, filter_value: float = -float("Inf")) -> torch.Tensor:
    """
    Filters a distribution of logits using top-k and/or nucleus (top-p) filtering.
    """
    logits = logits.clone()

    # Top-K filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # Top-P (Nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift indices to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter removed indices back to original tensor
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


@torch.no_grad()
def generate_tokens(
    model: ZEROModel,
    prompt_ids: List[int],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_id: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> List[int]:
    """
    Generates tokens given a prompt token list.
    Supports greedy decoding, temperature sampling, top-k, and top-p (nucleus sampling).
    """
    model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Truncate sequence if exceeding max_seq_len
        curr_input = input_ids[:, -model.config.max_seq_len :]

        # Forward pass to get logits for last token
        logits = model(curr_input)
        next_token_logits = logits[0, -1, :]

        if temperature == 0.0:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # Temperature scaling
            next_token_logits = next_token_logits / temperature

            # Top-k and Top-p filtering
            filtered_logits = sample_top_k_top_p(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        token_item = next_token.item()
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        if eos_id is not None and token_item == eos_id:
            break

    return input_ids[0].tolist()
