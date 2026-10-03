import random
import numpy as np
import torch
from typing import Dict, Any, Tuple


def set_seed(seed: int = 42):
    """
    Sets random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Returns CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Returns GPU memory allocated and reserved in MB if CUDA is available.
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
    }


def get_model_summary(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Returns key model metrics and parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config = getattr(model, "config", None)

    summary = {
        "architecture": "Decoder-Only Transformer",
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "initialization_method": "LLaMA/GPT style (Normal(0, 0.02) with 1/sqrt(2*N) residual scaling)",
    }

    if config is not None:
        summary.update({
            "number_of_layers": config.num_layers,
            "hidden_size": config.hidden_size,
            "attention_heads": config.num_attention_heads,
            "kv_heads": config.num_kv_heads,
            "intermediate_size": config.intermediate_size,
            "maximum_context_length": config.max_seq_len,
            "vocabulary_size": config.vocab_size,
        })

    return summary
