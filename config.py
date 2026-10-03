from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    num_kv_heads: int = 12
    intermediate_size: int = 2048
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def get_tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=4,
        num_attention_heads=4,
        num_kv_heads=4,
        intermediate_size=344,
        max_seq_len=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        dropout=0.0,
        tie_embeddings=False,
    )


def get_100m_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        num_kv_heads=12,
        intermediate_size=2048,
        max_seq_len=2048,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        dropout=0.0,
        tie_embeddings=False,
    )


def get_300m_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_layers=24,
        num_attention_heads=16,
        num_kv_heads=16,
        intermediate_size=2816,
        max_seq_len=2048,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        dropout=0.0,
        tie_embeddings=False,
    )


def get_500m_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32000,
        hidden_size=1536,
        num_layers=24,
        num_attention_heads=16,
        num_kv_heads=16,
        intermediate_size=4096,
        max_seq_len=2048,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        dropout=0.0,
        tie_embeddings=False,
    )


def get_1b_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32000,
        hidden_size=2048,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=5632,
        max_seq_len=2048,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        dropout=0.0,
        tie_embeddings=False,
    )
