import json
from typing import Dict, List, Tuple, Set, Optional


def bytes_to_unicode() -> Dict[int, str]:
    """
    Returns list of utf-8 byte values and mapping to printable unicode strings.
    Avoids mapping bytes to whitespace/control characters.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_stats(ids: List[int], counts: Optional[Dict[Tuple[int, int], int]] = None) -> Dict[Tuple[int, int], int]:
    counts = counts if counts is not None else {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


class Tokenizer:
    def __init__(self, special_tokens: Optional[List[str]] = None):
        if special_tokens is None:
            special_tokens = ["<unk>", "<pad>", "<s>", "</s>"]
        self.special_tokens = special_tokens

        # Byte encoder and decoder
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Initial vocabulary: special tokens + 256 byte tokens
        self.vocab: Dict[int, bytes] = {}
        self.encoder: Dict[bytes, int] = {}

        # Register special tokens
        for idx, token_str in enumerate(self.special_tokens):
            token_bytes = token_str.encode("utf-8")
            self.vocab[idx] = token_bytes
            self.encoder[token_bytes] = idx

        # Register 256 bytes
        start_idx = len(self.special_tokens)
        for b in range(256):
            b_bytes = bytes([b])
            self.vocab[start_idx + b] = b_bytes
            self.encoder[b_bytes] = start_idx + b

        # Merges: (int_id1, int_id2) -> new_int_id
        self.merges: Dict[Tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        if vocab_size < self.vocab_size:
            raise ValueError(f"Target vocab_size ({vocab_size}) must be >= initial size ({self.vocab_size})")

        num_merges = vocab_size - self.vocab_size
        text_bytes = text.encode("utf-8")
        byte_start_offset = len(self.special_tokens)

        # Convert text bytes into list of initial byte token IDs
        ids = [b + byte_start_offset for b in text_bytes]

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            # Find pair with max frequency
            best_pair = max(stats, key=stats.get)
            idx = self.vocab_size

            ids = merge(ids, best_pair, idx)
            self.merges[best_pair] = idx

            merged_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[idx] = merged_bytes
            self.encoder[merged_bytes] = idx

            if verbose:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {idx} ({merged_bytes})")

    def encode(self, text: str) -> List[int]:
        if not text:
            return []

        # Handle special tokens directly if text is exactly a special token
        if text in self.special_tokens:
            return [self.encoder[text.encode("utf-8")]]

        text_bytes = text.encode("utf-8")
        byte_start_offset = len(self.special_tokens)
        ids = [b + byte_start_offset for b in text_bytes]

        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find pair that was merged earliest
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids

    def decode(self, ids: List[int]) -> str:
        byte_chunks = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])
            else:
                byte_chunks.append("<unk>".encode("utf-8"))

        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, filepath: str):
        data = {
            "special_tokens": self.special_tokens,
            "merges": [
                [p[0], p[1], idx] for p, idx in self.merges.items()
            ],
            "vocab": {
                str(k): list(v) for k, v in self.vocab.items()
            }
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Tokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(special_tokens=data["special_tokens"])

        # Load vocab
        tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tok.encoder = {v: k for k, v in tok.vocab.items()}

        # Load merges
        tok.merges = {
            (m[0], m[1]): m[2] for m in data["merges"]
        }

        return tok
