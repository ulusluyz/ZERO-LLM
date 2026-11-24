import json
import re
import hashlib
from collections import Counter
from typing import Dict, List, Tuple, Optional


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
    def __init__(self, special_tokens: Optional[List[str]] = None, pretokenize: bool = True):
        if special_tokens is None:
            special_tokens = ["<unk>", "<pad>", "<s>", "</s>"]
        if any(not token for token in special_tokens) or len(set(special_tokens)) != len(special_tokens):
            raise ValueError("Special tokens must be nonempty and unique")
        if any(len(token.encode("utf-8")) == 1 for token in special_tokens):
            raise ValueError("Special tokens cannot overlap single-byte tokens")
        self.pretokenize = pretokenize
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

    @property
    def pad_id(self):
        return self.special_id("<pad>")

    @property
    def bos_id(self):
        return self.special_id("<s>")

    @property
    def eos_id(self):
        return self.special_id("</s>")

    def special_id(self, token):
        return self.special_tokens.index(token)

    def _pieces(self, text):
        return re.findall(r"\s+|[^\s]+", text) if self.pretokenize else [text]

    def train(self, text: str, vocab_size: int, verbose: bool = False, min_frequency: int = 2):
        if self.merges:
            raise ValueError("Train a fresh tokenizer; retraining changes existing token IDs")
        if vocab_size < self.vocab_size or min_frequency < 2:
            raise ValueError("Invalid vocabulary size or minimum pair frequency")
        # Count distinct pieces once; never merge across whitespace or special tokens.
        chunks = Counter()
        for special, part in self._parts(text):
            if special:
                continue
            for piece in self._pieces(part):
                chunks[tuple(b + len(self.special_tokens) for b in piece.encode("utf-8"))] += 1
        for i in range(vocab_size - self.vocab_size):
            stats = Counter()
            for ids, frequency in chunks.items():
                for pair, count in get_stats(ids).items():
                    stats[pair] += count * frequency
            if not stats:
                break
            best = max(stats, key=stats.get)
            if stats[best] < min_frequency:
                break
            idx = self.vocab_size
            updated = Counter()
            for ids, frequency in chunks.items():
                updated[tuple(merge(ids, best, idx))] += frequency
            chunks = updated
            self.merges[best] = idx
            value = self.vocab[best[0]] + self.vocab[best[1]]
            self.vocab[idx] = value
            self.encoder[value] = idx
            if verbose:
                print(f"Merge {i + 1}: {best} -> {idx}")

    def _parts(self, text):
        if not self.special_tokens:
            return [(False, text)]
        pattern = "(" + "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)) + ")"
        return [(part in self.special_tokens, part) for part in re.split(pattern, text) if part]

    def encode(self, text: str) -> List[int]:
        output = []
        for special, part in self._parts(text):
            if special:
                output.append(self.special_id(part))
                continue
            for piece in self._pieces(part):
                ids = [b + len(self.special_tokens) for b in piece.encode("utf-8")]
                while len(ids) >= 2:
                    pair = min(get_stats(ids), key=lambda p: self.merges.get(p, float("inf")))
                    if pair not in self.merges:
                        break
                    ids = merge(ids, pair, self.merges[pair])
                output.extend(ids)
        return output

    def decode(self, ids: List[int]) -> str:
        byte_chunks = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])
            else:
                byte_chunks.append("<unk>".encode("utf-8"))

        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def to_dict(self):
        return {
            "special_tokens": self.special_tokens,
            "pretokenize": self.pretokenize,
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }

    @property
    def fingerprint(self):
        data = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data):
        # Old tokenizer files used whole-stream BPE.
        tok = cls(data["special_tokens"], pretokenize=data.get("pretokenize", False))
        tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tok.encoder = {v: k for k, v in tok.vocab.items()}
        tok.merges = {(m[0], m[1]): m[2] for m in data["merges"]}
        if set(tok.vocab) != set(range(len(tok.vocab))):
            raise ValueError("Tokenizer vocabulary IDs must be contiguous")
        for i, token in enumerate(tok.special_tokens):
            if tok.vocab.get(i) != token.encode("utf-8"):
                raise ValueError("Tokenizer special token IDs are invalid")
        offset = len(tok.special_tokens)
        if any(tok.vocab.get(offset + b) != bytes([b]) for b in range(256)):
            raise ValueError("Tokenizer byte vocabulary is invalid")
        for (a, b), idx in tok.merges.items():
            if not (offset <= a < idx and offset <= b < idx and idx in tok.vocab):
                raise ValueError("Invalid BPE merge IDs")
            if tok.vocab[idx] != tok.vocab[a] + tok.vocab[b]:
                raise ValueError("Invalid BPE merge bytes")
        return tok

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
