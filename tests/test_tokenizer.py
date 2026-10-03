import os
import tempfile
import pytest
from tokenizer import Tokenizer


def test_special_tokens_and_vocab_size():
    special_tokens = ["<unk>", "<pad>", "<s>", "</s>"]
    tok = Tokenizer(special_tokens=special_tokens)
    assert tok.vocab_size == len(special_tokens) + 256


def test_turkish_character_encoding_and_decoding():
    tok = Tokenizer()
    turkish_text = "ç Ç ğ Ğ ı İ ö Ö ş Ş ü Ü"
    tok.train(turkish_text, vocab_size=300)

    encoded = tok.encode(turkish_text)
    decoded = tok.decode(encoded)

    assert decoded == turkish_text
    for char in turkish_text.split():
        assert tok.decode(tok.encode(char)) == char


def test_bpe_training_and_merges():
    tok = Tokenizer()
    text = "aaa bbb ccc aaabbbccc " * 5
    initial_vocab_size = tok.vocab_size
    target_vocab_size = initial_vocab_size + 10

    tok.train(text, vocab_size=target_vocab_size)
    assert tok.vocab_size == target_vocab_size
    assert len(tok.merges) == 10

    encoded = tok.encode("aaabbbccc")
    decoded = tok.decode(encoded)
    assert decoded == "aaabbbccc"


def test_tokenizer_save_and_load():
    tok = Tokenizer()
    text = "ZERO LLM Türkçe test verisi"
    tok.train(text, vocab_size=300)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        tok.save(tmp_path)
        loaded_tok = Tokenizer.load(tmp_path)

        assert loaded_tok.vocab_size == tok.vocab_size
        assert loaded_tok.special_tokens == tok.special_tokens

        test_str = "Türkçe test"
        assert tok.encode(test_str) == loaded_tok.encode(test_str)
        assert tok.decode(tok.encode(test_str)) == loaded_tok.decode(loaded_tok.encode(test_str))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
