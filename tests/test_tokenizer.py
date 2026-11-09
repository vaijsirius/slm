"""Tests for the Tokenizer class (Part 2)."""
import random

import numpy as np
import pytest

from minilm.tokenizer import Tokenizer, load_vocab_and_merges, save_vocab_and_merges, train_bpe

EOT = "<|endoftext|>"

TOY_VOCAB = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t", 6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
TOY_MERGES = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]

SAMPLES = [
    "",
    "Hello, world!",
    "Once upon a time, there was a little girl named Lily.",
    "  leading and trailing spaces  ",
    "tabs\tand\nnewlines\r\n\n\n",
    "नमस्ते दुनिया, यह एक कहानी है।",
    "emoji 😀🎉 and accents café naïve",
    "mixed: Hello नमस्ते 😀 你好",
    "<|endoftext|>",
    "a<|endoftext|>b<|endoftext|><|endoftext|>c",
]

TRAIN_TEXT = (
    "Once upon a time there was a little girl named Lily. She loved to play with her cat. " * 20
    + EOT
    + "The cat sat on the mat. The dog ran to the park and found a ball! " * 20
    + EOT
    + "café naïve नमस्ते 你好 😀 " * 5
)


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    p = tmp_path_factory.mktemp("tok") / "train.txt"
    p.write_text(TRAIN_TEXT, encoding="utf-8")
    vocab, merges = train_bpe(str(p), vocab_size=400, special_tokens=[EOT])
    return Tokenizer(vocab, merges, [EOT])


def toy():
    return Tokenizer(TOY_VOCAB, TOY_MERGES)


def test_hand_worked_encoding():
    # "the cat ate" -> pre-tokens "the", " cat", " ate" -> [the] [" c", a, t] [" at", e]
    assert toy().encode("the cat ate") == [9, 7, 1, 5, 10, 3]


def test_toy_decode():
    assert toy().decode([9, 7, 1, 5, 10, 3]) == "the cat ate"


def test_special_token_added_to_end_of_vocab():
    t = Tokenizer(TOY_VOCAB, TOY_MERGES, [EOT])
    assert t.vocab[11] == EOT.encode()
    assert t.encode(EOT) == [11]
    # an existing special token is not duplicated
    t2 = Tokenizer(t.vocab, TOY_MERGES, [EOT])
    assert len(t2.vocab) == len(t.vocab)


@pytest.mark.parametrize("text", SAMPLES)
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_random_unicode(tok):
    rng = random.Random(0)
    alphabet = "ab cd\n\t.,!?éü你नम😀 <|"
    for _ in range(200):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        assert tok.decode(tok.encode(s)) == s


def test_special_token_is_one_id(tok):
    eot_id = tok.special_ids[EOT]
    assert tok.encode(EOT) == [eot_id]
    assert tok.encode("abc" + EOT + "def").count(eot_id) == 1


def test_no_merge_across_special_token(tok):
    eot_id = tok.special_ids[EOT]
    a, b = "The cat sat", " on the mat"
    assert tok.encode(a + EOT + b) == tok.encode(a) + [eot_id] + tok.encode(b)


def test_overlapping_specials_prefer_longest():
    specials = ["<|eot|>", "<|eot|><|eot|>"]
    vocab = {i: bytes([i]) for i in range(256)}
    t = Tokenizer(vocab, [], specials)
    ids = t.encode("<|eot|><|eot|>")
    assert ids == [t.special_ids["<|eot|><|eot|>"]]
    assert t.encode("<|eot|>") == [t.special_ids["<|eot|>"]]
    assert t.encode("x<|eot|><|eot|><|eot|>y") == (
        t.encode("x") + [t.special_ids["<|eot|><|eot|>"], t.special_ids["<|eot|>"]] + t.encode("y"))


def test_decode_invalid_utf8_gives_replacement_char(tok):
    # byte 0x80 on its own is a continuation byte -> not valid UTF-8
    out = tok.decode([0x80])
    assert out == "�"
    assert tok.decode([0xE0, 0xA4]) == "�"  # truncated 3-byte sequence


def test_encode_iterable_matches_encode(tok):
    rng = random.Random(3)
    pieces = ["Hello ", "world", "   ", "\n\n", "  x", EOT, "é", "你好", "😀", "<|endo", "ftext|>", " tail  ", "\n"]
    for _ in range(60):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 40)))
        # chop at arbitrary points, including inside a special token or a multi-char unit
        chunks, i = [], 0
        while i < len(text):
            k = rng.randint(1, 9)
            chunks.append(text[i:i + k])
            i += k
        for flush in (1, 5, 17):
            assert list(tok.encode_iterable(chunks, flush_chars=flush)) == tok.encode(text)


def test_encode_iterable_over_file_matches_whole_file(tok, tmp_path):
    text = (TRAIN_TEXT + "\n\n  \n") * 30
    p = tmp_path / "f.txt"
    p.write_text(text, encoding="utf-8")
    with open(p, encoding="utf-8") as f:
        streamed = list(tok.encode_iterable(f, flush_chars=64))
    assert streamed == tok.encode(text)


def test_encode_iterable_is_lazy_and_memory_bounded(tok):
    """A generator of many pieces must be consumed incrementally (not materialised up front)."""
    consumed = []

    def gen():
        for i in range(2000):
            consumed.append(i)
            yield "hello world. "

    it = tok.encode_iterable(gen(), flush_chars=256)
    next(it)
    assert len(consumed) < 100


def test_save_and_load_round_trip(tok, tmp_path):
    vp, mp = tmp_path / "vocab.json", tmp_path / "merges.txt"
    save_vocab_and_merges(tok.vocab, tok.merges, str(vp), str(mp))
    vocab, merges = load_vocab_and_merges(str(vp), str(mp))
    assert vocab == tok.vocab and merges == tok.merges
    t2 = Tokenizer.from_files(str(vp), str(mp), [EOT])
    for s in SAMPLES:
        assert t2.encode(s) == tok.encode(s)


def test_ids_fit_uint16_and_dataset_dtype(tok):
    ids = tok.encode(TRAIN_TEXT)
    arr = np.array(ids, dtype=np.uint16)
    assert arr.dtype == np.uint16
    assert tok.vocab_size < 2 ** 16
    assert arr.tolist() == ids  # nothing was truncated by the cast
