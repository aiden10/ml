from collections import Counter
from pathlib import Path
import json
import re

import numpy as np
from datasets import load_dataset

DIR = Path(__file__).parent
VOCAB_PATH = DIR / "vocab.json"
CORPUS_PATH = DIR / "corpus.npy"

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
UNK = "<unk>"

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def create_wiki_vocab(
    output_path: str = VOCAB_PATH,
    lang: str = "20231101.en",
    max_articles: int = 1000,
    max_vocab_size: int = 32_000,
):
    dataset = load_dataset("wikimedia/wikipedia", lang, split="train", streaming=True)

    counts = Counter()
    for i, article in enumerate(dataset):
        if i >= max_articles:
            break
        counts.update(tokenize(article["text"]))

    most_common = [
        token
        for token, _ in counts.most_common(max_vocab_size - len(SPECIAL_TOKENS))
    ]
    vocab = SPECIAL_TOKENS + most_common

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def get_vocab(file_path: str = VOCAB_PATH) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_stoi(vocab: list[str]) -> dict[str, int]:
    return {token: i for i, token in enumerate(vocab)}


def encode(tokens: list[str], stoi: dict[str, int]) -> list[int]:
    unk = stoi[UNK]
    return [stoi.get(token, unk) for token in tokens]


def decode(token_ids: list[int], vocab: list[str]) -> str:
    text = " ".join(vocab[i] for i in token_ids)
    return re.sub(r"\s+([.,!?;:)'\"])", r"\1", text)


def create_corpus(
    output_path: str = CORPUS_PATH,
    lang: str = "20231101.en",
    max_articles: int = 1000,
    max_tokens: int = 2_000_000,
    vocab: list[str] | None = None,
) -> np.ndarray:
    stoi = build_stoi(vocab if vocab is not None else get_vocab())
    dataset = load_dataset("wikimedia/wikipedia", lang, split="train", streaming=True)

    ids: list[int] = []
    for i, article in enumerate(dataset):
        if i >= max_articles or len(ids) >= max_tokens:
            break
        ids.extend(encode(tokenize(article["text"]), stoi))

    corpus = np.array(ids[:max_tokens], dtype=np.int32)
    np.save(output_path, corpus)
    return corpus


def get_corpus(
    file_path: str = CORPUS_PATH, max_tokens: int = 2_000_000
) -> np.ndarray:
    if not Path(file_path).exists():
        return create_corpus(output_path=file_path, max_tokens=max_tokens)
    return np.load(file_path)


def sample_batch(
    corpus: np.ndarray,
    batch_size: int,
    context_len: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (inputs (batch, context_len), targets (batch,)) as int64 arrays."""
    rng = rng or np.random.default_rng()
    starts = rng.integers(0, len(corpus) - context_len, size=batch_size)
    columns = starts[:, None] + np.arange(context_len + 1)
    windows = corpus[columns]
    return windows[:, :-1].astype(np.int64), windows[:, -1].astype(np.int64)
