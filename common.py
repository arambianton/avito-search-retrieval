"""
Общие вещи: пути, загрузка данных, токенизация, BM25, эмбеддинги, метрика.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import Stemmer

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
EMB_DIR = ROOT / "embeddings"
CACHE.mkdir(exist_ok=True)

Q_COLS = ["search_query", "search_location_id", "search_is_delivery_search",
          "search_infm_params_text", "search_category"]
I_COLS = ["item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
          "item_category_id", "item_microcat_id", "item_price", "item_rating",
          "item_rating_reviews_count", "item_location_id", "item_latitude",
          "item_longitude", "item_is_phone_hidden", "item_is_message_forbidden"]


def _fix_types(df: pl.DataFrame) -> pl.DataFrame:
    casts = []
    for c in ("item_price", "item_latitude", "item_longitude"):
        if c in df.columns:
            casts.append(pl.col(c).cast(pl.Float64))
    for c in ("item_description_raw", "item_title_raw", "item_infm_params_text",
              "search_infm_params_text", "search_query"):
        if c in df.columns:
            casts.append(pl.col(c).fill_null(""))
    return df.with_columns(casts)


def load_train() -> pl.DataFrame:
    return _fix_types(pl.read_parquet(ROOT / "train.parquet"))


def load_bench_queries() -> pl.DataFrame:
    return _fix_types(pl.read_parquet(ROOT / "benchmark_queries.parquet"))


def load_bench_items() -> pl.DataFrame:
    return _fix_types(pl.read_parquet(ROOT / "benchmark_items.parquet"))


def log(*a):
    import time
    print(time.strftime("%H:%M:%S"), *a, flush=True)


_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")
_ru = Stemmer.Stemmer("russian")
_en = Stemmer.Stemmer("english")


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


@lru_cache(maxsize=2_000_000)
def stem(word: str) -> str:
    if word.isdigit():
        return word
    if word.isascii():
        return _en.stemWord(word)
    return _ru.stemWord(word)


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in _TOKEN_RE.findall(normalize(text))]


def tokenize_series(texts, max_chars: int | None = None) -> list[list[str]]:
    if max_chars:
        return [tokenize(t[:max_chars]) for t in texts]
    return [tokenize(t) for t in texts]


class Vocab:
    def __init__(self):
        self.index: dict[str, int] = {}

    def ids(self, tokens, grow: bool) -> list[int]:
        out = []
        for t in tokens:
            j = self.index.get(t)
            if j is None and grow:
                j = self.index[t] = len(self.index)
            if j is not None:
                out.append(j)
        return out

    def __len__(self):
        return len(self.index)


def count_matrix(token_lists, vocab: Vocab, grow: bool = True) -> sp.csr_matrix:
    """Списки токенов -> матрица частот (документ x терм)."""
    indptr, indices = [0], []
    for toks in token_lists:
        indices.extend(vocab.ids(toks, grow))
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    m = sp.csr_matrix((data, np.array(indices, dtype=np.int64), np.array(indptr, dtype=np.int64)),
                      shape=(len(token_lists), max(len(vocab), 1)))
    m.sum_duplicates()
    return m


def bm25_matrix(tf: sp.csr_matrix, k1: float = 1.2, b: float = 0.75) -> sp.csr_matrix:
    tf = tf.tocsr().astype(np.float32)
    n_docs = tf.shape[0]
    dl = np.asarray(tf.sum(axis=1)).ravel()
    avgdl = max(dl.mean(), 1e-9)
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
    norm = k1 * (1 - b + b * dl[rows] / avgdl)
    w = tf.data * (k1 + 1) / (tf.data + norm) * idf[tf.indices]
    out = sp.csr_matrix((w.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape)
    return out.T.tocsr()


def query_matrix(token_lists, vocab: Vocab) -> sp.csr_matrix:
    m = count_matrix([list(dict.fromkeys(t)) for t in token_lists], vocab, grow=False)
    m.data[:] = 1.0
    return m


class Embeddings:
    def __init__(self, mode: str, corpus_ids: np.ndarray):
        folder = EMB_DIR / mode
        ids = pl.read_parquet(folder / "item_ids.parquet")["item_id"].to_numpy()
        pos = {x: i for i, x in enumerate(ids)}
        self.items = np.load(folder / "item_emb.npy")[[pos[x] for x in corpus_ids]].astype(np.float32)
        keys = pl.read_parquet(folder / "query_keys.parquet")
        self.qpos = {k: i for i, k in enumerate(zip(keys["search_query"].to_list(),
                                                    keys["search_infm_params_text"].to_list()))}
        self.qemb = np.load(folder / "query_emb.npy")

    def queries(self, q: pl.DataFrame) -> np.ndarray:
        keys = zip(q["search_query"].to_list(), q["search_infm_params_text"].to_list())
        return self.qemb[[self.qpos[k] for k in keys]]


def recall_at_k(pred, relevant, k: int = 50) -> float:
    return float(np.mean([len(set(p[:k]) & set(r)) / len(r) for p, r in zip(pred, relevant)]))
