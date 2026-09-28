"""
Таблица объявлений с токенами и симуляция бенчмарка на train.
Результат: cache/items.parquet, cache/sim_*.parquet
"""
from __future__ import annotations

import numpy as np
import polars as pl

from common import CACHE, I_COLS, Q_COLS, load_bench_items, load_train, log, tokenize_series

DESC_MAX_CHARS = 3000     # для лексики хватает начала описания
SEED = 42
P_SHARED = 0.33
N_EVAL = 4000
N_RANKER = 40000
BENCH_EMPTY_PARAMS = 0.63


def build_items(tr: pl.DataFrame):
    """Одна строка на item_id (объявления train + корпуса) и токены полей."""
    bi = load_bench_items()
    tr_items = tr.select(I_COLS).unique("item_id", keep="first", maintain_order=True).with_columns(pl.lit(True).alias("in_train"))
    bi_items = bi.select(I_COLS).with_columns(pl.lit(True).alias("in_bench"))
    items = (
        pl.concat([bi_items, tr_items], how="diagonal")
        .group_by("item_id", maintain_order=True)
        .agg([pl.col(c).first() for c in I_COLS if c != "item_id"]
             + [pl.col("in_train").max(), pl.col("in_bench").max()])
        .with_columns(pl.col("in_train").fill_null(False), pl.col("in_bench").fill_null(False))
    )
    items = items.with_columns(
        pl.Series("tok_title", tokenize_series(items["item_title_raw"].to_list())),
        pl.Series("tok_params", tokenize_series(items["item_infm_params_text"].to_list())),
        pl.Series("tok_desc", tokenize_series(items["item_description_raw"].to_list(), DESC_MAX_CHARS)),
    )
    items.write_parquet(CACHE / "items.parquet")
    log("items:", items.shape)


def split_by_items(tr: pl.DataFrame, rng) -> np.ndarray:
    """Маска строк, попавших в B. Коды объявлений через np.unique, чтобы разбиение совпадало с train_encoder.py."""
    _, item_idx = np.unique(tr["item_id"].to_numpy(), return_inverse=True)
    n_items = item_idx.max() + 1
    side_b = rng.random(n_items) < 0.5          # куда уходит обычное объявление
    shared = rng.random(n_items) < P_SHARED     # долгоживущее объявление
    row_b = rng.random(tr.height) < 0.5         # у долгоживущих - построчно
    return np.where(shared[item_idx], row_b, side_b[item_idx])


def stratified(q: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """Выборка с долей пустых фильтров как в бенчмарке."""
    empty = q.filter(pl.col("search_infm_params_text") == "")
    nonempty = q.filter(pl.col("search_infm_params_text") != "")
    n = min(n, int(empty.height / BENCH_EMPTY_PARAMS), int(nonempty.height / (1 - BENCH_EMPTY_PARAMS)))
    n_empty = int(round(n * BENCH_EMPTY_PARAMS))
    return pl.concat([empty.sample(n=n_empty, seed=seed), nonempty.sample(n=n - n_empty, seed=seed)])


def simulate(tr: pl.DataFrame):
    tr = tr.select(Q_COLS + ["item_id"])
    is_b = split_by_items(tr, np.random.default_rng(SEED))
    A, B = tr.filter(pl.Series(~is_b)), tr.filter(pl.Series(is_b))

    groups = (B.group_by(Q_COLS, maintain_order=True)
              .agg(pl.col("item_id").unique(maintain_order=True).alias("relevant"))
              .with_row_index("qid")
              .with_columns(pl.format("sim{}", pl.col("qid")).alias("query_id")).drop("qid"))
    one_per_text = groups.sample(fraction=1.0, shuffle=True, seed=SEED).unique("search_query", keep="first", maintain_order=True)
    eval_q = stratified(one_per_text, N_EVAL, SEED)
    
    rest = one_per_text.filter(~pl.col("search_query").is_in(eval_q["search_query"].implode()))
    rank_q = stratified(rest, N_RANKER, SEED + 1)

    A.write_parquet(CACHE / "sim_A.parquet")
    pl.DataFrame({"item_id": B["item_id"].unique()}).write_parquet(CACHE / "sim_corpus.parquet")
    eval_q.write_parquet(CACHE / "sim_eval.parquet")
    rank_q.write_parquet(CACHE / "sim_rank.parquet")
    log(f"simulation: history {A.height} rows, corpus {B['item_id'].n_unique()} items, "
        f"eval {eval_q.height} queries, ranker {rank_q.height} queries")


if __name__ == "__main__":
    tr = load_train()
    build_items(tr)
    simulate(tr)
