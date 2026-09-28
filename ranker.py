"""
Ранкер поверх пула кандидатов: LightGBM LambdaRank (ndcg@50).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

NON_FEATURES = {"query_id", "item_row", "item_id", "label"}

PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[50],
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambdarank_truncation_level=60,
    verbose=-1,
    seed=42,
    num_threads=16,
    deterministic=True,
    force_col_wise=True,
)


def feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def train(train_df: pl.DataFrame, feats: list[str], num_rounds: int):
    # запросы без единого подходящего объявления в пуле ранкеру ничего не дают
    pos_q = train_df.group_by("query_id").agg(pl.col("label").max()).filter(pl.col("label") > 0)["query_id"]
    df = train_df.filter(pl.col("query_id").is_in(pos_q.implode())).sort("query_id", maintain_order=True)
    groups = df.group_by("query_id", maintain_order=True).len()["len"].to_numpy()
    X = df.select(feats).to_numpy().astype(np.float32)
    dtrain = lgb.Dataset(X, df["label"].to_numpy(), group=groups, feature_name=feats, free_raw_data=False)
    return lgb.train(PARAMS, dtrain, num_boost_round=num_rounds)


def predict_top(model, df: pl.DataFrame, feats: list[str], k: int = 50) -> dict[str, list[str]]:
    """Скор ранкера -> top-k item_id на запрос."""
    s = model.predict(df.select(feats).to_numpy().astype(np.float32), num_threads=16)
    top = (df.select(["query_id", "item_id"]).with_columns(pl.Series("s", s))
           .sort(["query_id", "s"], descending=[False, True], maintain_order=True)
           .group_by("query_id", maintain_order=True).head(k)
           .group_by("query_id", maintain_order=True).agg(pl.col("item_id")))
    return dict(zip(top["query_id"].to_list(), top["item_id"].to_list()))
