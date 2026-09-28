"""
Шаг 3. Ответ для бенчмарка -> answer.csv.

То же, что в симуляции, но история = весь train, корпус = benchmark_items,
эмбеддинги = энкодер, обученный на всём train (embeddings/full).
Ранкер учится на размеченных кандидатах симуляции из validate.py.
"""
from __future__ import annotations

import re

import polars as pl

from common import CACHE, ROOT, Q_COLS, Embeddings, load_bench_queries, load_train, log
from pipeline import Corpus, HistoryStats, LocationModel, MicrocatModel, build_candidates
from ranker import feature_cols, predict_top, train

N_ROUNDS = 350
K = 50


def main():
    items = pl.read_parquet(CACHE / "items.parquet")
    geo_cols = ["item_id", "item_location_id", "item_microcat_id", "item_latitude", "item_longitude"]
    H = load_train().select(Q_COLS + ["item_id"]).join(items.select(geo_cols), on="item_id", maintain_order="left")
    bq = load_bench_queries()
    corpus = Corpus(items.filter(pl.col("in_bench")))

    locm, hist, mcm = LocationModel(H, items), HistoryStats(H), MicrocatModel(H)
    emb = Embeddings("full", corpus.ids)
    cand = build_candidates(corpus, bq, locm, mcm.predict(bq), mcm.classes, hist,
                            q_emb=emb.queries(bq), i_emb=emb.items)

    sim = pl.concat([pl.read_parquet(CACHE / "sim_cand_rank.parquet"),
                     pl.read_parquet(CACHE / "sim_cand_eval.parquet")], how="vertical_relaxed")
    feats = feature_cols(sim)
    top = predict_top(train(sim, feats, N_ROUNDS), cand, feats, K)

    answer = pl.DataFrame({"query_id": bq["query_id"],
                           "answer": [" ".join(top.get(q, [])) for q in bq["query_id"].to_list()]})
    check_answer(answer, bq, set(corpus.ids))
    answer.write_csv(ROOT / "answer.csv")
    log("answer.csv saved")


def check_answer(answer: pl.DataFrame, bq: pl.DataFrame, corpus_ids: set):
    """Требования к файлу из условия."""
    assert answer.columns == ["query_id", "answer"]
    assert answer.height == bq.height and set(answer["query_id"]) == set(bq["query_id"])
    hexre = re.compile(r"^[0-9a-f]{16}$")
    for row in answer["answer"].to_list():
        ids = row.split(" ")
        assert 0 < len(ids) <= K and len(set(ids)) == len(ids)
        assert all(hexre.match(i) and i in corpus_ids for i in ids)


if __name__ == "__main__":
    main()
