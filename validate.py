from __future__ import annotations

import polars as pl

from common import CACHE, Embeddings, log, recall_at_k
from pipeline import Corpus, HistoryStats, LocationModel, MicrocatModel, build_candidates
from ranker import feature_cols, predict_top, train

N_RANK = 20000
N_ROUNDS = 400


def main():
    items = pl.read_parquet(CACHE / "items.parquet")
    geo_cols = ["item_id", "item_location_id", "item_microcat_id", "item_latitude", "item_longitude"]
    A = pl.read_parquet(CACHE / "sim_A.parquet").join(items.select(geo_cols), on="item_id", maintain_order="left")
    corpus_ids = pl.read_parquet(CACHE / "sim_corpus.parquet")["item_id"]
    ev = pl.read_parquet(CACHE / "sim_eval.parquet")
    rk = pl.read_parquet(CACHE / "sim_rank.parquet").head(N_RANK)

    corpus = Corpus(items.filter(pl.col("item_id").is_in(corpus_ids.implode())))
    locm, hist, mcm = LocationModel(A, items), HistoryStats(A), MicrocatModel(A)
    emb = Embeddings("sim", corpus.ids)

    cands = {}
    for name, q in [("eval", ev), ("rank", rk)]:
        cand = build_candidates(corpus, q, locm, mcm.predict(q), mcm.classes, hist,
                                q_emb=emb.queries(q), i_emb=emb.items)
        rel = q.select(["query_id", "relevant"]).explode("relevant").rename({"relevant": "item_id"}) \
               .with_columns(pl.lit(1).alias("label"))
        cand = cand.join(rel, on=["query_id", "item_id"], how="left", maintain_order="left").with_columns(pl.col("label").fill_null(0))
        cand.write_parquet(CACHE / f"sim_cand_{name}.parquet")
        cands[name] = cand

    ce = cands["eval"]
    feats = feature_cols(ce)
    model = train(cands["rank"], feats, N_ROUNDS)
    top = predict_top(model, ce, feats)
    qids, rel = ev["query_id"].to_list(), ev["relevant"].to_list()
    pool = ce.group_by("query_id").agg(pl.col("item_id"))
    pool = dict(zip(pool["query_id"].to_list(), pool["item_id"].to_list()))
    log(f"pool recall: {recall_at_k([pool.get(q, []) for q in qids], rel, k=10**6):.4f}")
    log(f"Recall@50 on simulation: {recall_at_k([top.get(q, []) for q in qids], rel):.4f}")


if __name__ == "__main__":
    main()
