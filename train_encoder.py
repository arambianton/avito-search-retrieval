"""
Дообучение энкодера intfloat/multilingual-e5-base на парах "запрос - выбранное объявление"
и расчёт эмбеддингов

Лосс: MultipleNegativesRankingLoss с GradCache (батч 256 при памяти как у 32),
в батче нет одинаковых текстов
"""
import os
import shutil
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)
from sentence_transformers.training_args import BatchSamplers

MODE = sys.argv[1]                  # "sim" или "full"
BASE_MODEL = "intfloat/multilingual-e5-base"
MAX_SEQ, BATCH, MINI_BATCH, LR = 128, 256, 32, 5e-5
SEED, P_SHARED = 42, 0.33           # как в prepare.py

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", ROOT)
OUT = os.path.join(ROOT, "embeddings", MODE)
os.makedirs(OUT, exist_ok=True)


def query_text(q: str, params: str) -> str:
    q = q.strip()
    return f"query: {q} | {params}" if params else f"query: {q}"


def item_text(title: str, params: str, desc: str) -> str:
    return f"passage: {title}\n{(params or '')[:300]}\n{(desc or '')[:700]}"


tr = pd.read_parquet(f"{DATA_DIR}/train.parquet",
                     columns=["search_query", "search_infm_params_text", "item_id",
                              "item_title_raw", "item_infm_params_text", "item_description_raw"]).fillna("")
if MODE == "sim":
    rng = np.random.default_rng(SEED)
    _, item_idx = np.unique(tr["item_id"].to_numpy(), return_inverse=True)
    n_items = item_idx.max() + 1
    side_b = rng.random(n_items) < 0.5
    shared = rng.random(n_items) < P_SHARED
    row_b = rng.random(len(tr)) < 0.5
    fit_rows = tr[~np.where(shared[item_idx], row_b, side_b[item_idx])]
else:
    fit_rows = tr

pairs = fit_rows.drop_duplicates(["search_query", "search_infm_params_text", "item_id"])
train_ds = Dataset.from_dict({
    "anchor": [query_text(q, p) for q, p in zip(pairs.search_query, pairs.search_infm_params_text)],
    "positive": [item_text(t, p, d) for t, p, d in zip(pairs.item_title_raw, pairs.item_infm_params_text,
                                                      pairs.item_description_raw)],
}).shuffle(seed=SEED)

model = SentenceTransformer(BASE_MODEL)
model.max_seq_length = MAX_SEQ
model[0].auto_model.embeddings.word_embeddings.weight.requires_grad = False
args = SentenceTransformerTrainingArguments(
    output_dir=f"{OUT}/ckpt", num_train_epochs=1, per_device_train_batch_size=BATCH,
    learning_rate=LR, warmup_ratio=0.05, weight_decay=0.01, fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=100, save_strategy="no",
    report_to="none", seed=SEED, dataloader_num_workers=2,
)
SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds,
                           loss=losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=MINI_BATCH)).train()
shutil.rmtree(f"{OUT}/ckpt", ignore_errors=True)
model.eval()


def encode(texts):
    return model.encode(texts, batch_size=512, convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=False).astype(np.float16)


if MODE == "sim":
    items = tr.drop_duplicates("item_id")
else:
    items = pd.read_parquet(f"{DATA_DIR}/benchmark_items.parquet",
                            columns=["item_id", "item_title_raw", "item_infm_params_text",
                                     "item_description_raw"]).fillna("")
np.save(f"{OUT}/item_emb.npy", encode([item_text(t, p, d) for t, p, d in zip(
    items.item_title_raw, items.item_infm_params_text, items.item_description_raw)]))
items[["item_id"]].to_parquet(f"{OUT}/item_ids.parquet", index=False)

bq = pd.read_parquet(f"{DATA_DIR}/benchmark_queries.parquet").fillna("")
qs = pd.concat([tr[["search_query", "search_infm_params_text"]],
                bq[["search_query", "search_infm_params_text"]]]).drop_duplicates().reset_index(drop=True)
np.save(f"{OUT}/query_emb.npy", encode([query_text(q, p) for q, p in zip(qs.search_query, qs.search_infm_params_text)]))
qs.to_parquet(f"{OUT}/query_keys.parquet", index=False)
