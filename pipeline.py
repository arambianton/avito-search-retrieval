"""
Кандидаты и признаки.

Работает в двух режимах:
  - симуляция: история = половина A из train, корпус = объявления из B;
  - бенчмарк:  история = весь train, корпус = benchmark_items.

Для каждого запроса несколько дешёвых источников скорят весь корпус (BM25 с
разной силой гео-приора, символьные n-граммы, профиль прошлых запросов,
подкатегория и популярность, эмбеддинги), топы объединяются в пул, по каждому кандидату считаются признаки для ранкера.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from common import Vocab, bm25_matrix, count_matrix, log, normalize, query_matrix, tokenize


# Локации
class LocationModel:
    """
    Насколько объявление из локации I подходит поиску из локации L.
    """
    def __init__(self, hist: pl.DataFrame, items: pl.DataFrame, alpha: float = 1.0, n_prior: int = 50):
        self.alpha, self.n_prior = alpha, n_prior
        self.nL = dict(hist.group_by("search_location_id").len().iter_rows())
        self.nLI = defaultdict(dict)
        for L, I, n in hist.group_by(["search_location_id", "item_location_id"]).len().iter_rows():
            self.nLI[L][I] = n
        ic = items.group_by("item_location_id").agg(pl.col("item_latitude").median(),
                                                    pl.col("item_longitude").median(), pl.len())
        self.item_cent = {r[0]: (r[1], r[2]) for r in ic.iter_rows() if r[1] is not None}
        self.item_loc_size = {r[0]: r[3] for r in ic.iter_rows()}
        sc = hist.group_by("search_location_id").agg(pl.col("item_latitude").median(),
                                                     pl.col("item_longitude").median())
        self.search_cent = {r[0]: (r[1], r[2]) for r in sc.iter_rows() if r[1] is not None}

    def is_region(self, L) -> bool:
        # у городских локаций есть собственные объявления
        return self.item_loc_size.get(L, 0) < 5

    def centroid(self, L):
        if not self.is_region(L) and L in self.item_cent:
            return self.item_cent[L]
        return self.search_cent.get(L) or self.item_cent.get(L)

    def vectors(self, L, uloc: np.ndarray):
        # для всех уникальных локаций корпуса: log P(I L) и счётчик n(L I)
        n = self.nL.get(L, 0)
        d = self.nLI.get(L, {})
        cnt = np.array([d.get(I, 0) for I in uloc], dtype=np.float32)
        logp = np.log((cnt + self.alpha) / (n + self.alpha * self.n_prior)).astype(np.float32)
        return logp, cnt

    @staticmethod
    def dist_km(c, lat, lon):
        if c is None:
            return np.full(len(lat), 3000.0, dtype=np.float32)
        return (111.0 * np.sqrt((lat - c[0]) ** 2 + ((lon - c[1]) * np.cos(np.radians(c[0]))) ** 2)).astype(np.float32)


# Подкатегория запроса
class MicrocatModel:
    """
    P(подкатегория объявления | текст запроса и фильтры), линейная модель на TF-IDF
    """

    def __init__(self, hist: pl.DataFrame, seed: int = 0):
        texts = self._texts(hist)
        y = hist["item_microcat_id"].to_numpy()
        self.word = TfidfVectorizer(analyzer=lambda s: self._stems(s), min_df=2, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                    sublinear_tf=True, max_features=300_000)
        X = sp.hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        self.clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=15, tol=None,
                                 n_jobs=-1, random_state=seed)
        self.clf.fit(X, y)
        self.classes = self.clf.classes_
        self.class_pos = {c: i for i, c in enumerate(self.classes)}

    @staticmethod
    def _texts(df):
        return [normalize(q + " " + p) for q, p in zip(df["search_query"].to_list(),
                                                       df["search_infm_params_text"].to_list())]

    @staticmethod
    def _stems(s):
        t = tokenize(s)
        return t + [a + "_" + b for a, b in zip(t, t[1:])]

    def predict(self, queries: pl.DataFrame) -> np.ndarray:
        texts = self._texts(queries)
        X = sp.hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
        return self.clf.predict_proba(X).astype(np.float32)


# Корпус и запросы
@dataclass
class Corpus:
    """Объявления, среди которых ищем, в удобном для нампай виде."""
    df: pl.DataFrame
    ids: np.ndarray = field(init=False)

    def __post_init__(self):
        d = self.df
        self.ids = d["item_id"].to_numpy()
        self.n = len(self.ids)
        self.loc = d["item_location_id"].to_numpy()
        self.uloc, self.loc_idx = np.unique(self.loc, return_inverse=True)
        self.lat = d["item_latitude"].fill_null(0).to_numpy().astype(np.float32)
        self.lon = d["item_longitude"].fill_null(0).to_numpy().astype(np.float32)
        self.cat = d["item_category_id"].to_numpy()
        self.mc = d["item_microcat_id"].to_numpy()
        self.id2row = {x: i for i, x in enumerate(self.ids)}
        # Лексика: BM25 по трём полям
        self.vocab = Vocab()
        tf = {f: count_matrix(d[f"tok_{f}"].to_list(), self.vocab) for f in ("title", "params", "desc")}
        V = len(self.vocab)
        for f in tf:
            tf[f].resize((self.n, V))
        self.bm25 = {f: bm25_matrix(m) for f, m in tf.items()}

        self.has = {}
        for f, m in tf.items():
            b = m.copy(); b.data[:] = 1.0
            self.has[f] = b.T.tocsr()
        anyf = tf["title"] + tf["params"] + tf["desc"]; anyf.data[:] = 1.0
        self.has["any"] = anyf.T.tocsr()
        # Итоговый лексический скор: заголовок + параметры + половина описания
        self.bm25_comb = (self.bm25["title"] + self.bm25["params"] + 0.5 * self.bm25["desc"]).tocsr()
        # Пары соседних стемов по заголовку, параметрам и началу описания
        self.bigram_codes, self.has_bigram = self._bigram_matrix(d)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2,
                                        sublinear_tf=True, dtype=np.float32)
        self.char_T = self.char_vec.fit_transform([normalize(t) for t in d["item_title_raw"].to_list()]).T.tocsr()
        self._init_item_feats(d)

    BIGRAM_DESC_TOKENS = 150

    def _bigram_matrix(self, d):
        V = len(self.vocab)
        rows, codes = [], []
        for i, (t, p, ds) in enumerate(zip(d["tok_title"].to_list(), d["tok_params"].to_list(), d["tok_desc"].to_list())):
            for toks in (t, p, ds[: self.BIGRAM_DESC_TOKENS]):
                ids = np.array(self.vocab.ids(toks, grow=False), dtype=np.int64)
                if len(ids) > 1:
                    codes.append(ids[:-1] * V + ids[1:])
                    rows.append(np.full(len(ids) - 1, i, dtype=np.int32))
        codes = np.concatenate(codes); rows = np.concatenate(rows)
        uniq, col = np.unique(codes, return_inverse=True)
        m = sp.csr_matrix((np.ones(len(col), np.float32), (rows, col)), shape=(self.n, len(uniq)))
        m.data[:] = 1.0
        return uniq, m.T.tocsr()

    def query_bigrams(self, token_lists) -> tuple[sp.csr_matrix, np.ndarray]:
        """Запросы -> бинарная матрица их биграмм в индексе корпуса и число биграмм запроса."""
        V = len(self.vocab)
        indptr, indices, nb = [0], [], []
        for toks in token_lists:
            ids = [self.vocab.index.get(t, -1) for t in toks]
            pairs = {(a, b) for a, b in zip(ids, ids[1:])}
            nb.append(len(pairs))
            codes = np.array([a * V + b for a, b in pairs if a >= 0 and b >= 0], dtype=np.int64)
            pos = np.searchsorted(self.bigram_codes, codes)
            ok = (pos < len(self.bigram_codes)) & (self.bigram_codes[np.minimum(pos, len(self.bigram_codes) - 1)] == codes)
            indices.extend(pos[ok].tolist()); indptr.append(len(indices))
        m = sp.csr_matrix((np.ones(len(indices), np.float32), indices, indptr),
                          shape=(len(token_lists), len(self.bigram_codes)))
        return m, np.array(nb, dtype=np.float32)

    def _init_item_feats(self, d):
        rc = d["item_rating_reviews_count"].fill_null(0).to_numpy()
        self.item_feats = {
            "log_reviews": np.log1p(rc).astype(np.float32),
            "rating": d["item_rating"].fill_null(-1).to_numpy().astype(np.float32),
            "log_price": np.log1p(d["item_price"].fill_null(0).clip(0, None).to_numpy()).astype(np.float32),
            "phone_hidden": d["item_is_phone_hidden"].cast(pl.Float32).to_numpy(),
            "msg_forbidden": d["item_is_message_forbidden"].cast(pl.Float32).to_numpy(),
            "title_len": d["tok_title"].list.len().to_numpy().astype(np.float32),
            "desc_len": np.log1p(d["tok_desc"].list.len().to_numpy()).astype(np.float32),
            "params_len": d["tok_params"].list.len().to_numpy().astype(np.float32),
            "is_services": (self.cat == 114).astype(np.float32),
        }


# История выборов
def text_key(q: str) -> str:
    return " ".join(tokenize(q))


class HistoryStats:

    def __init__(self, hist: pl.DataFrame, ref_size: int = 250_000):
        self.scale = ref_size / hist.height
        h = hist.with_columns(pl.col("search_query").map_elements(text_key, return_dtype=pl.String).alias("tkey"))
        self.item_cnt = dict(h.group_by("item_id").len().iter_rows())
        self.text_cnt = dict(h.group_by("tkey").len().iter_rows())
        self.text_item = defaultdict(dict)
        for k, i, n in h.group_by(["tkey", "item_id"]).len().iter_rows():
            self.text_item[k][i] = n
        self.textloc_item = defaultdict(dict)
        for k, L, i, n in h.group_by(["tkey", "search_location_id", "item_id"]).len().iter_rows():
            self.textloc_item[(k, L)][i] = n
        self.item_qtokens = {i: " ".join(ks).split() for i, ks in
                             h.group_by("item_id").agg(pl.col("tkey")).iter_rows()}


# Кандидаты + признаки
# Сколько объявлений берём из каждого источника в общий пул.
POOL = {"lex_loc": 300, "lex_same": 100, "lex_soft": 150, "lex_nogeo": 100, "char_loc": 100, "histq_loc": 50,
        "dense_loc": 300, "dense_soft": 100, "mc_pop": 100}
# Веса дешёвых скорингов для отбора. W_LOC и W_DIST подобраны перебором на симуляции: BM25 + 3*log P(I|L) - log(1 + км/10).
W_LOC, W_DIST = 3.0, 1.0
W_DENSE = 25.0


def _topk(score: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(score))
    idx = np.argpartition(-score, k - 1)[:k]
    return idx[np.argsort(-score[idx])]


def build_candidates(corpus: Corpus, queries: pl.DataFrame, locm: LocationModel, mc_proba: np.ndarray,
                     mc_classes: np.ndarray, hist: HistoryStats, q_emb: np.ndarray | None = None,
                     i_emb: np.ndarray | None = None, chunk: int = 256, pool: dict | None = None) -> pl.DataFrame:
    """
    Для каждого запроса отбирает пул кандидатов и считает признаки.
    """
    pool = pool or POOL
    n_q = queries.height
    qtok = [tokenize(q) for q in queries["search_query"].to_list()]
    ptok = [tokenize(p) for p in queries["search_infm_params_text"].to_list()]
    tkeys = [" ".join(t) for t in qtok]
    Q = query_matrix(qtok, corpus.vocab)
    QP = query_matrix(ptok, corpus.vocab)
    QB, q_nbigrams = corpus.query_bigrams(qtok)
    QC = corpus.char_vec.transform([normalize(q) for q in queries["search_query"].to_list()]).tocsr()
    q_nterms = np.asarray(Q.sum(axis=1)).ravel()
    p_nterms = np.asarray(QP.sum(axis=1)).ravel()
    locs = queries["search_location_id"].to_list()
    qcat = queries["search_category"].to_list()
    qids = queries["query_id"].to_list()

    # позиция подкатегории объявления в классах модели (-1, если класс неизвестен)
    cls_pos = {c: i for i, c in enumerate(mc_classes)}
    item_mc_pos = np.array([cls_pos.get(m, -1) for m in corpus.mc])
    known_mc = item_mc_pos >= 0
    item_mc_pos_safe = np.where(known_mc, item_mc_pos, 0)
    item_hist = np.array([hist.item_cnt.get(i, 0) for i in corpus.ids], dtype=np.float32) * hist.scale
    loc_size = {L: n for L, n in zip(*np.unique(corpus.loc, return_counts=True))}
    if i_emb is not None:
        i_emb = i_emb.astype(np.float32)
    # BM25 по профилю прошлых запросов (есть только у объявлений из истории)
    hvocab = Vocab()
    hq_bm25 = bm25_matrix(count_matrix([hist.item_qtokens.get(i, []) for i in corpus.ids], hvocab))
    HQ = query_matrix(qtok, hvocab)

    out = []
    t0 = time.time()
    for c0 in range(0, n_q, chunk):
        sl = slice(c0, min(c0 + chunk, n_q))
        Qc, QPc = Q[sl], QP[sl]
        S = {f: (Qc @ corpus.bm25[f]).toarray() for f in ("title", "params", "desc")}
        COV = {f: (Qc @ corpus.has[f]).toarray() for f in ("title", "params", "desc", "any")}
        SPc = (QPc @ corpus.has["params"]).toarray()
        CH = (QC[sl] @ corpus.char_T).toarray()
        BG = (QB[sl] @ corpus.has_bigram).toarray()
        HS = (HQ[sl] @ hq_bm25).toarray()
        E = q_emb[sl].astype(np.float32) @ i_emb.T if q_emb is not None else None

        for r, qi in enumerate(range(sl.start, sl.stop)):
            L = locs[qi]
            lp_u, cnt_u = locm.vectors(L, corpus.uloc)
            lp, lcnt = lp_u[corpus.loc_idx], cnt_u[corpus.loc_idx]
            dist = locm.dist_km(locm.centroid(L), corpus.lat, corpus.lon)
            logd = np.log1p(dist / 10.0)
            same = corpus.loc == L
            cat_ok = (corpus.cat == 114) if qcat[qi] == 114 else np.ones(corpus.n, bool)
            ban = np.where(cat_ok, 0.0, -1e6).astype(np.float32)

            comb = S["title"][r] + S["params"][r] + 0.5 * S["desc"][r]
            pmc = np.where(known_mc, mc_proba[qi][item_mc_pos_safe], 0.0).astype(np.float32)
            geo = W_LOC * lp - W_DIST * logd

            # Несколько "взглядов" на один и тот же корпус
            soft_geo = 0.3 * geo
            sources = {
                "lex_loc": comb + geo + ban,
                "lex_same": comb + 100.0 * same + ban,
                "lex_soft": comb + soft_geo + ban,
                "lex_nogeo": comb + ban,
                "char_loc": 10.0 * CH[r] + geo + ban,
                "histq_loc": HS[r] + geo + ban - 100.0 * (HS[r] == 0),
                "mc_pop": np.log(pmc + 1e-4) + geo + 0.3 * corpus.item_feats["log_reviews"] + ban,
            }
            if E is not None:
                sources["dense_loc"] = W_DENSE * E[r] + geo + ban
                sources["dense_soft"] = W_DENSE * E[r] + soft_geo + ban
            ranks, cand = {}, []
            for name, sc in sources.items():
                top = _topk(sc, pool[name])
                ranks[name] = top
                cand.append(top)
            hist_items = hist.text_item.get(tkeys[qi], {})
            hi = [corpus.id2row[i] for i in hist_items if i in corpus.id2row]
            if hi:
                cand.append(np.array(hi))
            j = np.unique(np.concatenate(cand))

            nq = max(q_nterms[qi], 1)
            f = {
                "query_id": np.full(len(j), qids[qi], dtype=object),
                "item_row": j,
                # лексика
                "bm25_title": S["title"][r][j], "bm25_params": S["params"][r][j],
                "bm25_desc": S["desc"][r][j], "bm25_comb": comb[j], "char_title": CH[r][j],
                "bm25_histq": HS[r][j],
                "cov_title": COV["title"][r][j] / nq, "cov_params": COV["params"][r][j] / nq,
                "cov_desc": COV["desc"][r][j] / nq, "cov_any": COV["any"][r][j] / nq,
                "sp_overlap": SPc[r][j] / max(p_nterms[qi], 1),
                # доля биграмм запроса, найденных в объявлении
                "bigram_cov": BG[r][j] / q_nbigrams[qi] if q_nbigrams[qi] > 0 else np.full(len(j), np.nan, np.float32),
                # география
                "same_loc": same[j].astype(np.float32), "loc_logp": lp[j], "loc_cnt": np.log1p(lcnt[j]),
                "log_dist": logd[j],
                # подкатегория
                "p_mc": pmc[j],
                # история
                "item_hist": np.log1p(item_hist[j]),
                "hist_text_item": np.log1p(np.array([hist_items.get(corpus.ids[x], 0) for x in j], np.float32) * hist.scale),
            }
            tl = hist.textloc_item.get((tkeys[qi], L), {})
            f["hist_textloc_item"] = np.log1p(np.array([tl.get(corpus.ids[x], 0) for x in j], np.float32) * hist.scale)
            if E is not None:
                e = E[r][j]
                f["dense"] = e
                f["dense_rel"] = e - E[r][ranks["dense_loc"][0]]
            # относительная сила лексики внутри запроса
            f["comb_rel"] = comb[j] / (comb[ranks["lex_loc"][0]] + 1e-6)
            # ранги в источниках (1000 = не попал в топ источника)
            for name, top in ranks.items():
                rk = np.full(corpus.n, 1000, np.int32); rk[top] = np.arange(len(top))
                f[f"rank_{name}"] = rk[j]
            for k, v in corpus.item_feats.items():
                f[k] = v[j]
            # признаки запроса (одинаковы для всех кандидатов, но важны в сочетании)
            f["q_len"] = np.full(len(j), q_nterms[qi], np.float32)
            f["q_has_params"] = np.full(len(j), float(p_nterms[qi] > 0), np.float32)
            f["q_region"] = np.full(len(j), float(locm.is_region(L)), np.float32)
            f["q_loc_hist"] = np.full(len(j), np.log1p(locm.nL.get(L, 0) * hist.scale), np.float32)
            f["q_loc_items"] = np.full(len(j), np.log1p(loc_size.get(L, 0)), np.float32)
            f["q_text_hist"] = np.full(len(j), np.log1p(hist.text_cnt.get(tkeys[qi], 0) * hist.scale), np.float32)
            f["q_cat0"] = np.full(len(j), float(qcat[qi] == 0), np.float32)
            f["q_mc_max"] = np.full(len(j), mc_proba[qi].max(), np.float32)
            out.append(pl.DataFrame(f))
        log(f"candidates: {sl.stop}/{n_q} queries, {time.time() - t0:.0f}s")
    df = pl.concat(out)
    return df.with_columns(pl.Series("item_id", corpus.ids[df["item_row"].to_numpy()]))
