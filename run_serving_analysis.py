"""
run_serving_analysis.py — Q4: Serving & Scale Analysis

Measures the two-stage pipeline the way it would actually be served:

  1. Index memory    embedding matrix, feature store, GBDT model, measured as
                     real byte counts rather than RSS deltas
  2. Latency         one request at a time, broken into stage-1 retrieval,
                     feature extraction and GBDT scoring, reported at p50/p95/p99
  3. Cost / QPS      back-of-envelope cost per 1000 queries at a p99 SLA
  4. Scale           what breaks first at 10x, argued from the measured numbers

The model here is the real trained Booster, not a stand-in: GBDT scoring is part
of what a request pays for, and timing a random-number generator in its place
would understate the tail.

Usage
-----
  python run_serving_analysis.py --dataset mind --ranker emb
  python run_serving_analysis.py --dataset ebnerd_small --ranker emb
  python run_serving_analysis.py --dataset both
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data.schema import (
    COL_ARTICLE_ID, COL_CLICK_TIME, COL_IMPRESSION_ID, COL_IMPRESSION_TIME,
    COL_SPLIT, COL_USER_ID, SPLIT_VAL,
)
from src.features.behavioural_features import BehaviouralFeatureExtractor
from src.features.embedding_index import EmbeddingIndex

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("serving")

BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PRED_DIR      = BASE_DIR / "data" / "predictions"
CACHE_DIR     = BASE_DIR / "data" / "cache"
MODELS_DIR    = BASE_DIR / "models"
RESULTS_DIR   = BASE_DIR / "results"

# Target service level the cost model is quoted against.
SLA_P99_MS = 100.0
# us-east-1 on-demand, 2 vCPU, Sep 2026 list price.
INSTANCE_HOURLY_USD = 0.068
INSTANCE_VCPUS      = 2


def _mb(nbytes) -> float:
    return nbytes / (1024 * 1024)


def _df_bytes(df: pd.DataFrame) -> int:
    return int(df.memory_usage(deep=True).sum())


def analyse(dataset_name: str, ranker: str, n_requests: int, seed: int = 42) -> dict:
    logger.info("=" * 64)
    logger.info(f"Serving analysis  |  {dataset_name}  |  stage-1 = {ranker}")
    logger.info("=" * 64)

    proc_dir = PROCESSED_DIR / dataset_name
    if not proc_dir.exists():
        logger.error(f"Missing processed data for {dataset_name}")
        return {}

    model_path = MODELS_DIR / f"lgbm_{dataset_name.lower()}_{ranker}_serving_model.txt"
    if not model_path.exists():
        logger.error(f"Missing {model_path}. Train with run_reranker.py first.")
        return {}

    articles    = pd.read_parquet(proc_dir / "articles.parquet")
    history     = pd.read_parquet(proc_dir / "history.parquet")
    impressions = pd.read_parquet(proc_dir / "impressions.parquet")

    for df, col in ((history, COL_CLICK_TIME), (impressions, COL_IMPRESSION_TIME)):
        s = pd.to_datetime(df[col], errors="coerce")
        if hasattr(s.dtype, "tz") and s.dtype.tz is not None:
            s = s.dt.tz_localize(None)
        df[col] = s

    # ── 1. Index memory ───────────────────────────────────────────────────────
    logger.info("[1] Index memory")

    index = EmbeddingIndex(
        articles_df=articles, dataset_name=dataset_name, zip_dir=BASE_DIR,
        cache_path=CACHE_DIR / f"emb_{dataset_name}.npz",
    ).build()
    emb_bytes = index.matrix.nbytes
    id_map_bytes = sum(len(k) + 56 for k in index.id_to_idx) + 64 * len(index.id_to_idx)

    booster = lgb.Booster(model_file=str(model_path))
    feat_path = model_path.with_suffix(".features.json")
    feature_cols = (json.loads(feat_path.read_text())
                    if feat_path.exists() else booster.feature_name())
    model_bytes = model_path.stat().st_size

    hist_bytes = _df_bytes(history)
    art_bytes  = _df_bytes(articles)

    logger.info(f"    embedding matrix   {index.matrix.shape[0]:>9,} x {index.dim:<4} "
                f"{_mb(emb_bytes):>9.1f} MB")
    logger.info(f"    id -> row map      {len(index.id_to_idx):>9,} entries "
                f"{_mb(id_map_bytes):>9.1f} MB")
    logger.info(f"    click history      {len(history):>9,} rows      "
                f"{_mb(hist_bytes):>9.1f} MB")
    logger.info(f"    article store      {len(articles):>9,} rows      "
                f"{_mb(art_bytes):>9.1f} MB")
    logger.info(f"    GBDT model         {booster.num_trees():>9,} trees     "
                f"{_mb(model_bytes):>9.1f} MB")
    total_bytes = emb_bytes + id_map_bytes + hist_bytes + art_bytes + model_bytes
    logger.info(f"    {'TOTAL':<18} {'':>9} {'':4} {_mb(total_bytes):>9.1f} MB")

    # Per-article cost is the number that actually scales with catalogue size.
    per_article_kb = (emb_bytes + art_bytes) / max(len(articles), 1) / 1024
    logger.info(f"    -> {per_article_kb:.2f} KB per article "
                f"(embedding + metadata)")

    # ── 2. Latency ────────────────────────────────────────────────────────────
    logger.info(f"[2] Latency over {n_requests} single-user requests")

    val = impressions[impressions[COL_SPLIT] == SPLIT_VAL]
    imp_ids = (val[COL_IMPRESSION_ID].drop_duplicates()
               .sample(min(n_requests, val[COL_IMPRESSION_ID].nunique()),
                       random_state=seed))
    requests = [g for _, g in val[val[COL_IMPRESSION_ID].isin(imp_ids)]
                .groupby(COL_IMPRESSION_ID, sort=False)]

    # Built once at startup in a real service, so excluded from per-request time.
    hist_idx = EmbeddingIndex.preindex_history(history)
    pop = history[COL_ARTICLE_ID].astype(str).value_counts().rank(pct=True)
    extractor = BehaviouralFeatureExtractor(history, articles, article_popularity=pop)

    t_stage1, t_feats, t_score, t_total, n_cands = [], [], [], [], []

    for grp in requests:
        uid = grp[COL_USER_ID].iloc[0]
        t   = grp[COL_IMPRESSION_TIME].iloc[0]
        cands = grp[COL_ARTICLE_ID].tolist()

        r0 = perf_counter()
        user_vec = index.make_user_vector(uid, hist_idx, t)
        scores = index.score_candidates(user_vec, cands)
        r1 = perf_counter()

        req = grp.copy()
        req[f"{ranker}_score"] = req[COL_ARTICLE_ID].map(scores)
        feats = extractor.extract_features(req)
        for c in feature_cols:
            if c not in feats.columns:
                feats[c] = 0.0
        r2 = perf_counter()

        booster.predict(feats[feature_cols].fillna(0).to_numpy(dtype=np.float32))
        r3 = perf_counter()

        t_stage1.append((r1 - r0) * 1000)
        t_feats.append((r2 - r1) * 1000)
        t_score.append((r3 - r2) * 1000)
        t_total.append((r3 - r0) * 1000)
        n_cands.append(len(cands))

    def pct(a, p):
        return float(np.percentile(a, p))

    logger.info(f"    mean candidates/request  {np.mean(n_cands):.1f}")
    logger.info(f"    {'phase':<22}{'p50':>9}{'p95':>9}{'p99':>9}   (ms)")
    for label, arr in (("stage-1 retrieval", t_stage1),
                       ("feature extraction", t_feats),
                       ("GBDT scoring", t_score),
                       ("END-TO-END", t_total)):
        logger.info(f"    {label:<22}{pct(arr,50):>9.2f}{pct(arr,95):>9.2f}"
                    f"{pct(arr,99):>9.2f}")

    p99 = pct(t_total, 99)
    p50 = pct(t_total, 50)

    # ── 3. Cost / QPS ─────────────────────────────────────────────────────────
    logger.info(f"[3] Cost / QPS at a p99 < {SLA_P99_MS:.0f} ms SLA")

    # Serve from the tail, not the mean: capacity planned on p50 misses the SLA
    # for the slowest requests.
    qps_per_core = 1000.0 / p99
    qps_instance = qps_per_core * INSTANCE_VCPUS
    cost_per_1k = (INSTANCE_HOURLY_USD / 3600.0) * (1000.0 / max(qps_instance, 1e-9))

    logger.info(f"    sustained QPS per core        {qps_per_core:>8.1f}")
    logger.info(f"    sustained QPS per 2-vCPU node {qps_instance:>8.1f}")
    logger.info(f"    cost per 1,000 queries        ${cost_per_1k:>8.5f}")
    logger.info(f"    cost per 1M queries           ${cost_per_1k*1000:>8.2f}")
    meets = "MEETS" if p99 < SLA_P99_MS else "MISSES"
    logger.info(f"    single-node p99 {p99:.1f} ms -> {meets} the {SLA_P99_MS:.0f} ms SLA")

    # ── 4. Scale ──────────────────────────────────────────────────────────────
    logger.info("[4] What breaks first at 10x")
    share_s1 = np.mean(t_stage1) / np.mean(t_total) * 100
    share_ft = np.mean(t_feats) / np.mean(t_total) * 100
    share_sc = np.mean(t_score) / np.mean(t_total) * 100
    logger.info(f"    time split: stage-1 {share_s1:.0f}%  features {share_ft:.0f}%  "
                f"GBDT {share_sc:.0f}%")
    logger.info(f"    10x catalogue -> embedding matrix "
                f"{_mb(emb_bytes*10):.0f} MB, still RAM-resident; brute-force "
                f"cosine is O(candidates) per request, not O(catalogue), so "
                f"latency is unchanged")
    logger.info(f"    10x users     -> history store {_mb(hist_bytes*10):.0f} MB. "
                f"This is the first thing to break: it is held as an in-process "
                f"pandas frame, so it must move to a key-value store")
    # Requests share no mutable state, so throughput scales by adding nodes.
    # The catch is that each node needs its own full copy of the indices.
    logger.info(f"    10x QPS       -> {qps_instance*10:.0f} QPS on 10 nodes at "
                f"${INSTANCE_HOURLY_USD*10:.2f}/hr, but each node carries its own "
                f"{_mb(total_bytes):.0f} MB of index, so memory cost grows with "
                f"throughput and not just with data")

    return {
        "dataset": dataset_name, "ranker": ranker,
        "emb_index_mb": round(_mb(emb_bytes), 2),
        "history_mb": round(_mb(hist_bytes), 2),
        "articles_mb": round(_mb(art_bytes), 2),
        "model_mb": round(_mb(model_bytes), 3),
        "total_mb": round(_mb(total_bytes), 2),
        "kb_per_article": round(per_article_kb, 3),
        "mean_candidates": round(float(np.mean(n_cands)), 1),
        "p50_ms": round(p50, 3), "p95_ms": round(pct(t_total, 95), 3),
        "p99_ms": round(p99, 3),
        "stage1_p99_ms": round(pct(t_stage1, 99), 3),
        "features_p99_ms": round(pct(t_feats, 99), 3),
        "gbdt_p99_ms": round(pct(t_score, 99), 3),
        "qps_per_core": round(qps_per_core, 2),
        "cost_per_1k_usd": round(cost_per_1k, 6),
        "meets_sla": bool(p99 < SLA_P99_MS),
    }


def main():
    parser = argparse.ArgumentParser(description="Q4: serving and scale analysis")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "ebnerd_small", "both"],
                        default="both")
    parser.add_argument("--ranker", choices=["bm25", "emb"], default="emb")
    parser.add_argument("--requests", type=int, default=300)
    args = parser.parse_args()

    if lgb is None:
        logger.error("lightgbm not installed.")
        sys.exit(1)

    names = {"mind": ["MIND"], "ebnerd": ["EBNERD_DEMO"],
             "ebnerd_small": ["EBNERD_SMALL"], "both": ["MIND", "EBNERD_SMALL"]}[args.dataset]

    rows = [r for n in names if (r := analyse(n, args.ranker, args.requests))]
    if rows:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "serving_analysis.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
