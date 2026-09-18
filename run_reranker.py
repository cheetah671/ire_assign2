"""
run_reranker.py — Q2 & Q3: Re-Ranker and Baseline Beaten

Orchestrator to:
1. Score stage-1 (BM25 or Embeddings) on the SAME impressions used for training
   and validation — no dummy placeholder scores.
2. Extract behavioural features.
3. Train LightGBM ranker on train impressions.
4. Score and evaluate on val impressions.
5. Perform an ablation study (serving-safe vs. full feature set).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.features.behavioural_features import BehaviouralFeatureExtractor
from src.features.embedding_index import EmbeddingIndex
from src.ranking.bm25_ranker import BM25Ranker
from src.ranking.reranker import LightGBMReranker
from src.data.schema import (
    COL_ARTICLE_ID, COL_CLICK_TIME, COL_IMPRESSION_ID, COL_IMPRESSION_TIME,
    COL_LABEL, COL_SPLIT, COL_USER_ID,
    SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_reranker")

BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PRED_DIR      = BASE_DIR / "data" / "predictions"
RESULTS_DIR   = BASE_DIR / "results"
MODELS_DIR    = BASE_DIR / "models"
CACHE_DIR     = BASE_DIR / "data" / "cache"

# Q9: features whose training-time value cannot be reconstructed at serving time.
# This is dataset-specific, not a universal list.
#
# MIND      behaviors.tsv gives the history as a bare article-id string with no
#           per-click timestamps, and news.tsv has no publication date, so every
#           time-derived feature collapses to a constant at serving.
# EB-NeRD   the test bundle ships history.parquet with impression_time_fixed and
#           read_time_fixed, and articles carry published_time, so the same
#           features are genuinely available.
SERVING_UNSAFE = {
    "MIND": [
        "user_hist_recency_sum", "session_clicks_1h", "session_clicks_24h",
        "freshness_days", "hist_mean_read_time", "hist_mean_scroll",
    ],
    "EBNERD_DEMO": [],
    "EBNERD_SMALL": [],
}


def _strip_tz(s):
    s = pd.to_datetime(s, errors="coerce")
    if hasattr(s.dtype, "tz") and s.dtype.tz is not None:
        return s.dt.tz_localize(None)
    return s


def popularity_percentile(history: pd.DataFrame, cutoff=None) -> pd.Series:
    """
    Article popularity as a rank-percentile in [0, 1] of how often the article
    appears in users' click history.

    Percentile rather than a raw count because the serving-time corpus
    (MIND-large-test) has a different user count than the training corpus, so
    raw counts are on incomparable scales while percentiles are not.

    `cutoff` drops clicks at or after that timestamp. Without it the count would
    be taken over the whole history table, 54% of which lands inside the
    validation period on MIND — that is future information relative to the
    impressions being scored, and it inflates validation metrics (Q1.4).
    """
    h = history
    if cutoff is not None:
        ct = pd.to_datetime(h[COL_CLICK_TIME], errors="coerce")
        if hasattr(ct.dtype, "tz") and ct.dtype.tz is not None:
            ct = ct.dt.tz_localize(None)
        h = h[ct < cutoff]
        logger.info(
            f"Popularity window: {len(h):,}/{len(history):,} clicks before {cutoff}"
        )
    counts = h[COL_ARTICLE_ID].astype(str).value_counts()
    return counts.rank(pct=True)


def load_data(dataset_name: str):
    proc_dir = PROCESSED_DIR / dataset_name
    if not proc_dir.exists():
        logger.error(f"Missing data for {dataset_name}. Expected at {proc_dir}")
        return None, None, None, None

    logger.info("Loading processed tables...")
    articles = pd.read_parquet(proc_dir / "articles.parquet")
    history = pd.read_parquet(proc_dir / "history.parquet")
    impressions = pd.read_parquet(proc_dir / "impressions.parquet")

    history["click_time"] = _strip_tz(history["click_time"])
    impressions["impression_time"] = _strip_tz(impressions["impression_time"])

    train_preds = impressions[impressions[COL_SPLIT] == SPLIT_TRAIN].copy()
    val_preds = impressions[impressions[COL_SPLIT] == SPLIT_VAL].copy()

    return articles, history, train_preds, val_preds


def score_stage1(ranker_type, index, df, history, hist_idx, desc):
    """
    Score every candidate in `df` with the stage-1 retriever and return `df`
    with `<ranker>_score` (and, for embeddings, `emb_max_sim`) attached.

    Applied identically to train and val so the stage-1 feature the reranker
    learns from is the same quantity it sees at inference.
    """
    score_col = f"{ranker_type}_score"
    grouped = list(df.groupby(COL_IMPRESSION_ID, sort=False))

    out = []
    for _imp_id, group in tqdm(grouped, desc=desc, unit="imp"):
        uid = group["user_id"].iloc[0]
        t = group["impression_time"].iloc[0]
        candidate_ids = group[COL_ARTICLE_ID].tolist()
        grp = group.copy()

        if ranker_type == "bm25":
            query_tokens = index.make_query(uid, history, t, history_index=hist_idx)
            scores = index.score_candidates(query_tokens, candidate_ids)
            grp[score_col] = grp[COL_ARTICLE_ID].map(scores)
        else:
            user_vec = index.make_user_vector(uid, hist_idx, t)
            scores = index.score_candidates(user_vec, candidate_ids)
            grp[score_col] = grp[COL_ARTICLE_ID].map(scores)
            grp["emb_max_sim"] = _max_sim(index, hist_idx, uid, t, candidate_ids)

        out.append(grp)

    return pd.concat(out, ignore_index=True)


def _max_sim(index, hist_idx, uid, imp_time, candidate_ids, max_history=50):
    """Max cosine similarity between each candidate and any single history article."""
    user_df = hist_idx.get(uid)
    if user_df is None or len(user_df) == 0:
        return np.zeros(len(candidate_ids), dtype=np.float32)

    hist = user_df[user_df["click_time"].notna() &
                   (user_df["click_time"] < imp_time)].head(max_history)

    h_rows = [index.matrix[index.id_to_idx[str(a)]]
              for a in hist[COL_ARTICLE_ID] if str(a) in index.id_to_idx]
    c_rows = [index.matrix[index.id_to_idx[str(c)]] if str(c) in index.id_to_idx
              else np.zeros(index.dim, dtype=np.float32) for c in candidate_ids]

    if not h_rows:
        return np.zeros(len(candidate_ids), dtype=np.float32)

    H = np.vstack(h_rows).astype(np.float32)
    C = np.vstack(c_rows).astype(np.float32)
    return (C @ H.T).max(axis=1)


def main():
    parser = argparse.ArgumentParser(description="Q2 & Q3: Re-Ranker and Baseline")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "ebnerd_small"], default="ebnerd")
    parser.add_argument("--ranker", choices=["bm25", "emb"], default="bm25")
    parser.add_argument("--train-imps", type=int, default=20_000)
    parser.add_argument("--val-imps", type=int, default=5_000)
    args = parser.parse_args()

    dataset_map = {"mind": "MIND", "ebnerd": "EBNERD_DEMO", "ebnerd_small": "EBNERD_SMALL"}
    dataset_name = dataset_map[args.dataset]
    score_col = f"{args.ranker}_score"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    articles, history, train_df, val_df = load_data(dataset_name)
    if articles is None:
        return

    n_train_imps = min(args.train_imps, train_df[COL_IMPRESSION_ID].nunique())
    n_val_imps   = min(args.val_imps,  val_df[COL_IMPRESSION_ID].nunique())
    logger.info(f"Sampling {n_train_imps:,} train impressions and {n_val_imps:,} val impressions …")
    sampled_train = train_df[train_df[COL_IMPRESSION_ID].isin(
        train_df[COL_IMPRESSION_ID].drop_duplicates().sample(n_train_imps, random_state=42)
    )].copy()

    sampled_val = val_df[val_df[COL_IMPRESSION_ID].isin(
        val_df[COL_IMPRESSION_ID].drop_duplicates().sample(n_val_imps, random_state=42)
    )].copy()

    # ── Stage 1: build the retriever and score BOTH splits ────────────────────
    logger.info(f"Building stage-1 index ({args.ranker}) …")
    t0 = perf_counter()
    if args.ranker == "bm25":
        index = BM25Ranker(articles).build()
        hist_idx = BM25Ranker.preindex_history(history)
    else:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        index = EmbeddingIndex(
            articles_df=articles,
            dataset_name=dataset_name,
            zip_dir=BASE_DIR,
            cache_path=CACHE_DIR / f"emb_{dataset_name}.npz",
        ).build()
        hist_idx = EmbeddingIndex.preindex_history(history)
    logger.info(f"Stage-1 index ready in {perf_counter()-t0:.1f}s")

    sampled_train = score_stage1(args.ranker, index, sampled_train, history, hist_idx, "stage-1 train")
    sampled_val   = score_stage1(args.ranker, index, sampled_val,   history, hist_idx, "stage-1 val")

    # Popularity from click history, windowed to strictly before the validation
    # period so val metrics stay honest (Q1.4). At serving the equivalent window
    # is the test file's own history, which is entirely in the past by construction.
    logger.info("Computing article popularity percentiles from click history...")
    val_start = pd.to_datetime(val_df[COL_IMPRESSION_TIME]).min()
    article_popularity = popularity_percentile(history, cutoff=val_start)

    extractor = BehaviouralFeatureExtractor(history, articles, article_popularity=article_popularity)

    logger.info("Extracting features for training set...")
    t0 = perf_counter()
    train_features_df = extractor.extract_features(sampled_train)
    logger.info(f"Done in {perf_counter() - t0:.1f}s")

    logger.info("Extracting features for validation set...")
    t0 = perf_counter()
    val_features_df = extractor.extract_features(sampled_val)
    logger.info(f"Done in {perf_counter() - t0:.1f}s")

    # Guard the bug this script used to have: a stage-1 score that is constant
    # across training rows teaches the ranker nothing and silently no-ops.
    train_var = train_features_df[score_col].var()
    if not train_var or train_var < 1e-12:
        logger.error(
            f"{score_col} has ~zero variance in the training set (var={train_var}). "
            "The reranker cannot learn from it. Aborting."
        )
        sys.exit(1)
    logger.info(f"{score_col} train variance = {train_var:.6f}  (non-degenerate)")

    # ── Q3: baseline = behavioural only; improved = + stage-1 semantic score ──
    behavioural = [
        "user_click_count", "user_hist_recency_sum",
        "session_clicks_1h", "session_clicks_24h",
        "category_match", "cat_affinity", "freshness_days", "article_popularity",
        "hist_mean_read_time", "hist_mean_scroll",   # EB-NeRD only
    ]
    baseline_features = [f for f in behavioural if f in train_features_df.columns]

    improved_features = baseline_features + [score_col]
    if args.ranker == "emb" and "emb_max_sim" in train_features_df.columns:
        improved_features.append("emb_max_sim")

    # Q9: features unavailable at serving time for THIS dataset are dropped from
    # the model that actually ships.
    unsafe = SERVING_UNSAFE.get(dataset_name, [])
    serving_features = [f for f in improved_features if f not in unsafe]

    logger.info(f"Baseline features ({len(baseline_features)}): {baseline_features}")
    logger.info(f"Improved features ({len(improved_features)}): {improved_features}")
    logger.info(f"Serving-safe features ({len(serving_features)}): {serving_features}")

    logger.info("Training BASELINE model (behavioural only)...")
    baseline_model = LightGBMReranker(feature_cols=baseline_features)
    baseline_model.train(train_features_df, val_features_df)
    val_features_df["baseline_score"] = baseline_model.predict(val_features_df)

    logger.info("Training IMPROVED model (+ stage-1 semantic score)...")
    improved_model = LightGBMReranker(feature_cols=improved_features)
    improved_model.train(train_features_df, val_features_df)
    val_features_df["improved_score"] = improved_model.predict(val_features_df)

    logger.info("Training SERVING-SAFE model (Q9: drops features absent at serving)...")
    serving_model = LightGBMReranker(feature_cols=serving_features)
    serving_model.train(train_features_df, val_features_df)
    val_features_df["serving_score"] = serving_model.predict(val_features_df)

    # ── Save models + their feature lists ─────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for label, model in [("improved", improved_model), ("serving", serving_model)]:
        stem = f"lgbm_{dataset_name.lower()}_{args.ranker}"
        suffix = "_model" if label == "improved" else "_serving_model"
        model_path = MODELS_DIR / f"{stem}{suffix}.txt"
        model.model.save_model(str(model_path))
        # The inference script reads this to build its feature matrix in the
        # exact order the model was trained with.
        model_path.with_suffix(".features.json").write_text(
            json.dumps(model.feature_cols, indent=2)
        )
        logger.info(f"{label} model saved → {model_path}")

    # Q2 "before re-ranking" = the stage-1 ranking, on exactly the same sampled
    # impressions as the reranked runs so the paired bootstrap is valid.
    val_features_df["stage1_score"] = val_features_df[score_col]

    for prefix in ["stage1", "baseline", "improved", "serving"]:
        val_features_df[f"{prefix}_rank"] = (
            val_features_df.groupby(COL_IMPRESSION_ID)[f"{prefix}_score"]
            .rank(ascending=False, method="first").astype(int)
        )

    pred_dir = PRED_DIR / dataset_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    for prefix in ["stage1", "baseline", "improved", "serving"]:
        out_path = pred_dir / f"{prefix}_val_predictions.parquet"
        val_features_df.to_parquet(out_path, index=False)
        logger.info(f"Saved {prefix} predictions to {out_path}")

    # Quick in-run AUC so the ablation is visible without a second command.
    from sklearn.metrics import roc_auc_score
    y = val_features_df[COL_LABEL]
    logger.info("=" * 60)
    logger.info("Validation AUC (flat, all candidates pooled)")
    for prefix in ["stage1", "baseline", "improved", "serving"]:
        logger.info(f"  {prefix:<9} = {roc_auc_score(y, val_features_df[f'{prefix}_score']):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
