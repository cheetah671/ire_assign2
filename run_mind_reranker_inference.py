"""
run_mind_reranker_inference.py
Two-Stage Retrieve-then-Rank Pipeline for MIND Large Test Set (Codabench)

Stage 1 - Retrieve:
    SentenceTransformer (all-MiniLM-L6-v2) encodes MIND large-test news.tsv.
    Per impression, cosine-similarity scores computed via recency-weighted
    mean-pool of the user's click history from behaviors.tsv.

Stage 2 - Rerank:
    LightGBM LambdaRank model trained on MIND small train impressions.
    Features: emb_score, user_click_count, category_match,
              freshness_days, article_popularity

Output:
    submissions/MIND_LARGE/reranker/prediction.txt
    mind_large_reranker_submission.zip   <-- upload to Codabench

Usage:
    python run_mind_reranker_inference.py              # full pipeline
    python run_mind_reranker_inference.py --skip-train # load saved model
"""

import argparse
import logging
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.data.parse_mind import _parse_news
from src.features.embedding_index import EmbeddingIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mind_reranker_inference")

LARGE_TEST_DIR = BASE_DIR / "data" / "raw" / "MIND" / "large_test" / "MINDlarge_test"
PROCESSED_MIND = BASE_DIR / "data" / "processed" / "MIND"
CACHE_DIR      = BASE_DIR / "data" / "cache"
MODEL_DIR      = BASE_DIR / "models"
SUBMIT_DIR     = BASE_DIR / "submissions" / "MIND_LARGE" / "reranker"
ZIP_OUT        = BASE_DIR / "mind_large_reranker_submission.zip"
EMB_CACHE      = CACHE_DIR / "emb_MIND_LARGE.npz"

FEATURE_COLS = [
    "emb_score",
    "user_click_count",
    "category_match",
    "freshness_days",
    "article_popularity",
]


def _zip_file(file_path: Path, zip_path: Path) -> None:
    logger.info(f"Zipping -> {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname=file_path.name)
    logger.info(f"  Zip ready: {zip_path.name}  ({zip_path.stat().st_size / 1e6:.1f} MB)")


def _build_embedding_index(news_df: pd.DataFrame) -> EmbeddingIndex:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    idx = EmbeddingIndex(articles_df=news_df, dataset_name="MIND", cache_path=EMB_CACHE)
    idx.build()
    return idx


def _build_article_metadata(news_df: pd.DataFrame) -> dict:
    logger.info("Building article metadata lookup ...")
    meta = {}
    for _, row in news_df.iterrows():
        aid = str(row["article_id"])
        pt  = row.get("published_time", None)
        if pt is not None and not pd.isna(pt):
            pt = pd.to_datetime(pt, errors="coerce")
            if hasattr(pt, "tzinfo") and pt.tzinfo is not None:
                pt = pt.tz_localize(None)
        else:
            pt = None
        meta[aid] = {
            "category":       str(row.get("category", "unknown") or "unknown"),
            "published_time": pt,
        }
    logger.info(f"  Metadata for {len(meta):,} articles.")
    return meta


def _build_article_popularity(processed_dir: Path) -> dict:
    logger.info("Computing article popularity from MIND small train ...")
    try:
        imps = pd.read_parquet(
            processed_dir / "impressions.parquet",
            filters=[("split", "==", "train"), ("label", "==", 1)],
            columns=["article_id"],
        )
        pop = imps["article_id"].value_counts().to_dict()
        logger.info(f"  Popularity for {len(pop):,} articles.")
        return pop
    except Exception as e:
        logger.warning(f"  Failed: {e} -- using zeros.")
        return {}


def _compute_emb_scores_for_df(
    impressions_df: pd.DataFrame,
    emb_idx: EmbeddingIndex,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    logger.info(
        f"  Computing embedding scores for "
        f"{impressions_df['impression_id'].nunique():,} impressions ..."
    )
    hist_idx = {}
    for uid, grp in history_df.groupby("user_id", sort=False):
        hist_idx[uid] = grp.sort_values("click_time", ascending=False)

    emb_scores = []
    for imp_id, grp in tqdm(
        impressions_df.groupby("impression_id", sort=False),
        desc="  Emb Scoring", leave=False
    ):
        uid        = str(grp["user_id"].iloc[0])
        imp_time   = pd.to_datetime(grp["impression_time"].iloc[0])
        candidates = grp["article_id"].tolist()

        user_hist = hist_idx.get(uid)
        if user_hist is not None and not user_hist.empty:
            past = user_hist[user_hist["click_time"] < imp_time]
        else:
            past = pd.DataFrame(columns=["click_time", "article_id"])

        vecs = []
        for aid in past["article_id"].tolist()[-50:]:
            idx = emb_idx.id_to_idx.get(str(aid))
            if idx is not None:
                vecs.append(emb_idx.matrix[idx])

        if vecs:
            n       = len(vecs)
            weights = np.linspace(0.5, 1.5, n, dtype=np.float32)
            u_vec   = np.average(vecs, axis=0, weights=weights).astype(np.float32)
            norm    = np.linalg.norm(u_vec)
            if norm > 0:
                u_vec /= norm
        else:
            u_vec = None

        scores = emb_idx.score_candidates(u_vec, candidates)
        for aid in candidates:
            emb_scores.append(scores.get(aid, 0.0))

    impressions_df = impressions_df.copy()
    impressions_df["emb_score"] = emb_scores
    return impressions_df


def _extract_behavioural_features(
    impressions_df: pd.DataFrame,
    history_df: pd.DataFrame,
    article_meta: dict,
    article_pop: dict,
) -> pd.DataFrame:
    hist_idx = {}
    for uid, grp in history_df.groupby("user_id", sort=False):
        hist_idx[uid] = grp.sort_values("click_time", ascending=False)

    rows = []
    for imp_id, grp in tqdm(
        impressions_df.groupby("impression_id", sort=False),
        desc="  Beh Features", leave=False
    ):
        uid      = str(grp["user_id"].iloc[0])
        imp_time = pd.to_datetime(grp["impression_time"].iloc[0])

        user_hist = hist_idx.get(uid)
        if user_hist is not None and not user_hist.empty:
            past = user_hist[user_hist["click_time"] < imp_time]
        else:
            past = pd.DataFrame(columns=["click_time", "article_id"])

        click_count = len(past)
        user_cats   = set()
        if click_count > 0:
            for aid in past["article_id"].values:
                m = article_meta.get(str(aid))
                if m:
                    user_cats.add(m["category"])

        for _, row in grp.iterrows():
            cand_id = str(row["article_id"])
            m       = article_meta.get(cand_id, {})
            cat_match = 1 if (m.get("category") in user_cats and user_cats) else 0
            pt = m.get("published_time")
            if pt is not None:
                freshness = max(0.0, (imp_time - pt).total_seconds() / 86400.0)
            else:
                freshness = -1.0
            rows.append({
                "impression_id":      imp_id,
                "article_id":         cand_id,
                "user_click_count":   click_count,
                "category_match":     cat_match,
                "freshness_days":     freshness,
                "article_popularity": article_pop.get(cand_id, 0),
            })

    feat_df = pd.DataFrame(rows)
    return pd.merge(impressions_df, feat_df, on=["impression_id", "article_id"], how="left")


def train_lgbm_model(
    emb_idx: EmbeddingIndex,
    article_meta: dict,
    article_pop: dict,
    processed_dir: Path,
    train_sample: int = 15000,
    val_sample: int = 1000,
):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm not installed -- run: pip install lightgbm")

    logger.info("=" * 60)
    logger.info("STAGE 2 -- Training LightGBM on MIND small ...")
    logger.info("=" * 60)

    history_df = pd.read_parquet(processed_dir / "history.parquet")
    logger.info(f"History loaded: {len(history_df):,} rows.")
    imps = pd.read_parquet(processed_dir / "impressions.parquet")

    train_imps  = imps[imps["split"] == "train"]
    sampled_ids = train_imps["impression_id"].drop_duplicates().sample(
        min(train_sample, train_imps["impression_id"].nunique()), random_state=42
    )
    train_df = train_imps[train_imps["impression_id"].isin(sampled_ids)].copy()

    val_imps        = imps[imps["split"] == "val"]
    sampled_val_ids = val_imps["impression_id"].drop_duplicates().sample(
        min(val_sample, val_imps["impression_id"].nunique()), random_state=42
    )
    val_df = val_imps[val_imps["impression_id"].isin(sampled_val_ids)].copy()

    logger.info(
        f"Train sample: {len(train_df):,} rows  "
        f"({train_df['impression_id'].nunique():,} impressions)"
    )
    logger.info(
        f"Val   sample: {len(val_df):,} rows  "
        f"({val_df['impression_id'].nunique():,} impressions)"
    )

    logger.info("Computing embedding scores for training data ...")
    train_df = _compute_emb_scores_for_df(train_df, emb_idx, history_df)
    val_df   = _compute_emb_scores_for_df(val_df,   emb_idx, history_df)

    logger.info("Extracting behavioural features ...")
    train_df = _extract_behavioural_features(train_df, history_df, article_meta, article_pop)
    val_df   = _extract_behavioural_features(val_df,   history_df, article_meta, article_pop)

    for col in FEATURE_COLS:
        if col not in train_df.columns:
            train_df[col] = 0.0
            val_df[col]   = 0.0
        train_df[col] = train_df[col].fillna(0)
        val_df[col]   = val_df[col].fillna(0)

    train_df = train_df.sort_values("impression_id")
    val_df   = val_df.sort_values("impression_id")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"].astype(int)
    g_train = train_df.groupby("impression_id").size().values

    X_val   = val_df[FEATURE_COLS]
    y_val   = val_df["label"].astype(int)
    g_val   = val_df.groupby("impression_id").size().values

    dtrain = lgb.Dataset(X_train, label=y_train, group=g_train)
    dval   = lgb.Dataset(X_val, label=y_val, group=g_val, reference=dtrain)

    params = {
        "objective":        "lambdarank",
        "metric":           "ndcg",
        "ndcg_eval_at":     [5, 10],
        "learning_rate":    0.05,
        "num_leaves":       63,
        "min_data_in_leaf": 20,
        "verbose":          -1,
        "random_state":     42,
    }

    logger.info(f"Training LightGBM | features: {FEATURE_COLS}")
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(25)],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "mind_large_lgbm_model.txt"
    booster.save_model(str(model_path))
    logger.info(f"  Model saved -> {model_path}")
    return booster


def run_inference(
    booster,
    emb_idx: EmbeddingIndex,
    article_meta: dict,
    article_pop: dict,
) -> None:
    behaviors_path = LARGE_TEST_DIR / "behaviors.tsv"
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMIT_DIR / "prediction.txt"

    logger.info("=" * 60)
    logger.info("STAGE 1+2 -- Streaming inference on MIND Large Test ...")
    logger.info(f"  Input : {behaviors_path}")
    logger.info(f"  Output: {out_path}")
    logger.info("=" * 60)

    logger.info("Counting total impressions ...")
    with open(behaviors_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    logger.info(f"  Total: {total_lines:,} impressions")

    t0 = perf_counter()

    with open(behaviors_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line in tqdm(
            fin, total=total_lines, desc="Reranking", unit="imp",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {percentage:.1f}%"
                       " [{elapsed}<{remaining}  {rate_fmt}]",
        ):
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue

            imp_id     = parts[0]
            time_str   = parts[2]
            history    = parts[3].split() if parts[3] else []
            candidates = parts[4].split() if parts[4] else []

            if not candidates:
                fout.write(f"{imp_id} []\n")
                continue

            try:
                imp_time = datetime.strptime(time_str.strip(), "%m/%d/%Y %I:%M:%S %p")
            except Exception:
                imp_time = None

            # Stage 1: embedding scoring
            history_trunc = history[-50:]
            vecs = []
            for aid in history_trunc:
                idx = emb_idx.id_to_idx.get(str(aid))
                if idx is not None:
                    vecs.append(emb_idx.matrix[idx])

            if vecs:
                n       = len(vecs)
                weights = np.linspace(0.5, 1.5, n, dtype=np.float32)
                u_vec   = np.average(vecs, axis=0, weights=weights).astype(np.float32)
                norm    = np.linalg.norm(u_vec)
                if norm > 0:
                    u_vec /= norm
            else:
                u_vec = None

            emb_scores = emb_idx.score_candidates(u_vec, candidates)

            # Stage 2: LightGBM rerank
            click_count = len(history_trunc)
            user_cats   = set()
            for aid in history_trunc:
                m = article_meta.get(str(aid))
                if m:
                    user_cats.add(m["category"])

            feat_rows = []
            for cand_raw in candidates:
                cand_id = str(cand_raw)
                m       = article_meta.get(cand_id, {})
                cat_match = 1 if (m.get("category") in user_cats and user_cats) else 0
                pt = m.get("published_time")
                if pt is not None and imp_time is not None:
                    freshness = max(0.0, (imp_time - pt).total_seconds() / 86400.0)
                else:
                    freshness = -1.0
                feat_rows.append([
                    emb_scores.get(cand_raw, 0.0),
                    click_count,
                    cat_match,
                    freshness,
                    article_pop.get(cand_id, 0),
                ])

            X           = np.array(feat_rows, dtype=np.float32)
            lgbm_scores = booster.predict(X)

            sorted_idx = np.argsort(lgbm_scores)[::-1]
            rank_map   = np.empty(len(candidates), dtype=int)
            for rank, orig_idx in enumerate(sorted_idx, start=1):
                rank_map[orig_idx] = rank

            fout.write(f"{imp_id} [{','.join(map(str, rank_map))}]\n")

    elapsed = perf_counter() - t0
    logger.info(
        f"  Done in {elapsed / 60:.1f} min  ({total_lines / elapsed:,.0f} imp/s)"
    )
    logger.info(f"  Written -> {out_path}")

    _zip_file(out_path, ZIP_OUT)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  SUBMISSION ZIP: {ZIP_OUT}")
    logger.info("  Upload this to Codabench MIND Large Test leaderboard.")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Two-Stage Retrieve-then-Rank for MIND Large Test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training and load model from models/mind_large_lgbm_model.txt",
    )
    args = parser.parse_args()

    for p in [LARGE_TEST_DIR / "news.tsv", LARGE_TEST_DIR / "behaviors.tsv"]:
        if not p.exists():
            logger.error(f"Missing required file: {p}")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Loading articles & building embedding index ...")
    logger.info("=" * 60)
    news_df      = _parse_news(LARGE_TEST_DIR / "news.tsv")
    logger.info(f"  {len(news_df):,} articles loaded.")
    emb_idx      = _build_embedding_index(news_df)
    article_meta = _build_article_metadata(news_df)
    article_pop  = _build_article_popularity(PROCESSED_MIND)

    model_path = MODEL_DIR / "mind_large_lgbm_model.txt"
    if args.skip_train and model_path.exists():
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed -- run: pip install lightgbm")
        logger.info(f"Loading saved model from {model_path} ...")
        booster = lgb.Booster(model_file=str(model_path))
        logger.info("  Model loaded.")
    else:
        if args.skip_train:
            logger.warning(
                f"--skip-train set but no saved model at {model_path}. Training ..."
            )
        booster = train_lgbm_model(emb_idx, article_meta, article_pop, PROCESSED_MIND)

    run_inference(booster, emb_idx, article_meta, article_pop)


if __name__ == "__main__":
    main()
