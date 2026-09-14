"""
run_reranker.py — Q2 & Q3: Re-Ranker and Baseline Beaten

Orchestrator to:
1. Load stage-1 predictions (BM25 or Embeddings).
2. Extract behavioural features.
3. Train LightGBM ranker on train impressions.
4. Score and evaluate on val impressions.
5. Perform an ablation study.
"""

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.features.behavioural_features import BehaviouralFeatureExtractor
from src.ranking.reranker import LightGBMReranker
from src.data.schema import (
    COL_ARTICLE_ID, COL_IMPRESSION_ID, COL_LABEL, COL_SPLIT,
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

def load_data(dataset_name: str, ranker_type: str):
    proc_dir = PROCESSED_DIR / dataset_name
    if not proc_dir.exists():
        logger.error(f"Missing data for {dataset_name}. Expected at {proc_dir}")
        return None, None, None, None

    logger.info("Loading processed tables...")
    articles = pd.read_parquet(proc_dir / "articles.parquet")
    history = pd.read_parquet(proc_dir / "history.parquet")
    impressions = pd.read_parquet(proc_dir / "impressions.parquet")

    # Load stage-1 predictions (for val split)
    val_pred_path = PRED_DIR / dataset_name / f"{ranker_type}_val_predictions.parquet"
    if val_pred_path.exists():
        val_preds = pd.read_parquet(val_pred_path)
    else:
        logger.warning(f"Validation predictions not found at {val_pred_path}. Falling back to impressions.")
        val_preds = impressions[impressions[COL_SPLIT] == SPLIT_VAL].copy()

    # Train data: We can just use impressions directly for candidate generation in stage 2 training
    # For a real pipeline we'd use negative sampling, but here we can just use the provided candidates in impressions
    train_preds = impressions[impressions[COL_SPLIT] == SPLIT_TRAIN].copy()
    
    # Merge stage-1 scores if present
    if f"{ranker_type}_score" in val_preds.columns:
        train_preds[f"{ranker_type}_score"] = 0.0 # dummy for train if not scored
    
    return articles, history, train_preds, val_preds

def main():
    parser = argparse.ArgumentParser(description="Q2 & Q3: Re-Ranker and Baseline")
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], default="ebnerd")
    parser.add_argument("--ranker", choices=["bm25", "emb"], default="bm25")
    args = parser.parse_args()

    dataset_map = {"mind": "MIND", "ebnerd": "EBNERD_DEMO"}
    dataset_name = dataset_map[args.dataset]
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    articles, history, train_df, val_df = load_data(dataset_name, args.ranker)
    if articles is None:
        return

    # To avoid memory explosion, sample training impressions
    sampled_train = train_df[train_df["impression_id"].isin(
        train_df["impression_id"].drop_duplicates().sample(2000, random_state=42)
    )].copy()
    
    sampled_val = val_df[val_df["impression_id"].isin(
        val_df["impression_id"].drop_duplicates().sample(500, random_state=42)
    )].copy()

    # Compute article popularity from training impressions (clicked only)
    logger.info("Computing article popularity from train set...")
    article_popularity = train_df[train_df["label"] == 1].groupby("article_id").size()
    
    extractor = BehaviouralFeatureExtractor(history, articles, article_popularity=article_popularity)
    
    logger.info("Extracting features for training set...")
    t0 = perf_counter()
    train_features_df = extractor.extract_features(sampled_train)
    logger.info(f"Done in {perf_counter() - t0:.1f}s")
    
    logger.info("Extracting features for validation set...")
    t0 = perf_counter()
    val_features_df = extractor.extract_features(sampled_val)
    logger.info(f"Done in {perf_counter() - t0:.1f}s")

    # Q3: Baseline (only basic features)
    baseline_features = ["user_click_count", "user_hist_recency_sum"]
    if f"{args.ranker}_score" in val_features_df.columns:
        baseline_features.append(f"{args.ranker}_score")
        
    logger.info("Training BASELINE model...")
    baseline_model = LightGBMReranker(feature_cols=baseline_features)
    baseline_model.train(train_features_df, val_features_df)
    
    val_features_df["baseline_score"] = baseline_model.predict(val_features_df)
    
    # Q3: Improved (with article freshness and category match, plus popularity and session features)
    improved_features = baseline_features + ["category_match", "freshness_days", "article_popularity", "session_clicks_1h"]
    logger.info("Training IMPROVED model (Ablation)...")
    improved_model = LightGBMReranker(feature_cols=improved_features)
    improved_model.train(train_features_df, val_features_df)
    
    val_features_df["improved_score"] = improved_model.predict(val_features_df)
    
    # Calculate ranks
    for prefix in ["baseline", "improved"]:
        val_features_df[f"{prefix}_rank"] = val_features_df.groupby("impression_id")[f"{prefix}_score"].rank(ascending=False, method="first").astype(int)
    
    # Save predictions for run_evaluation.py
    pred_dir = PRED_DIR / dataset_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_out = val_features_df.copy()
    baseline_out["baseline_score"] = val_features_df["baseline_score"]
    baseline_out["baseline_rank"] = val_features_df["baseline_rank"]
    out_path_base = pred_dir / f"baseline_val_predictions.parquet"
    baseline_out.to_parquet(out_path_base, index=False)
    
    improved_out = val_features_df.copy()
    improved_out["improved_score"] = val_features_df["improved_score"]
    improved_out["improved_rank"] = val_features_df["improved_rank"]
    out_path_imp = pred_dir / f"improved_val_predictions.parquet"
    improved_out.to_parquet(out_path_imp, index=False)
    
    logger.info(f"Saved baseline predictions to {out_path_base}")
    logger.info(f"Saved improved predictions to {out_path_imp}")

if __name__ == "__main__":
    main()
