"""
run_bm25.py -- Q2: BM25 Lexical Retrieval Baseline

What it does
------------
For each dataset (MIND, EBNERD_DEMO):
  1. Loads processed articles, history, impressions from data/processed/
  2. Builds (or loads cached) BM25 index over title + subtitle text
  3. Computes recall@K (K=50,100,200) on a sample of val impressions via
     full-corpus retrieval
  4. Reranks all val impression candidates for Q4 evaluation
     (saves impression_id / article_id / bm25_score / bm25_rank)
  5. Prints a results table

Usage
-----
  python run_bm25.py                        # both datasets, demo scale
  python run_bm25.py --dataset mind
  python run_bm25.py --dataset ebnerd
  python run_bm25.py --dataset both --recall-sample 2000
  python run_bm25.py --no-cache             # rebuild BM25 index even if cached
"""

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.data.schema import (
    COL_ARTICLE_ID,
    COL_CLICK_TIME,
    COL_IMPRESSION_ID,
    COL_IMPRESSION_TIME,
    COL_LABEL,
    COL_SPLIT,
    COL_USER_ID,
    SPLIT_VAL,
    SPLIT_TEST,
)
from src.ranking.bm25_ranker import BM25Ranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bm25")

BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PRED_DIR      = BASE_DIR / "data" / "predictions"
CACHE_DIR     = BASE_DIR / "data" / "cache"
RESULTS_DIR   = BASE_DIR / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Recall@K
# ─────────────────────────────────────────────────────────────────────────────

def compute_recall_at_k(
    ranker: BM25Ranker,
    impressions_val: pd.DataFrame,
    history: pd.DataFrame,
    ks: list = [50, 100, 200],
    n_sample: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Sample n_sample val impressions; for each do full-corpus BM25 retrieval
    and compute recall@K.

    recall@K per impression = |clicked ∩ top-K| / |clicked|
    Final metric = mean over all sampled impressions (with positives only).
    """
    # Deduplicate to one row per impression
    unique_imp = (
        impressions_val
        .drop_duplicates(subset=[COL_IMPRESSION_ID])
        [[COL_IMPRESSION_ID, COL_USER_ID, COL_IMPRESSION_TIME]]
        .reset_index(drop=True)
    )

    if len(unique_imp) > n_sample:
        unique_imp = unique_imp.sample(n_sample, random_state=seed).reset_index(drop=True)

    logger.info(f"Pre-indexing history by user …")
    hist_idx = BM25Ranker.preindex_history(history)

    logger.info(f"Computing recall@K on {len(unique_imp):,} sampled impressions …")

    max_k = max(ks)
    recall_lists = {k: [] for k in ks}

    # Build a lookup: impression_id -> set of clicked article_ids
    clicked_by_imp = (
        impressions_val[impressions_val[COL_LABEL] == 1]
        .groupby(COL_IMPRESSION_ID)[COL_ARTICLE_ID]
        .apply(set)
        .to_dict()
    )

    for _, imp_row in tqdm(unique_imp.iterrows(), total=len(unique_imp), desc="recall@K"):
        imp_id = imp_row[COL_IMPRESSION_ID]
        uid    = imp_row[COL_USER_ID]
        t      = imp_row[COL_IMPRESSION_TIME]

        clicked_set = clicked_by_imp.get(imp_id, set())
        if not clicked_set:
            continue

        # Build query using pre-indexed history (fast)
        query_tokens = ranker.make_query(uid, history, t, history_index=hist_idx)

        # Score all articles ONCE — slice for each K
        if query_tokens:
            scores   = ranker.bm25.get_scores(query_tokens)
            topk_idx = np.argsort(scores)[::-1]  # descending
        else:
            topk_idx = np.arange(len(ranker.article_ids))  # arbitrary order for cold-start

        article_ids_arr = np.array(ranker.article_ids)
        for k in ks:
            topk_set = set(article_ids_arr[topk_idx[:k]].tolist())
            r = len(clicked_set & topk_set) / len(clicked_set)
            recall_lists[k].append(r)

    return {k: float(np.mean(v)) if v else 0.0 for k, v in recall_lists.items()}



# ─────────────────────────────────────────────────────────────────────────────
# Per-impression reranking  (for Q4 / Q5)
# ─────────────────────────────────────────────────────────────────────────────

def rerank_impressions(
    ranker: BM25Ranker,
    impressions_val: pd.DataFrame,
    history: pd.DataFrame,
    max_impressions: int = None,
) -> pd.DataFrame:
    """
    For each val impression, score its candidate articles with BM25 and
    assign a rank.  Returns impressions_val with bm25_score and bm25_rank added.
    """
    logger.info("Pre-indexing history by user for reranking …")
    hist_idx = BM25Ranker.preindex_history(history)

    # Group candidates by impression
    grouped = list(impressions_val.groupby(COL_IMPRESSION_ID, sort=False))

    if max_impressions:
        grouped = grouped[:max_impressions]

    logger.info(f"Reranking {len(grouped):,} impressions …")

    results = []
    for imp_id, group in tqdm(grouped, desc="reranking"):
        uid = group[COL_USER_ID].iloc[0]
        t   = group[COL_IMPRESSION_TIME].iloc[0]

        query_tokens  = ranker.make_query(uid, history, t, history_index=hist_idx)
        candidate_ids = group[COL_ARTICLE_ID].tolist()
        scores        = ranker.score_candidates(query_tokens, candidate_ids)

        grp = group.copy()
        grp["bm25_score"] = grp[COL_ARTICLE_ID].map(scores)
        grp["bm25_rank"]  = grp["bm25_score"].rank(ascending=False, method="first").astype(int)
        results.append(grp)

    return pd.concat(results, ignore_index=True)



# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_bm25_for_dataset(
    dataset_name: str,
    recall_sample: int,
    use_cache: bool,
) -> dict:
    """Run the full Q2 BM25 pipeline for one processed dataset."""
    logger.info("=" * 60)
    logger.info(f"BM25 pipeline  │  dataset={dataset_name}")
    logger.info("=" * 60)

    proc_dir = PROCESSED_DIR / dataset_name
    if not proc_dir.exists():
        raise FileNotFoundError(
            f"Processed data not found at {proc_dir}. "
            "Run build_pipeline.py first."
        )

    # ── Load data ──────────────────────────────────────────────────────
    logger.info("Loading processed data …")
    articles    = pd.read_parquet(proc_dir / "articles.parquet")
    history     = pd.read_parquet(proc_dir / "history.parquet")
    impressions = pd.read_parquet(proc_dir / "impressions.parquet")

    # Ensure datetime types (tz-naive)
    def _strip_tz(s):
        s = pd.to_datetime(s, errors="coerce")
        if hasattr(s.dtype, "tz") and s.dtype.tz is not None:
            return s.dt.tz_localize(None)
        return s

    history["click_time"]         = _strip_tz(history["click_time"])
    impressions["impression_time"] = _strip_tz(impressions["impression_time"])

    val_imp = impressions[impressions[COL_SPLIT] == SPLIT_VAL].copy()
    test_imp = impressions[impressions[COL_SPLIT] == SPLIT_TEST].copy()
    logger.info(
        f"Loaded: {len(articles):,} articles | "
        f"{len(history):,} history | "
        f"{len(val_imp):,} val impressions | "
        f"{len(test_imp):,} test impressions"
    )

    # ── BM25 index ────────────────────────────────────────────────────
    cache_path = CACHE_DIR / f"bm25_{dataset_name}.pkl"

    if use_cache and cache_path.exists():
        logger.info(f"Loading cached BM25 index from {cache_path} …")
        ranker = BM25Ranker.load(cache_path)
    else:
        t0 = perf_counter()
        ranker = BM25Ranker(articles).build()
        logger.info(f"Index built in {perf_counter()-t0:.1f}s")
        ranker.save(cache_path)

    # ── Recall@K ──────────────────────────────────────────────────────
    t0 = perf_counter()
    recall = compute_recall_at_k(
        ranker, val_imp, history,
        ks=[50, 100, 200],
        n_sample=recall_sample,
    )
    recall_time = perf_counter() - t0
    logger.info(f"Recall@K ({recall_time:.1f}s):")
    for k, v in recall.items():
        logger.info(f"  recall@{k:<4} = {v:.4f}")

    # ── Rerank val and test impressions for Q4 / Q5 ───────────────────
    for imp_df, split_name in [(val_imp, "val"), (test_imp, "test")]:
        if len(imp_df) == 0:
            continue
        logger.info(f"Reranking {split_name} impression candidates …")
        t0 = perf_counter()
        ranked_df = rerank_impressions(ranker, imp_df, history)
        logger.info(f"Reranking done in {perf_counter()-t0:.1f}s")

        # Save predictions
        pred_dir = PRED_DIR / dataset_name
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / f"bm25_{split_name}_predictions.parquet"
        ranked_df.to_parquet(pred_path, index=False)
        logger.info(f"Predictions saved → {pred_path}  ({pred_path.stat().st_size/1e6:.1f} MB)")

    return {"dataset": dataset_name, **{f"recall@{k}": v for k, v in recall.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Q2: BM25 lexical retrieval baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["mind", "ebnerd", "both", "mind_large_test"],
        default="both",
        help="Which dataset to run.",
    )
    parser.add_argument(
        "--recall-sample",
        type=int,
        default=200,
        help="Number of val impressions to use for recall@K (full-corpus retrieval). "
             "Set lower for speed; 200 gives a reliable estimate.",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild BM25 index even if a cached pickle exists.",
    )
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_map = {
        "mind":   ["MIND"],
        "ebnerd": ["EBNERD_DEMO"],
        "both":   ["MIND", "EBNERD_DEMO"],
        "mind_large_test": ["MIND_LARGE_TEST"],
    }
    targets = dataset_map[args.dataset]

    all_results = []
    for name in targets:
        try:
            res = run_bm25_for_dataset(
                name,
                recall_sample=args.recall_sample,
                use_cache=not args.no_cache,
            )
            all_results.append(res)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

    # ── Summary table ─────────────────────────────────────────────────
    if all_results:
        results_df = pd.DataFrame(all_results)
        logger.info("\n" + "=" * 60)
        logger.info("Q2 BM25 RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info("\n" + results_df.to_string(index=False))

        out_path = RESULTS_DIR / "bm25_recall_at_k.csv"
        results_df.to_csv(out_path, index=False)
        logger.info(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
