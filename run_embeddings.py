"""
run_embeddings.py — Q3: Semantic Embedding Baseline

What it does
------------
For each dataset (MIND, EBNERD_DEMO):
  1. Loads processed articles, history, impressions from data/processed/
  2. Builds (or loads cached) embedding index:
       MIND       → SentenceTransformer('all-MiniLM-L6-v2')
       EBNERD     → Word2Vec document vectors from Ekstra_Bladet_word2vec.zip
  3. Computes recall@K (K=50,100,200) on a sample of val impressions
     via full-corpus cosine similarity retrieval
  4. Reranks all val impression candidates (per-impression cosine scores)
     and saves emb_val_predictions.parquet for Q4 evaluation
  5. Prints a results table and saves results/emb_recall_at_k.csv

Usage
-----
  python run_embeddings.py                        # both datasets
  python run_embeddings.py --dataset mind
  python run_embeddings.py --dataset ebnerd
  python run_embeddings.py --no-cache             # rebuild embeddings even if cached
  python run_embeddings.py --recall-sample 500
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
    COL_IMPRESSION_ID,
    COL_IMPRESSION_TIME,
    COL_LABEL,
    COL_SPLIT,
    COL_USER_ID,
    SPLIT_VAL,
    SPLIT_TEST,
)
from src.features.embedding_index import EmbeddingIndex

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("embeddings")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PRED_DIR      = BASE_DIR / "data" / "predictions"
CACHE_DIR     = BASE_DIR / "data" / "cache"
RESULTS_DIR   = BASE_DIR / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Recall@K
# ─────────────────────────────────────────────────────────────────────────────

def compute_recall_at_k(
    index: EmbeddingIndex,
    impressions_val: pd.DataFrame,
    history: pd.DataFrame,
    ks: list = [50, 100, 200],
    n_sample: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Sample n_sample val impressions; for each do full-corpus embedding retrieval
    and compute recall@K.

    recall@K per impression = |clicked ∩ top-K| / |clicked|
    Final metric = mean over impressions with at least one positive.
    """
    unique_imp = (
        impressions_val
        .drop_duplicates(subset=[COL_IMPRESSION_ID])
        [[COL_IMPRESSION_ID, COL_USER_ID, COL_IMPRESSION_TIME]]
        .reset_index(drop=True)
    )

    if len(unique_imp) > n_sample:
        unique_imp = unique_imp.sample(n_sample, random_state=seed).reset_index(drop=True)

    logger.info(f"Pre-indexing history by user …")
    hist_idx = EmbeddingIndex.preindex_history(history)

    # Clicked articles per impression
    clicked_by_imp = (
        impressions_val[impressions_val[COL_LABEL] == 1]
        .groupby(COL_IMPRESSION_ID)[COL_ARTICLE_ID]
        .apply(set)
        .to_dict()
    )

    max_k = max(ks)
    recall_lists = {k: [] for k in ks}

    logger.info(f"Computing recall@K on {len(unique_imp):,} sampled impressions …")
    for _, imp_row in tqdm(unique_imp.iterrows(), total=len(unique_imp), desc="recall@K"):
        imp_id = imp_row[COL_IMPRESSION_ID]
        uid    = imp_row[COL_USER_ID]
        t      = imp_row[COL_IMPRESSION_TIME]

        clicked_set = clicked_by_imp.get(imp_id, set())
        if not clicked_set:
            continue

        user_vec = index.make_user_vector(uid, hist_idx, t)
        topk_ids = index.retrieve_topk(user_vec, k=max_k)
        topk_set_all = topk_ids  # list, ordered

        for k in ks:
            topk_set = set(topk_set_all[:k])
            r = len(clicked_set & topk_set) / len(clicked_set)
            recall_lists[k].append(r)

    return {k: float(np.mean(v)) if v else 0.0 for k, v in recall_lists.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Per-impression reranking  (for Q4)
# ─────────────────────────────────────────────────────────────────────────────

def rerank_impressions(
    index: EmbeddingIndex,
    impressions_val: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each val impression, score its candidate articles by cosine similarity
    and assign a rank. Returns impressions_val with emb_score and emb_rank added.
    """
    logger.info("Pre-indexing history by user for reranking …")
    hist_idx = EmbeddingIndex.preindex_history(history)

    grouped = list(impressions_val.groupby(COL_IMPRESSION_ID, sort=False))
    logger.info(f"Reranking {len(grouped):,} impressions …")

    results = []
    for imp_id, group in tqdm(grouped, desc="reranking"):
        uid = group[COL_USER_ID].iloc[0]
        t   = group[COL_IMPRESSION_TIME].iloc[0]

        user_vec      = index.make_user_vector(uid, hist_idx, t)
        candidate_ids = group[COL_ARTICLE_ID].tolist()
        scores        = index.score_candidates(user_vec, candidate_ids)

        grp = group.copy()
        grp["emb_score"] = grp[COL_ARTICLE_ID].map(scores)
        grp["emb_rank"]  = grp["emb_score"].rank(ascending=False, method="first").astype(int)
        results.append(grp)

    return pd.concat(results, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_embeddings_for_dataset(
    dataset_name: str,
    recall_sample: int,
    use_cache: bool,
) -> dict:
    """Run the full Q3 embedding pipeline for one processed dataset."""
    logger.info("=" * 60)
    logger.info(f"Embedding pipeline  │  dataset={dataset_name}")
    logger.info("=" * 60)

    proc_dir = PROCESSED_DIR / dataset_name
    if not proc_dir.exists():
        raise FileNotFoundError(
            f"Processed data not found at {proc_dir}. Run build_pipeline.py first."
        )

    # ── Load data ──────────────────────────────────────────────────────
    logger.info("Loading processed data …")
    articles    = pd.read_parquet(proc_dir / "articles.parquet")
    history     = pd.read_parquet(proc_dir / "history.parquet")
    impressions = pd.read_parquet(proc_dir / "impressions.parquet")

    def _strip_tz(s):
        s = pd.to_datetime(s, errors="coerce")
        if hasattr(s.dtype, "tz") and s.dtype.tz is not None:
            return s.dt.tz_localize(None)
        return s

    history["click_time"]          = _strip_tz(history["click_time"])
    impressions["impression_time"] = _strip_tz(impressions["impression_time"])

    val_imp = impressions[impressions[COL_SPLIT] == SPLIT_VAL].copy()
    test_imp = impressions[impressions[COL_SPLIT] == SPLIT_TEST].copy()
    logger.info(
        f"Loaded: {len(articles):,} articles | "
        f"{len(history):,} history | "
        f"{len(val_imp):,} val impressions | "
        f"{len(test_imp):,} test impressions"
    )

    # ── Build embedding index ──────────────────────────────────────────
    cache_path = CACHE_DIR / f"emb_{dataset_name}.npz"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()
    index = EmbeddingIndex(
        articles_df=articles,
        dataset_name=dataset_name,
        zip_dir=BASE_DIR,
        cache_path=cache_path if use_cache else None,
    ).build()
    logger.info(f"Embedding index ready in {perf_counter()-t0:.1f}s  (dim={index.dim})")

    # ── Recall@K ──────────────────────────────────────────────────────
    t0 = perf_counter()
    recall = compute_recall_at_k(
        index, val_imp, history,
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
        ranked_df = rerank_impressions(index, imp_df, history)
        logger.info(f"Reranking done in {perf_counter()-t0:.1f}s")
    
        pred_dir = PRED_DIR / dataset_name
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / f"emb_{split_name}_predictions.parquet"
        ranked_df.to_parquet(pred_path, index=False)
        logger.info(f"Predictions saved → {pred_path}  ({pred_path.stat().st_size/1e6:.1f} MB)")

    return {"dataset": dataset_name, **{f"recall@{k}": v for k, v in recall.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Q3: Semantic embedding baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["mind", "ebnerd", "both", "mind_large_test"],
        default="both",
        help="Which dataset to run.",
    )
    parser.add_argument(
        "--recall-sample", type=int, default=500,
        help="Number of val impressions to use for recall@K.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Rebuild embedding index even if a cached .npz exists.",
    )
    args = parser.parse_args()

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
            res = run_embeddings_for_dataset(
                name,
                recall_sample=args.recall_sample,
                use_cache=not args.no_cache,
            )
            all_results.append(res)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

    if all_results:
        results_df = pd.DataFrame(all_results)
        logger.info("\n" + "=" * 60)
        logger.info("Q3 EMBEDDING RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info("\n" + results_df.to_string(index=False))

        out_path = RESULTS_DIR / "emb_recall_at_k.csv"
        results_df.to_csv(out_path, index=False)
        logger.info(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
