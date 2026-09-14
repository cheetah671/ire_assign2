"""
make_mindlarge_submission.py
============================
End-to-end pipeline that:
  1. Parses the MINDlarge_test raw TSV files  →  data/processed/MIND_LARGE_TEST/
  2. Builds a BM25 ranker over large-test articles
  3. Scores all test impressions (per-candidate reranking)
  4. Writes a Codabench-ready prediction.txt  →  submissions/MIND_LARGE_TEST/bm25/
  5. Zips the prediction.txt  →  mindlarge_test_bm25_submission.zip

Usage
-----
  python make_mindlarge_submission.py

All progress is shown with tqdm progress bars and % completion.
Intermediate results are cached so re-running is fast after the first pass.
"""

import logging
import sys
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Make src importable ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data.parse_mind_large import parse_mind_large_test
from src.data.schema import (
    COL_ARTICLE_ID,
    COL_CLICK_TIME,
    COL_IMPRESSION_ID,
    COL_IMPRESSION_TIME,
    COL_LABEL,
    COL_SPLIT,
    COL_USER_ID,
    SPLIT_TEST,
)
from src.ranking.bm25_ranker import BM25Ranker

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mindlarge_submission")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
RAW_TEST_DIR   = BASE_DIR / "data" / "raw" / "MIND" / "large_test" / "MINDlarge_test"
PROCESSED_DIR  = BASE_DIR / "data" / "processed" / "MIND_LARGE_TEST"
PRED_DIR       = BASE_DIR / "data" / "predictions" / "MIND_LARGE_TEST"
CACHE_DIR      = BASE_DIR / "data" / "cache"
SUBMIT_DIR     = BASE_DIR / "submissions" / "MIND_LARGE_TEST" / "bm25"
ZIP_OUT        = BASE_DIR / "mindlarge_test_bm25_submission.zip"

BEHAVIORS_TSV  = RAW_TEST_DIR / "behaviors.tsv"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Parse MINDlarge_test
# ─────────────────────────────────────────────────────────────────────────────

def step1_parse():
    """Parse MINDlarge_test raw TSVs → data/processed/MIND_LARGE_TEST/."""
    articles_p   = PROCESSED_DIR / "articles.parquet"
    history_p    = PROCESSED_DIR / "history.parquet"
    impressions_p = PROCESSED_DIR / "impressions.parquet"

    if articles_p.exists() and history_p.exists() and impressions_p.exists():
        logger.info("Step 1: Processed data already exists — loading from cache.")
        articles    = pd.read_parquet(articles_p)
        history     = pd.read_parquet(history_p)
        impressions = pd.read_parquet(impressions_p)
        logger.info(
            f"  Loaded: {len(articles):,} articles | "
            f"{len(history):,} history rows | "
            f"{len(impressions):,} impression rows"
        )
        return articles, history, impressions

    logger.info("=" * 60)
    logger.info("Step 1: Parsing MINDlarge_test raw files")
    logger.info("=" * 60)

    if not RAW_TEST_DIR.exists():
        raise FileNotFoundError(
            f"Raw test dir not found: {RAW_TEST_DIR}\n"
            "Please ensure MINDlarge_test.zip has been extracted to "
            "data/raw/MIND/large_test/MINDlarge_test/"
        )

    t0 = perf_counter()
    articles, history, impressions = parse_mind_large_test(RAW_TEST_DIR)
    logger.info(f"Parsing done in {perf_counter()-t0:.1f}s")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(articles_p, index=False)
    history.to_parquet(history_p, index=False)
    impressions.to_parquet(impressions_p, index=False)

    logger.info(
        f"Saved to {PROCESSED_DIR}\n"
        f"  articles:    {len(articles):>10,} rows  ({articles_p.stat().st_size/1e6:.1f} MB)\n"
        f"  history:     {len(history):>10,} rows  ({history_p.stat().st_size/1e6:.1f} MB)\n"
        f"  impressions: {len(impressions):>10,} rows  ({impressions_p.stat().st_size/1e6:.1f} MB)"
    )
    return articles, history, impressions


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Build BM25 index
# ─────────────────────────────────────────────────────────────────────────────

def step2_build_bm25(articles: pd.DataFrame) -> BM25Ranker:
    """Build (or load cached) BM25 index over MINDlarge_test articles."""
    logger.info("=" * 60)
    logger.info("Step 2: Building BM25 index")
    logger.info("=" * 60)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "bm25_MIND_LARGE_TEST.pkl"

    if cache_path.exists():
        logger.info(f"Loading cached BM25 index from {cache_path} …")
        ranker = BM25Ranker.load(cache_path)
        return ranker

    t0 = perf_counter()
    ranker = BM25Ranker(articles).build()
    logger.info(f"BM25 index built in {perf_counter()-t0:.1f}s")
    ranker.save(cache_path)
    return ranker


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Score test impressions
# ─────────────────────────────────────────────────────────────────────────────

def step3_score(
    ranker: BM25Ranker,
    impressions: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Score all test impression candidates with BM25."""
    logger.info("=" * 60)
    logger.info("Step 3: Scoring test impressions with BM25")
    logger.info("=" * 60)

    pred_path = PRED_DIR / "bm25_test_predictions.parquet"

    if pred_path.exists():
        logger.info(f"Predictions already exist — loading from {pred_path}")
        return pd.read_parquet(pred_path)

    # Ensure datetime types are tz-naive
    def _strip_tz(s):
        s = pd.to_datetime(s, errors="coerce")
        if hasattr(s.dtype, "tz") and s.dtype.tz is not None:
            return s.dt.tz_localize(None)
        return s

    history[COL_CLICK_TIME]         = _strip_tz(history[COL_CLICK_TIME])
    impressions[COL_IMPRESSION_TIME] = _strip_tz(impressions[COL_IMPRESSION_TIME])

    test_imp = impressions[impressions[COL_SPLIT] == SPLIT_TEST].copy()
    logger.info(f"Test impressions: {len(test_imp):,} rows across "
                f"{test_imp[COL_IMPRESSION_ID].nunique():,} unique impressions")

    logger.info("Pre-indexing history by user …")
    hist_idx = BM25Ranker.preindex_history(history)

    grouped = list(test_imp.groupby(COL_IMPRESSION_ID, sort=False))
    logger.info(f"Reranking {len(grouped):,} impressions …")

    results = []
    t0 = perf_counter()
    for imp_id, group in tqdm(grouped, desc="BM25 reranking", unit="imp",
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
        uid = group[COL_USER_ID].iloc[0]
        t   = group[COL_IMPRESSION_TIME].iloc[0]

        query_tokens  = ranker.make_query(uid, history, t, history_index=hist_idx)
        candidate_ids = group[COL_ARTICLE_ID].tolist()
        scores        = ranker.score_candidates(query_tokens, candidate_ids)

        grp = group.copy()
        grp["bm25_score"] = grp[COL_ARTICLE_ID].map(scores)
        grp["bm25_rank"]  = grp["bm25_score"].rank(ascending=False, method="first").astype(int)
        results.append(grp)

    ranked_df = pd.concat(results, ignore_index=True)
    logger.info(f"Scoring done in {perf_counter()-t0:.1f}s")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    ranked_df.to_parquet(pred_path, index=False)
    logger.info(f"Predictions saved → {pred_path}  ({pred_path.stat().st_size/1e6:.1f} MB)")
    return ranked_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Write prediction.txt + zip
# ─────────────────────────────────────────────────────────────────────────────

def step4_write_submission(ranked_df: pd.DataFrame) -> None:
    """
    Write Codabench MIND prediction.txt and zip it.

    MIND format:
      <impression_id> [rank_1,rank_2,...]
    Ranks are 1-indexed positions from score descending.
    Output order follows behaviors.tsv original row order.
    """
    logger.info("=" * 60)
    logger.info("Step 4: Writing Codabench submission")
    logger.info("=" * 60)

    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMIT_DIR / "prediction.txt"

    score_col = "bm25_score"

    # Compute per-impression ranks
    ranked_df["_rank"] = (
        ranked_df.groupby(COL_IMPRESSION_ID, sort=False)[score_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Build lookup: impression_id → ordered list of rank strings
    logger.info("Building rank lookup …")
    ranked_by_imp = {}
    for imp_id, grp in tqdm(ranked_df.groupby(COL_IMPRESSION_ID, sort=False),
                             desc="Building rank map",
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        ranked_by_imp[str(imp_id)] = grp["_rank"].astype(str).tolist()

    # Read behaviors.tsv for canonical impression ordering
    logger.info(f"Reading {BEHAVIORS_TSV} for output ordering …")
    beh = pd.read_csv(
        BEHAVIORS_TSV, sep="\t", header=None,
        names=["imp_id", "user", "time", "hist", "candidates"],
        dtype=str,
    )
    beh["candidates"] = beh["candidates"].fillna("")
    logger.info(f"  {len(beh):,} impressions in behaviors.tsv")

    lines = []
    scored_count = 0
    dummy_count  = 0

    logger.info("Writing prediction lines …")
    for _, row in tqdm(beh.iterrows(), total=len(beh), desc="Writing predictions",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
        imp_id     = str(row["imp_id"])
        candidates = row["candidates"].split() if row["candidates"] else []
        n          = len(candidates)

        if imp_id in ranked_by_imp:
            rank_list = "[" + ",".join(ranked_by_imp[imp_id]) + "]"
            scored_count += 1
        else:
            # Fallback: sequential ranks (should not happen)
            rank_list = "[" + ",".join(str(i) for i in range(1, n + 1)) + "]"
            dummy_count += 1

        lines.append(f"{imp_id} {rank_list}")

    logger.info(
        f"  Total lines: {len(lines):,}  "
        f"(scored={scored_count:,}, dummy={dummy_count:,})"
    )

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    size_kb = out_path.stat().st_size / 1024
    logger.info(f"  Written {len(lines):,} lines  ({size_kb:.0f} KB) → {out_path}")

    # Zip for Codabench upload
    logger.info(f"Creating zip → {ZIP_OUT} …")
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname="prediction.txt")
    logger.info(
        f"  ✓ ZIP created: {ZIP_OUT}  "
        f"({ZIP_OUT.stat().st_size/1024:.0f} KB)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("MINDlarge_test → Codabench Submission Pipeline")
    logger.info("=" * 60)
    t_total = perf_counter()

    # Step 1: Parse
    articles, history, impressions = step1_parse()

    # Step 2: BM25 index
    ranker = step2_build_bm25(articles)

    # Step 3: Score test impressions
    ranked_df = step3_score(ranker, impressions, history)

    # Step 4: Write submission + zip
    step4_write_submission(ranked_df)

    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {perf_counter()-t_total:.1f}s")
    logger.info(f"Submit this file to Codabench: {ZIP_OUT}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
