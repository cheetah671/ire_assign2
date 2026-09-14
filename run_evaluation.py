"""
run_evaluation.py — Q4: Offline Evaluation Harness

Loads BM25 (and optionally embedding) val predictions and computes:
  • Ranking metrics: AUC, MRR, nDCG@5, nDCG@10
  • Beyond-accuracy : Novelty, Coverage, ILD@10
  • User slicing    : All users / Cold-start (≤5 clicks) / Warm (>5 clicks)
  • Bootstrap 95% CI on ranking metrics (N=1000 user resample)

Usage
-----
  python run_evaluation.py                        # both datasets, BM25
  python run_evaluation.py --dataset mind
  python run_evaluation.py --dataset ebnerd
  python run_evaluation.py --ranker bm25         # default
  python run_evaluation.py --no-bootstrap        # skip CIs (faster)
  python run_evaluation.py --bootstrap-n 200     # fewer iterations (faster)
"""

import argparse
import logging
import sys
from pathlib import Path
from functools import partial

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data.schema import (
    COL_ARTICLE_ID,
    COL_IMPRESSION_ID,
    COL_LABEL,
    COL_SPLIT,
    COL_USER_ID,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_TEST,
)
from src.evaluation.metrics import (
    compute_ranking_metrics,
    compute_beyond_accuracy_metrics,
)
from src.evaluation.slicing import split_cold_warm
from src.evaluation.bootstrap import bootstrap_ci

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PRED_DIR      = BASE_DIR / "data" / "predictions"
RESULTS_DIR   = BASE_DIR / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _print_metrics(label: str, metrics: dict, ci: dict = None) -> None:
    """Print a metrics dict with optional CI."""
    logger.info(f"\n  ── {label} ──")
    for k, v in metrics.items():
        ci_str = ""
        if ci and k in ci:
            lo, hi = ci[k]
            ci_str = f"  [95% CI: {lo:.4f} – {hi:.4f}]"
        logger.info(f"    {k:<12} = {_fmt(v)}{ci_str}")


def _results_row(dataset, ranker, slice_name, metrics, ci=None):
    row = {"dataset": dataset, "ranker": ranker, "slice": slice_name}
    for k, v in metrics.items():
        row[k] = round(v, 6)
    if ci:
        for k, (lo, hi) in ci.items():
            row[f"{k}_ci_lo"] = round(lo, 6)
            row[f"{k}_ci_hi"] = round(hi, 6)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset pipeline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(
    dataset_name: str,
    ranker: str,
    run_bootstrap: bool,
    bootstrap_n: int,
) -> list:
    """
    Full Q4 evaluation for one (dataset, ranker) pair.
    Returns a list of result-row dicts.
    """
    logger.info("=" * 60)
    logger.info(f"Evaluating  │  dataset={dataset_name}  │  ranker={ranker}")
    logger.info("=" * 60)

    # ── Load prediction file ───────────────────────────────────────────
    score_col = f"{ranker}_score"
    rank_col  = f"{ranker}_rank"
    pred_path = PRED_DIR / dataset_name / f"{ranker}_val_predictions.parquet"

    if not pred_path.exists():
        logger.error(f"Predictions not found: {pred_path}")
        logger.error(f"Run run_{ranker}.py first.")
        return []

    logger.info(f"Loading predictions from {pred_path} …")
    preds = pd.read_parquet(pred_path)
    logger.info(f"  {len(preds):,} rows, {preds[COL_IMPRESSION_ID].nunique():,} impressions")

    # ── Load processed tables ──────────────────────────────────────────
    proc_dir = PROCESSED_DIR / dataset_name
    articles    = pd.read_parquet(proc_dir / "articles.parquet")
    history_all = pd.read_parquet(proc_dir / "history.parquet")
    impressions = pd.read_parquet(proc_dir / "impressions.parquet")

    train_impressions = impressions[impressions[COL_SPLIT] == SPLIT_TRAIN].copy()

    logger.info(
        f"  Articles: {len(articles):,} | "
        f"History rows: {len(history_all):,} | "
        f"Train impressions: {len(train_impressions):,}"
    )

    # ── Ranking metrics helper (bound to this score/rank/label cols) ───
    def _ranking_metrics(df):
        return compute_ranking_metrics(
            df,
            score_col=score_col,
            rank_col=rank_col,
            impression_col=COL_IMPRESSION_ID,
            label_col=COL_LABEL,
        )

    # ── Beyond-accuracy metrics ────────────────────────────────────────
    logger.info("Computing beyond-accuracy metrics …")
    beyond = compute_beyond_accuracy_metrics(
        predictions=preds,
        articles=articles,
        impressions_train=train_impressions,
        top_k=10,
        score_col=score_col,
        impression_col=COL_IMPRESSION_ID,
        article_col=COL_ARTICLE_ID,
        label_col=COL_LABEL,
        category_col="category",
    )

    # ── All-users ranking metrics ──────────────────────────────────────
    logger.info("Computing ranking metrics (all users) …")
    all_metrics = _ranking_metrics(preds)
    all_metrics.update(beyond)
    _print_metrics("All users", all_metrics)

    all_ci = None
    if run_bootstrap:
        logger.info(f"Bootstrap CI (N={bootstrap_n}) for all users …")
        all_ci = bootstrap_ci(preds, _ranking_metrics,
                              n_iterations=bootstrap_n, seed=42)

    rows = [_results_row(dataset_name, ranker, "all", all_metrics, all_ci)]

    # ── Cold / Warm slicing ────────────────────────────────────────────
    logger.info("Slicing into cold-start vs warm users …")
    cold_preds, warm_preds = split_cold_warm(preds, history_all)

    for slice_name, slice_df in [("cold", cold_preds), ("warm", warm_preds)]:
        if len(slice_df) == 0:
            logger.warning(f"No predictions for {slice_name} slice — skipping.")
            continue

        logger.info(f"Computing ranking metrics ({slice_name} users) …")
        slice_metrics = _ranking_metrics(slice_df)

        # Beyond-accuracy for this slice
        beyond_slice = compute_beyond_accuracy_metrics(
            predictions=slice_df,
            articles=articles,
            impressions_train=train_impressions,
            top_k=10,
            score_col=score_col,
            impression_col=COL_IMPRESSION_ID,
            article_col=COL_ARTICLE_ID,
            label_col=COL_LABEL,
            category_col="category",
        )
        slice_metrics.update(beyond_slice)

        slice_ci = None
        if run_bootstrap:
            logger.info(f"Bootstrap CI ({slice_name} users, N={bootstrap_n}) …")
            slice_ci = bootstrap_ci(slice_df, _ranking_metrics,
                                    n_iterations=bootstrap_n, seed=42)

        _print_metrics(f"{slice_name.capitalize()} users", slice_metrics, slice_ci)
        rows.append(_results_row(dataset_name, ranker, slice_name,
                                 slice_metrics, slice_ci))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q4: Offline evaluation harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=["mind", "ebnerd", "both"], default="both",
        help="Which dataset to evaluate.",
    )
    parser.add_argument(
        "--ranker", choices=["bm25", "emb", "both"], default="bm25",
        help="Which ranker's predictions to evaluate.",
    )
    parser.add_argument(
        "--no-bootstrap", action="store_true",
        help="Skip bootstrap CIs (faster, good for quick iteration).",
    )
    parser.add_argument(
        "--bootstrap-n", type=int, default=1000,
        help="Number of bootstrap iterations.",
    )
    args = parser.parse_args()

    dataset_map = {
        "mind":   ["MIND"],
        "ebnerd": ["EBNERD_DEMO"],
        "both":   ["MIND", "EBNERD_DEMO"],
    }
    targets = dataset_map[args.dataset]

    ranker_map = {
        "bm25": ["bm25"],
        "emb":  ["emb"],
        "both": ["bm25", "emb"],
    }
    rankers = ranker_map[args.ranker]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for ranker in rankers:
        all_rows = []
        for name in targets:
            rows = evaluate_dataset(
                dataset_name=name,
                ranker=ranker,
                run_bootstrap=not args.no_bootstrap,
                bootstrap_n=args.bootstrap_n,
            )
            all_rows.extend(rows)

        if not all_rows:
            logger.error(f"No results produced for ranker={ranker}.")
            continue

        results_df = pd.DataFrame(all_rows)

        # ── Print summary table ──────────────────────────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info(f"Q4 EVALUATION RESULTS SUMMARY  [{ranker.upper()}]")
        logger.info("=" * 80)

        core_cols = ["dataset", "ranker", "slice", "auc", "mrr", "ndcg5", "ndcg10",
                     "novelty", "coverage", "ild"]
        display_cols = [c for c in core_cols if c in results_df.columns]
        logger.info("\n" + results_df[display_cols].to_string(index=False))

        # ── Save full results (including CI bounds) ────────────────────────────
        out_path = RESULTS_DIR / f"evaluation_{ranker}.csv"
        results_df.to_csv(out_path, index=False)
        logger.info(f"\nFull results saved → {out_path}")


if __name__ == "__main__":
    main()
