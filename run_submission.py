"""
run_submission.py — Q5: Generate Codabench submission files

Reads test predictions and writes submission files in competition format.

MIND format (prediction.txt):
  <impression_id> [rank_1,rank_2,...]
  ALL dev impressions are written in behaviors.tsv original order.
  Test impressions get real ranks; val impressions get dummy sequential ranks.

EB-NeRD format (prediction.txt):
  <impression_id> [article_id_1,article_id_2,...]

Usage
-----
  python run_submission.py                        # all datasets, both rankers
  python run_submission.py --dataset mind
  python run_submission.py --ranker bm25
  python run_submission.py --ranker emb
"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.submission.format_mind   import write_mind_predictions
from src.submission.format_ebnerd import write_ebnerd_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("submission")

BASE_DIR    = Path(__file__).parent
PRED_DIR    = BASE_DIR / "data" / "predictions"
SUBMIT_DIR  = BASE_DIR / "submissions"
RAW_DIR     = BASE_DIR / "data" / "raw"

SCORE_COL = {"bm25": "bm25_score", "emb": "emb_score"}

# Paths to behaviors.tsv files (used to establish the correct output ordering)
BEHAVIORS_TSV = {
    "MIND":        RAW_DIR / "MIND" / "dev" / "MINDsmall_dev" / "behaviors.tsv",
    "EBNERD_DEMO": RAW_DIR / "EBNERD" / "demo" / "validation" / "behaviors.parquet",
}


def generate_submission(dataset_name: str, ranker: str) -> None:
    """Generate submission file for one dataset + ranker combination."""
    score_col = SCORE_COL[ranker]
    pred_path = PRED_DIR / dataset_name / f"{ranker}_test_predictions.parquet"

    if not pred_path.exists():
        logger.warning(f"Predictions not found: {pred_path} — skipping.")
        return

    logger.info(f"Loading {pred_path.name} …")
    preds = pd.read_parquet(pred_path)
    logger.info(f"  {len(preds):,} rows, {preds['impression_id'].nunique():,} impressions")

    out_dir  = SUBMIT_DIR / dataset_name / ranker
    out_path = out_dir / "prediction.txt"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "EBNERD" in dataset_name.upper():
        write_ebnerd_predictions(preds, out_path, score_col=score_col)
    else:
        # Also load val predictions so ALL dev impressions get real scores
        val_pred_path = PRED_DIR / dataset_name / f"{ranker}_val_predictions.parquet"
        extra_preds = None
        if val_pred_path.exists():
            logger.info(f"Loading {val_pred_path.name} (val split) …")
            extra_preds = pd.read_parquet(val_pred_path)
            logger.info(f"  {len(extra_preds):,} rows, {extra_preds['impression_id'].nunique():,} impressions")
        behaviors_path = BEHAVIORS_TSV[dataset_name]
        write_mind_predictions(
            preds,
            output_path=out_path,
            behaviors_tsv_path=behaviors_path,
            score_col=score_col,
            extra_predictions=extra_preds,
        )

    logger.info(f"  ✓ Saved → {out_path}")

    # ── Also create a ready-to-submit zip ─────────────────────────────────────
    zip_path = BASE_DIR / f"{dataset_name.lower()}_{ranker}_submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname="prediction.txt")
    logger.info(f"  ✓ Zip  → {zip_path}  ({zip_path.stat().st_size/1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Q5: Generate Codabench submission prediction.txt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=["mind", "ebnerd", "both", "mind_large_test"], default="both",
        help="Which dataset to process."
    )
    parser.add_argument(
        "--ranker", choices=["bm25", "emb", "both"], default="both",
    )
    args = parser.parse_args()

    datasets = {
        "mind": ["MIND"],
        "ebnerd": ["EBNERD_DEMO", "EBNERD_SMALL"],
        "both": ["MIND", "EBNERD_DEMO", "EBNERD_SMALL"],
        "mind_large_test": ["MIND_LARGE_TEST"],
    }[args.dataset]

    rankers = {
        "bm25": ["bm25"],
        "emb":  ["emb"],
        "both": ["bm25", "emb"],
    }[args.ranker]

    logger.info("=" * 60)
    logger.info("Q5: Generating Codabench submission files")
    logger.info("=" * 60)

    for dataset in datasets:
        for ranker in rankers:
            logger.info(f"\n── {dataset} / {ranker} ──")
            generate_submission(dataset, ranker)

    logger.info(f"\nAll submissions written to: {SUBMIT_DIR}/")
    logger.info("Submission zips:")
    for p in sorted(BASE_DIR.glob("*_submission.zip")):
        size_kb = p.stat().st_size / 1024
        logger.info(f"  {p.name}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
