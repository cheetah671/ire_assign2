"""
format_mind.py — Q5: Generate MIND Codabench submission file

MIND submission format (prediction.txt):
  One line per impression:
    <impression_id> [rank_1,rank_2,...]
  Ranks are 1-indexed positions assigned by score descending (rank 1 = most recommended).

  APPROACH: Submit ALL 73,152 dev impressions in behaviors.tsv original order with
  real scores (val + test combined). This is robust regardless of how the instructor's
  truth.txt was built (whether it contains all impressions or just a subset).

Usage
-----
  from src.submission.format_mind import write_mind_predictions
  write_mind_predictions(predictions_df, output_path, behaviors_tsv_path, score_col,
                         extra_predictions_df=None)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

IMPRESSION_COL = "impression_id"
ARTICLE_COL    = "article_id"
SCORE_COL_BM25 = "bm25_score"
SCORE_COL_EMB  = "emb_score"


def write_mind_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
    behaviors_tsv_path: Path,
    score_col: str = "bm25_score",
    extra_predictions: pd.DataFrame = None,
) -> None:
    """
    Write MIND-format prediction.txt from a predictions DataFrame.

    Outputs ALL impressions from behaviors_tsv_path in their original file order.
    If extra_predictions is provided, it is merged so that ALL dev impressions
    (both val and test splits) get real rankings.

    Parameters
    ----------
    predictions        : DataFrame with impression_id, article_id, score_col columns (test split)
    output_path        : path to write prediction.txt
    behaviors_tsv_path : path to the dev behaviors.tsv
    score_col          : column to rank by (bm25_score or emb_score)
    extra_predictions  : Optional extra predictions DataFrame (val split) to merge in
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Merge val + test predictions if provided ──────────────────────────────
    if extra_predictions is not None:
        all_preds = pd.concat([predictions, extra_predictions], ignore_index=True)
    else:
        all_preds = predictions.copy()

    # ── Compute per-impression ranks ───────────────────────────────────────────
    all_preds["_rank"] = (
        all_preds.groupby(IMPRESSION_COL, sort=False)[score_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Build a lookup: impression_id → ordered list of rank strings
    ranked_by_imp = {}
    for imp_id, grp in all_preds.groupby(IMPRESSION_COL, sort=False):
        ranked_by_imp[str(imp_id)] = grp["_rank"].astype(str).tolist()

    logger.info(
        f"Writing MIND predictions: "
        f"{len(ranked_by_imp):,} impressions scored → {output_path}"
    )

    # ── Read behaviors.tsv for canonical impression ordering ──────────────────
    beh = pd.read_csv(
        behaviors_tsv_path, sep="\t", header=None,
        names=["imp_id", "user", "time", "hist", "candidates"],
        dtype=str,
    )
    beh["candidates"] = beh["candidates"].fillna("")

    lines = []
    scored_count = 0
    dummy_count = 0

    for _, row in beh.iterrows():
        imp_id = str(row["imp_id"])
        candidates = row["candidates"].split() if row["candidates"] else []
        n = len(candidates)

        if imp_id in ranked_by_imp:
            rank_list = "[" + ",".join(ranked_by_imp[imp_id]) + "]"
            scored_count += 1
        else:
            # Should not happen when val+test are both provided
            rank_list = "[" + ",".join(str(i) for i in range(1, n + 1)) + "]"
            dummy_count += 1

        lines.append(f"{imp_id} {rank_list}")

    logger.info(
        f"  Total lines: {len(lines):,}  "
        f"(scored={scored_count:,}, dummy={dummy_count:,})"
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    size_kb = output_path.stat().st_size / 1024
    logger.info(f"  Written {len(lines):,} lines  ({size_kb:.0f} KB)")
