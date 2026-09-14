"""
format_ebnerd.py — Q5: Generate EB-NeRD Codabench submission file

EB-NeRD submission format (prediction.txt):
  One line per impression:
    <impression_id> [article_id_1,article_id_2,...]
  Articles are ordered by score descending.

This mirrors the MIND format but uses integer article_ids as used in EB-NeRD.

Usage
-----
  from src.submission.format_ebnerd import write_ebnerd_predictions
  write_ebnerd_predictions(predictions_df, output_path)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def write_ebnerd_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
    score_col: str = "bm25_score",
    impression_col: str = "impression_id",
    article_col: str = "article_id",
) -> None:
    """
    Write EB-NeRD-format prediction.txt from a predictions DataFrame.

    Parameters
    ----------
    predictions  : DataFrame with impression_id, article_id, score_col columns
    output_path  : path to write prediction.txt
    score_col    : column to rank by (bm25_score or emb_score)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranked = (
        predictions
        .sort_values([impression_col, score_col], ascending=[True, False])
    )

    logger.info(
        f"Writing EB-NeRD predictions: "
        f"{ranked[impression_col].nunique():,} impressions → {output_path}"
    )

    lines = []
    # groupby(sort=False) is CRITICAL here to preserve exact line-by-line order for the evaluation script
    for imp_id, grp in ranked.groupby(impression_col, sort=False):
        article_list = ",".join(grp[article_col].astype(str).tolist())
        lines.append(f"{imp_id} [{article_list}]")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    size_kb = output_path.stat().st_size / 1024
    logger.info(f"  Written {len(lines):,} lines  ({size_kb:.0f} KB)")
