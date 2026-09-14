"""
slicing.py — Q4: User slicing for evaluation

Splits users into:
  - Cold-start users: ≤ COLD_THRESHOLD clicks in history
  - Warm users     : >  COLD_THRESHOLD clicks in history

The slicing is done on the TRAIN history so there is no leakage.
"""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

COLD_THRESHOLD = 5  # users with ≤ 5 train clicks are "cold"


def get_user_click_counts(history: pd.DataFrame, user_col: str = "user_id") -> pd.Series:
    """Return a Series: user_id → number of click rows in history."""
    return history.groupby(user_col).size()


def split_cold_warm(
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    threshold: int = COLD_THRESHOLD,
    user_col: str = "user_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split prediction rows into cold-start and warm subsets.

    Parameters
    ----------
    predictions : val-split predictions DataFrame
    history     : train history DataFrame (used to count clicks per user)
    threshold   : users with <= threshold clicks are cold-start
    user_col    : column name for user ID

    Returns
    -------
    (cold_preds, warm_preds) — two DataFrames, subsets of predictions
    """
    click_counts = get_user_click_counts(history, user_col)

    cold_users = set(click_counts[click_counts <= threshold].index)
    warm_users = set(click_counts[click_counts >  threshold].index)

    # Users with zero history at all are also cold-start
    all_pred_users = set(predictions[user_col].unique())
    no_history_users = all_pred_users - set(click_counts.index)
    cold_users = cold_users | no_history_users

    cold_preds = predictions[predictions[user_col].isin(cold_users)].copy()
    warm_preds = predictions[predictions[user_col].isin(warm_users)].copy()

    logger.info(
        f"Slicing: cold users (≤{threshold} clicks) = {len(cold_users):,} | "
        f"warm users (>{threshold} clicks) = {len(warm_users):,}"
    )
    logger.info(
        f"  Cold predictions: {len(cold_preds):,} rows | "
        f"Warm predictions: {len(warm_preds):,} rows"
    )

    return cold_preds, warm_preds
