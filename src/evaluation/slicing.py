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

def split_head_tail(
    predictions: pd.DataFrame,
    train_impressions: pd.DataFrame,
    top_p: float = 0.2,
    article_col: str = "article_id",
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split prediction rows into head and tail subsets based on article popularity.
    """
    # Calculate popularity
    pop = train_impressions[train_impressions[label_col] == 1].groupby(article_col).size()
    
    # Identify head articles (top 20%)
    if len(pop) == 0:
        return pd.DataFrame(columns=predictions.columns), predictions.copy()
        
    threshold = pop.quantile(1.0 - top_p)
    head_articles = set(pop[pop >= threshold].index)
    
    head_preds = predictions[predictions[article_col].isin(head_articles)].copy()
    tail_preds = predictions[~predictions[article_col].isin(head_articles)].copy()
    
    logger.info(
        f"Slicing: head articles (top {top_p*100}%) = {len(head_articles):,} | "
        f"tail articles = {len(predictions[article_col].unique()) - len(head_articles):,}"
    )
    logger.info(
        f"  Head predictions: {len(head_preds):,} rows | "
        f"Tail predictions: {len(tail_preds):,} rows"
    )
    
    return head_preds, tail_preds
