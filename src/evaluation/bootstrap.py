"""
bootstrap.py — Q4: Bootstrap Confidence Intervals

Computes 95% bootstrap CIs by resampling users with replacement N times.

Usage
-----
    from src.evaluation.bootstrap import bootstrap_ci

    ci = bootstrap_ci(
        predictions=preds_df,
        metric_fn=compute_ranking_metrics,
        n_iterations=1000,
        ci_level=0.95,
        seed=42,
    )
    # ci = {"auc": (lower, upper), "mrr": (lower, upper), ...}
"""

import logging
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def bootstrap_ci(
    predictions: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], Dict[str, float]],
    n_iterations: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    user_col: str = "user_id",
) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap confidence intervals by resampling users with replacement.

    Parameters
    ----------
    predictions   : DataFrame with user_id, impression_id, article_id, label, scores
    metric_fn     : callable that takes a predictions DataFrame and returns
                    a dict {metric_name: float}
    n_iterations  : number of bootstrap samples (default 1000)
    ci_level      : confidence level (default 0.95 → 2.5th–97.5th percentiles)
    seed          : random seed for reproducibility
    user_col      : column name for user IDs

    Returns
    -------
    dict {metric_name: (lower_bound, upper_bound)}
    """
    rng   = np.random.default_rng(seed)
    users = predictions[user_col].unique()
    n     = len(users)

    logger.info(f"Bootstrap: {n_iterations} iterations over {n:,} users …")

    # Pre-build a user → row-index mapping for O(1) fast lookup
    user_to_idx: Dict[str, np.ndarray] = {}
    for u, grp in predictions.groupby(user_col, sort=False):
        user_to_idx[u] = grp.index.values

    bootstrap_scores: Dict[str, list] = {}

    for i in range(n_iterations):
        # Resample users with replacement → gather all their rows
        sampled_users = rng.choice(users, size=n, replace=True)
        row_indices   = np.concatenate([user_to_idx[u] for u in sampled_users])
        sample_df     = predictions.loc[row_indices].reset_index(drop=True)

        try:
            metrics = metric_fn(sample_df)
        except Exception as e:
            logger.debug(f"Bootstrap iteration {i} failed: {e}")
            continue

        for k, v in metrics.items():
            bootstrap_scores.setdefault(k, []).append(v)

    alpha = (1.0 - ci_level) / 2.0
    lo_pct = alpha * 100
    hi_pct = (1.0 - alpha) * 100

    ci = {}
    for k, vals in bootstrap_scores.items():
        arr = np.array(vals)
        ci[k] = (float(np.percentile(arr, lo_pct)),
                  float(np.percentile(arr, hi_pct)))

    return ci
