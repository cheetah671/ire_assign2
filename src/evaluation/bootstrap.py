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

def paired_bootstrap_ci(
    preds_baseline: pd.DataFrame,
    preds_improved: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], Dict[str, float]],
    n_iterations: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    user_col: str = "user_id",
) -> Dict[str, Tuple[float, float]]:
    """
    Paired Bootstrap confidence intervals for the difference between improved and baseline models.
    Returns dict {metric_name: (lower_bound, upper_bound)} representing the 95% CI of the delta.
    """
    rng   = np.random.default_rng(seed)
    users = np.intersect1d(preds_baseline[user_col].unique(), preds_improved[user_col].unique())
    n     = len(users)

    logger.info(f"Paired Bootstrap: {n_iterations} iterations over {n:,} shared users …")

    base_idx: Dict[str, np.ndarray] = {}
    for u, grp in preds_baseline.groupby(user_col, sort=False):
        if u in users: base_idx[u] = grp.index.values
        
    imp_idx: Dict[str, np.ndarray] = {}
    for u, grp in preds_improved.groupby(user_col, sort=False):
        if u in users: imp_idx[u] = grp.index.values

    bootstrap_deltas: Dict[str, list] = {}

    for i in range(n_iterations):
        sampled_users = rng.choice(users, size=n, replace=True)
        
        base_rows = np.concatenate([base_idx[u] for u in sampled_users])
        base_sample = preds_baseline.loc[base_rows].reset_index(drop=True)
        
        imp_rows = np.concatenate([imp_idx[u] for u in sampled_users])
        imp_sample = preds_improved.loc[imp_rows].reset_index(drop=True)

        try:
            m_base = metric_fn(base_sample)
            m_imp = metric_fn(imp_sample)
        except Exception as e:
            continue

        for k in m_base.keys():
            if k in m_imp:
                bootstrap_deltas.setdefault(k, []).append(m_imp[k] - m_base[k])

    alpha = (1.0 - ci_level) / 2.0
    lo_pct = alpha * 100
    hi_pct = (1.0 - alpha) * 100

    ci = {}
    for k, vals in bootstrap_deltas.items():
        arr = np.array(vals)
        ci[k] = (float(np.percentile(arr, lo_pct)),
                 float(np.percentile(arr, hi_pct)))

    return ci
