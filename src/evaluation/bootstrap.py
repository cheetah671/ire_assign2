"""
bootstrap.py — Q4/Q5: Bootstrap Confidence Intervals

Computes 95% bootstrap CIs by resampling users with replacement.

Why impression ids are rewritten on every draw
----------------------------------------------
Resampling users with replacement means a user can be drawn several times, so
their rows appear several times in the resample. The ranking metrics group rows
by impression_id. If the duplicated rows keep their original impression_id, the
copies collapse back into a single oversized impression group — one impression
holding each candidate two or three times — instead of counting as separate
impressions.

MRR and nDCG are computed per impression group, so that collapse biases them
downward, and the resulting interval can sit entirely below the point estimate.
AUC is computed flat over all rows and is invariant to proportional duplication,
which is why only the ranking metrics looked wrong.

Each draw therefore gets its own replica id, and impression ids are made unique
per replica before the metric function sees them.

Usage
    from src.evaluation.bootstrap import bootstrap_ci, paired_bootstrap_ci
    ci = bootstrap_ci(predictions, metric_fn, n_iterations=1000)
"""

import logging
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _user_row_positions(df: pd.DataFrame, user_col: str) -> Dict[object, np.ndarray]:
    """{user: positional row indices} — positional so .iloc can be used."""
    return {u: idx for u, idx in df.groupby(user_col, sort=False).indices.items()}


def _resample(
    df: pd.DataFrame,
    pos_by_user: Dict[object, np.ndarray],
    sampled_users: np.ndarray,
    imp_codes: np.ndarray,
    n_imps: int,
    impression_col: str,
) -> pd.DataFrame:
    """Gather the sampled users' rows, giving each draw its own impression ids."""
    parts = [pos_by_user[u] for u in sampled_users]
    rows = np.concatenate(parts)
    replica = np.repeat(np.arange(len(parts), dtype=np.int64),
                        [len(p) for p in parts])

    out = df.iloc[rows].copy()
    # Unique per (original impression, replica) without string formatting.
    out[impression_col] = imp_codes[rows] + replica * n_imps
    return out


def bootstrap_ci(
    predictions: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], Dict[str, float]],
    n_iterations: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    user_col: str = "user_id",
    impression_col: str = "impression_id",
) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap confidence intervals by resampling users with replacement.

    Returns {metric_name: (lower_bound, upper_bound)}.
    """
    preds = predictions.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    users = preds[user_col].unique()
    n = len(users)

    logger.info(f"Bootstrap: {n_iterations} iterations over {n:,} users ...")

    pos_by_user = _user_row_positions(preds, user_col)
    imp_codes, uniq = pd.factorize(preds[impression_col])
    n_imps = len(uniq)

    bootstrap_scores: Dict[str, list] = {}

    for i in range(n_iterations):
        sampled_users = rng.choice(users, size=n, replace=True)
        sample_df = _resample(preds, pos_by_user, sampled_users,
                              imp_codes, n_imps, impression_col)

        try:
            metrics = metric_fn(sample_df)
        except Exception as e:
            logger.debug(f"Bootstrap iteration {i} failed: {e}")
            continue

        for k, v in metrics.items():
            bootstrap_scores.setdefault(k, []).append(v)

    alpha = (1.0 - ci_level) / 2.0
    return {
        k: (float(np.percentile(v, alpha * 100)),
            float(np.percentile(v, (1.0 - alpha) * 100)))
        for k, v in bootstrap_scores.items()
    }


def paired_bootstrap_ci(
    preds_baseline: pd.DataFrame,
    preds_improved: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], Dict[str, float]],
    n_iterations: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    user_col: str = "user_id",
    impression_col: str = "impression_id",
) -> Dict[str, Tuple[float, float]]:
    """
    Paired bootstrap CI for (improved - baseline).

    Both systems are resampled with the *same* user draw each iteration, so the
    difference is measured on identical users and the variance of the delta is
    not inflated by sampling the two sides independently.

    A CI that excludes zero is the significance claim A2 Q3.4 asks for.
    """
    base = preds_baseline.reset_index(drop=True)
    imp = preds_improved.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    users = np.intersect1d(base[user_col].unique(), imp[user_col].unique())
    n = len(users)

    logger.info(f"Paired bootstrap: {n_iterations} iterations over {n:,} shared users ...")

    base_pos = _user_row_positions(base, user_col)
    imp_pos = _user_row_positions(imp, user_col)

    base_codes, base_uniq = pd.factorize(base[impression_col])
    imp_codes, imp_uniq = pd.factorize(imp[impression_col])

    bootstrap_deltas: Dict[str, list] = {}

    for i in range(n_iterations):
        sampled_users = rng.choice(users, size=n, replace=True)

        base_sample = _resample(base, base_pos, sampled_users,
                                base_codes, len(base_uniq), impression_col)
        imp_sample = _resample(imp, imp_pos, sampled_users,
                               imp_codes, len(imp_uniq), impression_col)

        try:
            m_base = metric_fn(base_sample)
            m_imp = metric_fn(imp_sample)
        except Exception as e:
            logger.debug(f"Paired bootstrap iteration {i} failed: {e}")
            continue

        for k, v in m_base.items():
            if k in m_imp:
                bootstrap_deltas.setdefault(k, []).append(m_imp[k] - v)

    alpha = (1.0 - ci_level) / 2.0
    return {
        k: (float(np.percentile(v, alpha * 100)),
            float(np.percentile(v, (1.0 - alpha) * 100)))
        for k, v in bootstrap_deltas.items()
    }
