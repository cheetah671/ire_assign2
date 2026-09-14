"""
test_bootstrap.py -- Q5: the confidence intervals must actually contain the estimate.

Regression guard for a bug that made every reported CI wrong without failing.

Resampling users with replacement duplicates a drawn user's rows. Those copies
originally kept their impression_id, so the metric function's
groupby(impression_id) merged them back into one oversized impression instead of
counting them as separate impressions. MRR and nDCG are per-impression, so their
bootstrap distribution shifted downward far enough that the interval could sit
entirely below the point estimate. AUC is computed flat across rows, is
invariant to proportional duplication, and looked fine throughout - which is
what made the bug easy to miss.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.bootstrap import bootstrap_ci, paired_bootstrap_ci
from src.evaluation.metrics import compute_ranking_metrics


def _synthetic(n_users=150, n_cands=10, seed=0):
    """Impressions where the score is informative, so metrics sit well above chance."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        clicked = rng.integers(0, n_cands)
        for c in range(n_cands):
            label = int(c == clicked)
            rows.append({
                "user_id": f"U{u}",
                "impression_id": f"I{u}",
                "article_id": f"N{c}",
                "label": label,
                "score": rng.normal(1.5 if label else 0.0, 1.0),
            })
    df = pd.DataFrame(rows)
    df["rank"] = df.groupby("impression_id")["score"].rank(ascending=False, method="first").astype(int)
    return df


def _metrics(df):
    return compute_ranking_metrics(
        df, score_col="score", rank_col="rank",
        impression_col="impression_id", label_col="label",
    )


def test_point_estimate_lies_inside_ci():
    df = _synthetic()
    point = _metrics(df)
    ci = bootstrap_ci(df, _metrics, n_iterations=120, seed=1)

    for metric, value in point.items():
        if metric not in ci:
            continue
        lo, hi = ci[metric]
        assert lo <= value <= hi, (
            f"{metric}: point estimate {value:.4f} outside its own 95% CI "
            f"[{lo:.4f}, {hi:.4f}]. The resample is not reproducing the "
            f"impression structure of the original data."
        )


def test_resampling_preserves_impression_size():
    """
    Every impression in a resample must hold the same number of candidates as
    the original. If duplicated users collapse into one group, sizes inflate.
    """
    df = _synthetic(n_users=40, n_cands=10)
    seen = {}

    def _capture(sample):
        sizes = sample.groupby("impression_id").size()
        seen["max"] = max(seen.get("max", 0), int(sizes.max()))
        return _metrics(sample)

    bootstrap_ci(df, _capture, n_iterations=25, seed=3)

    assert seen["max"] == 10, (
        f"Largest resampled impression held {seen['max']} candidates, expected 10. "
        f"Duplicate draws of a user are merging into a single impression."
    )


def test_paired_ci_on_identical_systems_contains_zero():
    """Comparing a system against itself must not produce a significant delta."""
    df = _synthetic()
    ci = paired_bootstrap_ci(df, df.copy(), _metrics, n_iterations=120, seed=5)

    for metric, (lo, hi) in ci.items():
        assert lo <= 0.0 <= hi, (
            f"{metric}: CI [{lo:.4f}, {hi:.4f}] for a system against itself "
            f"excludes zero, so the paired test would report false significance."
        )


def test_paired_ci_detects_a_real_improvement():
    """A genuinely better system must produce a CI that excludes zero."""
    base = _synthetic(seed=0)
    better = base.copy()
    # Push the clicked item up without making the ranking perfect.
    better["score"] = base["score"] + 1.2 * base["label"]
    better["rank"] = (better.groupby("impression_id")["score"]
                      .rank(ascending=False, method="first").astype(int))

    ci = paired_bootstrap_ci(base, better, _metrics, n_iterations=200, seed=7)
    lo, hi = ci["ndcg5"]
    assert lo > 0, (
        f"ndcg5 CI [{lo:.4f}, {hi:.4f}] failed to detect a real improvement."
    )
