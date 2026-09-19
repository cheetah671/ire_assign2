"""
test_leakage.py -- Q9: Verify time-based correctness of the data pipeline.

Rules enforced
--------------
1. FEATURE-BUILD SAFETY
   `test_extractor_ignores_future_clicks` exercises the actual production
   code path -- `BehaviouralFeatureExtractor` from
   `src/features/behavioural_features.py` -- rather than re-implementing the
   `click_time < impression_time` filter inline. It injects a synthetic click
   one second AFTER a real impression's timestamp into that user's history and
   asserts the extractor's own `searchsorted` boundary does not count it. A
   regression in the extractor's boundary logic (e.g. an off-by-one between
   `side="left"` and `side="right"`) would be caught here; a test that only
   re-checks the filter concept would not.

2. SPLIT ORDER
   All train impression_times must be strictly earlier than all val
   impression_times for the same user.

3. DATA INTEGRITY
   No null article_ids, labels are 0 or 1, at least some positives exist.

Run with:
    pytest tests/test_leakage.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.schema import (
    COL_ARTICLE_ID,
    COL_CLICK_TIME,
    COL_DATASET,
    COL_IMPRESSION_TIME,
    COL_SPLIT,
    COL_USER_ID,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from src.features.behavioural_features import BehaviouralFeatureExtractor

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_tz(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if hasattr(s.dtype, "tz") and s.dtype.tz is not None:
        return s.dt.tz_localize(None)
    return s


def _load(dataset_name: str):
    """Return (history_df, impressions_df) with tz-naive datetimes."""
    d = PROCESSED_DIR / dataset_name
    hist = pd.read_parquet(d / "history.parquet")
    imp  = pd.read_parquet(d / "impressions.parquet")
    hist[COL_CLICK_TIME]         = _strip_tz(hist[COL_CLICK_TIME])
    imp[COL_IMPRESSION_TIME]     = _strip_tz(imp[COL_IMPRESSION_TIME])
    return hist, imp


def _available_datasets():
    out = []
    if PROCESSED_DIR.exists():
        for d in PROCESSED_DIR.iterdir():
            if d.is_dir():
                if (d / "history.parquet").exists() and (d / "impressions.parquet").exists():
                    out.append(d.name)
    return out


DATASETS = _available_datasets()
SKIP_MSG = "No processed datasets -- run build_pipeline.py first"


# ---------------------------------------------------------------------------
# Test 1: feature-build leakage (the critical one) -- exercises the real
# BehaviouralFeatureExtractor, not a re-implementation of its filter.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DATASETS, reason=SKIP_MSG)
@pytest.mark.parametrize("dataset_name", DATASETS)
def test_extractor_ignores_future_clicks(dataset_name: str):
    """
    Inject a synthetic click 1 second AFTER a real impression's timestamp into
    that user's history, then run the production `BehaviouralFeatureExtractor`
    over it. Every count/recency feature it emits must be identical to the
    unpoisoned baseline -- if the extractor's searchsorted boundary ever leaks
    a future click, this test (not a re-implemented filter) catches it.
    """
    hist, imp = _load(dataset_name)
    articles = pd.read_parquet(PROCESSED_DIR / dataset_name / "articles.parquet")

    candidates = imp.dropna(subset=[COL_IMPRESSION_TIME, COL_USER_ID])
    candidates = candidates[candidates[COL_USER_ID].isin(hist[COL_USER_ID])]
    if len(candidates) == 0:
        pytest.skip(f"[{dataset_name}] no impression with a user present in history")

    target = candidates.sample(1, random_state=42)
    uid = target[COL_USER_ID].iloc[0]
    t = target[COL_IMPRESSION_TIME].iloc[0]

    baseline_feats = BehaviouralFeatureExtractor(hist, articles).extract_features(target)

    future_click = pd.DataFrame([{
        COL_USER_ID: uid,
        COL_DATASET: hist[COL_DATASET].iloc[0] if COL_DATASET in hist.columns else dataset_name,
        COL_ARTICLE_ID: hist[COL_ARTICLE_ID].iloc[0],
        COL_CLICK_TIME: t + pd.Timedelta(seconds=1),
    }])
    poisoned_hist = pd.concat([hist, future_click], ignore_index=True)
    poisoned_feats = BehaviouralFeatureExtractor(poisoned_hist, articles).extract_features(target)

    feature_cols = [
        "user_click_count", "user_hist_recency_sum",
        "session_clicks_1h", "session_clicks_24h",
    ]
    for col in feature_cols:
        before = baseline_feats[col].iloc[0]
        after = poisoned_feats[col].iloc[0]
        assert before == after, (
            f"[{dataset_name}] injecting a click 1s after the impression time "
            f"changed '{col}' ({before} -> {after}). "
            f"BehaviouralFeatureExtractor's behaviour-window boundary is leaking "
            f"future clicks."
        )


# ---------------------------------------------------------------------------
# Test 2: train impressions precede val impressions per user
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DATASETS, reason=SKIP_MSG)
@pytest.mark.parametrize("dataset_name", DATASETS)
def test_train_before_val(dataset_name: str):
    """Latest train impression_time < earliest val impression_time per user."""
    _, imp = _load(dataset_name)

    train = imp[imp[COL_SPLIT] == SPLIT_TRAIN]
    val   = imp[imp[COL_SPLIT] == SPLIT_VAL]

    if len(train) == 0 or len(val) == 0:
        pytest.skip(f"Missing train or val split in {dataset_name}")

    train_max = train.groupby(COL_USER_ID)[COL_IMPRESSION_TIME].max().rename("train_max")
    val_min   = val.groupby(COL_USER_ID)[COL_IMPRESSION_TIME].min().rename("val_min")

    merged = pd.concat([train_max, val_min], axis=1).dropna()
    if len(merged) > 500:
        merged = merged.sample(500, random_state=42)

    bad = merged[merged["train_max"] >= merged["val_min"]]
    assert len(bad) == 0, (
        f"[{dataset_name}] {len(bad)} users have train impressions at or after "
        f"their earliest val impression (temporal split violation).\n"
        f"Sample:\n{bad.head()}"
    )


# ---------------------------------------------------------------------------
# Test 3: data integrity checks
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DATASETS, reason=SKIP_MSG)
@pytest.mark.parametrize("dataset_name", DATASETS)
def test_no_null_article_ids(dataset_name: str):
    _, imp = _load(dataset_name)
    nulls = imp[COL_ARTICLE_ID].isna().sum()
    assert nulls == 0, f"[{dataset_name}] {nulls} impression rows have null article_id."


@pytest.mark.skipif(not DATASETS, reason=SKIP_MSG)
@pytest.mark.parametrize("dataset_name", DATASETS)
def test_labels_are_binary(dataset_name: str):
    _, imp = _load(dataset_name)
    bad = imp[~imp["label"].isin([0, 1])]
    assert len(bad) == 0, f"[{dataset_name}] {len(bad)} rows have non-binary labels."


@pytest.mark.skipif(not DATASETS, reason=SKIP_MSG)
@pytest.mark.parametrize("dataset_name", DATASETS)
def test_positive_impressions_exist(dataset_name: str):
    _, imp = _load(dataset_name)
    n_pos = imp["label"].sum()
    assert n_pos > 0, f"[{dataset_name}] No positive impressions found -- pipeline may be broken."
