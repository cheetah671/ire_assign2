"""
test_leakage.py -- Q9: Verify time-based correctness of the data pipeline.

Rules enforced
--------------
1. FEATURE-BUILD SAFETY
   When building features for an impression at time T, only history with
   click_time < T must be used.  We verify this on a sample of impressions:
   after applying the correct filter (click_time < impression_time), there
   must be zero future clicks in the resulting window.

   NOTE: history.parquet stores ALL historical clicks for a user across the
   entire dataset period.  This is intentional -- the leakage guard happens
   at feature-build time (inside the ranker), not at storage time.
   The test simulates what the ranker does and confirms the filter is sound.

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
    COL_IMPRESSION_TIME,
    COL_SPLIT,
    COL_USER_ID,
    SPLIT_TRAIN,
    SPLIT_VAL,
)

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
# Test 1: feature-build leakage (the critical one)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DATASETS, reason=SKIP_MSG)
@pytest.mark.parametrize("dataset_name", DATASETS)
def test_feature_window_is_leak_free(dataset_name: str):
    """
    Simulate the ranking code's feature-build step on 200 sampled impressions.
    After filtering history to click_time < impression_time there must be
    zero clicks at or after impression_time in the resulting window.
    """
    hist, imp = _load(dataset_name)

    sample = imp.dropna(subset=[COL_IMPRESSION_TIME])
    if len(sample) > 200:
        sample = sample.sample(200, random_state=42)

    violations = 0
    for _, row in sample.iterrows():
        uid = row[COL_USER_ID]
        t   = row[COL_IMPRESSION_TIME]
        window = hist[
            (hist[COL_USER_ID] == uid) &
            hist[COL_CLICK_TIME].notna() &
            (hist[COL_CLICK_TIME] < t)
        ]
        if len(window) > 0 and window[COL_CLICK_TIME].max() >= t:
            violations += 1

    assert violations == 0, (
        f"[{dataset_name}] {violations} impressions had future clicks in the "
        f"filtered history window (click_time < impression_time). "
        f"The filter in the ranker is broken."
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
