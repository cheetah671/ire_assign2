"""
test_feature_parity.py -- Q9: guard the train/serve contract.

Two failures that produced a silently-broken submission before, both of which
ran to completion without raising:

1. The stage-1 score column was filled with a constant 0.0 on the training
   split. LightGBM cannot split on a zero-variance feature, so the model listed
   it in feature_names while having learned nothing from it, and the real
   values it met at inference had no effect.

2. The inference script's feature list drifted out of sync with the feature
   list the model was trained on, so feature i at serving time was not the
   feature the model learned at position i.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
PRED_DIR   = BASE_DIR / "data" / "predictions"

lgb = pytest.importorskip("lightgbm")

MODELS = sorted(MODELS_DIR.glob("lgbm_*.txt")) if MODELS_DIR.exists() else []


@pytest.mark.skipif(not MODELS, reason="No trained models -- run run_reranker.py first")
@pytest.mark.parametrize("model_path", MODELS, ids=lambda p: p.name)
def test_model_feature_list_matches_sidecar(model_path: Path):
    """The .features.json the inference script reads must match the model itself."""
    feat_path = model_path.with_suffix(".features.json")
    assert feat_path.exists(), (
        f"{model_path.name} has no {feat_path.name}. The inference script reads "
        f"that file to order its feature matrix; without it the order is a guess."
    )

    booster = lgb.Booster(model_file=str(model_path))
    declared = json.loads(feat_path.read_text())

    assert declared == booster.feature_name(), (
        f"Feature list mismatch for {model_path.name}.\n"
        f"  model   : {booster.feature_name()}\n"
        f"  sidecar : {declared}"
    )


@pytest.mark.skipif(not MODELS, reason="No trained models -- run run_reranker.py first")
@pytest.mark.parametrize("model_path", MODELS, ids=lambda p: p.name)
def test_serving_model_excludes_unavailable_features(model_path: Path):
    """
    The model that ships must not depend on features the MIND test set cannot
    provide: behaviors.tsv has no per-click timestamps and news.tsv no pub date.
    """
    if "serving" not in model_path.name:
        pytest.skip("Only the serving model is held to this contract")

    from run_reranker import SERVING_UNSAFE

    # models/lgbm_<dataset>_<ranker>_serving_model.txt
    dataset = model_path.stem.replace("lgbm_", "").rsplit("_", 3)[0].upper()
    unsafe = SERVING_UNSAFE.get(dataset)
    if unsafe is None:
        pytest.skip(f"No serving-safety contract declared for {dataset}")

    booster = lgb.Booster(model_file=str(model_path))
    leaked = set(booster.feature_name()) & set(unsafe)
    assert not leaked, (
        f"{model_path.name} uses {sorted(leaked)}, which cannot be reconstructed "
        f"at serving time and would arrive as a constant."
    )


@pytest.mark.skipif(
    not (PRED_DIR / "MIND" / "improved_val_predictions.parquet").exists(),
    reason="No reranker predictions -- run run_reranker.py first",
)
def test_stage1_score_is_not_degenerate():
    """A stage-1 score with no variance teaches the reranker nothing."""
    preds = pd.read_parquet(PRED_DIR / "MIND" / "improved_val_predictions.parquet")

    score_cols = [c for c in ("emb_score", "bm25_score") if c in preds.columns]
    assert score_cols, "No stage-1 score column found in the reranker predictions."

    for col in score_cols:
        assert preds[col].var() > 1e-12, (
            f"{col} is constant ({preds[col].iloc[0]}). A constant stage-1 score "
            f"means the two-stage pipeline collapsed to a one-stage one."
        )


@pytest.mark.skipif(
    not (PRED_DIR / "MIND" / "improved_val_predictions.parquet").exists(),
    reason="No reranker predictions -- run run_reranker.py first",
)
def test_features_vary_within_impression():
    """
    A feature that is constant inside an impression cannot reorder that
    impression's candidates, so it contributes nothing to a ranking metric no
    matter how much training gain it shows.
    """
    preds = pd.read_parquet(PRED_DIR / "MIND" / "improved_val_predictions.parquet")

    must_vary = [c for c in ("emb_score", "category_match", "article_popularity")
                 if c in preds.columns]
    assert must_vary, "None of the expected within-impression features are present."

    for col in must_vary:
        within = preds.groupby("impression_id")[col].nunique()
        frac = (within > 1).mean()
        assert frac > 0.5, (
            f"{col} varies within only {frac:.1%} of impressions; it cannot "
            f"meaningfully reorder candidates."
        )
