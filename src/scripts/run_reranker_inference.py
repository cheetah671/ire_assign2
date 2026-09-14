"""
run_reranker_inference.py -- Two-Stage LightGBM Inference on Test Sets
=====================================================================

Implements memory-safe, chunked inference for the full two-stage pipeline:
  Stage 1: Embedding score  (sentence-transformers all-MiniLM-L6-v2, for MIND)
  Stage 2: LightGBM reranker using behavioural + embedding features

Memory strategy:
  - Reads behaviors.tsv in streaming chunks of CHUNK_SIZE impressions
  - Extracts features and scores each chunk with LightGBM, writes ranks immediately
  - Never materialises more than chunk_size impressions in RAM at once
  - History is pre-indexed into a dict for O(1) per-user lookup

Usage
-----
  # Train first (writes the model AND its feature list):
  #   python run_reranker.py --dataset mind --ranker emb
  python src/scripts/run_reranker_inference.py --dataset mind_large

Output
------
  submissions/MIND_LARGE_RERANKER/prediction.txt
  mind_large_reranker_submission.zip    <-- upload this to Codabench
"""

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

# -- Make project root importable ---------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.parse_mind import _parse_news
from src.features.embedding_index import EmbeddingIndex

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reranker_inference")

CHUNK_SIZE  = 10_000   # impressions processed per batch before flushing to disk
MAX_HISTORY = 50       # most-recent clicked articles used per impression


# -----------------------------------------------------------------------------
# Pre-index user history + article popularity from behaviors.tsv
# -----------------------------------------------------------------------------

def count_lines(path: Path) -> int:
    """Total impressions, so every progress bar can show a real percentage."""
    logger.info(f"Counting impressions in {path.name} ...")
    with open(path, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    logger.info(f"  {n:,} impressions")
    return n


def preindex_from_tsv(behaviors_path: Path, total_lines: int = None):
    """
    Stream behaviors.tsv once to build:
      - {user_id: [article_id, ...]}  (last MAX_HISTORY articles, most recent last)
      - {article_id: popularity_percentile in [0,1]}

    Popularity is counted over each user's history exactly once, matching how
    run_reranker.py derives it from history.parquet at training time, and is
    converted to a percentile so the two corpora are on a comparable scale.
    """
    logger.info("[1/4] Pre-indexing user click history from behaviors.tsv ...")
    user_hist: dict = {}

    with open(behaviors_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="  history index", unit="imp",
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                     "[{elapsed}<{remaining}, {rate_fmt}]"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            user_id     = parts[1]
            history_str = parts[3].strip() if parts[3] else ""
            if history_str:
                user_hist[user_id] = history_str.split()[-MAX_HISTORY:]

    logger.info(f"  Indexed {len(user_hist):,} unique users")

    logger.info("[2/4] Computing article popularity percentiles ...")
    counts: dict = {}
    for arts in user_hist.values():
        for aid in arts:
            counts[aid] = counts.get(aid, 0) + 1

    pop = pd.Series(counts, dtype="float64").rank(pct=True).to_dict() if counts else {}
    logger.info(f"  Popularity computed over {len(pop):,} distinct articles")
    return user_hist, pop


# -----------------------------------------------------------------------------
# Per-impression feature extraction  (pure numpy, no DataFrame overhead)
# -----------------------------------------------------------------------------

def extract_features_for_impression(
    candidates: list,            # [article_id str, ...]
    user_hist_articles: list,    # pre-looked-up history articles for this user
    emb_index: EmbeddingIndex,   # built embedding index
    article_cat_map: dict,       # {article_id str -> category str}
    article_popularity: dict,    # {article_id str -> popularity percentile}
) -> dict:
    """
    Compute Stage-2 features for one impression. Returns {article_id -> features}.

    Only features reconstructible from behaviors.tsv + news.tsv are produced.
    Click timestamps and publication dates do not exist in the MIND test set, so
    recency/session/freshness features are deliberately absent here and excluded
    from the shipped model's feature list (see SERVING_UNSAFE in run_reranker.py).
    """
    click_count = len(user_hist_articles)

    hist_cat_counts: dict = {}
    for aid in user_hist_articles:
        cat = article_cat_map.get(aid, "")
        if cat:
            hist_cat_counts[cat] = hist_cat_counts.get(cat, 0) + 1
    n_cat_hist = sum(hist_cat_counts.values())

    # -- Stage-1 user vector: plain mean-pool, matching EmbeddingIndex.make_user_vector
    # used at training time. Any other pooling here would shift the emb_score
    # distribution away from what the reranker was fitted on.
    hist_rows = []
    for aid in user_hist_articles:
        idx = emb_index.id_to_idx.get(aid)
        if idx is not None:
            hist_rows.append(emb_index.matrix[idx])

    if hist_rows:
        H = np.vstack(hist_rows).astype(np.float32)
        user_vec = H.mean(axis=0)
        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec = (user_vec / norm).astype(np.float32)
    else:
        H = None
        user_vec = None

    cand_rows = []
    for cid in candidates:
        idx = emb_index.id_to_idx.get(cid)
        cand_rows.append(
            emb_index.matrix[idx] if idx is not None
            else np.zeros(emb_index.dim, dtype=np.float32)
        )
    C = np.vstack(cand_rows).astype(np.float32)

    emb_scores = C @ user_vec if user_vec is not None else np.zeros(len(candidates), dtype=np.float32)
    max_sims   = (C @ H.T).max(axis=1) if H is not None else np.zeros(len(candidates), dtype=np.float32)

    result = {}
    for i, cand_id in enumerate(candidates):
        cand_cat = article_cat_map.get(cand_id, "")
        result[cand_id] = {
            "user_click_count":   click_count,
            "emb_score":          float(emb_scores[i]),
            "emb_max_sim":        float(max_sims[i]),
            "category_match":     1 if cand_cat in hist_cat_counts else 0,
            "cat_affinity":       hist_cat_counts.get(cand_cat, 0) / n_cat_hist if n_cat_hist else 0.0,
            "article_popularity": article_popularity.get(cand_id, 0.0),
        }
    return result


# -----------------------------------------------------------------------------
# Chunk flusher: batch-predict with LightGBM and write ranks to file
# -----------------------------------------------------------------------------

def _flush_chunk(rows: list, lgbm_model, feature_cols: list, emb_alpha: float, fout) -> None:
    """Score one chunk with LightGBM and write one prediction line per impression."""
    if not rows:
        return

    all_feats = []
    for _, candidates, feat_dicts in rows:
        for cand_id in candidates:
            fv = feat_dicts[cand_id]
            all_feats.append([fv.get(col, 0.0) for col in feature_cols])

    X = np.array(all_feats, dtype=np.float32)
    np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    lgbm_scores = lgbm_model.predict(X)

    cursor = 0
    for imp_id, candidates, feat_dicts in rows:
        n = len(candidates)
        scores = np.asarray(lgbm_scores[cursor:cursor + n], dtype=np.float64)
        cursor += n

        if emb_alpha > 0.0:
            # Min-max the model score within this impression so it is on the same
            # [0,1] footing as cosine similarity before blending.
            lo, hi = scores.min(), scores.max()
            norm = (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)
            emb = np.array([feat_dicts[c]["emb_score"] for c in candidates])
            scores = emb_alpha * emb + (1.0 - emb_alpha) * norm

        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(1, n + 1)
        fout.write(f"{imp_id} [{','.join(map(str, ranks.tolist()))}]\n")


# -----------------------------------------------------------------------------
# Streaming inference over MIND behaviors.tsv
# -----------------------------------------------------------------------------

def run_mind_inference(
    behaviors_path: Path,
    emb_index: EmbeddingIndex,
    lgbm_model,
    feature_cols: list,
    emb_alpha: float,
    article_cat_map: dict,
    article_popularity: dict,
    user_hist: dict,
    out_path: Path,
    total_lines: int,
) -> None:
    """Stream behaviors.tsv in CHUNK_SIZE batches, write prediction.txt."""

    logger.info(f"[4/4] Scoring {total_lines:,} impressions (stage-1 embedding "
                f"+ stage-2 LightGBM) ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0         = perf_counter()
    chunk_rows = []
    imps_done  = 0
    skipped    = 0
    malformed  = 0

    with open(behaviors_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        pbar = tqdm(
            total=total_lines, desc="  reranking", unit="imp", mininterval=2.0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                if parts and parts[0]:
                    fout.write(f"{parts[0]} []\n")
                malformed += 1
                pbar.update(1)
                continue

            imp_id     = parts[0]
            user_id    = parts[1]
            candidates = parts[4].split() if parts[4] else []

            # Codabench matches predictions to impressions by line, so every
            # input line must produce exactly one output line.
            if not candidates:
                fout.write(f"{imp_id} []\n")
                skipped += 1
                pbar.update(1)
                continue

            feat_dicts = extract_features_for_impression(
                candidates         = candidates,
                user_hist_articles = user_hist.get(user_id, []),
                emb_index          = emb_index,
                article_cat_map    = article_cat_map,
                article_popularity = article_popularity,
            )

            chunk_rows.append((imp_id, candidates, feat_dicts))
            imps_done += 1
            pbar.update(1)

            if len(chunk_rows) >= CHUNK_SIZE:
                _flush_chunk(chunk_rows, lgbm_model, feature_cols, emb_alpha, fout)
                chunk_rows.clear()

        _flush_chunk(chunk_rows, lgbm_model, feature_cols, emb_alpha, fout)
        pbar.close()

    elapsed = perf_counter() - t0
    size_mb = out_path.stat().st_size / 1e6
    logger.info(
        f"Inference complete: {imps_done:,} impressions scored in {elapsed:.1f}s "
        f"({imps_done/elapsed:,.0f} imp/s)  ->  {out_path}  ({size_mb:.1f} MB)"
    )
    if skipped or malformed:
        logger.warning(f"  {skipped:,} impressions had no candidates, "
                       f"{malformed:,} lines were malformed (written as empty)")

    with open(out_path, "r", encoding="utf-8") as f:
        written = sum(1 for _ in f)
    if written != total_lines:
        logger.error(
            f"Line-count mismatch: wrote {written:,} but behaviors.tsv has "
            f"{total_lines:,}. Codabench matches predictions to impressions by "
            f"line, so this file would be scored against the wrong impressions."
        )
        sys.exit(1)
    logger.info(f"  Validated: {written:,} prediction lines == {total_lines:,} impressions")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage Embedding+LightGBM inference on MIND-large test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["mind_large"], default="mind_large")
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Trained LightGBM .txt model. Default: models/lgbm_mind_emb_serving_model.txt",
    )
    parser.add_argument(
        "--emb-alpha", type=float, default=0.0,
        help="Blend weight on raw cosine similarity: "
             "final = alpha*emb_score + (1-alpha)*normalised_model_score. "
             "0.0 = pure reranker (emb_score is already a trained feature).",
    )
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if lgb is None:
        logger.error("lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)
    if not 0.0 <= args.emb_alpha <= 1.0:
        logger.error("--emb-alpha must be in [0, 1]")
        sys.exit(1)

    TEST_RAW_DIR   = BASE_DIR / "data" / "raw" / "MIND" / "large_test" / "MINDlarge_test"
    BEHAVIORS_PATH = TEST_RAW_DIR / "behaviors.tsv"
    NEWS_PATH      = TEST_RAW_DIR / "news.tsv"
    CACHE_DIR      = BASE_DIR / "data" / "cache"
    ZIP_OUT        = BASE_DIR / "mind_large_reranker_submission.zip"
    model_path     = args.model or (BASE_DIR / "models" / "lgbm_mind_emb_serving_model.txt")

    for p, label in [(BEHAVIORS_PATH, "behaviors.tsv"), (NEWS_PATH, "news.tsv")]:
        if not p.exists():
            logger.error(f"Missing {label}: {p}")
            sys.exit(1)

    if not model_path.exists():
        logger.error(
            f"LightGBM model not found: {model_path}\n"
            "Train it first:\n  python run_reranker.py --dataset mind --ranker emb"
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Two-Stage Embedding + LightGBM Inference  |  MIND-large")
    logger.info("=" * 60)

    # -- Load model and the exact feature list it was trained with -------------
    logger.info(f"Loading LightGBM model from {model_path} ...")
    lgbm_model = lgb.Booster(model_file=str(model_path))

    feat_path = model_path.with_suffix(".features.json")
    if feat_path.exists():
        feature_cols = json.loads(feat_path.read_text())
    else:
        feature_cols = lgbm_model.feature_name()
        logger.warning(f"No {feat_path.name}; falling back to model feature names.")

    # A silent order/count mismatch between this list and the model is exactly
    # what produced a garbage submission before; fail loudly instead.
    model_names = lgbm_model.feature_name()
    if feature_cols != model_names:
        logger.error(
            "Feature mismatch between model and feature list.\n"
            f"  model : {model_names}\n"
            f"  script: {feature_cols}\n"
            "Retrain with run_reranker.py so the two agree."
        )
        sys.exit(1)
    logger.info(f"  Model loaded with {len(feature_cols)} features: {feature_cols}")

    # -- Load articles --------------------------------------------------------
    logger.info(f"Loading articles from {NEWS_PATH.name} ...")
    articles_df = _parse_news(NEWS_PATH)
    logger.info(f"  {len(articles_df):,} articles")

    article_cat_map = dict(zip(
        articles_df["article_id"].astype(str),
        articles_df.get("category", pd.Series("", index=articles_df.index)).fillna(""),
    ))

    total_lines = count_lines(BEHAVIORS_PATH)

    # -- Build embedding index (cached as .npz) --------------------------------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    emb_cache = CACHE_DIR / "emb_MIND_LARGE_TEST.npz"

    logger.info("[3/4] Building / loading embedding index ...")
    emb_index = EmbeddingIndex(
        articles_df  = articles_df,
        dataset_name = "MIND",
        cache_path   = None if args.no_cache else emb_cache,
    )
    t0 = perf_counter()
    emb_index.build()
    logger.info(f"  Embedding index ready in {perf_counter()-t0:.1f}s  "
                f"(shape={emb_index.matrix.shape})")

    if args.no_cache or not emb_cache.exists():
        emb_index._save_cache()

    # -- Pre-index user history + popularity -----------------------------------
    user_hist, article_popularity = preindex_from_tsv(BEHAVIORS_PATH, total_lines)

    out_dir  = BASE_DIR / "submissions" / "MIND_LARGE_RERANKER"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prediction.txt"

    run_mind_inference(
        behaviors_path     = BEHAVIORS_PATH,
        emb_index          = emb_index,
        lgbm_model         = lgbm_model,
        feature_cols       = feature_cols,
        emb_alpha          = args.emb_alpha,
        article_cat_map    = article_cat_map,
        article_popularity = article_popularity,
        user_hist          = user_hist,
        out_path           = out_path,
        total_lines        = total_lines,
    )

    logger.info(f"Zipping  ->  {ZIP_OUT} ...")
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname="prediction.txt")
    logger.info(f"  Done: {ZIP_OUT}  ({ZIP_OUT.stat().st_size/1e6:.1f} MB)")

    logger.info("=" * 60)
    logger.info("Upload this file to Codabench MIND leaderboard:")
    logger.info(f"  {ZIP_OUT}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
