"""
run_ebnerd_reranker_inference.py -- Two-Stage Inference on the EB-NeRD Test Set
==============================================================================

The EB-NeRD counterpart of run_reranker_inference.py.

  Stage 1  Ekstra Bladet Word2Vec document vectors -> cosine similarity
  Stage 2  LightGBM lambdarank over behavioural + semantic features

Scale
-----
The test bundle is 13.5M impressions over 807k users whose histories average
248 clicks. Nothing is materialised whole:

  - articles, categories and publication times become flat numpy arrays indexed
    by article id, so a candidate lookup is an array index rather than a dict hit
  - history is stored CSR-style (one flat array per field plus per-user offsets)
    and capped to the most recent 50 clicks, matching the training-time cap
  - behaviors.parquet is streamed one row group at a time and scored in batches

Unlike MIND, EB-NeRD's test bundle ships click timestamps, read time, scroll
depth and article publication dates, so every feature the model was trained on
is reconstructible here and SERVING_UNSAFE is empty for this dataset.

Usage
-----
  python run_reranker.py --dataset ebnerd_small --ranker emb      # train first
  python src/scripts/run_ebnerd_reranker_inference.py

  # quick smoke test over the first 20k impressions
  python src/scripts/run_ebnerd_reranker_inference.py --limit 20000

Output
------
  submissions/EBNERD_LARGE_RERANKER/predictions.txt
  ebnerd_large_reranker_submission.zip     <-- upload this to Codabench
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
import pyarrow.parquet as pq
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.parse_ebnerd import _parse_articles
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
logger = logging.getLogger("ebnerd_inference")

MAX_HISTORY    = 50
HALF_LIFE_DAYS = 7.0
BATCH_IMPS     = 20_000          # impressions per LightGBM predict call
NS_PER_DAY     = 86_400_000_000_000
NS_PER_HOUR    = 3_600_000_000_000

TEST_DIR       = BASE_DIR / "data" / "raw" / "EBNERD" / "test" / "ebnerd_testset"
BEHAVIORS_PATH = TEST_DIR / "test" / "behaviors.parquet"
HISTORY_PATH   = TEST_DIR / "test" / "history.parquet"
ARTICLES_PATH  = TEST_DIR / "articles.parquet"


# -----------------------------------------------------------------------------
# Article side tables
# -----------------------------------------------------------------------------

def build_article_tables(emb_index: EmbeddingIndex, articles: pd.DataFrame):
    """
    Flat arrays indexed by a dense article row, plus `row_of_id` mapping a raw
    article id straight to that row. Candidate lookups are then pure numpy
    fancy-indexing instead of per-candidate dict hits, which matters at 13.5M
    impressions.
    """
    logger.info("[2/5] Building article lookup tables ...")
    ids = articles["article_id"].astype(np.int64).to_numpy()
    max_id = int(ids.max())
    row_of_id = np.full(max_id + 2, -1, dtype=np.int32)
    row_of_id[ids] = np.arange(len(ids), dtype=np.int32)

    # Categories as integer codes so history counting is a bincount.
    cat_codes, cat_uniq = pd.factorize(articles["category"].fillna("unknown").astype(str))
    cat_codes = cat_codes.astype(np.int32)

    pub = pd.to_datetime(articles["published_time"], errors="coerce")
    if hasattr(pub.dtype, "tz") and pub.dtype.tz is not None:
        pub = pub.dt.tz_localize(None)
    pub_ns = pub.to_numpy(dtype="datetime64[ns]").astype(np.int64).astype(np.float64)
    pub_ns[pd.isna(pub).to_numpy()] = np.nan

    # article row -> row in the embedding matrix (-1 when the article has no vector)
    emb_row = np.full(len(ids), -1, dtype=np.int32)
    hits = 0
    for i, aid in enumerate(ids):
        j = emb_index.id_to_idx.get(str(aid))
        if j is not None:
            emb_row[i] = j
            hits += 1

    logger.info(f"  {len(ids):,} articles | {len(cat_uniq)} categories | "
                f"{hits:,} with a Word2Vec vector "
                f"({hits/len(ids)*100:.1f}%)")
    return row_of_id, cat_codes, len(cat_uniq), pub_ns, emb_row


# -----------------------------------------------------------------------------
# History (CSR-style, capped)
# -----------------------------------------------------------------------------

def build_history_index(row_of_id: np.ndarray, n_articles: int, cache_path: Path = None):
    """
    Returns (user_slot, offsets, times, art_rows, reads, scrolls, pop_pct).

    Popularity is counted over the *full* history, matching how
    run_reranker.popularity_percentile counts every history row at training
    time, while the per-user window kept for features is capped at MAX_HISTORY.

    The build walks 807k users' arrays in Python and takes ~9 minutes, so the
    result is cached; it depends only on the test bundle, which never changes.
    """
    if cache_path and cache_path.exists():
        logger.info(f"[3/5] Loading cached history index from {cache_path.name} ...")
        d = np.load(cache_path, allow_pickle=False)
        user_ids = d["user_ids"]
        user_slot = {int(u): i for i, u in enumerate(user_ids)}
        logger.info(f"  {len(user_slot):,} users | {len(d['times']):,} clicks")
        return (user_slot, d["offsets"], d["times"], d["art_rows"],
                d["reads"], d["scrolls"], d["pop_pct"])

    logger.info("[3/5] Indexing user history ...")
    pf = pq.ParquetFile(HISTORY_PATH)
    cols = ["user_id", "article_id_fixed", "impression_time_fixed",
            "read_time_fixed", "scroll_percentage_fixed"]

    pop_counts = np.zeros(n_articles, dtype=np.int64)
    user_ids_all, t_chunks, a_chunks, r_chunks, s_chunks, len_chunks = [], [], [], [], [], []

    for rg in tqdm(range(pf.metadata.num_row_groups), desc="  history", unit="rg"):
        tbl = pf.read_row_group(rg, columns=cols).to_pandas()

        for uid, arts, times, reads, scrolls in zip(
            tbl["user_id"].to_numpy(),
            tbl["article_id_fixed"], tbl["impression_time_fixed"],
            tbl["read_time_fixed"], tbl["scroll_percentage_fixed"],
        ):
            if arts is None or len(arts) == 0:
                continue

            a_full = np.asarray(arts, dtype=np.int64)
            rows_full = row_of_id[np.clip(a_full, 0, len(row_of_id) - 1)]
            valid_full = rows_full >= 0
            if valid_full.any():
                pop_counts += np.bincount(rows_full[valid_full], minlength=n_articles)

            a = a_full[-MAX_HISTORY:]
            t = np.asarray(times, dtype="datetime64[ns]")[-MAX_HISTORY:].astype(np.int64)
            r = np.asarray(reads, dtype=np.float32)[-MAX_HISTORY:]
            s = np.asarray(scrolls, dtype=np.float32)[-MAX_HISTORY:]
            rows = row_of_id[np.clip(a, 0, len(row_of_id) - 1)]

            order = np.argsort(t, kind="stable")   # searchsorted needs sorted times
            user_ids_all.append(uid)
            len_chunks.append(len(a))
            t_chunks.append(t[order]); a_chunks.append(rows[order])
            r_chunks.append(r[order]); s_chunks.append(s[order])

        del tbl

    times = np.concatenate(t_chunks); art_rows = np.concatenate(a_chunks)
    reads = np.concatenate(r_chunks); scrolls = np.concatenate(s_chunks)
    del t_chunks, a_chunks, r_chunks, s_chunks

    offsets = np.zeros(len(len_chunks) + 1, dtype=np.int64)
    np.cumsum(np.asarray(len_chunks, dtype=np.int64), out=offsets[1:])
    user_slot = {int(u): i for i, u in enumerate(user_ids_all)}

    logger.info(f"  {len(user_slot):,} users | {len(times):,} clicks kept "
                f"(capped at {MAX_HISTORY}) | {times.nbytes/1e6:.0f} MB timestamps")

    # Percentile over articles that were actually clicked; unseen articles stay 0.
    pop_pct = np.zeros(n_articles, dtype=np.float32)
    seen = pop_counts > 0
    if seen.any():
        pop_pct[seen] = (pd.Series(pop_counts[seen]).rank(pct=True)
                         .to_numpy(dtype=np.float32))
    logger.info(f"  popularity computed over {int(seen.sum()):,} distinct articles")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path,
                 user_ids=np.asarray(user_ids_all, dtype=np.int64),
                 offsets=offsets, times=times, art_rows=art_rows,
                 reads=reads, scrolls=scrolls, pop_pct=pop_pct)
        logger.info(f"  History index cached -> {cache_path.name} "
                    f"({cache_path.stat().st_size/1e6:.0f} MB)")

    return user_slot, offsets, times, art_rows, reads, scrolls, pop_pct


# -----------------------------------------------------------------------------
# Per-impression features
# -----------------------------------------------------------------------------

def impression_features(
    cand_ids, t_ns, slot, offsets, h_times, h_rows, h_reads, h_scrolls,
    row_of_id, cat_codes, n_cats, pub_ns, pop_pct, emb_row, emb_matrix, emb_dim,
):
    """Feature matrix (n_candidates, 12) in FEATURE_ORDER."""
    n = len(cand_ids)
    cand_rows = row_of_id[np.clip(cand_ids, 0, len(row_of_id) - 1)]
    known = cand_rows >= 0
    safe_rows = np.where(known, cand_rows, 0)

    click_count = 0
    recency_sum = 0.0
    sess_1h = sess_24h = 0
    mean_read = np.nan
    mean_scroll = np.nan
    cat_counts = None
    n_cat = 0
    H = None

    if slot is not None:
        lo_u, hi_u = offsets[slot], offsets[slot + 1]
        if hi_u > lo_u:
            seg_t = h_times[lo_u:hi_u]
            # Behaviour-window boundary: only clicks strictly before the impression.
            k = int(np.searchsorted(seg_t, t_ns, side="left"))
            if k > 0:
                lo = lo_u + max(0, k - MAX_HISTORY)
                hi = lo_u + k
                w_t = h_times[lo:hi]
                age_ns = t_ns - w_t

                click_count = hi - lo
                recency_sum = float(np.exp(
                    -np.log(2) * (age_ns / NS_PER_DAY) / HALF_LIFE_DAYS).sum())
                sess_1h = int((age_ns <= NS_PER_HOUR).sum())
                sess_24h = int((age_ns <= 24 * NS_PER_HOUR).sum())

                w_rows = h_rows[lo:hi]
                good = w_rows >= 0
                if good.any():
                    cat_counts = np.bincount(cat_codes[w_rows[good]], minlength=n_cats)
                    n_cat = int(cat_counts.sum())
                    e = emb_row[w_rows[good]]
                    e = e[e >= 0]
                    if len(e):
                        H = emb_matrix[e]

                w_read = h_reads[lo:hi]; w_scroll = h_scrolls[lo:hi]
                if np.isfinite(w_read).any():
                    mean_read = float(np.nanmean(w_read))
                if np.isfinite(w_scroll).any():
                    mean_scroll = float(np.nanmean(w_scroll))

    # -- semantic --------------------------------------------------------------
    cand_emb = emb_row[safe_rows]
    has_emb = known & (cand_emb >= 0)
    C = np.zeros((n, emb_dim), dtype=np.float32)
    if has_emb.any():
        C[has_emb] = emb_matrix[cand_emb[has_emb]]

    if H is not None and len(H):
        user_vec = H.mean(axis=0)
        nrm = np.linalg.norm(user_vec)
        if nrm > 0:
            user_vec = user_vec / nrm
        emb_score = C @ user_vec.astype(np.float32)
        emb_max = (C @ H.T).max(axis=1)
    else:
        emb_score = np.zeros(n, dtype=np.float32)
        emb_max = np.zeros(n, dtype=np.float32)

    # -- category --------------------------------------------------------------
    if cat_counts is not None and n_cat:
        cc = cat_codes[safe_rows]
        hits = cat_counts[cc].astype(np.float32)
        cat_match = ((hits > 0) & known).astype(np.float32)
        cat_aff = np.where(known, hits / n_cat, 0.0).astype(np.float32)
    else:
        cat_match = np.zeros(n, dtype=np.float32)
        cat_aff = np.zeros(n, dtype=np.float32)

    # -- freshness / popularity ------------------------------------------------
    pubs = np.where(known, pub_ns[safe_rows], np.nan)
    fresh = np.full(n, -1.0, dtype=np.float32)
    ok = ~np.isnan(pubs)
    if ok.any():
        fresh[ok] = np.maximum(0.0, (t_ns - pubs[ok]) / NS_PER_DAY)

    pop = np.where(known, pop_pct[safe_rows], 0.0).astype(np.float32)

    out = np.empty((n, 12), dtype=np.float32)
    out[:, 0] = click_count
    out[:, 1] = recency_sum
    out[:, 2] = sess_1h
    out[:, 3] = sess_24h
    out[:, 4] = cat_match
    out[:, 5] = cat_aff
    out[:, 6] = fresh
    out[:, 7] = pop
    out[:, 8] = mean_read
    out[:, 9] = mean_scroll
    out[:, 10] = emb_score
    out[:, 11] = emb_max
    return out


FEATURE_ORDER = [
    "user_click_count", "user_hist_recency_sum", "session_clicks_1h",
    "session_clicks_24h", "category_match", "cat_affinity", "freshness_days",
    "article_popularity", "hist_mean_read_time", "hist_mean_scroll",
    "emb_score", "emb_max_sim",
]


def main():
    parser = argparse.ArgumentParser(description="Two-stage inference on the EB-NeRD test set.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="Score only the first N impressions (smoke test). "
                             "The resulting file is NOT a valid submission.")
    args = parser.parse_args()

    if lgb is None:
        logger.error("lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)

    for p in (BEHAVIORS_PATH, HISTORY_PATH, ARTICLES_PATH):
        if not p.exists():
            logger.error(f"Missing {p}. Extract ebnerd_testset.zip first.")
            sys.exit(1)

    model_path = args.model or (BASE_DIR / "models" / "lgbm_ebnerd_small_emb_serving_model.txt")
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}\n"
                     "Train it first:\n"
                     "  python run_reranker.py --dataset ebnerd_small --ranker emb")
        sys.exit(1)

    logger.info("=" * 62)
    logger.info("Two-Stage Word2Vec + LightGBM Inference  |  EB-NeRD test set")
    logger.info("=" * 62)

    # -- model + feature contract ---------------------------------------------
    logger.info("[1/5] Loading LightGBM model ...")
    booster = lgb.Booster(model_file=str(model_path))
    feat_path = model_path.with_suffix(".features.json")
    declared = json.loads(feat_path.read_text()) if feat_path.exists() else booster.feature_name()

    if declared != booster.feature_name() or declared != FEATURE_ORDER:
        logger.error(
            "Feature mismatch. This script builds its matrix in a fixed order, so "
            "any divergence would feed the model the wrong columns.\n"
            f"  model  : {booster.feature_name()}\n"
            f"  sidecar: {declared}\n"
            f"  script : {FEATURE_ORDER}"
        )
        sys.exit(1)
    logger.info(f"  {len(FEATURE_ORDER)} features verified against the model")

    # -- stage-1 index ---------------------------------------------------------
    articles_df = _parse_articles(ARTICLES_PATH)
    cache = BASE_DIR / "data" / "cache" / "emb_EBNERD_TEST.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    logger.info("  Building / loading Word2Vec index ...")
    t0 = perf_counter()
    emb_index = EmbeddingIndex(articles_df=articles_df, dataset_name="EBNERD",
                               zip_dir=BASE_DIR, cache_path=cache).build()
    logger.info(f"  Embedding index ready in {perf_counter()-t0:.1f}s "
                f"(shape={emb_index.matrix.shape})")

    row_of_id, cat_codes, n_cats, pub_ns, emb_row = build_article_tables(
        emb_index, articles_df)
    n_articles = len(cat_codes)
    del articles_df

    (user_slot, offsets, h_times, h_rows,
     h_reads, h_scrolls, pop_pct) = build_history_index(
        row_of_id, n_articles,
        cache_path=BASE_DIR / "data" / "cache" / "ebnerd_test_history.npz")

    emb_matrix = emb_index.matrix
    emb_dim = emb_index.dim

    # -- stream and score ------------------------------------------------------
    out_dir = BASE_DIR / "submissions" / "EBNERD_LARGE_RERANKER"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.txt"

    pf = pq.ParquetFile(BEHAVIORS_PATH)
    total = pf.metadata.num_rows
    if args.limit:
        total = min(total, args.limit)
    logger.info(f"[4/5] Scoring {total:,} impressions ...")

    written = 0
    t0 = perf_counter()
    pbar = tqdm(total=total, desc="  reranking", unit="imp", mininterval=2.0,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]")

    with open(out_path, "w", encoding="utf-8") as fout:
        stop = False
        for rg in range(pf.metadata.num_row_groups):
            if stop:
                break
            tbl = pf.read_row_group(
                rg, columns=["impression_id", "user_id", "impression_time",
                             "article_ids_inview"]).to_pandas()

            imp_ids = tbl["impression_id"].to_numpy()
            uids = tbl["user_id"].to_numpy()
            t_all = tbl["impression_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            inview = tbl["article_ids_inview"].to_numpy()

            start = 0
            while start < len(tbl):
                end = min(start + BATCH_IMPS, len(tbl))
                blocks, meta = [], []

                for i in range(start, end):
                    cands = np.asarray(inview[i], dtype=np.int64)
                    if len(cands) == 0:
                        meta.append((imp_ids[i], 0))
                        continue
                    blocks.append(impression_features(
                        cands, t_all[i], user_slot.get(int(uids[i])), offsets,
                        h_times, h_rows, h_reads, h_scrolls, row_of_id, cat_codes,
                        n_cats, pub_ns, pop_pct, emb_row, emb_matrix, emb_dim))
                    meta.append((imp_ids[i], len(cands)))

                scores = booster.predict(np.vstack(blocks)) if blocks else np.empty(0)

                cur = 0
                for imp_id, n in meta:
                    if n == 0:
                        # One output line per input impression, always.
                        fout.write(f"{imp_id} []\n")
                        written += 1
                        continue
                    s = scores[cur:cur + n]; cur += n
                    order = np.argsort(-s, kind="stable")
                    ranks = np.empty(n, dtype=np.int64)
                    ranks[order] = np.arange(1, n + 1)
                    fout.write(f"{imp_id} [{','.join(map(str, ranks.tolist()))}]\n")
                    written += 1

                pbar.update(end - start)
                start = end

                if args.limit and written >= args.limit:
                    stop = True
                    break
            del tbl
    pbar.close()

    elapsed = perf_counter() - t0
    logger.info(f"  Scored {written:,} impressions in {elapsed/60:.1f} min "
                f"({written/max(elapsed,1):,.0f} imp/s) -> {out_path} "
                f"({out_path.stat().st_size/1e6:.1f} MB)")

    if args.limit:
        logger.warning(f"--limit was set; {out_path.name} covers only "
                       f"{written:,} impressions and is not a valid submission.")
        return

    if written != pf.metadata.num_rows:
        logger.error(f"Line-count mismatch: wrote {written:,}, expected "
                     f"{pf.metadata.num_rows:,}. Codabench matches predictions to "
                     f"impressions by line; this file would be misaligned.")
        sys.exit(1)
    logger.info(f"  Validated: {written:,} lines == {pf.metadata.num_rows:,} impressions")

    # -- zip -------------------------------------------------------------------
    zip_out = BASE_DIR / "ebnerd_large_reranker_submission.zip"
    logger.info(f"[5/5] Zipping -> {zip_out.name} ...")
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname="predictions.txt")
    logger.info(f"  Done: {zip_out}  ({zip_out.stat().st_size/1e6:.1f} MB)")
    logger.info("=" * 62)
    logger.info("Upload this file to the RecSys 2024 Codabench leaderboard:")
    logger.info(f"  {zip_out}")
    logger.info("=" * 62)


if __name__ == "__main__":
    main()
