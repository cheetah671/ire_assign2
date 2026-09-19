"""
parse_ebnerd.py — Parse EB-NeRD Parquet files into the unified schema.

Confirmed column names from inspecting the actual demo zip:

articles.parquet:
    article_id (int32), title, subtitle, body, category (int16),
    category_str, published_time

train/behaviors.parquet:
    impression_id (uint32), user_id (uint32), impression_time,
    article_ids_inview (numpy array of int32),
    article_ids_clicked (numpy array of int32)

train/history.parquet:
    user_id (uint32),
    article_id_fixed      (numpy array of int32)  — past clicked articles
    impression_time_fixed (numpy array of datetime64) — click timestamps

All article/user IDs are stored as integers in parquet; we cast to str for the
unified schema so MIND (e.g. "N12345") and EB-NeRD ("9774516") IDs remain
unambiguous when the `dataset` column is present.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    ARTICLES_COLUMNS,
    COL_ARTICLE_ID,
    COL_BODY,
    COL_CATEGORY,
    COL_CLICK_TIME,
    COL_DATASET,
    COL_IMPRESSION_ID,
    COL_IMPRESSION_TIME,
    COL_LABEL,
    COL_PUBLISHED_TIME,
    COL_SPLIT,
    COL_SUBTITLE,
    COL_TITLE,
    COL_USER_ID,
    DATASET_EBNERD,
    HISTORY_COLUMNS,
    IMPRESSIONS_COLUMNS,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_TEST,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Articles
# ─────────────────────────────────────────────────────────────────────────────

def _parse_articles(path: Path) -> pd.DataFrame:
    """Read articles.parquet and return a DataFrame in the unified articles schema."""
    raw = pd.read_parquet(path)
    logger.info(f"EB-NeRD articles raw columns: {list(raw.columns)}")

    out = pd.DataFrame()
    out[COL_ARTICLE_ID] = raw["article_id"].astype(str)
    out[COL_DATASET]    = DATASET_EBNERD
    out[COL_TITLE]      = raw["title"].fillna("") if "title" in raw.columns else ""
    out[COL_SUBTITLE]   = raw["subtitle"].fillna("") if "subtitle" in raw.columns else ""
    out[COL_BODY]       = raw["body"].fillna("") if "body" in raw.columns else np.nan

    # Prefer human-readable category_str; fall back to numeric category
    if "category_str" in raw.columns:
        out[COL_CATEGORY] = raw["category_str"].fillna("unknown").astype(str)
    elif "category" in raw.columns:
        out[COL_CATEGORY] = raw["category"].astype(str).fillna("unknown")
    else:
        out[COL_CATEGORY] = "unknown"

    if "published_time" in raw.columns:
        out[COL_PUBLISHED_TIME] = pd.to_datetime(raw["published_time"], errors="coerce")
    else:
        out[COL_PUBLISHED_TIME] = pd.NaT

    return out[ARTICLES_COLUMNS]


# ─────────────────────────────────────────────────────────────────────────────
# One split (train or validation)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_split(split_dir: Path, split_name: str):
    """
    Parse behaviors.parquet and history.parquet for one EB-NeRD split.

    Returns
    -------
    history_df     : pd.DataFrame  (HISTORY_COLUMNS)
    impressions_df : pd.DataFrame  (IMPRESSIONS_COLUMNS)
    """
    behaviors_path = split_dir / "behaviors.parquet"
    history_path   = split_dir / "history.parquet"

    if not behaviors_path.exists():
        raise FileNotFoundError(f"behaviors.parquet not found: {behaviors_path}")

    # ── Impressions ──────────────────────────────────────────────────────────
    beh = pd.read_parquet(behaviors_path)
    logger.info(f"EB-NeRD behaviors [{split_name}] shape: {beh.shape}")

    beh["impression_time"] = pd.to_datetime(beh["impression_time"], errors="coerce")

    # article_ids_inview is stored as a numpy array per row → explode
    beh_exp = beh[
        ["impression_id", "user_id", "impression_time",
         "article_ids_inview", "article_ids_clicked"]
    ].copy()

    # Convert array cells to lists so pandas .explode works cleanly
    beh_exp["article_ids_inview"] = beh_exp["article_ids_inview"].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else (x if isinstance(x, list) else [])
    )
    beh_exp["article_ids_clicked"] = beh_exp["article_ids_clicked"].apply(
        lambda x: set(x.tolist()) if isinstance(x, np.ndarray) else (set(x) if isinstance(x, list) else set())
    )

    # Explode one row per candidate article
    inview = beh_exp.explode("article_ids_inview").dropna(subset=["article_ids_inview"])

    # Vectorised labelling: merge clicked sets
    inview = inview.copy()
    inview["label"] = inview.apply(
        lambda row: 1 if row["article_ids_inview"] in row["article_ids_clicked"] else 0,
        axis=1,
    )

    impressions_df = pd.DataFrame({
        COL_IMPRESSION_ID:   inview["impression_id"].astype(str),
        COL_USER_ID:         inview["user_id"].astype(str),
        COL_DATASET:         DATASET_EBNERD,
        COL_IMPRESSION_TIME: inview["impression_time"],
        COL_ARTICLE_ID:      inview["article_ids_inview"].astype(str),
        COL_LABEL:           inview["label"].astype(int),
        COL_SPLIT:           split_name,
    })[IMPRESSIONS_COLUMNS].reset_index(drop=True)

    # ── History ──────────────────────────────────────────────────────────────
    history_df = pd.DataFrame(columns=HISTORY_COLUMNS)

    if history_path.exists():
        hist = pd.read_parquet(history_path)
        logger.info(f"EB-NeRD history [{split_name}] shape: {hist.shape}")

        # article_id_fixed and impression_time_fixed are parallel numpy arrays
        rows = []
        for _, row in hist.iterrows():
            uid       = str(row["user_id"])
            art_ids   = row["article_id_fixed"]
            click_ts  = row["impression_time_fixed"]

            if art_ids is None or len(art_ids) == 0:
                continue

            # Pair each article_id with its click timestamp
            if click_ts is None or len(click_ts) != len(art_ids):
                click_ts = [pd.NaT] * len(art_ids)

            for art_id, ts in zip(art_ids, click_ts):
                rows.append({
                    COL_USER_ID:   uid,
                    COL_DATASET:   DATASET_EBNERD,
                    COL_ARTICLE_ID: str(int(art_id)),
                    COL_CLICK_TIME: pd.to_datetime(ts, errors="coerce") if ts is not None else pd.NaT,
                })

        if rows:
            history_df = pd.DataFrame(rows, columns=HISTORY_COLUMNS)

    logger.info(
        f"EB-NeRD [{split_name}]: "
        f"{len(history_df):,} history rows, "
        f"{len(impressions_df):,} impression rows"
    )
    return history_df, impressions_df


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_ebnerd(bundle_dir: Path):
    """
    Parse one EB-NeRD bundle (demo or small).

    Parameters
    ----------
    bundle_dir : Path   root directory of the extracted bundle
                        (contains articles.parquet, train/, validation/)

    Returns
    -------
    articles_df    : pd.DataFrame
    history_df     : pd.DataFrame
    impressions_df : pd.DataFrame
    """
    articles_path = bundle_dir / "articles.parquet"
    if not articles_path.exists():
        raise FileNotFoundError(f"articles.parquet not found: {articles_path}")

    logger.info("Parsing EB-NeRD articles …")
    articles_df = _parse_articles(articles_path)
    logger.info(f"EB-NeRD articles: {len(articles_df):,} rows")

    all_hist = []
    all_imp  = []

    for split_name, subdir in [(SPLIT_TRAIN, "train"), (SPLIT_VAL, "validation")]:
        split_dir = bundle_dir / subdir
        if not split_dir.exists():
            logger.warning(f"Split directory not found, skipping: {split_dir}")
            continue
        h, i = _parse_split(split_dir, split_name)
        
        if split_name == SPLIT_VAL:
            # ── Chronological Split of Validation into Val and Test ──
            i = i.sort_values(COL_IMPRESSION_TIME)
            unique_imps = i[COL_IMPRESSION_ID].unique()
            split_idx = len(unique_imps) // 2
            
            val_imps = set(unique_imps[:split_idx])
            is_test = ~i[COL_IMPRESSION_ID].isin(val_imps)
            i.loc[is_test, COL_SPLIT] = SPLIT_TEST
            
        all_hist.append(h)
        all_imp.append(i)

    history_df     = pd.concat(all_hist, ignore_index=True) if all_hist else pd.DataFrame(columns=HISTORY_COLUMNS)
    impressions_df = pd.concat(all_imp,  ignore_index=True) if all_imp  else pd.DataFrame(columns=IMPRESSIONS_COLUMNS)

    # Deduplicate history (same user may appear in both splits)
    history_df = history_df.drop_duplicates(subset=[COL_USER_ID, COL_ARTICLE_ID]).reset_index(drop=True)

    logger.info(
        f"EB-NeRD total: {len(articles_df):,} articles | "
        f"{len(history_df):,} history | "
        f"{len(impressions_df):,} impressions"
    )
    return articles_df, history_df, impressions_df
