"""
parse_mind.py — Parse MIND-small raw TSV files into the unified schema.

Raw file format (no header row, tab-separated):
  news.tsv:      news_id | category | subcategory | title | abstract | url | title_entities | abstract_entities
  behaviors.tsv: impression_id | user_id | time | history | impressions
                   history    = space-separated news_ids clicked before this impression
                   impressions = space-separated "news_id-label" pairs (1=clicked, 0=not)

MIND does not provide per-click timestamps for the history field.  We use
  click_time = impression_time - 1 second
as a conservative proxy (guarantees click_time < impression_time for every row,
which satisfies the leakage test in Q9).  For any user that appears in multiple
impression rows we keep the *earliest* proxy time for each article, which is the
best estimate of when the article was first known to be clicked.
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
    DATASET_MIND,
    HISTORY_COLUMNS,
    IMPRESSIONS_COLUMNS,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_TEST,
)

logger = logging.getLogger(__name__)

# Column names for the raw TSV files (no header in original files)
_NEWS_COLS = [
    "news_id", "category", "subcategory", "title", "subtitle",
    "url", "title_entities", "abstract_entities",
]
_BEHAVIORS_COLS = ["impression_id", "user_id", "time", "history", "impressions"]

# MIND timestamp format: "11/11/2019 9:05:58 AM"
_MIND_TIME_FMT = "%m/%d/%Y %I:%M:%S %p"


# ─────────────────────────────────────────────────────────────────────────────
# Articles
# ─────────────────────────────────────────────────────────────────────────────

def _parse_news(path: Path) -> pd.DataFrame:
    """Read news.tsv and return a DataFrame in the unified articles schema."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=_NEWS_COLS,
        usecols=["news_id", "category", "title", "subtitle"],
        dtype=str,
    )
    out = pd.DataFrame()
    out[COL_ARTICLE_ID]     = df["news_id"]
    out[COL_DATASET]        = DATASET_MIND
    out[COL_TITLE]          = df["title"].fillna("")
    out[COL_SUBTITLE]       = df["subtitle"].fillna("")
    out[COL_BODY]           = np.nan          # MIND does not ship body text
    out[COL_CATEGORY]       = df["category"].fillna("unknown")
    out[COL_PUBLISHED_TIME] = pd.NaT          # MIND does not provide pub time
    return out[ARTICLES_COLUMNS]


# ─────────────────────────────────────────────────────────────────────────────
# Behaviors → history + impressions
# ─────────────────────────────────────────────────────────────────────────────

def _parse_behaviors(path: Path, split: str):
    """
    Parse behaviors.tsv for one MIND split.

    Returns
    -------
    history_df : pd.DataFrame   (HISTORY_COLUMNS)
    impressions_df : pd.DataFrame  (IMPRESSIONS_COLUMNS)
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=_BEHAVIORS_COLS,
        dtype={"impression_id": str, "user_id": str,
               "time": str, "history": str, "impressions": str},
    )

    # Parse timestamps
    df[COL_IMPRESSION_TIME] = pd.to_datetime(
        df["time"], format=_MIND_TIME_FMT, errors="coerce"
    )
    df["history"]     = df["history"].fillna("").str.strip()
    df["impressions"] = df["impressions"].fillna("").str.strip()

    # ── History ──────────────────────────────────────────────────────────────
    df_hist = df[df["history"] != ""].copy()
    df_hist["history_list"] = df_hist["history"].str.split()

    hist_expanded = (
        df_hist[["user_id", COL_IMPRESSION_TIME, "history_list"]]
        .explode("history_list")
        .rename(columns={"history_list": COL_ARTICLE_ID, "user_id": COL_USER_ID})
    )
    # proxy click_time = impression_time - 1 s  →  guarantees < impression_time
    hist_expanded[COL_CLICK_TIME] = (
        hist_expanded[COL_IMPRESSION_TIME] - pd.Timedelta(seconds=1)
    )
    hist_expanded[COL_DATASET] = DATASET_MIND
    hist_expanded = hist_expanded[[COL_USER_ID, COL_DATASET, COL_ARTICLE_ID, COL_CLICK_TIME]]

    # Keep earliest proxy time per (user, article) — best estimate of first click
    history_df = (
        hist_expanded
        .groupby([COL_USER_ID, COL_DATASET, COL_ARTICLE_ID], as_index=False)
        [COL_CLICK_TIME]
        .min()
    )[HISTORY_COLUMNS].copy()

    # ── Impressions ──────────────────────────────────────────────────────────
    df_imp = df[df["impressions"] != ""].copy()
    df_imp["imp_list"] = df_imp["impressions"].str.split()

    imp_expanded = (
        df_imp[["impression_id", "user_id", COL_IMPRESSION_TIME, "imp_list"]]
        .explode("imp_list")
    )

    # "N1234-0" → article_id="N1234", label=0
    split_cols = imp_expanded["imp_list"].str.rsplit("-", n=1, expand=True)
    imp_expanded = imp_expanded.copy()
    imp_expanded[COL_ARTICLE_ID] = split_cols[0]
    imp_expanded[COL_LABEL]      = pd.to_numeric(split_cols[1], errors="coerce").fillna(0).astype(int)
    imp_expanded[COL_IMPRESSION_ID] = imp_expanded["impression_id"].astype(str)
    imp_expanded[COL_USER_ID]       = imp_expanded["user_id"]
    imp_expanded[COL_DATASET]       = DATASET_MIND
    imp_expanded[COL_SPLIT]         = split

    impressions_df = imp_expanded[IMPRESSIONS_COLUMNS].dropna(subset=[COL_ARTICLE_ID]).copy()

    logger.info(
        f"MIND [{split}]: "
        f"{len(history_df):,} history rows, "
        f"{len(impressions_df):,} impression rows"
    )
    return history_df, impressions_df


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_mind(train_dir: Path, dev_dir: Path):
    """
    Parse the full MIND-small corpus (train + dev splits).

    Parameters
    ----------
    train_dir : Path   directory extracted from MINDsmall_train.zip
    dev_dir   : Path   directory extracted from MINDsmall_dev.zip

    Returns
    -------
    articles_df    : pd.DataFrame
    history_df     : pd.DataFrame
    impressions_df : pd.DataFrame
    """
    logger.info("Parsing MIND articles …")
    # Articles exist in both splits; merge and deduplicate
    train_articles = _parse_news(train_dir / "news.tsv")
    dev_articles   = _parse_news(dev_dir   / "news.tsv")
    articles_df = (
        pd.concat([train_articles, dev_articles], ignore_index=True)
        .drop_duplicates(subset=[COL_ARTICLE_ID], keep="first")
        .reset_index(drop=True)
    )
    logger.info(f"MIND articles: {len(articles_df):,} unique articles")

    logger.info("Parsing MIND train behaviors …")
    train_hist, train_imp = _parse_behaviors(train_dir / "behaviors.tsv", SPLIT_TRAIN)

    logger.info("Parsing MIND dev behaviors …")
    dev_hist, dev_imp = _parse_behaviors(dev_dir / "behaviors.tsv", SPLIT_VAL)

    # ── Chronological Split of Dev into Val and Test ──
    dev_imp = dev_imp.sort_values(COL_IMPRESSION_TIME)
    unique_imps = dev_imp[COL_IMPRESSION_ID].unique()
    split_idx = len(unique_imps) // 2
    
    val_imps = set(unique_imps[:split_idx])
    is_test = ~dev_imp[COL_IMPRESSION_ID].isin(val_imps)
    dev_imp.loc[is_test, COL_SPLIT] = SPLIT_TEST

    history_df = (
        pd.concat([train_hist, dev_hist], ignore_index=True)
        .groupby([COL_USER_ID, COL_DATASET, COL_ARTICLE_ID], as_index=False)
        [COL_CLICK_TIME].min()
    )[HISTORY_COLUMNS].copy()

    impressions_df = pd.concat([train_imp, dev_imp], ignore_index=True)

    logger.info(
        f"MIND total: {len(articles_df):,} articles | "
        f"{len(history_df):,} history | "
        f"{len(impressions_df):,} impressions"
    )
    return articles_df, history_df, impressions_df
