"""
parse_mind_large_test.py — Parse MIND-large test set into the unified schema.
"""

import logging
from pathlib import Path

import pandas as pd
from src.data.parse_mind import _parse_news, _parse_behaviors
from src.data.schema import SPLIT_TEST

logger = logging.getLogger(__name__)

def parse_mind_large_test(test_dir: Path):
    """
    Parse the MIND-large test corpus.

    Parameters
    ----------
    test_dir : Path   directory extracted from MINDlarge_test.zip

    Returns
    -------
    articles_df    : pd.DataFrame
    history_df     : pd.DataFrame
    impressions_df : pd.DataFrame
    """
    logger.info("Parsing MIND-large test articles …")
    articles_df = _parse_news(test_dir / "news.tsv")
    logger.info(f"MIND-large articles: {len(articles_df):,} unique articles")

    logger.info("Parsing MIND-large test behaviors …")
    history_df, impressions_df = _parse_behaviors(test_dir / "behaviors.tsv", SPLIT_TEST)

    logger.info(
        f"MIND-large test total: {len(articles_df):,} articles | "
        f"{len(history_df):,} history | "
        f"{len(impressions_df):,} impressions"
    )
    return articles_df, history_df, impressions_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dir = Path("data/raw/MIND/large_test/MINDlarge_test")
    articles, history, impressions = parse_mind_large_test(test_dir)
    
    out = Path("data/processed/MIND_LARGE_TEST")
    out.mkdir(parents=True, exist_ok=True)
    
    articles.to_parquet(out / "articles.parquet", index=False)
    history.to_parquet(out / "history.parquet", index=False)
    impressions.to_parquet(out / "impressions.parquet", index=False)
    
    logger.info(f"Saved to {out}")
