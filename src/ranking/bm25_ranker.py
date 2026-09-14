"""
bm25_ranker.py -- Q2: BM25 Lexical Retrieval

BM25Ranker wraps rank_bm25.BM25Okapi and exposes two modes:
  score_candidates(query, candidate_ids)
      Score a specific list of article IDs (used for per-impression reranking
      in Q4: AUC, MRR, nDCG).

  retrieve_topk(query, k)
      Score the entire article corpus and return the top-K IDs (used for
      recall@K in Q2).

Query construction
------------------
A user's query is the concatenated titles (+ subtitles) of their most recent
N clicked articles, filtered to clicks strictly before the impression timestamp.
This enforces the behaviour-window boundary from Q9.

Tokenization
------------
Simple regex: keep alphabetic tokens of length >= 2, lowercased.  Works for
both English (MIND) and Danish (EB-NeRD) without installing language-specific
tokenizers, keeping the pipeline self-contained.

Cold-start handling
-------------------
Users with no click history before the impression time get an empty query and
therefore score 0.0 on all articles (they appear at the bottom of the ranking).
"""

import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# Import schema constants (relative import, package is src/)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.schema import (
    COL_ARTICLE_ID,
    COL_CLICK_TIME,
    COL_IMPRESSION_TIME,
    COL_SUBTITLE,
    COL_TITLE,
    COL_USER_ID,
)

logger = logging.getLogger(__name__)

# Regex: keep word chars (covers Latin + Nordic letters via \w)
_TOKEN_RE = re.compile(r'\b\w{2,}\b')


def tokenize(text: str) -> list:
    """Lowercase + word-level tokenize.  Returns [] for empty / NaN input."""
    if not text or not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


class BM25Ranker:
    """
    BM25 ranker over an article corpus.

    Parameters
    ----------
    articles_df : pd.DataFrame
        Must contain COL_ARTICLE_ID, COL_TITLE, COL_SUBTITLE columns.
    max_history : int
        Maximum number of recently clicked articles to include in the query.
    """

    def __init__(self, articles_df: pd.DataFrame, max_history: int = 50):
        self.articles_df  = articles_df.copy().reset_index(drop=True)
        self.max_history  = max_history
        self.article_ids  = None  # aligned list of article IDs
        self.id_to_idx    = None  # article_id -> corpus index
        self.bm25         = None  # BM25Okapi instance
        self._tokenized   = None  # raw tokenized corpus (kept for inspection)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _article_text(self, row) -> str:
        """Concatenate title and subtitle for one article row."""
        parts = []
        title = row.get(COL_TITLE)
        sub   = row.get(COL_SUBTITLE)
        if title and isinstance(title, str):
            parts.append(title)
        if sub and isinstance(sub, str):
            parts.append(sub)
        return " ".join(parts)

    def build(self) -> "BM25Ranker":
        """Tokenize the corpus and fit BM25Okapi.  Returns self for chaining."""
        logger.info(f"Tokenizing {len(self.articles_df):,} articles …")
        texts = self.articles_df.apply(self._article_text, axis=1)
        self._tokenized = [tokenize(t) for t in texts]

        self.article_ids = self.articles_df[COL_ARTICLE_ID].tolist()
        self.id_to_idx   = {aid: i for i, aid in enumerate(self.article_ids)}

        # Precompute article_id → plain text string for fast query building
        # (avoids expensive DataFrame merge inside make_query)
        self.article_text_map = {
            row[COL_ARTICLE_ID]: self._article_text(row)
            for _, row in self.articles_df.iterrows()
        }

        logger.info("Fitting BM25Okapi …")
        self.bm25 = BM25Okapi(self._tokenized)
        logger.info("BM25 index ready.")
        return self

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    @staticmethod
    def preindex_history(history_df: pd.DataFrame) -> dict:
        """
        Pre-group history by user_id into a dict for O(1) per-user lookup.
        Call this once before looping over impressions.

        Returns
        -------
        dict {user_id: pd.DataFrame}  sorted by click_time descending
        """
        idx = {}
        for uid, grp in history_df.groupby(COL_USER_ID, sort=False):
            idx[uid] = grp.sort_values(COL_CLICK_TIME, ascending=False)
        return idx

    def make_query(
        self,
        user_id: str,
        history_df: pd.DataFrame,
        impression_time,
        history_index: dict = None,
    ) -> list:
        """
        Build query tokens from user click history before *impression_time*.

        Parameters
        ----------
        history_index : dict, optional
            Pre-built index from preindex_history().  When provided, lookup
            is O(1) instead of O(N) scan over the full history DataFrame.

        Returns
        -------
        list of str tokens  (empty list if user has no history)
        """
        if history_index is not None:
            user_df = history_index.get(user_id)
            if user_df is None or len(user_df) == 0:
                return []
            user_hist = user_df[
                user_df[COL_CLICK_TIME].notna() &
                (user_df[COL_CLICK_TIME] < impression_time)
            ].head(self.max_history)
        else:
            mask = (
                (history_df[COL_USER_ID] == user_id) &
                history_df[COL_CLICK_TIME].notna() &
                (history_df[COL_CLICK_TIME] < impression_time)
            )
            user_hist = (
                history_df[mask]
                .sort_values(COL_CLICK_TIME, ascending=False)
                .head(self.max_history)
            )

        if len(user_hist) == 0:
            return []

        # Fast O(1) dict lookup — no DataFrame merge
        text_parts = [
            self.article_text_map.get(aid, "")
            for aid in user_hist[COL_ARTICLE_ID]
            if self.article_text_map.get(aid, "")
        ]

        return tokenize(" ".join(text_parts))[:100]  # cap query length for BM25 speed


    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_doc(self, query_tokens: list, doc_idx: int) -> float:
        """
        Compute the BM25 score for a single document using the BM25Okapi formula.
        Much faster than get_scores() when only a few candidates need scoring,
        because it avoids scanning the entire 65K-document corpus.
        """
        bm25 = self.bm25
        doc_freq = bm25.doc_freqs[doc_idx]
        doc_len  = bm25.doc_len[doc_idx]
        k1, b    = bm25.k1, bm25.b
        avgdl    = bm25.avgdl

        score = 0.0
        for q in query_tokens:
            tf = doc_freq.get(q, 0)
            if tf == 0:
                continue
            idf = bm25.idf.get(q, 0.0)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1.0 - b + b * doc_len / avgdl))
        return score

    def score_candidates(
        self,
        query_tokens: list,
        candidate_ids: list,
    ) -> dict:
        """
        Score a specific list of article IDs using BM25 per-doc formula.
        Cost: O(|query| × |candidates|) instead of O(|query| × |corpus|).
        For 37 candidates vs 65K corpus this is ~1750× faster than get_scores().

        Returns
        -------
        dict {article_id: score}  — articles not in the index get score 0.0
        """
        if not query_tokens or not candidate_ids:
            return {aid: 0.0 for aid in candidate_ids}

        return {
            aid: self._score_doc(query_tokens, self.id_to_idx[aid])
            if aid in self.id_to_idx
            else 0.0
            for aid in candidate_ids
        }

    def retrieve_topk(self, query_tokens: list, k: int) -> list:
        """
        Score ALL articles and return the top-K article_ids.
        Used for recall@K computation in Q2.

        Returns
        -------
        list of article_id str, length <= k
        """
        if not query_tokens:
            return []

        scores   = self.bm25.get_scores(query_tokens)
        topk_idx = np.argsort(scores)[::-1][:k]
        return [self.article_ids[i] for i in topk_idx]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"BM25Ranker saved → {path}  ({path.stat().st_size / 1e6:.1f} MB)")

    @classmethod
    def load(cls, path: Path) -> "BM25Ranker":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"BM25Ranker loaded from {path}")
        return obj
