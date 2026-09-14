"""
behavioural_features.py — Q1: Click-History & Session Features

Engineers behavioural features from click-logs for both datasets:

  Click-history   user_click_count, user_hist_recency_sum (exponential decay)
  Session         session_clicks_1h, session_clicks_24h
  Dwell time      hist_mean_read_time, hist_mean_scroll  (EB-NeRD only)
  Article         article_popularity, freshness_days, category_match, cat_affinity

Behaviour-window boundary (Q1.4)
--------------------------------
Every feature for an impression at time T is computed from clicks strictly
before T. History is pre-sorted per user, so the boundary is a binary search
(`np.searchsorted`) rather than a filter over the whole table.

Only the most recent `max_history` clicks before T are used. This cap is not
cosmetic: MIND's test `behaviors.tsv` gives at most a truncated history string,
so training on an uncapped history would fit a user representation that cannot
be rebuilt at serving time.

Position bias (Q1.2)
--------------------
Deliberately not a feature. Candidate lists in both MIND and EB-NeRD are
shuffled, not display-ordered: once list length is held fixed, click-through
rate is flat across positions. The decay visible when positions are pooled is a
list-length artifact (long lists have a lower per-candidate rate by
construction). See `scripts/analyse_position_bias.py`.
"""

import logging

import numpy as np
import pandas as pd

from src.data.schema import (
    COL_ARTICLE_ID,
    COL_CATEGORY,
    COL_CLICK_TIME,
    COL_IMPRESSION_ID,
    COL_IMPRESSION_TIME,
    COL_PUBLISHED_TIME,
    COL_READ_TIME,
    COL_SCROLL_PCT,
    COL_USER_ID,
)

logger = logging.getLogger(__name__)

NS_PER_DAY = 86_400_000_000_000
NS_PER_HOUR = 3_600_000_000_000


def _to_ns(series) -> np.ndarray:
    """Datetime series → int64 nanoseconds, timezone-stripped."""
    s = pd.to_datetime(series, errors="coerce")
    if hasattr(s.dtype, "tz") and s.dtype.tz is not None:
        s = s.dt.tz_localize(None)
    return s.to_numpy(dtype="datetime64[ns]").astype(np.int64)


class BehaviouralFeatureExtractor:
    def __init__(
        self,
        history_df: pd.DataFrame,
        articles_df: pd.DataFrame,
        article_popularity=None,
        max_history: int = 50,
        half_life_days: float = 7.0,
    ):
        self.max_history = max_history
        self.half_life_days = half_life_days

        aid = articles_df[COL_ARTICLE_ID].astype(str)
        cats = (
            articles_df[COL_CATEGORY].fillna("").astype(str)
            if COL_CATEGORY in articles_df.columns
            else pd.Series("", index=articles_df.index)
        )
        self.cat_map = dict(zip(aid, cats))

        if COL_PUBLISHED_TIME in articles_df.columns:
            pub_ns = _to_ns(articles_df[COL_PUBLISHED_TIME])
            self.pub_map = {a: (np.nan if p == np.iinfo(np.int64).min else float(p))
                            for a, p in zip(aid, pub_ns)}
        else:
            self.pub_map = {}

        if article_popularity is None:
            self.pop_map = {}
        elif isinstance(article_popularity, pd.Series):
            self.pop_map = {str(k): float(v) for k, v in article_popularity.items()}
        else:
            self.pop_map = {str(k): float(v) for k, v in dict(article_popularity).items()}

        self.has_dwell = COL_READ_TIME in history_df.columns
        self._build_history_index(history_df)

    # ------------------------------------------------------------------

    def _build_history_index(self, history_df: pd.DataFrame) -> None:
        """Pre-sort history per user into numpy arrays for O(log n) windowing."""
        h = history_df[[COL_USER_ID, COL_ARTICLE_ID, COL_CLICK_TIME]
                       + ([COL_READ_TIME, COL_SCROLL_PCT] if self.has_dwell else [])].copy()
        h[COL_CLICK_TIME] = pd.to_datetime(h[COL_CLICK_TIME], errors="coerce")
        if hasattr(h[COL_CLICK_TIME].dtype, "tz") and h[COL_CLICK_TIME].dtype.tz is not None:
            h[COL_CLICK_TIME] = h[COL_CLICK_TIME].dt.tz_localize(None)

        # A click with no timestamp cannot be placed relative to the impression,
        # so it cannot be shown to be in-window and is dropped.
        h = h.dropna(subset=[COL_CLICK_TIME])
        h = h.sort_values([COL_USER_ID, COL_CLICK_TIME], kind="mergesort")

        if len(h) == 0:
            self._hist = {}
            return

        uids = h[COL_USER_ID].astype(str).to_numpy()
        times = _to_ns(h[COL_CLICK_TIME])
        arts = h[COL_ARTICLE_ID].astype(str).to_numpy()
        cats = np.array([self.cat_map.get(a, "") for a in arts], dtype=object)
        reads = h[COL_READ_TIME].to_numpy(dtype=float) if self.has_dwell else None
        scrolls = h[COL_SCROLL_PCT].to_numpy(dtype=float) if self.has_dwell else None

        starts = np.flatnonzero(np.r_[True, uids[1:] != uids[:-1]])
        ends = np.r_[starts[1:], len(uids)]

        self._hist = {
            uids[s]: (
                times[s:e], arts[s:e], cats[s:e],
                reads[s:e] if reads is not None else None,
                scrolls[s:e] if scrolls is not None else None,
            )
            for s, e in zip(starts, ends)
        }
        logger.info(f"History index: {len(self._hist):,} users, {len(h):,} clicks")

    # ------------------------------------------------------------------

    def extract_features(self, impressions_df: pd.DataFrame, half_life_days: float = None) -> pd.DataFrame:
        """
        Return `impressions_df` with feature columns appended, same rows, same order.
        """
        half_life = half_life_days or self.half_life_days
        out = impressions_df.copy()
        n = len(out)

        imp_ns = _to_ns(out[COL_IMPRESSION_TIME])
        users = out[COL_USER_ID].astype(str).to_numpy()
        cands = out[COL_ARTICLE_ID].astype(str).to_numpy()

        click_count   = np.zeros(n, dtype=np.int32)
        recency_sum   = np.zeros(n, dtype=np.float32)
        sess_1h       = np.zeros(n, dtype=np.int32)
        sess_24h      = np.zeros(n, dtype=np.int32)
        cat_match     = np.zeros(n, dtype=np.int8)
        cat_affinity  = np.zeros(n, dtype=np.float32)
        freshness     = np.full(n, -1.0, dtype=np.float32)
        popularity    = np.zeros(n, dtype=np.float32)
        mean_read     = np.full(n, np.nan, dtype=np.float32)
        mean_scroll   = np.full(n, np.nan, dtype=np.float32)

        groups = out.groupby(COL_IMPRESSION_ID, sort=False).indices

        for idx in groups.values():
            first = idx[0]
            uid = users[first]
            t = imp_ns[first]

            entry = self._hist.get(uid)
            cat_counts = {}
            n_cat = 0

            if entry is not None:
                times, _arts, cats, reads, scrolls = entry

                # Behaviour-window boundary: strictly before the impression.
                k = int(np.searchsorted(times, t, side="left"))
                lo = max(0, k - self.max_history)

                if k > lo:
                    w_times = times[lo:k]
                    age_days = (t - w_times) / NS_PER_DAY

                    click_count[idx] = k - lo
                    recency_sum[idx] = np.exp(-np.log(2) * age_days / half_life).sum()
                    sess_1h[idx]  = int(((t - w_times) <= NS_PER_HOUR).sum())
                    sess_24h[idx] = int(((t - w_times) <= 24 * NS_PER_HOUR).sum())

                    w_cats = cats[lo:k]
                    uniq, cnt = np.unique(w_cats.astype(str), return_counts=True)
                    cat_counts = dict(zip(uniq.tolist(), cnt.tolist()))
                    cat_counts.pop("", None)
                    n_cat = sum(cat_counts.values())

                    if reads is not None:
                        w_read = reads[lo:k]
                        w_scroll = scrolls[lo:k]
                        if np.isfinite(w_read).any():
                            mean_read[idx] = np.nanmean(w_read)
                        if np.isfinite(w_scroll).any():
                            mean_scroll[idx] = np.nanmean(w_scroll)

            for i in idx:
                cid = cands[i]
                ccat = self.cat_map.get(cid, "")
                cat_match[i] = 1 if ccat and ccat in cat_counts else 0
                cat_affinity[i] = (cat_counts.get(ccat, 0) / n_cat) if n_cat else 0.0
                popularity[i] = self.pop_map.get(cid, 0.0)

                pub = self.pub_map.get(cid, np.nan)
                if pub == pub:  # not NaN
                    freshness[i] = max(0.0, (t - pub) / NS_PER_DAY)

        out["user_click_count"]      = click_count
        out["user_hist_recency_sum"] = recency_sum
        out["session_clicks_1h"]     = sess_1h
        out["session_clicks_24h"]    = sess_24h
        out["category_match"]        = cat_match
        out["cat_affinity"]          = cat_affinity
        out["freshness_days"]        = freshness
        out["article_popularity"]    = popularity
        if self.has_dwell:
            out["hist_mean_read_time"] = mean_read
            out["hist_mean_scroll"]    = mean_scroll

        return out
