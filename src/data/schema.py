"""
schema.py — Unified column name constants shared by every module.

Both MIND and EB-NeRD are normalised into three tables:
  articles    — article metadata
  history     — per-user click history (one click per row)
  impressions — candidate articles per impression with click labels
"""

# ── Articles ──────────────────────────────────────────────────────────────────
COL_ARTICLE_ID    = "article_id"
COL_DATASET       = "dataset"
COL_TITLE         = "title"
COL_SUBTITLE      = "subtitle"
COL_BODY          = "body"
COL_CATEGORY      = "category"
COL_PUBLISHED_TIME = "published_time"

ARTICLES_COLUMNS = [
    COL_ARTICLE_ID,
    COL_DATASET,
    COL_TITLE,
    COL_SUBTITLE,
    COL_BODY,
    COL_CATEGORY,
    COL_PUBLISHED_TIME,
]

# ── History ───────────────────────────────────────────────────────────────────
COL_USER_ID   = "user_id"
COL_CLICK_TIME = "click_time"
# EB-NeRD-only dwell-time signals (raw columns: read_time_fixed, scroll_percentage_fixed).
# Not in HISTORY_COLUMNS: the unified history schema is shared with MIND, which
# has no equivalent field. BehaviouralFeatureExtractor checks for their presence
# (`has_dwell`) rather than assuming every history_df carries them.
COL_READ_TIME  = "read_time_fixed"
COL_SCROLL_PCT = "scroll_percentage_fixed"

HISTORY_COLUMNS = [
    COL_USER_ID,
    COL_DATASET,
    COL_ARTICLE_ID,
    COL_CLICK_TIME,
]

# ── Impressions ───────────────────────────────────────────────────────────────
COL_IMPRESSION_ID   = "impression_id"
COL_IMPRESSION_TIME = "impression_time"
COL_LABEL           = "label"
COL_SPLIT           = "split"

IMPRESSIONS_COLUMNS = [
    COL_IMPRESSION_ID,
    COL_USER_ID,
    COL_DATASET,
    COL_IMPRESSION_TIME,
    COL_ARTICLE_ID,
    COL_LABEL,
    COL_SPLIT,
]

# ── Dataset identifiers ────────────────────────────────────────────────────────
DATASET_MIND   = "MIND"
DATASET_EBNERD = "EBNERD"

# ── Split names ────────────────────────────────────────────────────────────────
SPLIT_TRAIN = "train"
SPLIT_VAL   = "val"
SPLIT_TEST  = "test"
