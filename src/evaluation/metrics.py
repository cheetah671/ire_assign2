"""
metrics.py — Q4: Offline Evaluation Metrics

Ranking metrics (per impression, then averaged):
    auc        — Area Under the ROC Curve
    mrr        — Mean Reciprocal Rank of the first clicked article
    ndcg_at_5  — nDCG at cutoff 5
    ndcg_at_10 — nDCG at cutoff 10

Beyond-accuracy metrics (over all top-K recommended articles):
    novelty    — mean −log2(article_popularity)
    coverage   — fraction of catalog covered at least once
    ild        — Intra-List Diversity (mean pairwise category distance)

All functions operate on a predictions DataFrame that must contain:
    impression_id, user_id, article_id, label, bm25_score (or emb_score),
    bm25_rank (or emb_rank)

and an articles DataFrame with article_id, category.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dcg(relevances: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain at cutoff k (binary relevance)."""
    rel = relevances[:k]
    if len(rel) == 0:
        return 0.0
    positions = np.arange(1, len(rel) + 1)
    return float(np.sum(rel / np.log2(positions + 1)))


def _ndcg(labels_sorted_by_score: np.ndarray, k: int) -> float:
    """
    nDCG@k for one impression.

    Parameters
    ----------
    labels_sorted_by_score : 1-D array of 0/1 labels, sorted by model score
                             descending (position 0 = highest ranked).
    k : cutoff
    """
    dcg_val  = _dcg(labels_sorted_by_score, k)
    # Ideal: sort labels descending (all 1s first)
    ideal    = np.sort(labels_sorted_by_score)[::-1]
    idcg_val = _dcg(ideal, k)
    if idcg_val == 0:
        return 0.0
    return dcg_val / idcg_val


# ─────────────────────────────────────────────────────────────────────────────
# Per-impression ranking metrics  (fully vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ranking_metrics(
    predictions: pd.DataFrame,
    score_col: str = "bm25_score",
    rank_col: str = "bm25_rank",
    impression_col: str = "impression_id",
    label_col: str = "label",
) -> Dict[str, float]:
    """
    Vectorised computation of AUC, MRR, nDCG@5, nDCG@10 across all impressions.

    Strategy
    --------
    1. Assign a within-impression rank (1 = highest score) using groupby + rank().
    2. Compute per-impression metrics using groupby aggregations (no Python loop).
    3. Average over impressions that have at least one positive label.

    AUC per impression = (# discordant pairs + 0.5 * tied pairs) / (pos * neg)
    which equals sklearn.roc_auc_score but computed via the Mann-Whitney U formula
    using sorted ranks — vectorised over all impressions simultaneously.
    """
    df = predictions[[impression_col, label_col, score_col]].copy()
    df[label_col] = df[label_col].astype(float)

    # ── Rank within each impression (ascending rank = highest score at rank 1) ─
    df["_rank"] = (
        df.groupby(impression_col)[score_col]
        .rank(ascending=False, method="first")
    )

    # ── Filter to impressions with at least one positive ──────────────────────
    pos_per_imp = df.groupby(impression_col)[label_col].sum()
    valid_imps  = pos_per_imp[pos_per_imp > 0].index
    df = df[df[impression_col].isin(valid_imps)]

    if len(df) == 0:
        return {"auc": 0.0, "mrr": 0.0, "ndcg5": 0.0, "ndcg10": 0.0}

    n_imp = len(valid_imps)

    # ── MRR ───────────────────────────────────────────────────────────────────
    # For each impression, find the minimum rank among positives
    pos_df  = df[df[label_col] == 1].copy()
    min_pos_rank = pos_df.groupby(impression_col)["_rank"].min()
    mrr = float((1.0 / min_pos_rank).mean())

    # ── nDCG@K ───────────────────────────────────────────────────────────────
    def _ndcg_vec(df_: pd.DataFrame, k: int) -> float:
        """Vectorised nDCG@k — uses log2 discount on sorted labels."""
        # DCG: only positions 1..k contribute
        top_k = df_[df_["_rank"] <= k].copy()
        top_k["_disc"] = np.log2(top_k["_rank"] + 1)
        top_k["_dcg"]  = top_k[label_col] / top_k["_disc"]
        dcg_per_imp = top_k.groupby(impression_col)["_dcg"].sum()

        # IDCG: sort actual labels desc within each impression, assign ideal ranks
        # Number of positives per impression (capped at k)
        n_pos = df_.groupby(impression_col)[label_col].sum().clip(upper=k)
        # IDCG for each impression: Σ 1/log2(i+1) for i in 1..n_pos
        # Precompute lookup table up to max possible n_pos (capped at k)
        max_n = int(n_pos.max()) if len(n_pos) > 0 else 0
        idcg_lut = {0: 0.0}
        for _i in range(1, max_n + 1):
            idcg_lut[_i] = idcg_lut[_i - 1] + 1.0 / np.log2(_i + 1)

        idcg_per_imp = n_pos.map(idcg_lut)

        # Align and compute nDCG
        dcg_per_imp  = dcg_per_imp.reindex(valid_imps, fill_value=0.0)
        idcg_per_imp = idcg_per_imp.reindex(valid_imps, fill_value=0.0)
        mask = idcg_per_imp > 0
        ndcg = (dcg_per_imp[mask] / idcg_per_imp[mask]).mean()
        return float(ndcg) if not np.isnan(ndcg) else 0.0

    ndcg5  = _ndcg_vec(df, k=5)
    ndcg10 = _ndcg_vec(df, k=10)

    # ── AUC (Mann-Whitney U formulation, vectorised) ──────────────────────────
    # For each impression:
    #   AUC = (sum of ranks of positives − n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    # where ranks are the score-based ranks (rank 1 = best)
    # We use the complement: convert rank to "score rank" as (n+1 - rank)
    imp_size    = df.groupby(impression_col)["_rank"].count().rename("n")
    n_pos_s     = df.groupby(impression_col)[label_col].sum().rename("n_pos")
    n_neg_s     = (imp_size - n_pos_s).rename("n_neg")

    # Sum of ascending score-ranks for positives (higher score = lower _rank = better)
    # Ascending score rank = n_total + 1 - _rank
    df["_asc_rank"] = df.groupby(impression_col)["_rank"].transform(
        lambda x: x.max() + 1 - x
    )
    pos_rank_sum = (
        df[df[label_col] == 1]
        .groupby(impression_col)["_asc_rank"]
        .sum()
    )

    auc_df = pd.concat([pos_rank_sum.rename("pos_rank_sum"), n_pos_s, n_neg_s], axis=1).dropna()
    auc_df = auc_df[auc_df["n_neg"] > 0]  # skip all-positive impressions
    if len(auc_df) > 0:
        u_stat = auc_df["pos_rank_sum"] - auc_df["n_pos"] * (auc_df["n_pos"] + 1) / 2
        auc_vals = u_stat / (auc_df["n_pos"] * auc_df["n_neg"])
        auc = float(auc_vals.mean())
    else:
        auc = 0.5

    return {"auc": auc, "mrr": mrr, "ndcg5": ndcg5, "ndcg10": ndcg10}


def impression_metrics(
    group: pd.DataFrame,
    score_col: str = "bm25_score",
    rank_col: str = "bm25_rank",
    label_col: str = "label",
) -> Optional[Dict[str, float]]:
    """
    Compute AUC, MRR, nDCG@5, nDCG@10 for a single impression group.
    Kept for backward compatibility / per-impression inspection.
    """
    labels = group[label_col].values.astype(float)
    scores = group[score_col].values.astype(float)
    if labels.sum() == 0:
        return None
    order        = np.argsort(scores)[::-1]
    labels_ranked = labels[order]
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5
    hit_positions = np.where(labels_ranked == 1)[0]
    mrr    = 1.0 / (hit_positions[0] + 1) if len(hit_positions) > 0 else 0.0
    ndcg5  = _ndcg(labels_ranked, k=5)
    ndcg10 = _ndcg(labels_ranked, k=10)
    return {"auc": auc, "mrr": mrr, "ndcg5": ndcg5, "ndcg10": ndcg10}


# ─────────────────────────────────────────────────────────────────────────────
# Beyond-accuracy metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_article_popularity(
    impressions_train: pd.DataFrame,
    article_col: str = "article_id",
    label_col: str = "label",
) -> pd.Series:
    """
    Compute article popularity = click count in train split.
    Returns a Series indexed by article_id (articles with 0 clicks get count 0).
    """
    pop = (
        impressions_train[impressions_train[label_col] == 1]
        .groupby(article_col)
        .size()
    )
    return pop


def novelty(
    top_k_per_impression: pd.DataFrame,
    popularity: pd.Series,
    article_col: str = "article_id",
    impression_col: str = "impression_id",
) -> float:
    """
    Novelty = mean over all recommended (impression, article) pairs of
              −log2(P(article)) where P = popularity / total_clicks.

    Unpopular articles → high novelty score.
    """
    total = popularity.sum()
    if total == 0:
        return 0.0

    # Map article_id → probability
    def _novelty_score(aid):
        pop_count = popularity.get(aid, 0)
        if pop_count == 0:
            # Never seen in train → maximum novelty; use 1/(total+1) as floor
            p = 1.0 / (total + 1)
        else:
            p = pop_count / total
        return -np.log2(p)

    scores = top_k_per_impression[article_col].map(_novelty_score)
    return float(scores.mean())


def coverage(
    top_k_per_impression: pd.DataFrame,
    total_catalog_size: int,
    article_col: str = "article_id",
) -> float:
    """
    Coverage = fraction of catalog recommended at least once across all impressions.
    """
    n_recommended = top_k_per_impression[article_col].nunique()
    return n_recommended / total_catalog_size if total_catalog_size > 0 else 0.0


def intra_list_diversity(
    predictions: pd.DataFrame,
    articles: pd.DataFrame,
    top_k: int = 10,
    score_col: str = "bm25_score",
    impression_col: str = "impression_id",
    article_col: str = "article_id",
    category_col: str = "category",
) -> float:
    """
    Intra-List Diversity (ILD) — mean pairwise category diversity within top-K
    recommended list per impression, then averaged across impressions.

    Diversity between two items = 0 if same category, 1 if different.
    ILD for a list of size n = (# pairs with different categories) / C(n, 2)

    Vectorised implementation: for each impression, count categories in top-K,
    then use the identity:
        diverse_pairs = C(n,2) − Σ C(count_c, 2)
    which avoids any Python-level nested loops.
    """
    # Build article → category mapping
    cat_map = articles.set_index(article_col)[category_col].to_dict()

    # Take top-K per impression (vectorised groupby head)
    top_k_df = (
        predictions
        .sort_values(score_col, ascending=False)
        .groupby(impression_col, sort=False)
        .head(top_k)
        [[impression_col, article_col]]
        .copy()
    )
    top_k_df["category"] = top_k_df[article_col].map(cat_map)
    top_k_df = top_k_df.dropna(subset=["category"])

    # Count items per impression
    n_per_imp = top_k_df.groupby(impression_col)["category"].count()

    # Count same-category pairs per impression using value_counts per group
    # C(k, 2) = k*(k-1)/2
    same_pairs = (
        top_k_df.groupby([impression_col, "category"])
        .size()
        .reset_index(name="cnt")
    )
    same_pairs["same"] = same_pairs["cnt"] * (same_pairs["cnt"] - 1) // 2
    same_per_imp = same_pairs.groupby(impression_col)["same"].sum()

    # Align
    n_per_imp   = n_per_imp.reindex(same_per_imp.index, fill_value=0)
    total_pairs = n_per_imp * (n_per_imp - 1) // 2
    diverse_pairs = total_pairs - same_per_imp

    # Only count impressions with at least 2 items
    mask = total_pairs >= 1
    if mask.sum() == 0:
        return 0.0

    ild = (diverse_pairs[mask] / total_pairs[mask]).mean()
    return float(ild)


def compute_beyond_accuracy_metrics(
    predictions: pd.DataFrame,
    articles: pd.DataFrame,
    impressions_train: pd.DataFrame,
    top_k: int = 10,
    score_col: str = "bm25_score",
    impression_col: str = "impression_id",
    article_col: str = "article_id",
    label_col: str = "label",
    category_col: str = "category",
) -> Dict[str, float]:
    """
    Compute Novelty, Coverage, and ILD on top-K recommended articles.

    Returns dict with keys: novelty, coverage, ild.
    """
    # Build top-K per impression
    top_k_rows = (
        predictions
        .sort_values(score_col, ascending=False)
        .groupby(impression_col)
        .head(top_k)
    )

    pop         = compute_article_popularity(impressions_train, article_col, label_col)
    catalog_size = articles[article_col].nunique()

    nov  = novelty(top_k_rows, pop, article_col, impression_col)
    cov  = coverage(top_k_rows, catalog_size, article_col)
    ild  = intra_list_diversity(predictions, articles, top_k=top_k,
                                score_col=score_col,
                                impression_col=impression_col,
                                article_col=article_col,
                                category_col=category_col)

    return {"novelty": nov, "coverage": cov, "ild": ild}
