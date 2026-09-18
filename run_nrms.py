"""
run_nrms.py — Q3.1: train and evaluate the reproduced NRMS baseline.

NRMS (Wu et al., 2019) is the official/starter neural baseline for both
leaderboards. Q3 asks for it to be reproduced first, then beaten by a principled
change — here the two-stage retrieve-then-rank pipeline in run_reranker.py.

Predictions are written for exactly the impressions that
`stage1_val_predictions.parquet` covers, so NRMS, the stage-1 retriever and the
re-ranker are all scored on the same impressions and the paired bootstrap in
run_evaluation.py is valid.

Behaviour-window boundary (Q1.4) is enforced the same way as the feature
extractor: a user's history is truncated by binary search to clicks strictly
before the impression timestamp.

Usage
-----
  python run_nrms.py --dataset mind
  python run_nrms.py --dataset mind --epochs 2 --train-imps 20000
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data.schema import (
    COL_ARTICLE_ID, COL_CLICK_TIME, COL_IMPRESSION_ID, COL_IMPRESSION_TIME,
    COL_LABEL, COL_SPLIT, COL_USER_ID, SPLIT_TRAIN, SPLIT_VAL,
)
from src.ranking.nrms import NRMS, build_vocab, encode_titles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_nrms")

BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PRED_DIR      = BASE_DIR / "data" / "predictions"
MODELS_DIR    = BASE_DIR / "models"


def _to_ns(series) -> np.ndarray:
    s = pd.to_datetime(series, errors="coerce")
    if hasattr(s.dtype, "tz") and s.dtype.tz is not None:
        s = s.dt.tz_localize(None)
    return s.to_numpy(dtype="datetime64[ns]").astype(np.int64)


def build_history_index(history: pd.DataFrame) -> dict:
    """{user_id: (click_times_ns sorted, article_row_indices)} for O(log n) windowing."""
    h = history[[COL_USER_ID, COL_ARTICLE_ID, COL_CLICK_TIME]].copy()
    h[COL_CLICK_TIME] = pd.to_datetime(h[COL_CLICK_TIME], errors="coerce")
    h = h.dropna(subset=[COL_CLICK_TIME])
    h = h.sort_values([COL_USER_ID, COL_CLICK_TIME], kind="mergesort")

    if len(h) == 0:
        return {}

    uids  = h[COL_USER_ID].astype(str).to_numpy()
    times = _to_ns(h[COL_CLICK_TIME])
    arts  = h[COL_ARTICLE_ID].astype(str).to_numpy()

    starts = np.flatnonzero(np.r_[True, uids[1:] != uids[:-1]])
    ends   = np.r_[starts[1:], len(uids)]
    return {uids[s]: (times[s:e], arts[s:e]) for s, e in zip(starts, ends)}


def user_history_rows(hist_idx, aid_to_row, uid, t_ns, max_history):
    """Row indices of the user's most recent clicks strictly before t_ns."""
    entry = hist_idx.get(uid)
    if entry is None:
        return []
    times, arts = entry
    k = int(np.searchsorted(times, t_ns, side="left"))
    lo = max(0, k - max_history)
    return [aid_to_row[a] for a in arts[lo:k] if a in aid_to_row]


class TrainSet(Dataset):
    """One sample per clicked article: 1 positive + npratio sampled negatives."""

    def __init__(self, samples, title_ids, max_history, npratio, seed=42):
        self.samples = samples
        self.title_ids = title_ids
        self.max_history = max_history
        self.npratio = npratio
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        hist_rows, pos_row, neg_rows = self.samples[i]

        if len(neg_rows) >= self.npratio:
            negs = self.rng.choice(neg_rows, self.npratio, replace=False)
        elif len(neg_rows) > 0:
            negs = self.rng.choice(neg_rows, self.npratio, replace=True)
        else:
            negs = self.rng.integers(0, len(self.title_ids), self.npratio)

        cand_rows = np.concatenate([[pos_row], negs]).astype(np.int64)

        hist = np.zeros((self.max_history, self.title_ids.shape[1]), dtype=np.int64)
        if hist_rows:
            h = self.title_ids[hist_rows[-self.max_history:]]
            hist[-len(h):] = h

        return torch.from_numpy(hist), torch.from_numpy(self.title_ids[cand_rows].astype(np.int64))


def main():
    parser = argparse.ArgumentParser(description="Q3.1: reproduced NRMS baseline")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "ebnerd_small"], default="mind")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-imps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-history", type=int, default=30)
    parser.add_argument("--max-title-len", type=int, default=20)
    parser.add_argument("--npratio", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0, help="0 = torch default")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(42)

    dataset_name = {"mind": "MIND", "ebnerd": "EBNERD_DEMO",
                    "ebnerd_small": "EBNERD_SMALL"}[args.dataset]
    proc = PROCESSED_DIR / dataset_name

    logger.info("Loading processed tables ...")
    articles    = pd.read_parquet(proc / "articles.parquet")
    history     = pd.read_parquet(proc / "history.parquet")
    impressions = pd.read_parquet(proc / "impressions.parquet")

    # ── Article title matrix ─────────────────────────────────────────────────
    titles = (articles.get("title", pd.Series("", index=articles.index)).fillna("")
              + " " + articles.get("subtitle", pd.Series("", index=articles.index)).fillna(""))
    vocab = build_vocab(titles.tolist())
    title_ids = encode_titles(titles.tolist(), vocab, args.max_title_len)
    aid_to_row = {str(a): i for i, a in enumerate(articles[COL_ARTICLE_ID])}
    logger.info(f"Title matrix: {title_ids.shape}")

    hist_idx = build_history_index(history)

    # ── Build training samples ───────────────────────────────────────────────
    train = impressions[impressions[COL_SPLIT] == SPLIT_TRAIN]
    keep = train[COL_IMPRESSION_ID].drop_duplicates()
    if len(keep) > args.train_imps:
        keep = keep.sample(args.train_imps, random_state=42)
    train = train[train[COL_IMPRESSION_ID].isin(keep)]

    logger.info(f"Building training samples from {train[COL_IMPRESSION_ID].nunique():,} impressions ...")
    t_ns = _to_ns(train[COL_IMPRESSION_TIME])
    users = train[COL_USER_ID].astype(str).to_numpy()
    arts  = train[COL_ARTICLE_ID].astype(str).to_numpy()
    labels = train[COL_LABEL].to_numpy()

    samples = []
    for idx in tqdm(train.groupby(COL_IMPRESSION_ID, sort=False).indices.values(),
                    desc="train samples", unit="imp"):
        first = idx[0]
        hist_rows = user_history_rows(hist_idx, aid_to_row, users[first],
                                      t_ns[first], args.max_history)
        pos = [aid_to_row[arts[i]] for i in idx if labels[i] == 1 and arts[i] in aid_to_row]
        neg = [aid_to_row[arts[i]] for i in idx if labels[i] == 0 and arts[i] in aid_to_row]
        if not pos:
            continue
        for p in pos:
            samples.append((hist_rows, p, np.array(neg, dtype=np.int64)))

    logger.info(f"  {len(samples):,} training samples (1 positive + {args.npratio} negatives each)")

    loader = DataLoader(
        TrainSet(samples, title_ids, args.max_history, args.npratio),
        batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    model = NRMS(len(vocab), emb_dim=args.emb_dim, n_heads=args.n_heads)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"NRMS parameters: {n_params:,}")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total, nb = 0.0, 0
        t0 = perf_counter()
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}", unit="batch")
        for hist, cand in pbar:
            optim.zero_grad()
            logits = model(hist, cand)                       # (B, 1+npratio)
            # The positive is always slot 0, so the target is the zero vector.
            loss = F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total += loss.item(); nb += 1
            if nb % 20 == 0:
                pbar.set_postfix(loss=f"{total/nb:.4f}")
        logger.info(f"epoch {epoch+1}: mean loss {total/max(nb,1):.4f} "
                    f"({perf_counter()-t0:.0f}s)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = MODELS_DIR / f"nrms_{dataset_name.lower()}.pt"
    torch.save({"state_dict": model.state_dict(), "vocab_size": len(vocab),
                "config": vars(args)}, ckpt)
    logger.info(f"Model saved -> {ckpt}")

    # ── Score the same val impressions the re-ranker was scored on ───────────
    stage1_path = PRED_DIR / dataset_name / "stage1_val_predictions.parquet"
    if stage1_path.exists():
        target_imps = set(pd.read_parquet(stage1_path, columns=[COL_IMPRESSION_ID])
                          [COL_IMPRESSION_ID].unique())
        logger.info(f"Scoring the {len(target_imps):,} val impressions from {stage1_path.name}")
    else:
        target_imps = None
        logger.warning("No stage1 predictions found; scoring a fresh val sample "
                       "(paired bootstrap against the re-ranker will NOT be valid).")

    val = impressions[impressions[COL_SPLIT] == SPLIT_VAL]
    if target_imps is not None:
        val = val[val[COL_IMPRESSION_ID].isin(target_imps)]
    else:
        keep = val[COL_IMPRESSION_ID].drop_duplicates().sample(
            min(5000, val[COL_IMPRESSION_ID].nunique()), random_state=42)
        val = val[val[COL_IMPRESSION_ID].isin(keep)]
    val = val.copy()

    v_ns   = _to_ns(val[COL_IMPRESSION_TIME])
    v_user = val[COL_USER_ID].astype(str).to_numpy()
    v_art  = val[COL_ARTICLE_ID].astype(str).to_numpy()
    scores = np.zeros(len(val), dtype=np.float32)

    model.eval()
    groups = list(val.groupby(COL_IMPRESSION_ID, sort=False).indices.values())
    with torch.no_grad():
        for idx in tqdm(groups, desc="scoring val", unit="imp"):
            first = idx[0]
            hist_rows = user_history_rows(hist_idx, aid_to_row, v_user[first],
                                          v_ns[first], args.max_history)

            hist = np.zeros((1, args.max_history, args.max_title_len), dtype=np.int64)
            if hist_rows:
                h = title_ids[hist_rows[-args.max_history:]]
                hist[0, -len(h):] = h

            cand_rows = [aid_to_row.get(a, 0) for a in v_art[idx]]
            cand = title_ids[cand_rows].astype(np.int64)[None, ...]

            out = model(torch.from_numpy(hist), torch.from_numpy(cand))
            scores[idx] = out[0].numpy()

    val["nrms_score"] = scores
    val["nrms_rank"] = (val.groupby(COL_IMPRESSION_ID)["nrms_score"]
                        .rank(ascending=False, method="first").astype(int))

    out_dir = PRED_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nrms_val_predictions.parquet"
    val.to_parquet(out_path, index=False)
    logger.info(f"Saved NRMS predictions -> {out_path}")

    from sklearn.metrics import roc_auc_score
    logger.info("=" * 60)
    logger.info(f"NRMS val AUC (flat) = {roc_auc_score(val[COL_LABEL], val['nrms_score']):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
