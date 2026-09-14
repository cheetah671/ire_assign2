"""
nrms.py — Q3.1: NRMS, the reproduced neural baseline.

Neural News Recommendation with Multi-Head Self-Attention (Wu et al., EMNLP 2019),
the standard baseline for both the MIND and RecSys/EB-NeRD leaderboards.

Architecture
------------
News encoder   word embeddings -> multi-head self-attention -> additive attention
               -> one vector per article
User encoder   the same two stages applied over the user's clicked-article
               vectors -> one vector per user
Score          dot product between the user vector and each candidate vector

Training follows the paper: for each click, sample K non-clicked articles from
the same impression and optimise softmax cross-entropy over the 1 + K scores.
This turns ranking into (K+1)-way classification and is why the model is trained
per impression rather than per candidate.

Deviations from the paper, all made to fit CPU training, are configurable and
recorded in the run's config so the comparison stays honest:
  - embeddings are learned from scratch rather than initialised from GloVe 300d
  - smaller embedding dim and shorter history than the paper's 300 / 50
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PAD = 0


class AdditiveAttention(nn.Module):
    """Pool a sequence into one vector with learned per-token weights."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.query = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, D)   mask: (B, L) True where valid
        e = self.query(torch.tanh(self.proj(x))).squeeze(-1)      # (B, L)
        if mask is not None:
            # Fully-padded rows would make softmax produce NaN, so give them a
            # uniform distribution and let the zero vectors carry no signal.
            empty = ~mask.any(dim=-1, keepdim=True)
            e = e.masked_fill(~mask, torch.finfo(e.dtype).min)
            e = torch.where(empty.expand_as(e), torch.zeros_like(e), e)
        a = torch.softmax(e, dim=-1).unsqueeze(1)                 # (B, 1, L)
        return torch.bmm(a, x).squeeze(1)                         # (B, D)


class NewsEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, n_heads: int,
                 attn_hidden: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.self_attn = nn.MultiheadAttention(
            emb_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.additive = AdditiveAttention(emb_dim, attn_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, titles: torch.Tensor) -> torch.Tensor:
        # titles: (B, L) token ids
        valid = titles != PAD
        x = self.dropout(self.embedding(titles))

        # An all-padding title yields an all-True key_padding_mask, which makes
        # nn.MultiheadAttention emit NaN. Let those rows attend freely; the
        # additive stage masks them out afterwards.
        key_pad = ~valid
        key_pad = torch.where(
            valid.any(dim=-1, keepdim=True), key_pad, torch.zeros_like(key_pad)
        )

        attn, _ = self.self_attn(x, x, x, key_padding_mask=key_pad, need_weights=False)
        return self.additive(self.dropout(attn), mask=valid)


class NRMS(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, n_heads: int = 4,
                 attn_hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.news_encoder = NewsEncoder(vocab_size, emb_dim, n_heads, attn_hidden, dropout)
        self.user_attn = nn.MultiheadAttention(
            emb_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.user_additive = AdditiveAttention(emb_dim, attn_hidden)
        self.dropout = nn.Dropout(dropout)

    def encode_user(self, hist_titles: torch.Tensor) -> torch.Tensor:
        # hist_titles: (B, H, L)
        B, H, L = hist_titles.shape
        news_vecs = self.news_encoder(hist_titles.reshape(B * H, L)).reshape(B, H, -1)

        valid = (hist_titles != PAD).any(dim=-1)                  # (B, H)
        key_pad = ~valid
        key_pad = torch.where(
            valid.any(dim=-1, keepdim=True), key_pad, torch.zeros_like(key_pad)
        )

        attn, _ = self.user_attn(
            news_vecs, news_vecs, news_vecs, key_padding_mask=key_pad, need_weights=False
        )
        return self.user_additive(self.dropout(attn), mask=valid)

    def forward(self, hist_titles: torch.Tensor, cand_titles: torch.Tensor) -> torch.Tensor:
        # hist: (B, H, L)   cand: (B, C, L)   ->   (B, C) scores
        user_vec = self.encode_user(hist_titles)                  # (B, D)
        B, C, L = cand_titles.shape
        cand_vecs = self.news_encoder(cand_titles.reshape(B * C, L)).reshape(B, C, -1)
        return torch.bmm(cand_vecs, user_vec.unsqueeze(-1)).squeeze(-1)


def build_vocab(titles, max_vocab: int = 50_000, min_count: int = 2) -> dict:
    """Word -> id map built from article titles. Ids 0 (pad) and 1 (unk) reserved."""
    from collections import Counter

    counter = Counter()
    for t in titles:
        counter.update(_tokenize(t))

    vocab = {"<pad>": PAD, "<unk>": 1}
    for word, count in counter.most_common(max_vocab):
        if count < min_count:
            break
        vocab[word] = len(vocab)
    logger.info(f"Vocabulary: {len(vocab):,} tokens (from {len(counter):,} unique)")
    return vocab


def _tokenize(text: str) -> list:
    if not isinstance(text, str):
        return []
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w]


def encode_titles(titles, vocab: dict, max_len: int) -> np.ndarray:
    """Article titles -> (N, max_len) int32 token-id matrix, right-padded."""
    out = np.zeros((len(titles), max_len), dtype=np.int32)
    for i, t in enumerate(titles):
        ids = [vocab.get(w, 1) for w in _tokenize(t)][:max_len]
        out[i, :len(ids)] = ids
    return out
