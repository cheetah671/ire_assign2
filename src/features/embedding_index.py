"""
embedding_index.py — Q3: Article Embedding Index

Builds and caches a {article_id → embedding_vector} map for both datasets.

Strategy per dataset
--------------------
MIND (English):
    Uses sentence-transformers (all-MiniLM-L6-v2) to encode
    title + subtitle for every article. Batch encodes for speed.

EB-NeRD (Danish):
    Uses pre-computed Word2Vec document vectors from
    Ekstra_Bladet_word2vec.zip → Ekstra_Bladet_word2vec/document_vector.parquet
    These are 300-dim float32 vectors, one per article_id.

Output
------
EmbeddingIndex.embeddings : dict {article_id → np.ndarray shape (D,)}
EmbeddingIndex.dim        : embedding dimension D

Caching
-------
Embeddings are cached as .npz files so MIND is only encoded once:
    data/cache/emb_<dataset_name>.npz
"""

import logging
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sentence-transformer model used for MIND (English)
SBERT_MODEL = "all-MiniLM-L6-v2"

# Word2Vec zip location (relative to project root)
W2V_ZIP_NAME = "Ekstra_Bladet_word2vec.zip"
W2V_PARQUET  = "Ekstra_Bladet_word2vec/document_vector.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _text_for_article(row) -> str:
    """Concatenate title + subtitle into a single string for encoding."""
    parts = []
    title = row.get("title")
    sub   = row.get("subtitle")
    if title and isinstance(title, str):
        parts.append(title.strip())
    if sub and isinstance(sub, str):
        parts.append(sub.strip())
    return " ".join(parts)


def _encode_with_sbert(texts: list, batch_size: int = 256) -> np.ndarray:
    """Encode a list of strings with SentenceTransformer. Returns (N, D) float32."""
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading SentenceTransformer model: {SBERT_MODEL} …")
    model = SentenceTransformer(SBERT_MODEL)
    logger.info(f"Encoding {len(texts):,} texts (batch_size={batch_size}) …")
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
    )
    return vecs.astype(np.float32)


def _load_word2vec_vectors(zip_path: Path) -> Dict[int, np.ndarray]:
    """
    Load EB-NeRD Word2Vec document vectors from the zip file.
    Returns dict {article_id (int) → np.ndarray shape (300,)}.
    """
    logger.info(f"Loading Word2Vec vectors from {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(W2V_PARQUET) as f:
            df = pd.read_parquet(f)

    logger.info(f"  Word2Vec parquet: {len(df):,} rows")

    # document_vector column is a list/array per row → stack to matrix
    vecs = np.vstack(df["document_vector"].values).astype(np.float32)

    # L2-normalize so cosine similarity = dot product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs /= norms

    return {int(aid): vecs[i] for i, aid in enumerate(df["article_id"].values)}


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingIndex
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingIndex:
    """
    Article embedding index for cosine-similarity ranking.

    Parameters
    ----------
    articles_df : DataFrame with columns article_id, title, subtitle
    dataset_name: 'MIND' or 'EBNERD_DEMO' — determines encoding strategy
    zip_dir     : directory containing Ekstra_Bladet_word2vec.zip (for EBNERD)
    cache_path  : .npz file to save/load embeddings (optional)
    """

    def __init__(
        self,
        articles_df: pd.DataFrame,
        dataset_name: str,
        zip_dir: Optional[Path] = None,
        cache_path: Optional[Path] = None,
    ):
        self.articles_df  = articles_df.copy().reset_index(drop=True)
        self.dataset_name = dataset_name.upper()
        self.zip_dir      = Path(zip_dir) if zip_dir else Path(".")
        self.cache_path   = Path(cache_path) if cache_path else None

        self.article_ids: list = []          # aligned list of article_ids
        self.id_to_idx: dict   = {}          # article_id → matrix row index
        self.matrix: Optional[np.ndarray] = None   # (N, D) embedding matrix
        self.dim: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> "EmbeddingIndex":
        """Build the embedding index. Loads from cache if available."""
        if self.cache_path and self.cache_path.exists():
            self._load_cache()
            return self

        if "EBNERD" in self.dataset_name:
            self._build_ebnerd()
        else:
            self._build_mind()

        if self.cache_path:
            self._save_cache()

        return self

    def _build_mind(self) -> None:
        """Encode MIND articles using SentenceTransformer."""
        texts = self.articles_df.apply(_text_for_article, axis=1).tolist()
        # Replace empty strings with a single space (sbert handles them fine)
        texts = [t if t.strip() else "." for t in texts]

        vecs = _encode_with_sbert(texts)

        self.article_ids = [str(aid) for aid in self.articles_df["article_id"].tolist()]
        self.id_to_idx   = {aid: i for i, aid in enumerate(self.article_ids)}
        self.matrix      = vecs
        self.dim         = vecs.shape[1]
        logger.info(f"MIND embedding matrix: {self.matrix.shape}")

    def _build_ebnerd(self) -> None:
        """Load EB-NeRD Word2Vec vectors and align to articles_df."""
        zip_path = self.zip_dir / W2V_ZIP_NAME
        if not zip_path.exists():
            raise FileNotFoundError(
                f"Word2Vec zip not found: {zip_path}\n"
                "Place Ekstra_Bladet_word2vec.zip in the project root."
            )

        w2v = _load_word2vec_vectors(zip_path)

        # Align to articles_df — only keep articles that have a W2V vector
        article_ids = []
        vecs = []
        missing = 0
        for _, row in self.articles_df.iterrows():
            aid_str = str(row["article_id"])
            aid_int = int(row["article_id"])
            if aid_int in w2v:
                article_ids.append(aid_str)   # store as STRING to match impressions
                vecs.append(w2v[aid_int])
            else:
                missing += 1

        if missing:
            logger.warning(
                f"EB-NeRD: {missing:,} articles have no Word2Vec vector "
                f"(they will score 0 as candidates)."
            )

        self.article_ids = article_ids   # all strings
        self.id_to_idx   = {aid: i for i, aid in enumerate(article_ids)}
        self.matrix      = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, 300), dtype=np.float32)
        self.dim         = self.matrix.shape[1] if self.matrix.size > 0 else 300
        logger.info(f"EBNERD embedding matrix: {self.matrix.shape}")

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            matrix=self.matrix,
            article_ids=np.array(self.article_ids),
        )
        size_mb = self.cache_path.stat().st_size / 1e6
        logger.info(f"Embedding cache saved → {self.cache_path}  ({size_mb:.1f} MB)")

    def _load_cache(self) -> None:
        logger.info(f"Loading embedding cache from {self.cache_path} …")
        data = np.load(self.cache_path, allow_pickle=False)
        self.matrix      = data["matrix"].astype(np.float32)
        self.article_ids = [str(aid) for aid in data["article_ids"].tolist()]  # always strings
        self.id_to_idx   = {aid: i for i, aid in enumerate(self.article_ids)}
        self.dim         = self.matrix.shape[1]
        logger.info(f"  Loaded {self.matrix.shape[0]:,} embeddings (dim={self.dim})")

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    def make_user_vector(
        self,
        user_id: str,
        history_index: dict,
        impression_time,
        max_history: int = 50,
    ) -> Optional[np.ndarray]:
        """
        Mean-pool embeddings of user's clicked articles before impression_time.

        Returns a (D,) L2-normalised vector, or None if no history is available.
        """
        user_df = history_index.get(user_id)
        if user_df is None or len(user_df) == 0:
            return None

        user_hist = user_df[
            user_df["click_time"].notna() &
            (user_df["click_time"] < impression_time)
        ].head(max_history)

        if len(user_hist) == 0:
            return None

        vecs = []
        for aid in user_hist["article_id"]:
            idx = self.id_to_idx.get(str(aid))   # always normalise to str
            if idx is not None:
                vecs.append(self.matrix[idx])

        if not vecs:
            return None

        user_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec /= norm
        return user_vec.astype(np.float32)

    @staticmethod
    def preindex_history(history_df: pd.DataFrame) -> dict:
        """Pre-group history by user_id. Same interface as BM25Ranker."""
        idx = {}
        for uid, grp in history_df.groupby("user_id", sort=False):
            idx[uid] = grp.sort_values("click_time", ascending=False)
        return idx

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        user_vec: Optional[np.ndarray],
        candidate_ids: list,
    ) -> dict:
        """
        Score candidate articles by cosine similarity to user_vec.
        Since vectors are L2-normalised, cosine sim = dot product.

        Returns dict {article_id: score}, 0.0 for missing/cold-start.
        """
        if user_vec is None or len(candidate_ids) == 0:
            return {aid: 0.0 for aid in candidate_ids}

        scores = {}
        for aid in candidate_ids:
            idx = self.id_to_idx.get(str(aid))   # normalise to str
            if idx is not None:
                scores[aid] = float(np.dot(user_vec, self.matrix[idx]))
            else:
                scores[aid] = 0.0
        return scores

    def retrieve_topk(
        self,
        user_vec: Optional[np.ndarray],
        k: int,
    ) -> list:
        """
        Score ALL articles via matrix-vector dot product and return top-K ids.
        Used for recall@K computation (Q3).
        """
        if user_vec is None or self.matrix.shape[0] == 0:
            return []

        # Matrix-vector dot: (N, D) @ (D,) → (N,) — very fast with numpy
        scores   = self.matrix @ user_vec          # cosine sim (vecs are normalised)
        topk_idx = np.argpartition(scores, -k)[-k:]  # unordered top-K
        topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]  # sort descending
        return [self.article_ids[i] for i in topk_idx]
