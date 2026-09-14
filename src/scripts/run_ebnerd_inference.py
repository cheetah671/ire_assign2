"""
run_ebnerd_inference.py - Streaming Inference for EB-NeRD Test Set

This script implements memory-efficient inference for the 1.6GB EB-NeRD test set.
It reads the pre-computed Ekstra Bladet Word2Vec embeddings and loads them into a fast dictionary.
Then it pre-computes mean-pooled user vectors from history.parquet.
Finally, it streams behaviors.parquet, computes scores using fast numpy dot products,
and outputs the `predictions.txt` file directly inside a `.zip` archive.
"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ebnerd_inference")

def _zip_file(file_path: Path, zip_path: Path):
    """Zip a file into an archive."""
    logger.info(f"Zipping to {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname=file_path.name)
    logger.info(f"✓ Zip complete: {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB)")

def load_word2vec_vectors(zip_path: Path) -> dict:
    """Load Word2Vec vectors directly from zip into a dict {article_id_int -> np.array}"""
    logger.info(f"Loading Word2Vec vectors from {zip_path.name} ...")
    w2v = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("Ekstra_Bladet_word2vec/document_vector.parquet") as f:
            df = pd.read_parquet(f)
            
    vecs = np.vstack(df["document_vector"].values).astype(np.float32)
    
    # L2-normalize vectors so that dot product == cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs /= norms
    
    for aid, vec in zip(df["article_id"].values, vecs):
        w2v[int(aid)] = vec
        
    logger.info(f"Loaded {len(w2v):,} Word2Vec vectors.")
    return w2v

def build_user_vectors(history_path: Path, w2v: dict) -> dict:
    """Load history.parquet and compute mean-pooled vectors per user."""
    logger.info(f"Loading history from {history_path.name} ...")
    hist = pd.read_parquet(history_path, columns=["user_id", "article_id_fixed", "read_time_fixed", "scroll_percentage_fixed"])
    
    user_vecs = {}
    logger.info("Computing user vectors...")
    
    # Process using fast tuple iteration
    for row in tqdm(hist.itertuples(index=False), total=len(hist), desc="User Vectors"):
        uid = row.user_id
        art_ids = row.article_id_fixed
        read_times = row.read_time_fixed
        scrolls = row.scroll_percentage_fixed
        
        if art_ids is None or len(art_ids) == 0:
            continue
            
        clicked_vecs = []
        weights = []
        
        # Time decay weight (chronological order, more recent = higher weight)
        n = len(art_ids)
        base_weights = np.linspace(0.5, 1.5, n)
        
        for i, aid in enumerate(art_ids):
            if aid in w2v:
                # Engagement weighting
                rt = read_times[i] if not np.isnan(read_times[i]) else 10.0
                rt_w = min(rt / 30.0, 2.0)  # Cap read time weight multiplier
                
                sc = scrolls[i] if not np.isnan(scrolls[i]) else 50.0
                sc_w = sc / 100.0
                
                w = base_weights[i] * (1.0 + rt_w + sc_w)
                
                clicked_vecs.append(w2v[aid])
                weights.append(w)
        
        if clicked_vecs:
            u_vec = np.average(clicked_vecs, axis=0, weights=weights)
            norm = np.linalg.norm(u_vec)
            if norm > 0:
                u_vec /= norm
            user_vecs[int(uid)] = u_vec.astype(np.float32)
            
    logger.info(f"Built {len(user_vecs):,} user vectors.")
    del hist
    import gc
    gc.collect()
    return user_vecs

def main():
    test_dir = BASE_DIR / "data" / "raw" / "EBNERD" / "test" / "ebnerd_testset" / "test"
    behaviors_path = test_dir / "behaviors.parquet"
    history_path = test_dir / "history.parquet"
    w2v_zip_path = BASE_DIR / "Ekstra_Bladet_word2vec.zip"

    if not behaviors_path.exists() or not history_path.exists():
        logger.error(f"Missing test data in {test_dir}. Run unzip first.")
        sys.exit(1)
        
    if not w2v_zip_path.exists():
        logger.error(f"Missing {w2v_zip_path.name}.")
        sys.exit(1)

    # 1. Load Embeddings & User History
    w2v = load_word2vec_vectors(w2v_zip_path)
    user_vecs = build_user_vectors(history_path, w2v)

    # 2. Streaming Inference on Behaviors
    out_dir = BASE_DIR / "submissions" / "EBNERD_LARGE"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prediction.txt"

    logger.info(f"Loading behaviors from {behaviors_path.name} ...")
    beh = pd.read_parquet(behaviors_path, columns=["impression_id", "user_id", "article_ids_inview"])
    logger.info(f"Total impressions to process: {len(beh):,}")

    logger.info(f"Starting inference -> {out_path}")
    t0 = perf_counter()
    
    with open(out_path, 'w', encoding='utf-8') as fout:
        for row in tqdm(beh.itertuples(index=False), total=len(beh), desc="Scoring"):
            imp_id = row.impression_id
            uid = row.user_id
            candidates = row.article_ids_inview
            
            # Fetch User Vector
            u_vec = user_vecs.get(int(uid))
            
            scored_candidates = []
            
            for aid in candidates:
                aid_int = int(aid)
                score = 0.0
                if u_vec is not None and aid_int in w2v:
                    # Dot product (Cosine Similarity)
                    score = float(np.dot(u_vec, w2v[aid_int]))
                scored_candidates.append((aid_int, score))
            
            # Codabench EB-NeRD expects ranks for the candidates, matching original candidate order
            sorted_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
            
            # Map candidate back to rank (1 = best)
            rank_map = {aid: rank for rank, (aid, score) in enumerate(sorted_candidates, start=1)}
            
            # Extract ranks in the original order of candidates
            ranks = [str(rank_map[int(aid)]) for aid in candidates]
            
            fout.write(f"{imp_id} [{','.join(ranks)}]\n")

    elapsed = perf_counter() - t0
    logger.info(f"Inference complete in {elapsed:.1f}s ({len(beh)/elapsed:,.0f} it/s)")
    
    # 3. Zip it up for Codabench
    zip_path = BASE_DIR / "ebnerd_submission.zip"
    _zip_file(out_path, zip_path)

if __name__ == "__main__":
    main()
