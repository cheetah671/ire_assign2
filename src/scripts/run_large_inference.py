"""
run_large_inference.py - Streaming Inference for MIND-Large Test Set

This script completely bypasses Pandas to avoid the massive RAM explosion 
caused by exploding 2.37 million impressions × 50 candidates (100M+ rows).
It streams the test behaviors.tsv line-by-line, scoring and formatting 
the Codabench prediction.txt directly.
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

from src.data.parse_mind import _parse_news
from src.ranking.bm25_ranker import BM25Ranker, tokenize
from src.features.embedding_index import EmbeddingIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("large_inference")

def _zip_file(file_path: Path, zip_path: Path):
    """Zip a file into an archive."""
    logger.info(f"Zipping to {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname=file_path.name)
    logger.info(f"✓ Zip complete: {zip_path.stat().st_size / 1e6:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="Streaming Inference for MIND-Large")
    parser.add_argument("--ranker", choices=["bm25", "emb"], required=True, help="Ranker to use")
    args = parser.parse_args()

    test_dir = BASE_DIR / "data" / "raw" / "MIND" / "large_test" / "MINDlarge_test"
    news_path = test_dir / "news.tsv"
    behaviors_path = test_dir / "behaviors.tsv"

    if not news_path.exists() or not behaviors_path.exists():
        logger.error(f"Missing {news_path} or {behaviors_path}. Run unzip first.")
        sys.exit(1)

    # 1. Parse Articles & Build Index
    logger.info(f"Loading articles from {news_path.name} ...")
    articles_df = _parse_news(news_path)
    logger.info(f"Loaded {len(articles_df):,} articles.")

    if args.ranker == "bm25":
        logger.info("Building BM25 Index ...")
        t0 = perf_counter()
        ranker = BM25Ranker(articles_df, max_history=50).build()
        logger.info(f"BM25 Index built in {perf_counter()-t0:.1f}s")
    else:
        logger.info("Building Embedding Index ...")
        t0 = perf_counter()
        ranker = EmbeddingIndex(articles_df, dataset_name="MIND").build()
        logger.info(f"Embedding Index built in {perf_counter()-t0:.1f}s")

    # 2. Streaming Inference
    out_dir = BASE_DIR / "submissions" / "MIND_LARGE" / args.ranker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prediction.txt"

    # Count lines for tqdm
    logger.info("Counting total impressions...")
    with open(behaviors_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    logger.info(f"Total impressions to process: {total_lines:,}")

    logger.info(f"Starting streaming inference -> {out_path}")
    t0 = perf_counter()
    
    with open(behaviors_path, 'r', encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, total=total_lines, desc="Scoring"):
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
                
            imp_id = parts[0]
            # parts[1] = user_id
            # parts[2] = time
            history = parts[3].split() if parts[3] else []
            candidates = parts[4].split() if parts[4] else []
            
            # Trim history to max_history (most recent are at the end usually, but wait, we'll just take last max_history)
            history = history[-50:] 

            if args.ranker == "bm25":
                # Build query string
                text_parts = [
                    ranker.article_text_map.get(aid, "")
                    for aid in history
                    if ranker.article_text_map.get(aid, "")
                ]
                query_tokens = tokenize(" ".join(text_parts))[:100]
                
                # Score candidates
                scores_dict = ranker.score_candidates(query_tokens, candidates)
            else:
                # Embedding Ranker
                # Recency-weighted mean-pool: articles at end of history are more recent
                vecs = []
                for aid in history:
                    idx = ranker.id_to_idx.get(str(aid))
                    if idx is not None:
                        vecs.append(ranker.matrix[idx])
                
                if vecs:
                    n = len(vecs)
                    # Linear recency weights: oldest=0.5, newest=1.5
                    weights = np.linspace(0.5, 1.5, n, dtype=np.float32)
                    vecs_arr = np.array(vecs, dtype=np.float32)
                    user_vec = np.average(vecs_arr, axis=0, weights=weights)
                    norm = np.linalg.norm(user_vec)
                    if norm > 0:
                        user_vec /= norm
                    user_vec = user_vec.astype(np.float32)
                else:
                    user_vec = None
                    
                scores_dict = ranker.score_candidates(user_vec, candidates)

            # Assign ranks based on descending score
            # Higher score -> better rank (1 = best)
            scored_candidates = [(aid, scores_dict.get(aid, 0.0)) for aid in candidates]
            # Sort by score descending (if tie, preserve original order)
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Map candidate back to rank
            rank_map = {aid: rank for rank, (aid, score) in enumerate(scored_candidates, start=1)}
            
            # Codabench requires the ranks to be in the exact order of the original candidates list
            ranks = [str(rank_map[aid]) for aid in candidates]
            rank_list = "[" + ",".join(ranks) + "]"
            
            fout.write(f"{imp_id} {rank_list}\n")

    elapsed = perf_counter() - t0
    logger.info(f"Inference complete in {elapsed:.1f}s ({total_lines/elapsed:,.0f} it/s)")
    
    # Zip it up
    zip_path = BASE_DIR / f"mind_large_{args.ranker}_submission.zip"
    _zip_file(out_path, zip_path)

if __name__ == "__main__":
    main()
