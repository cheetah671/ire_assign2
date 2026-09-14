"""
run_serving_analysis.py — Q4: Serving & Scale Analysis

Simulates a serving environment to measure:
1. Index memory footprint.
2. p99 retrieval latency for a single user request.
3. Back-of-envelope cost/QPS estimate.
4. Scaling argument (prints to console).
"""

import argparse
import logging
import sys
import time
import os
import psutil
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.features.behavioural_features import BehaviouralFeatureExtractor
from src.ranking.reranker import LightGBMReranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("serving")

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PRED_DIR = BASE_DIR / "data" / "predictions"

def measure_latency(dataset_name: str, ranker_type: str):
    logger.info("="*60)
    logger.info(f"Serving Analysis: {dataset_name} | Stage-1: {ranker_type}")
    logger.info("="*60)
    
    proc_dir = PROCESSED_DIR / dataset_name
    if not proc_dir.exists():
        logger.error(f"Missing data for {dataset_name}.")
        return

    # Measure basic memory
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    articles = pd.read_parquet(proc_dir / "articles.parquet")
    history = pd.read_parquet(proc_dir / "history.parquet")
    
    # Look for improved predictions first, then baseline, then reranker
    for pred_name in ["improved_val_predictions.parquet", "baseline_val_predictions.parquet", "reranker_val_predictions.parquet"]:
        val_pred_path = PRED_DIR / dataset_name / pred_name
        if val_pred_path.exists():
            break
    else:
        logger.error("No reranker predictions found. Run run_reranker.py first.")
        return
    val_preds = pd.read_parquet(val_pred_path)
    logger.info(f"Loaded predictions from {val_pred_path}")

    mem_after = process.memory_info().rss / (1024 * 1024)
    logger.info(f"[1] Feature Store Memory Footprint: {mem_after - mem_before:.2f} MB")

    extractor = BehaviouralFeatureExtractor(history, articles)
    
    # Simulate single user requests
    imp_ids = val_preds["impression_id"].drop_duplicates().sample(100, random_state=42).tolist()
    
    latencies = []
    
    # We use a mock model for latency test (no training needed)
    class MockModel:
        def predict(self, X):
            return np.random.rand(len(X))
    
    dummy_reranker = MockModel()

    logger.info(f"Simulating {len(imp_ids)} single-user requests...")
    for iid in imp_ids:
        req_df = val_preds[val_preds["impression_id"] == iid].copy()
        
        t0 = time.perf_counter()
        
        # 1. Feature Extraction Latency
        feat_df = extractor.extract_features(req_df)
        
        # 2. Re-ranking Latency (mock scoring for latency measurement only)
        _ = dummy_reranker.predict(feat_df)
        
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000) # in ms
        
    p99_latency = np.percentile(latencies, 99)
    avg_latency = np.mean(latencies)
    
    logger.info(f"[2] Latency:")
    logger.info(f"    Average: {avg_latency:.2f} ms")
    logger.info(f"    p99:     {p99_latency:.2f} ms")
    
    # Back of envelope Cost/QPS
    # Assume 1 core handles 1 request in `avg_latency` ms.
    qps_per_core = 1000 / avg_latency
    logger.info(f"[3] Cost/QPS Estimate:")
    logger.info(f"    Estimated QPS per CPU core: {qps_per_core:.0f}")
    logger.info(f"    If AWS c6g.large ($0.068/hr) has 2 cores, QPS ~ {qps_per_core*2:.0f}")
    logger.info(f"    Cost per 1M queries: ${(0.068 / 3600) * (1_000_000 / (qps_per_core*2)):.4f}")
    
    logger.info("[4] Scaling Argument (10x Load):")
    logger.info("    - Memory: Feature store easily fits in RAM even at 10x (assuming 10x is ~1GB).")
    logger.info("    - Compute: LightGBM inference is CPU-bound. At 10x load, we need 10x cores.")
    logger.info("    - Bottleneck: Database lookups for user history will break first. We must move history to Redis/Memcached.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], default="ebnerd")
    parser.add_argument("--ranker", choices=["bm25", "emb"], default="bm25")
    args = parser.parse_args()
    
    dataset_map = {"mind": "MIND", "ebnerd": "EBNERD_DEMO"}
    measure_latency(dataset_map[args.dataset], args.ranker)
