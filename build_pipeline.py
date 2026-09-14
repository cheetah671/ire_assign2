"""
build_pipeline.py — One-command data pipeline for CS4.406 Assignment 1.

Usage
-----
# Process everything (EB-NeRD demo scale by default)
python build_pipeline.py

# Only MIND
python build_pipeline.py --dataset mind

# Only EB-NeRD, full small scale
python build_pipeline.py --dataset ebnerd --scale small

# Both datasets, both EB-NeRD scales
python build_pipeline.py --dataset both --scale both

What it does
------------
1. Extracts all raw zip archives into data/raw/
2. Parses MIND (news.tsv + behaviors.tsv) via src/data/parse_mind.py
3. Parses EB-NeRD (parquet files) via src/data/parse_ebnerd.py
4. Saves three unified tables per dataset/scale to data/processed/<name>/
     articles.parquet
     history.parquet
     impressions.parquet

The processed tables share the same column names (see src/data/schema.py)
so every downstream module (BM25, embeddings, evaluation) works identically
on both datasets.
"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd

# Make src importable without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.data.parse_ebnerd import parse_ebnerd
from src.data.parse_mind import parse_mind
from src.data.schema import (
    COL_ARTICLE_ID,
    COL_DATASET,
    HISTORY_COLUMNS,
    IMPRESSIONS_COLUMNS,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_zip(zip_path: Path, target_dir: Path, label: str) -> Path:
    """
    Extract *zip_path* into *target_dir*.
    Skips extraction if the directory is already non-empty.
    """
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"[{label}] Already extracted → {target_dir}")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{label}] Extracting {zip_path.name} → {target_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    logger.info(f"[{label}] Done.")
    return target_dir


def _find_file(root: Path, filename: str) -> Path:
    """Recursively find *filename* under *root*.  Raises if not found."""
    for p in root.rglob(filename):
        return p
    raise FileNotFoundError(f"Could not find '{filename}' under {root}")


def find_mind_split_dir(extracted_root: Path) -> Path:
    """Return the directory that directly contains news.tsv."""
    return _find_file(extracted_root, "news.tsv").parent


def find_ebnerd_bundle_dir(extracted_root: Path) -> Path:
    """Return the directory that directly contains articles.parquet."""
    return _find_file(extracted_root, "articles.parquet").parent


def save_processed(name: str, articles: pd.DataFrame,
                   history: pd.DataFrame, impressions: pd.DataFrame) -> None:
    """Write the three unified tables to data/processed/<name>/."""
    out = PROCESSED_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    ap = out / "articles.parquet"
    hp = out / "history.parquet"
    ip = out / "impressions.parquet"

    articles.to_parquet(ap,    index=False)
    history.to_parquet(hp,     index=False)
    impressions.to_parquet(ip, index=False)

    logger.info(f"Saved → {out}")
    logger.info(f"  articles:    {len(articles):>10,} rows  ({ap.stat().st_size/1e6:.1f} MB)")
    logger.info(f"  history:     {len(history):>10,} rows  ({hp.stat().st_size/1e6:.1f} MB)")
    logger.info(f"  impressions: {len(impressions):>10,} rows  ({ip.stat().st_size/1e6:.1f} MB)")


def print_summary(name: str, articles: pd.DataFrame,
                  history: pd.DataFrame, impressions: pd.DataFrame) -> None:
    """Log a human-readable summary of the processed tables."""
    n_users  = history["user_id"].nunique() if len(history) else 0
    n_splits = impressions["split"].value_counts().to_dict() if len(impressions) else {}
    n_pos    = int(impressions["label"].sum()) if len(impressions) else 0
    logger.info(
        f"\n{'─'*60}\n"
        f"  Dataset      : {name}\n"
        f"  Articles     : {len(articles):,}\n"
        f"  Users        : {n_users:,}\n"
        f"  History rows : {len(history):,}\n"
        f"  Impressions  : {len(impressions):,}  (clicks={n_pos:,})\n"
        f"  Splits       : {n_splits}\n"
        f"{'─'*60}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset pipelines
# ─────────────────────────────────────────────────────────────────────────────

def build_mind(zip_dir: Path) -> None:
    """End-to-end MIND-small pipeline: extract → parse → save."""
    logger.info("=" * 60)
    logger.info("MIND pipeline")
    logger.info("=" * 60)

    train_zip = zip_dir / "MINDsmall_train.zip"
    dev_zip   = zip_dir / "MINDsmall_dev.zip"

    for zp in (train_zip, dev_zip):
        if not zp.exists():
            raise FileNotFoundError(f"Required zip not found: {zp}")

    train_root = extract_zip(train_zip, RAW_DIR / "MIND" / "train", "MIND train")
    dev_root   = extract_zip(dev_zip,   RAW_DIR / "MIND" / "dev",   "MIND dev")

    train_dir = find_mind_split_dir(train_root)
    dev_dir   = find_mind_split_dir(dev_root)

    articles, history, impressions = parse_mind(train_dir, dev_dir)
    print_summary("MIND", articles, history, impressions)
    save_processed("MIND", articles, history, impressions)


def build_ebnerd(zip_dir: Path, scale: str) -> None:
    """End-to-end EB-NeRD pipeline for a given scale: extract → parse → save."""
    logger.info("=" * 60)
    logger.info(f"EB-NeRD pipeline  [scale={scale}]")
    logger.info("=" * 60)

    zip_path = zip_dir / f"ebnerd_{scale}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Required zip not found: {zip_path}")

    extracted = extract_zip(zip_path, RAW_DIR / "EBNERD" / scale, f"EB-NeRD {scale}")
    bundle_dir = find_ebnerd_bundle_dir(extracted)

    articles, history, impressions = parse_ebnerd(bundle_dir)
    name = f"EBNERD_{scale.upper()}"
    print_summary(name, articles, history, impressions)
    save_processed(name, articles, history, impressions)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the news recommendation data pipeline (Q1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=["mind", "ebnerd", "both"], default="both",
        help="Which dataset(s) to process.",
    )
    parser.add_argument(
        "--scale", choices=["demo", "small", "both"], default="demo",
        help="EB-NeRD bundle scale to process.",
    )
    parser.add_argument(
        "--zip-dir", type=Path, default=BASE_DIR,
        help="Directory that contains the raw .zip files.",
    )
    args = parser.parse_args()

    logger.info(
        f"Pipeline starting │ dataset={args.dataset} │ "
        f"scale={args.scale} │ zip_dir={args.zip_dir}"
    )

    if args.dataset in ("mind", "both"):
        build_mind(args.zip_dir)

    if args.dataset in ("ebnerd", "both"):
        scales = ["demo", "small"] if args.scale == "both" else [args.scale]
        for scale in scales:
            zp = args.zip_dir / f"ebnerd_{scale}.zip"
            if not zp.exists():
                logger.warning(f"EB-NeRD {scale} zip not found ({zp}), skipping.")
                continue
            build_ebnerd(args.zip_dir, scale)

    logger.info("Pipeline complete.  Processed data is in data/processed/")


if __name__ == "__main__":
    main()
