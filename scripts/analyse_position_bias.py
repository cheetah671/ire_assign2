"""
analyse_position_bias.py — Q1.2: is candidate position a usable signal?

Q1.2 asks for position bias as a session feature. Before engineering one, this
checks whether either dataset actually exposes it.

Pooling all impressions together suggests strong bias — click-through rate falls
steadily with position. That is an artifact. Impressions have different candidate
list lengths, and a longer list has a lower per-candidate click rate by
construction (roughly one click spread over more candidates). Long lists are the
only ones that reach high positions, so deep positions are dominated by
low-rate impressions.

Holding list length fixed removes the confound. If click rate is then flat
across positions, the list order carries no display-order information and a
position feature would be noise.

Usage
-----
  python scripts/analyse_position_bias.py
  python scripts/analyse_position_bias.py --dataset MIND
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"


def analyse(dataset_name: str, split: str = "val", min_imps: int = 100) -> pd.DataFrame:
    path = PROCESSED_DIR / dataset_name / "impressions.parquet"
    if not path.exists():
        print(f"[skip] {dataset_name}: {path} not found")
        return pd.DataFrame()

    df = pd.read_parquet(path, columns=["impression_id", "article_id", "label", "split"])
    df = df[df["split"] == split].copy()
    if df.empty:
        print(f"[skip] {dataset_name}: no {split} rows")
        return pd.DataFrame()

    df["pos"] = df.groupby("impression_id").cumcount()
    df["n_cand"] = df.groupby("impression_id")["article_id"].transform("size")

    print("=" * 70)
    print(f"{dataset_name}  ({split})   overall CTR = {df['label'].mean():.4f}")
    print("=" * 70)

    pooled = df.groupby("pos")["label"].mean().head(12)
    print("\nPooled over all list lengths (CONFOUNDED):")
    print("  " + "  ".join(f"{v:.3f}" for v in pooled.values))

    rows = []
    print("\nHolding candidate-list length fixed:")
    for n in sorted(df["n_cand"].unique()):
        sub = df[df["n_cand"] == n]
        n_imps = sub["impression_id"].nunique()
        if n_imps < min_imps:
            continue

        ctr = sub.groupby("pos")["label"].mean()
        # Spearman correlation between position and click rate: a real display
        # order would give a clear negative trend.
        rho = np.corrcoef(ctr.index.values, ctr.values)[0, 1] if len(ctr) > 2 else np.nan

        print(f"  len={n:<3} ({n_imps:>6,} imps)  rho={rho:+.3f}   "
              + " ".join(f"{v:.3f}" for v in ctr.values[:8]))
        rows.append({"dataset": dataset_name, "n_candidates": n,
                     "n_impressions": n_imps, "pos_ctr_corr": rho})

    out = pd.DataFrame(rows)
    if not out.empty:
        mean_rho = out["pos_ctr_corr"].mean()
        print(f"\n  mean position/CTR correlation across list lengths: {mean_rho:+.3f}")
        verdict = ("NO usable position bias - list order is shuffled"
                   if abs(mean_rho) < 0.3 else
                   "position bias present - worth a feature")
        print(f"  verdict: {verdict}\n")
    return out


def main():
    parser = argparse.ArgumentParser(description="Q1.2 position-bias check")
    parser.add_argument("--dataset", default="both", choices=["MIND", "EBNERD_DEMO", "both"])
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    targets = (["MIND", "EBNERD_DEMO"] if args.dataset == "both" else [args.dataset])
    frames = [analyse(name, args.split) for name in targets]
    frames = [f for f in frames if not f.empty]

    if frames:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "position_bias_check.csv"
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
