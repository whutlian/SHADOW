from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_hgc.pipeline.toy import run_toy_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the toy Shadow-HGC-R-1 pipeline.")
    parser.add_argument("--output", default="experiments/logs/toy/summary.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--M-tau", type=int, default=4)
    parser.add_argument("--M-r", type=int, default=3)
    parser.add_argument("--k-s", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=4)
    args = parser.parse_args()
    summary = run_toy_experiment(
        output_path=args.output,
        seed=args.seed,
        epochs=args.epochs,
        M_tau=args.M_tau,
        M_r=args.M_r,
        k_s=args.k_s,
        feature_dim=args.feature_dim,
    )
    print(f"wrote {args.output}")
    print(f"accuracy={summary['accuracy']:.4f}")


if __name__ == "__main__":
    main()
