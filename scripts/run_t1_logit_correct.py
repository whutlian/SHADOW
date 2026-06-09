from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t1_logit_affinity_stage import LOGIT_CORRECT, SAFE_BASES, _blocked_method_rows
from shadow_hgc.fullgraph.sfb_logging import write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Write T1 LogitCorrectLite rows.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = _blocked_method_rows("logit_correct_lite", SAFE_BASES)
    for row in rows:
        row["seed"] = int(args.seed)
    write_csv(LOGIT_CORRECT, rows)


if __name__ == "__main__":
    main()
