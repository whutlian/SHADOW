from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_hgc.eval.tables import (
    build_medium_ablation_rows_from_logs,
    build_medium_main_rows_from_logs,
    build_small_main_rows_from_logs,
    write_rows_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CSV tables from JSON logs.")
    parser.add_argument("--small-log-dir", default="experiments/logs/small")
    parser.add_argument("--small-output", default="experiments/tables/small_main.csv")
    parser.add_argument("--medium-log-dir", default=None)
    parser.add_argument("--medium-output", default="experiments/tables/medium_main.csv")
    parser.add_argument("--medium-ablation-output", default="experiments/tables/medium_ablation.csv")
    args = parser.parse_args()

    write_rows_csv(args.small_output, build_small_main_rows_from_logs(args.small_log_dir))
    print(f"wrote {args.small_output}")
    if args.medium_log_dir is not None:
        write_rows_csv(args.medium_output, build_medium_main_rows_from_logs(args.medium_log_dir))
        write_rows_csv(args.medium_ablation_output, build_medium_ablation_rows_from_logs(args.medium_log_dir))
        print(f"wrote {args.medium_output}")
        print(f"wrote {args.medium_ablation_output}")


if __name__ == "__main__":
    main()
