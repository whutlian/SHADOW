from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_hgc.eval.tables import (
    build_medium_ablation_rows_from_logs,
    build_medium_main_rows_from_logs,
    build_ratio_budget_summary_rows,
    build_ratio_main_rows_from_logs,
    build_small_main_rows_from_logs,
    write_rows_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CSV tables from JSON logs.")
    parser.add_argument("--small-log-dir", nargs="+", default=["experiments/logs/small"])
    parser.add_argument("--small-output", default="experiments/tables/small_main.csv")
    parser.add_argument("--small-ratio-output", default="experiments/tables/small_ratio_main.csv")
    parser.add_argument("--medium-log-dir", nargs="+", default=None)
    parser.add_argument("--medium-output", default="experiments/tables/medium_main.csv")
    parser.add_argument("--medium-ratio-output", default="experiments/tables/medium_ratio_main.csv")
    parser.add_argument("--medium-ablation-output", default="experiments/tables/medium_ablation.csv")
    parser.add_argument("--ratio-budget-summary-output", default="experiments/tables/ratio_budget_summary.csv")
    args = parser.parse_args()

    small_legacy_rows = []
    small_ratio_rows = []
    for log_dir in args.small_log_dir:
        small_legacy_rows.extend(build_small_main_rows_from_logs(log_dir))
        small_ratio_rows.extend(build_ratio_main_rows_from_logs(log_dir))
    write_rows_csv(args.small_output, small_legacy_rows)
    write_rows_csv(args.small_ratio_output, small_ratio_rows)
    print(f"wrote {args.small_output}")
    print(f"wrote {args.small_ratio_output}")
    if args.medium_log_dir is not None:
        medium_legacy_rows = []
        medium_ratio_rows = []
        medium_ablation_rows = []
        for log_dir in args.medium_log_dir:
            medium_legacy_rows.extend(build_medium_main_rows_from_logs(log_dir))
            medium_ratio_rows.extend(build_ratio_main_rows_from_logs(log_dir))
            medium_ablation_rows.extend(build_medium_ablation_rows_from_logs(log_dir))
        write_rows_csv(args.medium_output, medium_legacy_rows)
        write_rows_csv(args.medium_ratio_output, medium_ratio_rows)
        write_rows_csv(args.medium_ablation_output, medium_ablation_rows)
        write_rows_csv(
            args.ratio_budget_summary_output,
            build_ratio_budget_summary_rows([*args.small_log_dir, *args.medium_log_dir]),
        )
        print(f"wrote {args.medium_output}")
        print(f"wrote {args.medium_ratio_output}")
        print(f"wrote {args.medium_ablation_output}")
        print(f"wrote {args.ratio_budget_summary_output}")


if __name__ == "__main__":
    main()
