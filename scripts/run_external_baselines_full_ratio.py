"""Run external baselines with Shadow-HGC full-node ratio schedules.

Examples for the server:

  conda activate slian
  cd /data1/data_1/slian/Shadow-HGC
  python scripts/run_external_baselines_full_ratio.py \
    --data-root /data1/data_1/slian/data \
    --gpu 0 \
    --run

The default mode is a dry-run that writes planned command summaries without
executing the external repositories.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shadow_hgc.baselines.external_full_ratio import (  # noqa: E402
    BASELINES,
    FULL_RATIO_SCHEDULES,
    build_run_plan,
    execute_plan,
    path_arg,
    plan_to_record,
    write_records_csv,
    write_records_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute GECC/DeepCGC/TGCC/WbGC/ClustGDD runs using "
            "full-node compression ratios."
        )
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(BASELINES),
        choices=list(BASELINES),
        help="Baselines to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(FULL_RATIO_SCHEDULES),
        choices=list(FULL_RATIO_SCHEDULES),
        help="Dataset schedules to run. ogbn-products-low reuses ogbn-products data.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=None,
        help="Override the full-node ratio schedule for every selected dataset.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Random seeds.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Physical GPU id. The launcher sets CUDA_VISIBLE_DEVICES and passes gpu 0 to each baseline.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Dataset root. For WbGC/TGCC/ClustGDD this is linked as repo/data; "
            "for DeepCGC it should contain GraphSAINT/."
        ),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help="Root containing GECC, DeepCGC, TGCC, WbGC, and ClustGDD clones.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for per-run logs and aggregate JSONL/CSV.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run external baseline scripts.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="Optional timeout for each external run.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute commands. Without this flag the script only writes a dry-run plan.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="In --run mode, reuse an existing per-run summary.json if present.",
    )
    return parser.parse_args()


def resolve_data_root(args: argparse.Namespace) -> Path:
    if args.data_root is not None:
        return args.data_root
    env_root = os.environ.get("SHADOW_HGC_DATA_ROOT")
    if env_root:
        return Path(env_root)
    if args.run:
        raise SystemExit("--data-root is required in --run mode unless SHADOW_HGC_DATA_ROOT is set")
    return Path("<DATA_ROOT>")


def resolve_baseline_root(args: argparse.Namespace) -> Path:
    if args.baseline_root is not None:
        return args.baseline_root
    env_root = os.environ.get("SHADOW_HGC_BASELINE_ROOT")
    if env_root:
        return Path(env_root)
    return REPO_ROOT / "baselines" / "external_repos"


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "experiments" / "logs" / "external_baselines" / stamp


def load_existing_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    args = parse_args()
    data_root = resolve_data_root(args)
    baseline_root = resolve_baseline_root(args)
    output_root = resolve_output_root(args)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for baseline in args.baselines:
        for schedule_dataset in args.datasets:
            ratios = args.ratios if args.ratios is not None else FULL_RATIO_SCHEDULES[schedule_dataset]
            for full_ratio in ratios:
                for seed in args.seeds:
                    plan = build_run_plan(
                        baseline=baseline,
                        dataset=schedule_dataset,
                        full_ratio=full_ratio,
                        seed=seed,
                        gpu=args.gpu,
                        data_root=data_root,
                        baseline_root=baseline_root,
                        output_root=output_root,
                        python_executable=args.python,
                    )

                    if args.run:
                        existing = load_existing_summary(plan.summary_path) if args.skip_existing else None
                        if existing is not None:
                            existing["status"] = f"skipped_existing:{existing.get('status', 'unknown')}"
                            records.append(existing)
                            print(f"[skip] {path_arg(plan.summary_path)}")
                            continue
                        record = execute_plan(plan, timeout_sec=args.timeout_sec)
                        records.append(record)
                        print(
                            f"[{record['status']}] {baseline} {schedule_dataset} "
                            f"full={full_ratio:g} seed={seed} summary={path_arg(plan.summary_path)}"
                        )
                    else:
                        record = plan_to_record(plan)
                        records.append(record)
                        if plan.status == "unsupported":
                            print(
                                f"[unsupported] {baseline} {schedule_dataset} "
                                f"full={full_ratio:g}: {plan.failure_reason}"
                            )
                        else:
                            print(
                                f"[dry-run] {baseline} {schedule_dataset} full={full_ratio:g} "
                                f"seed={seed} cwd={path_arg(plan.cwd)}"
                            )
                            print(f"          {shlex.join(plan.command)}")

    jsonl_path = output_root / "external_baseline_runs.jsonl"
    csv_path = output_root / "external_baseline_runs.csv"
    write_records_jsonl(jsonl_path, records)
    write_records_csv(csv_path, records)
    print(f"[aggregate] {path_arg(jsonl_path)}")
    print(f"[aggregate] {path_arg(csv_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
