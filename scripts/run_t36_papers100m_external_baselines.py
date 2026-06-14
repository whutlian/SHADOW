from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import read_csv, write_csv
from scripts.run_t36_papers100m_disco_parity import _base_row, _ensure_policy_bank, _run_backend
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.ultra.papers100m_condensed import materialize_condensed_table
from shadow_hgc.ultra.papers100m_disco_parity import attach_disco_metrics, ensure_disco_baseline_csv, load_disco_baseline
from shadow_hgc.ultra.papers100m_ratio_policy_v2 import ratio_policy_v2
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_t36_contract import T36_REQUIRED_FIELDS, make_t36_row


def main() -> None:
    parser = argparse.ArgumentParser(description="T36 one-cache external baseline runner.")
    parser.add_argument("--cache-root", default="caches/papers100m/stt_v1")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.00005, 0.00010, 0.00020, 0.00050])
    parser.add_argument("--methods", nargs="+", default=["random_onecache", "herding_onecache", "kcenter_onecache"])
    parser.add_argument("--backends", nargs="+", default=["sgc"])
    parser.add_argument("--baseline-csv", default="baselines/disco_papers100m_sgc.csv")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--tables-dir", default="experiments/tables")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    ensure_disco_baseline_csv(args.baseline_csv)
    baseline = load_disco_baseline(args.baseline_csv)
    rows = read_csv(Path(args.tables_dir) / "t36_papers100m_external_baselines.csv")
    max_ratio = max(float(value) for value in args.ratios)
    for method in args.methods:
        policy = _ensure_policy_bank(cache_root, str(method), int(args.seed), max_ratio)
        for ratio in [float(value) for value in args.ratios]:
            for backend in args.backends:
                try:
                    ctx = Papers100MCacheContext(cache_root, selection_policy=policy, seed=int(args.seed))
                    materialized = materialize_condensed_table(ctx, ratio, policy=policy, seed=int(args.seed))
                    row = _base_row(ctx, method=str(method), backend=str(backend).lower(), ratio=ratio, seed=int(args.seed), policy=policy, materialized=materialized)
                    if bool(args.run_long):
                        metrics = _run_backend(ctx, ratio, str(backend), ratio_policy_v2(ratio), str(args.device))
                        row.update(metrics)
                        row["promotion_status"] = "diagnostic"
                        row["failure_reason"] = "external_baseline_local_proxy"
                    else:
                        row["promotion_status"] = "diagnostic"
                        row["failure_reason"] = "materialized_only_without_run_long"
                    row["peak_cpu_ram"] = current_cpu_ram_bytes()
                    row["peak_gpu_ram"] = current_gpu_ram_bytes()
                    row = attach_disco_metrics(row, baseline)
                except RuntimeError as exc:
                    reason = "OOM" if "out of memory" in str(exc).lower() else "runtime_error"
                    row = make_t36_row(method=str(method), backend=str(backend).lower(), seed=int(args.seed), requested_full_node_ratio=ratio, promotion_status="not_promoted", failure_reason=reason, notes=str(exc)[:300])
                rows.append(row)
                write_csv(Path(args.tables_dir) / "t36_papers100m_external_baselines.csv", rows, T36_REQUIRED_FIELDS)
    print("status=completed")


if __name__ == "__main__":
    main()
