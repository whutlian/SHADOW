from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import read_csv, write_csv
from scripts.t37_papers100m_common import (
    audit_rows_for_policy,
    ensure_t37_bank,
    finalize_t37_row,
    load_t37_disco_references,
    run_t37_backend,
    t37_materialized_row,
)
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.ultra.papers100m_condensed import materialize_condensed_table
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_t37_contract import T37_REQUIRED_FIELDS, make_t37_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T37 SCR DisCo-parity SGC table.")
    parser.add_argument("--cache-root", default="caches/papers100m/stt_v1")
    parser.add_argument("--backend", default="sgc")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.00005, 0.0001, 0.0002, 0.0005])
    parser.add_argument("--methods", nargs="+", default=["random_onecache", "scr_class_random", "scr_full_stochastic_coverage"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--comparison-type", default="disco_parity")
    parser.add_argument("--baseline-csv", default="baselines/disco_papers100m_sgc.csv")
    parser.add_argument("--feature-lsh-dim", type=int, default=64)
    parser.add_argument("--feature-lsh-bits", type=int, default=16)
    parser.add_argument("--teacher-weight-eta", type=float, default=0.10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-bank", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--tables-dir", default="experiments/tables")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    refs = load_t37_disco_references(args.baseline_csv)
    out_path = Path(args.out) if str(args.out) else Path(args.tables_dir) / "t37_papers100m_disco_parity_scr_seed42.csv"
    rows = read_csv(out_path)
    max_ratio = max(float(value) for value in args.ratios)
    for seed in [int(value) for value in args.seeds]:
        for method in [str(value) for value in args.methods]:
            try:
                policy, bank = ensure_t37_bank(
                    cache_root,
                    method=method,
                    seed=seed,
                    max_ratio=max_ratio,
                    feature_lsh_dim=int(args.feature_lsh_dim),
                    feature_lsh_bits=int(args.feature_lsh_bits),
                    teacher_weight_eta=float(args.teacher_weight_eta),
                    force=bool(args.force_bank),
                )
                audit_by_ratio = audit_rows_for_policy(cache_root, policy=policy, seed=seed, ratios=[float(v) for v in args.ratios])
            except Exception as exc:
                for ratio in [float(v) for v in args.ratios]:
                    rows.append(
                        make_t37_row(
                            method=method,
                            seed=seed,
                            backend=str(args.backend).lower(),
                            comparison_type=str(args.comparison_type),
                            requested_full_node_ratio=ratio,
                            promotion_status="not_promoted",
                            failure_reason="bank_build_exception",
                            notes=(str(exc) + "\n" + traceback.format_exc())[:500],
                        )
                    )
                write_csv(out_path, rows, T37_REQUIRED_FIELDS)
                continue
            for ratio in [float(v) for v in args.ratios]:
                try:
                    ctx = Papers100MCacheContext(cache_root, selection_policy=policy, seed=seed)
                    materialized = materialize_condensed_table(ctx, ratio, policy=policy, seed=seed)
                    row = t37_materialized_row(
                        ctx,
                        method=method,
                        backend=str(args.backend).lower(),
                        ratio=ratio,
                        seed=seed,
                        policy=policy,
                        bank_manifest=bank,
                        materialized=materialized,
                        comparison_type=str(args.comparison_type),
                        audit_row=audit_by_ratio.get(ratio),
                    )
                    if args.run_long:
                        metrics = run_t37_backend(ctx, ratio=ratio, backend=str(args.backend), method=method, device=str(args.device))
                        row.update(metrics)
                    else:
                        row["promotion_status"] = "diagnostic"
                        row["failure_reason"] = "materialized_only_without_run_long"
                    row["peak_cpu_ram"] = current_cpu_ram_bytes()
                    row["peak_gpu_ram"] = current_gpu_ram_bytes()
                    rows.append(finalize_t37_row(row, refs=refs, promoted=bool(args.run_long)))
                except RuntimeError as exc:
                    reason = "OOM" if "out of memory" in str(exc).lower() else "runtime_error"
                    rows.append(make_t37_row(method=method, seed=seed, backend=str(args.backend).lower(), comparison_type=str(args.comparison_type), requested_full_node_ratio=ratio, promotion_status="not_promoted", failure_reason=reason, notes=str(exc)[:300]))
                except Exception as exc:
                    rows.append(make_t37_row(method=method, seed=seed, backend=str(args.backend).lower(), comparison_type=str(args.comparison_type), requested_full_node_ratio=ratio, promotion_status="not_promoted", failure_reason="exception", notes=(str(exc) + "\n" + traceback.format_exc())[:500]))
                write_csv(out_path, rows, T37_REQUIRED_FIELDS)
    print("status=completed")


if __name__ == "__main__":
    main()
