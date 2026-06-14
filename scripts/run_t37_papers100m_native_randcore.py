from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import read_csv, write_csv
from scripts.t37_papers100m_common import ensure_t37_bank, finalize_t37_row, run_t37_backend, t37_materialized_row
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.ultra.papers100m_condensed import materialize_condensed_table
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_t37_contract import T36_NATIVE_REFERENCE, T37_REQUIRED_FIELDS, make_t37_row


def _native_reference_for_ratio(ratio: float) -> float | str:
    return T36_NATIVE_REFERENCE.get(float(ratio), "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T37 native STT-RandCore scale-fidelity table.")
    parser.add_argument("--cache-root", default="caches/papers100m/stt_v1")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0005, 0.001, 0.002, 0.005, 0.01])
    parser.add_argument("--methods", nargs="+", default=["stt_randcore_gamlp", "stt_randcore_dual_loss"])
    parser.add_argument("--backends", nargs="+", default=["gamlp_table"])
    parser.add_argument("--feature-lsh-dim", type=int, default=64)
    parser.add_argument("--feature-lsh-bits", type=int, default=16)
    parser.add_argument("--teacher-weight-eta", type=float, default=0.10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-bank", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--tables-dir", default="experiments/tables")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    out_path = Path(args.out) if str(args.out) else Path(args.tables_dir) / "t37_papers100m_native_randcore_seed42.csv"
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
            except Exception as exc:
                for ratio in [float(v) for v in args.ratios]:
                    rows.append(make_t37_row(method=method, seed=seed, backend="", comparison_type="ours_native_scale_fidelity", requested_full_node_ratio=ratio, promotion_status="not_promoted", failure_reason="bank_build_exception", notes=(str(exc) + "\n" + traceback.format_exc())[:500]))
                write_csv(out_path, rows, T37_REQUIRED_FIELDS)
                continue
            for ratio in [float(value) for value in args.ratios]:
                for backend in [str(value).lower() for value in args.backends]:
                    if method == "stt_randcore_sagn" and backend != "sagn_table":
                        continue
                    try:
                        ctx = Papers100MCacheContext(cache_root, selection_policy=policy, seed=seed)
                        materialized = materialize_condensed_table(ctx, ratio, policy=policy, seed=seed)
                        row = t37_materialized_row(
                            ctx,
                            method=method,
                            backend=backend,
                            ratio=ratio,
                            seed=seed,
                            policy=policy,
                            bank_manifest=bank,
                            materialized=materialized,
                            comparison_type="ours_native_scale_fidelity",
                        )
                        if args.run_long:
                            metrics = run_t37_backend(ctx, ratio=ratio, backend=backend, method=method, device=str(args.device))
                            row.update(metrics)
                            row["student_model"] = metrics.get("student", "")
                            row["hidden_dim"] = metrics.get("hidden_dim", "")
                            row["epochs"] = metrics.get("epochs", "")
                            row["temperature"] = metrics.get("temperature", "")
                            row["lambda_hard"] = 0.75 if method == "stt_randcore_dual_loss" and ratio <= 0.001 else (0.25 if method == "stt_randcore_dual_loss" and ratio >= 0.005 else 0.5)
                            row["lambda_soft"] = metrics.get("lambda_soft", "")
                            row["lambda_prior"] = metrics.get("lambda_prior", "")
                            row["uses_teacher_probs_as_soft_targets"] = True
                        else:
                            row["promotion_status"] = "diagnostic"
                            row["failure_reason"] = "materialized_only_without_run_long"
                        ref = _native_reference_for_ratio(ratio)
                        if ref != "" and row.get("accuracy", "") != "":
                            row["notes"] = f"T36_native_reference={ref}; " + str(row.get("notes", ""))
                        row["peak_cpu_ram"] = current_cpu_ram_bytes()
                        row["peak_gpu_ram"] = current_gpu_ram_bytes()
                        rows.append(finalize_t37_row(row, promoted=bool(args.run_long)))
                    except RuntimeError as exc:
                        reason = "OOM" if "out of memory" in str(exc).lower() else "runtime_error"
                        rows.append(make_t37_row(method=method, seed=seed, backend=backend, comparison_type="ours_native_scale_fidelity", requested_full_node_ratio=ratio, promotion_status="not_promoted", failure_reason=reason, notes=str(exc)[:300]))
                    except Exception as exc:
                        rows.append(make_t37_row(method=method, seed=seed, backend=backend, comparison_type="ours_native_scale_fidelity", requested_full_node_ratio=ratio, promotion_status="not_promoted", failure_reason="exception", notes=(str(exc) + "\n" + traceback.format_exc())[:500]))
                    write_csv(out_path, rows, T37_REQUIRED_FIELDS)
    print("status=completed")


if __name__ == "__main__":
    main()
