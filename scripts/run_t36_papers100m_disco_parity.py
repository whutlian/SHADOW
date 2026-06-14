from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import write_csv
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.ultra.papers100m_condensed import materialize_condensed_table, train_and_eval_condensed_table
from shadow_hgc.ultra.papers100m_disco_parity import attach_disco_metrics, ensure_disco_baseline_csv, load_disco_baseline
from shadow_hgc.ultra.papers100m_memmap import directory_bytes
from shadow_hgc.ultra.papers100m_nested_bank import NESTED_BANK_POLICY, build_external_onecache_bank, build_nested_bank_v2
from shadow_hgc.ultra.papers100m_ratio_policy_v2 import ratio_policy_v2
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_sgc_backend import train_and_eval_sgc_condensed
from shadow_hgc.ultra.papers100m_t36_contract import T36_REQUIRED_FIELDS, make_t36_row, validate_t36_row


def _policy_for_method(method: str) -> str:
    if method in {"random_onecache", "herding_onecache", "kcenter_onecache"}:
        return f"{method}_t36"
    if method == "stt_current_t35":
        return "stt_ratio_v2"
    return NESTED_BANK_POLICY


def _ensure_policy_bank(cache_root: Path, method: str, seed: int, max_ratio: float) -> str:
    policy = _policy_for_method(method)
    ctx = Papers100MCacheContext(cache_root, selection_policy=policy, seed=seed)
    if method in {"random_onecache", "herding_onecache", "kcenter_onecache"}:
        build_external_onecache_bank(ctx, method=method, seed=seed, max_ratio=max_ratio)
    elif policy == NESTED_BANK_POLICY:
        build_nested_bank_v2(ctx, policy=policy, seed=seed, max_ratio=max(0.002, max_ratio), teacher_id="current_teacher")
    return policy


def _base_row(ctx: Papers100MCacheContext, *, method: str, backend: str, ratio: float, seed: int, policy: str, materialized: dict[str, Any]) -> dict[str, Any]:
    ids = ctx.cache_ids()
    teacher_test = ctx.teacher.get("accuracy", "")
    teacher_valid = ctx.teacher.get("valid_acc", "")
    policy_values = ratio_policy_v2(float(ratio))
    row = make_t36_row(
        method=method,
        backend=backend,
        seed=seed,
        requested_full_node_ratio=float(ratio),
        full_node_denominator=int(ctx.manifest["num_nodes"]),
        condensed_nodes=int(materialized.get("condensed_nodes", 0)),
        target_universe_size=int(ctx.manifest["target_universe_size"]),
        cache_build_id=ids["cache_build_id"],
        edge_cache_id=ids["edge_slice_cache_id"],
        sft_cache_id=ids["sft_cache_id"],
        teacher_cache_id=ids["teacher_cache_id"],
        selection_bank_id=ids["selection_bank_id"],
        nested_bank_id=ctx.bank.get("nested_bank_id", ids["selection_bank_id"]),
        teacher_id=ctx.teacher.get("teacher_id", ctx.teacher.get("installed_from_teacher_upgrade", "current_teacher")),
        teacher_test_acc=teacher_test,
        teacher_valid_acc=teacher_valid,
        teacher_cache_mode=ctx.teacher.get("teacher_cache_mode", ""),
        teacher_cache_bytes=ctx.teacher.get("teacher_cache_bytes", ""),
        uses_streaming_logits=str(ctx.teacher.get("teacher_topk_build_mode", "")) == "streaming_logits",
        uses_dense_teacher_cache_in_ram=ctx.teacher.get("uses_dense_teacher_cache_in_ram", False),
        uses_dense_all_node_teacher_cache=ctx.teacher.get("uses_dense_all_node_teacher_cache", False),
        materialize_time=materialized.get("condensed_materialize_time", ""),
        condensed_bytes=materialized.get("condensed_cache_bytes", ""),
        notes=f"selection_policy={policy}",
        **policy_values.as_row_fields(),
    )
    return row


def _run_backend(ctx: Papers100MCacheContext, ratio: float, backend: str, policy_values, device: str) -> dict[str, Any]:
    if str(backend).lower() == "sgc":
        return train_and_eval_sgc_condensed(
            ctx,
            ratio,
            epochs=180,
            temperature=policy_values.soft_temperature,
            lambda_hard=policy_values.lambda_hard,
            lambda_prior=policy_values.lambda_prior,
            device=device,
        )
    if str(backend).lower() in {"gamlp_table", "gamlp", "native"}:
        return train_and_eval_condensed_table(
            ctx,
            ratio,
            student="papers100m_gamlp_table",
            hidden_dim=512,
            epochs=260,
            temperature=policy_values.soft_temperature,
            lambda_hard=policy_values.lambda_hard,
            lambda_prior=policy_values.lambda_prior,
            device=device,
        )
    if str(backend).lower() in {"sagn_table", "sagn"}:
        return train_and_eval_condensed_table(
            ctx,
            ratio,
            student="papers100m_sagn_table",
            hidden_dim=384,
            epochs=260,
            temperature=policy_values.soft_temperature,
            lambda_hard=policy_values.lambda_hard,
            lambda_prior=policy_values.lambda_prior,
            device=device,
        )
    raise ValueError(f"unknown backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="T36 papers100M DisCo-parity runner.")
    parser.add_argument("--cache-root", default="caches/papers100m/stt_v1")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.00005, 0.00010, 0.00020, 0.00050])
    parser.add_argument("--methods", nargs="+", default=["stt_nested_bank_v2", "stt_hard_anchor_v2"])
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
    rows: list[dict[str, Any]] = []
    max_ratio = max(float(v) for v in args.ratios)
    for method in args.methods:
        policy = _ensure_policy_bank(cache_root, str(method), int(args.seed), max_ratio)
        for ratio in [float(v) for v in args.ratios]:
            for backend in args.backends:
                try:
                    ctx = Papers100MCacheContext(cache_root, selection_policy=policy, seed=int(args.seed))
                    materialized = materialize_condensed_table(ctx, ratio, policy=policy, seed=int(args.seed))
                    row = _base_row(ctx, method=str(method), backend=str(backend).lower(), ratio=ratio, seed=int(args.seed), policy=policy, materialized=materialized)
                    if bool(args.run_long):
                        policy_values = ratio_policy_v2(ratio)
                        metrics = _run_backend(ctx, ratio, str(backend), policy_values, str(args.device))
                        row.update(metrics)
                        row["promotion_status"] = "promoted" if float(row.get("teacher_test_acc", 0.0) or 0.0) >= 0.55 else "diagnostic"
                        if row["promotion_status"] != "promoted":
                            row["failure_reason"] = "teacher_below_0p55_gate"
                    else:
                        row["promotion_status"] = "diagnostic"
                        row["failure_reason"] = "materialized_only_without_run_long"
                    row["peak_cpu_ram"] = current_cpu_ram_bytes()
                    row["peak_gpu_ram"] = current_gpu_ram_bytes()
                    row["condensed_bytes"] = row.get("condensed_bytes", "") or directory_bytes(cache_root / "condensed")
                    row = attach_disco_metrics(row, baseline)
                    if validate_t36_row(row)["valid"] is False and row["promotion_status"] == "promoted":
                        row["promotion_status"] = "not_promoted"
                        row["failure_reason"] = ",".join(validate_t36_row(row)["forbidden_flags"])
                except RuntimeError as exc:
                    reason = "OOM" if "out of memory" in str(exc).lower() else "runtime_error"
                    row = make_t36_row(method=str(method), backend=str(backend).lower(), seed=int(args.seed), requested_full_node_ratio=ratio, promotion_status="not_promoted", failure_reason=reason, notes=str(exc)[:300])
                except Exception as exc:
                    row = make_t36_row(method=str(method), backend=str(backend).lower(), seed=int(args.seed), requested_full_node_ratio=ratio, promotion_status="not_promoted", failure_reason="exception", notes=(str(exc) + "\n" + traceback.format_exc())[:500])
                rows.append(row)
                write_csv(Path(args.tables_dir) / "t36_papers100m_disco_parity.csv", rows, T36_REQUIRED_FIELDS)
    print("status=completed")


if __name__ == "__main__":
    main()
