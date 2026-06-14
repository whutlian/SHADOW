from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import read_csv, write_csv
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.ultra.papers100m_ant import materialize_ant_edges, train_or_load_ant_link_predictor
from shadow_hgc.ultra.papers100m_condensed import materialize_condensed_table, train_and_eval_condensed_table
from shadow_hgc.ultra.papers100m_nested_bank import NESTED_BANK_POLICY, build_nested_bank_v2
from shadow_hgc.ultra.papers100m_ratio_policy_v2 import ratio_policy_v2
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_sgc_backend import train_and_eval_sgc_condensed
from shadow_hgc.ultra.papers100m_t36_contract import T36_REQUIRED_FIELDS, make_t36_row, validate_t36_row


def _ratio_dir_name(ratio: float) -> str:
    return f"ratio={float(ratio):.12g}".replace("+", "")


def _run_backend(ctx: Papers100MCacheContext, ratio: float, backend: str, policy_values, device: str, ant_edge_dir: Path | None = None) -> dict[str, Any]:
    if str(backend).lower() == "sgc":
        return train_and_eval_sgc_condensed(
            ctx,
            ratio,
            epochs=180,
            temperature=policy_values.soft_temperature,
            lambda_hard=policy_values.lambda_hard,
            lambda_prior=policy_values.lambda_prior,
            device=device,
            ant_edge_dir=ant_edge_dir,
        )
    if str(backend).lower() in {"gamlp_table", "gamlp"}:
        return train_and_eval_condensed_table(
            ctx,
            ratio,
            student="papers100m_gamlp_table",
            hidden_dim=512,
            epochs=300,
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
    parser = argparse.ArgumentParser(description="T36 papers100M scale-fidelity runner.")
    parser.add_argument("--cache-root", default="caches/papers100m/stt_v1")
    parser.add_argument("--teacher-id", default="current_teacher")
    parser.add_argument("--nested-bank-id", default="current_nested_bank")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0005, 0.001, 0.002, 0.005, 0.010])
    parser.add_argument("--methods", nargs="+", default=["stt_nested_bank_v2", "stt_hard_anchor_v2"])
    parser.add_argument("--backends", nargs="+", default=["gamlp_table", "sgc"])
    parser.add_argument("--policy", default=NESTED_BANK_POLICY)
    parser.add_argument("--force-bank", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--tables-dir", default="experiments/tables")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    max_ratio = max(float(value) for value in args.ratios)
    policy = str(args.policy)
    ctx = Papers100MCacheContext(cache_root, selection_policy=policy, seed=int(args.seed))
    bank = build_nested_bank_v2(ctx, policy=policy, seed=int(args.seed), max_ratio=max_ratio, teacher_id=str(args.teacher_id), force=bool(args.force_bank))
    link = train_or_load_ant_link_predictor(cache_root, nested_bank_id=bank["nested_bank_id"], teacher_id=str(args.teacher_id), seed=int(args.seed))
    rows: list[dict[str, Any]] = read_csv(Path(args.tables_dir) / "t36_papers100m_scale_fidelity.csv")
    for method in args.methods:
        for ratio in [float(value) for value in args.ratios]:
            ctx = Papers100MCacheContext(cache_root, selection_policy=policy, seed=int(args.seed))
            materialized = materialize_condensed_table(ctx, ratio, policy=policy, seed=int(args.seed))
            policy_values = ratio_policy_v2(ratio)
            ant_manifest: dict[str, Any] | None = None
            ant_edge_dir: Path | None = None
            if str(method).startswith("stt_ant"):
                ant_manifest = materialize_ant_edges(cache_root, policy=policy, seed=int(args.seed), ratio=ratio, edge_topk=16, link_predictor_id=link["ant_link_predictor_id"])
                ant_edge_dir = cache_root / "condensed" / _ratio_dir_name(ratio) / "ant_edges_topk16"
            for backend in args.backends:
                row = make_t36_row(
                    method=str(method),
                    backend=str(backend).lower(),
                    seed=int(args.seed),
                    requested_full_node_ratio=ratio,
                    full_node_denominator=int(ctx.manifest["num_nodes"]),
                    condensed_nodes=int(materialized.get("condensed_nodes", 0)),
                    target_universe_size=int(ctx.manifest["target_universe_size"]),
                    cache_build_id=ctx.cache_ids()["cache_build_id"],
                    edge_cache_id=ctx.cache_ids()["edge_slice_cache_id"],
                    sft_cache_id=ctx.cache_ids()["sft_cache_id"],
                    teacher_cache_id=ctx.cache_ids()["teacher_cache_id"],
                    selection_bank_id=ctx.cache_ids()["selection_bank_id"],
                    nested_bank_id=bank["nested_bank_id"],
                    teacher_id=str(args.teacher_id),
                    teacher_test_acc=ctx.teacher.get("accuracy", ""),
                    teacher_valid_acc=ctx.teacher.get("valid_acc", ""),
                    teacher_cache_mode=ctx.teacher.get("teacher_cache_mode", ""),
                    teacher_cache_bytes=ctx.teacher.get("teacher_cache_bytes", ""),
                    ant_enabled=ant_manifest is not None,
                    ant_edge_topk=16 if ant_manifest else 0,
                    ant_link_predictor_id=link["ant_link_predictor_id"] if ant_manifest else "",
                    ant_edges=ant_manifest.get("ant_edges", 0) if ant_manifest else 0,
                    ant_candidate_count=ant_manifest.get("ant_candidate_count", 0) if ant_manifest else 0,
                    materialize_time=materialized.get("condensed_materialize_time", ""),
                    condensed_bytes=materialized.get("condensed_cache_bytes", ""),
                    **policy_values.as_row_fields(),
                )
                if bool(args.run_long):
                    try:
                        metrics = _run_backend(ctx, ratio, str(backend), policy_values, str(args.device), ant_edge_dir if str(backend).lower() == "sgc" else None)
                        row.update(metrics)
                        row["promotion_status"] = "promoted"
                    except RuntimeError as exc:
                        row["promotion_status"] = "not_promoted"
                        row["failure_reason"] = "OOM" if "out of memory" in str(exc).lower() else "runtime_error"
                        row["notes"] = str(exc)[:300]
                else:
                    row["promotion_status"] = "diagnostic"
                    row["failure_reason"] = "materialized_only_without_run_long"
                row["peak_cpu_ram"] = current_cpu_ram_bytes()
                row["peak_gpu_ram"] = current_gpu_ram_bytes()
                if row["promotion_status"] == "promoted" and not validate_t36_row(row)["valid"]:
                    row["promotion_status"] = "not_promoted"
                    row["failure_reason"] = ",".join(validate_t36_row(row)["forbidden_flags"])
                rows.append(row)
                write_csv(Path(args.tables_dir) / "t36_papers100m_scale_fidelity.csv", rows, T36_REQUIRED_FIELDS)
    print("status=completed")


if __name__ == "__main__":
    main()
