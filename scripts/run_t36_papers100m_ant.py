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
from shadow_hgc.ultra.papers100m_teacher_upgrade import install_teacher_upgrade
from shadow_hgc.ultra.papers100m_memmap import read_json


def _resolve_teacher(cache_root: Path, teacher_id: str) -> str:
    if teacher_id == "best_t36_teacher":
        best_path = cache_root / "teacher_upgrade" / "best_teacher.json"
        if best_path.exists():
            best = read_json(best_path)
            install_teacher_upgrade(cache_root, str(best["teacher_id"]))
            return str(best["teacher_id"])
    return str(teacher_id)


def _ratio_dir_name(ratio: float) -> str:
    return f"ratio={float(ratio):.12g}".replace("+", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="T36 papers100M ANT branch runner.")
    parser.add_argument("--cache-root", default="caches/papers100m/stt_v1")
    parser.add_argument("--teacher-id", default="current_teacher")
    parser.add_argument("--nested-bank-id", default="current_nested_bank")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.00005, 0.00010, 0.00020, 0.00050])
    parser.add_argument("--edge-topks", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--backends", nargs="+", default=["sgc"])
    parser.add_argument("--candidate-builders", nargs="*", default=["same_pred_class_knn", "anchor_bucket", "neighbor_sketch", "degree_bucket"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--tables-dir", default="experiments/tables")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    teacher_id = _resolve_teacher(cache_root, str(args.teacher_id))
    ctx = Papers100MCacheContext(cache_root, selection_policy=NESTED_BANK_POLICY, seed=int(args.seed))
    bank = build_nested_bank_v2(ctx, policy=NESTED_BANK_POLICY, seed=int(args.seed), max_ratio=max(float(v) for v in args.ratios), teacher_id=teacher_id)
    link = train_or_load_ant_link_predictor(cache_root, nested_bank_id=bank["nested_bank_id"], teacher_id=teacher_id, seed=int(args.seed))
    rows: list[dict[str, Any]] = read_csv(Path(args.tables_dir) / "t36_papers100m_ant.csv")
    for ratio in [float(v) for v in args.ratios]:
        ctx = Papers100MCacheContext(cache_root, selection_policy=NESTED_BANK_POLICY, seed=int(args.seed))
        materialized = materialize_condensed_table(ctx, ratio, policy=NESTED_BANK_POLICY, seed=int(args.seed))
        policy_values = ratio_policy_v2(ratio)
        for edge_topk in [int(v) for v in args.edge_topks]:
            ant = materialize_ant_edges(
                cache_root,
                policy=NESTED_BANK_POLICY,
                seed=int(args.seed),
                ratio=ratio,
                edge_topk=edge_topk,
                link_predictor_id=link["ant_link_predictor_id"],
            )
            edge_dir = cache_root / "condensed" / _ratio_dir_name(ratio) / f"ant_edges_topk{edge_topk}"
            for backend in args.backends:
                row = make_t36_row(
                    method=f"stt_ant_topk{edge_topk}",
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
                    teacher_id=teacher_id,
                    teacher_test_acc=ctx.teacher.get("accuracy", ""),
                    teacher_valid_acc=ctx.teacher.get("valid_acc", ""),
                    teacher_cache_mode=ctx.teacher.get("teacher_cache_mode", ""),
                    teacher_cache_bytes=ctx.teacher.get("teacher_cache_bytes", ""),
                    ant_enabled=True,
                    ant_edge_topk=edge_topk,
                    ant_link_predictor_id=link["ant_link_predictor_id"],
                    ant_edges=ant["ant_edges"],
                    ant_candidate_count=ant["ant_candidate_count"],
                    materialize_time=materialized.get("condensed_materialize_time", ""),
                    condensed_bytes=materialized.get("condensed_cache_bytes", ""),
                    notes=f"candidate_builders={','.join(args.candidate_builders)};ant_bounded={ant['ant_bounded']}",
                    **policy_values.as_row_fields(),
                )
                if bool(args.run_long):
                    if str(backend).lower() == "sgc":
                        metrics = train_and_eval_sgc_condensed(
                            ctx,
                            ratio,
                            epochs=180,
                            temperature=policy_values.soft_temperature,
                            lambda_hard=policy_values.lambda_hard,
                            lambda_prior=policy_values.lambda_prior,
                            device=str(args.device),
                            ant_edge_dir=edge_dir,
                        )
                    else:
                        metrics = train_and_eval_condensed_table(
                            ctx,
                            ratio,
                            student="papers100m_gamlp_table",
                            hidden_dim=512,
                            epochs=260,
                            temperature=policy_values.soft_temperature,
                            lambda_hard=policy_values.lambda_hard,
                            lambda_prior=policy_values.lambda_prior,
                            device=str(args.device),
                        )
                    row.update(metrics)
                    row["promotion_status"] = "promoted" if str(backend).lower() == "sgc" and float(row.get("teacher_test_acc", 0.0) or 0.0) >= 0.55 else "diagnostic"
                    if row["promotion_status"] != "promoted":
                        row["failure_reason"] = "teacher_below_0p55_gate_or_non_sgc_backend"
                else:
                    row["promotion_status"] = "diagnostic"
                    row["failure_reason"] = "materialized_only_without_run_long"
                row["peak_cpu_ram"] = current_cpu_ram_bytes()
                row["peak_gpu_ram"] = current_gpu_ram_bytes()
                if row["promotion_status"] == "promoted" and not validate_t36_row(row)["valid"]:
                    row["promotion_status"] = "not_promoted"
                    row["failure_reason"] = ",".join(validate_t36_row(row)["forbidden_flags"])
                rows.append(row)
                write_csv(Path(args.tables_dir) / "t36_papers100m_ant.csv", rows, T36_REQUIRED_FIELDS)
    print("status=completed")


if __name__ == "__main__":
    main()
