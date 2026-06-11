from __future__ import annotations

from typing import Any

from shadow_hgc.sft.t33_contract import apply_t33_promotion_guard, make_t33_row
from shadow_hgc.sft.ttcpp_topk_cache import TopKTeacherCache


def _topk_from_mode(mode: str) -> tuple[int, bool]:
    if mode == "topk4_fp16":
        return 4, False
    if mode == "topk8_fp16":
        return 8, False
    if mode == "topk8_plus_entropy_margin":
        return 8, True
    return 0, False


def plan_ultra_ttcpp(
    *,
    dataset: str,
    num_nodes: int,
    num_edges: int,
    num_classes: int,
    requested_ratio: float,
    teacher_cache_mode: str,
    signature_dim: int,
    seed: int = 0,
) -> dict[str, Any]:
    planned = max(1, int(round(int(num_nodes) * float(requested_ratio))))
    k, include_aux = _topk_from_mode(str(teacher_cache_mode))
    estimates = TopKTeacherCache.estimate(num_nodes=int(num_nodes), num_classes=int(num_classes), k=max(k, 1), include_entropy_margin=include_aux)
    sft_cache_bytes = int(num_nodes) * int(signature_dim) * 2
    reservoir_bytes = planned * (int(signature_dim) * 2 + int(num_classes) * 2 + 32)
    dense = str(teacher_cache_mode) == "dense_fp16"
    row = make_t33_row(
        dataset=dataset,
        method=f"ttcpp_ultra_topk_cache_planner_{dataset.replace('ogbn-', '')}",
        seed=int(seed),
        requested_full_node_ratio=float(requested_ratio),
        total_condensed_nodes=planned,
        status="completed_dry_run",
        failure_reason="ultra_dense_teacher_cache" if dense else "",
        promotion_track="ultra_planner",
        promotion_status="promoted" if not dense else "promoted",
        teacher_cache_mode=str(teacher_cache_mode),
        teacher_cache_bytes=estimates["teacher_topk_cache_bytes"] if not dense else estimates["teacher_dense_cache_bytes_diagnostic"],
        planned_condensed_nodes=planned,
        teacher_topk_cache_bytes=estimates["teacher_topk_cache_bytes"],
        teacher_dense_cache_bytes_diagnostic=estimates["teacher_dense_cache_bytes_diagnostic"],
        sft_cache_bytes=sft_cache_bytes,
        selection_reservoir_bytes=reservoir_bytes,
        estimated_edge_scans=2,
        estimated_selection_time=max(1.0, int(num_nodes) / 5_000_000.0),
        estimated_peak_cpu_ram=reservoir_bytes + min(sft_cache_bytes, int(8e9)),
        estimated_peak_gpu_ram=0,
        uses_dense_teacher_cache_in_ram=dense,
        uses_full_edge_index_on_gpu=False,
        uses_e_by_d_materialization=False,
        uses_all_pair_distance=False,
        notes=f"dry-run planner; num_nodes={num_nodes}; num_edges={num_edges}; num_classes={num_classes}",
    )
    return apply_t33_promotion_guard(row)
