from __future__ import annotations

from typing import Any

from shadow_hgc.sft.stt_cache import estimate_stt_cache_bytes
from shadow_hgc.sft.t34_contract import apply_t34_promotion_guard, make_t34_row


def plan_ultra_stt(
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
    estimates = estimate_stt_cache_bytes(num_nodes=int(num_nodes), num_classes=int(num_classes), mode=str(teacher_cache_mode))
    dense = str(teacher_cache_mode) == "dense_fp16"
    signature_cache_bytes = int(num_nodes) * int(signature_dim) * 2
    reservoir_bytes = planned * (int(signature_dim) * 2 + min(int(num_classes), 16) * 4 + 64)
    row = make_t34_row(
        dataset=dataset,
        method=f"ultra_stt_planner_{dataset.replace('ogbn-', '')}",
        seed=int(seed),
        requested_full_node_ratio=float(requested_ratio),
        condensed_nodes=planned,
        num_nodes=int(num_nodes),
        num_edges=int(num_edges),
        num_classes=int(num_classes),
        status="completed_dry_run",
        promotion_track="ultra_planner",
        promotion_status="promoted",
        teacher_cache_mode=str(teacher_cache_mode),
        teacher_cache_bytes=estimates["teacher_cache_bytes"],
        teacher_dense_cache_bytes_diagnostic=estimates["teacher_dense_cache_bytes_diagnostic"],
        cache_compression_ratio=estimates["cache_compression_ratio"],
        uses_dense_nxc_teacher_cache=dense,
        uses_dense_p2=False,
        uses_e_by_d_materialization=False,
        uses_e_by_d=False,
        uses_full_edge_index_on_gpu=False,
        uses_all_pair=False,
        planned_condensed_nodes=planned,
        teacher_topk_cache_bytes=estimates["teacher_cache_bytes"],
        selection_reservoir_bytes=reservoir_bytes,
        sft_cache_bytes_estimate=signature_cache_bytes,
        semantic_cache_bytes_estimate_if_any=0,
        selection_passes=1,
        edge_scans=2,
        peak_ram_estimate=reservoir_bytes + min(signature_cache_bytes, int(8e9)),
        estimated_edge_scans=2,
        estimated_peak_cpu_ram=reservoir_bytes + min(signature_cache_bytes, int(8e9)),
        estimated_peak_gpu_ram=0,
        notes=f"dry-run planner; num_nodes={num_nodes}; num_edges={num_edges}; num_classes={num_classes}",
    )
    return apply_t34_promotion_guard(row)
