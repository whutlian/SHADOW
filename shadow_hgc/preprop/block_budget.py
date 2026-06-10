from __future__ import annotations

from typing import Any, Sequence


def _block_bytes(block: str, *, rows: int, feature_dim: int, num_classes: int, dtype_bytes: int = 2) -> tuple[str, int]:
    name = str(block)
    if name.startswith("Y"):
        return "label", int(rows) * int(num_classes) * int(dtype_bytes)
    if name == "structure":
        return "structure", int(rows) * 8 * int(dtype_bytes)
    return "feature", int(rows) * int(feature_dim) * int(dtype_bytes)


def estimate_block_budget(
    *,
    dataset: str,
    num_target_nodes: int,
    num_train_target_nodes: int,
    num_edges: int,
    num_classes: int,
    feature_dim: int,
    selected_blocks: Sequence[str],
    dtype_bytes: int = 2,
) -> list[dict[str, Any]]:
    rows_out = []
    for mode, rows in (("all_target_rows", int(num_target_nodes)), ("train_target_only", int(num_train_target_nodes))):
        feature_bytes = 0
        label_bytes = 0
        structure_bytes = 0
        for block in selected_blocks:
            kind, value = _block_bytes(block, rows=rows, feature_dim=feature_dim, num_classes=num_classes, dtype_bytes=dtype_bytes)
            if kind == "feature":
                feature_bytes += value
            elif kind == "label":
                label_bytes += value
            else:
                structure_bytes += value
        scans = sum(1 for block in selected_blocks if str(block).startswith(("X1", "X2", "X3", "X4", "Y1", "Y2", "Y3")))
        rows_out.append(
            {
                "dataset": dataset,
                "num_nodes": int(num_target_nodes),
                "num_edges": int(num_edges),
                "num_classes": int(num_classes),
                "block_set": ",".join(str(block) for block in selected_blocks),
                "cache_mode": mode,
                "total_cache_bytes": int(feature_bytes + label_bytes + structure_bytes),
                "feature_cache_bytes": int(feature_bytes),
                "label_cache_bytes": int(label_bytes),
                "structure_cache_bytes": int(structure_bytes),
                "full_edge_scans": int(scans),
                "peak_cpu_ram_estimate_gb": float(max(2.0, min(64.0, (feature_bytes + label_bytes + structure_bytes) / (1024**3) * 0.15 + 2.0))),
                "peak_gpu_ram_estimate_gb": float(max(1.0, int(feature_dim) * len(selected_blocks) * 16384 * 4 / (1024**3) * 4.0)),
                "wall_time_category": "server_recommended" if int(num_edges) >= 150_000_000 else ("local_long" if int(num_edges) >= 5_000_000 else "local_short"),
                "server_recommended": bool(int(num_edges) >= 150_000_000),
                "uses_logits_as_input": False,
                "uses_kd": False,
                "uses_dense_p2": False,
                "uses_e_by_d_materialization": False,
            }
        )
    return rows_out


def select_blocks_for_budget(
    *,
    requested_blocks: Sequence[str],
    num_rows: int,
    feature_dim: int,
    num_classes: int,
    max_cache_gb: float,
    dtype_bytes: int = 2,
) -> list[str]:
    selected: list[str] = []
    used = 0
    max_bytes = float(max_cache_gb) * (1024**3)
    for block in requested_blocks:
        _, cost = _block_bytes(block, rows=int(num_rows), feature_dim=int(feature_dim), num_classes=int(num_classes), dtype_bytes=int(dtype_bytes))
        if used + cost <= max_bytes or str(block) == "X0":
            selected.append(str(block))
            used += int(cost)
    return selected
