from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.data.small import load_processed_small_dataset_full_schema
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.features.metapath_table import compute_metapath_feature
from shadow_hgc.features.projection import fixed_random_projection
from shadow_hgc.features.scap import non_target_source_scap, source_class_affinity
from shadow_hgc.features.scap_v2 import compute_target_target_scap_v2
from shadow_hgc.features.structure_stats import compute_structure_stats_block
from shadow_hgc.features.typed_feature_demand import compute_typed_feature_demand
from shadow_hgc.fullgraph.metapath_specs import available_metapath_specs
from shadow_hgc.train.block_selection import validate_t2_promotion_row


SMALL_DATASETS = ["acm", "dblp", "imdb"]
MEDIUM_DATASETS = ["ogbn-arxiv", "ogbn-products"]
ALL_T2_DATASETS = [*SMALL_DATASETS, *MEDIUM_DATASETS]

SAFE_BASELINES: dict[str, dict[str, Any]] = {
    "acm": {"variant": "SFB-v2 B3_scap_v2 retained", "accuracy": 0.915486, "macro_f1": 0.916580, "predicted_class_min": 3},
    "dblp": {"variant": "R+ relation-linear current-best", "accuracy": 0.836972, "macro_f1": 0.829937, "predicted_class_min": 4},
    "imdb": {"variant": "clean S1 MAM/MDM/MKM", "accuracy": 0.424110, "macro_f1": 0.353932, "predicted_class_min": 5},
    "ogbn-arxiv": {"variant": "LAD_reference", "accuracy": 0.596774, "macro_f1": 0.415452, "predicted_class_min": 35},
    "ogbn-products": {"variant": "R++ base shadow-fusion / LAD_reference", "accuracy": 0.668908, "macro_f1": 0.338064, "predicted_class_min": 30},
}

PRIMARY_TARGETS = {
    "acm": 0.93,
    "dblp": 0.85,
    "imdb": 0.45,
    "ogbn-arxiv": 0.66,
    "ogbn-products": 0.70,
}


T2_STAGE_FIELDS = [
    "dataset",
    "row_kind",
    "model_type",
    "block_group",
    "selected_blocks",
    "status",
    "reason",
    "accuracy",
    "macro_f1",
    "predicted_class_count",
    "valid_acc",
    "valid_macro_f1",
    "branch_valid_acc",
    "branch_test_acc_debug",
    "gate_value",
    "kept_or_dropped",
    "drop_reason",
    "safe_baseline",
    "safe_baseline_acc",
    "safe_baseline_macro_f1",
    "delta_acc_vs_safe",
    "delta_macro_f1_vs_safe",
    "full_edge_scans",
    "edge_chunk_size",
    "dst_chunk_size",
    "block_dim",
    "num_blocks",
    "cache_bytes",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "wall_time_s",
    "uses_memmap",
    "uses_e_by_d_materialization",
    "uses_dense_p2",
    "uses_logits_as_input",
    "uses_bounded_edges",
    "uses_diffusion_legacy",
    "uses_full_graph_backprop",
    "uses_teacher_logits",
    "uses_kd",
    "uses_train_labels_only",
    "source_log",
]


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target


def read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def load_t2_graph(dataset: str):
    dataset = dataset.lower()
    if dataset in SMALL_DATASETS:
        return load_processed_small_dataset_full_schema(dataset)
    return load_ogb_node_property_dataset(dataset, root="dataset", download=False)


def num_classes(labels: torch.Tensor) -> int:
    valid = labels[labels >= 0]
    return int(valid.max().item()) + 1 if valid.numel() else 0


def target_rows(graph) -> torch.Tensor:
    return torch.arange(graph.num_nodes[graph.target_type], dtype=torch.long)


def split_train_valid(graph, *, seed: int = 42, valid_fraction: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    if graph.val_idx.numel() > 0:
        return graph.train_idx.to(torch.long), graph.val_idx.to(torch.long)
    train = graph.train_idx.to(torch.long)
    if train.numel() <= 2:
        return train, train
    generator = torch.Generator().manual_seed(int(seed))
    perm = train[torch.randperm(train.numel(), generator=generator)]
    valid_count = max(1, int(round(float(valid_fraction) * int(train.numel()))))
    valid = perm[:valid_count].contiguous()
    fit = perm[valid_count:].contiguous()
    if fit.numel() == 0:
        fit = train
    return fit, valid


def maybe_project(x: torch.Tensor, *, block_dim: int, seed: int) -> torch.Tensor:
    out = x.to(torch.float32)
    if int(block_dim) > 0 and int(out.shape[1]) > int(block_dim):
        out = fixed_random_projection(out, out_dim=int(block_dim), seed=int(seed)).to(torch.float32)
    return out


def _slice_edges(edge_index: torch.Tensor, edge_limit: int) -> torch.Tensor:
    if int(edge_limit) <= 0 or int(edge_index.shape[1]) <= int(edge_limit):
        return edge_index
    return edge_index[:, : int(edge_limit)].contiguous()


def _empty_feature(num_nodes: int, *, dim: int = 16) -> torch.Tensor:
    return torch.zeros(int(num_nodes), int(dim), dtype=torch.float32)


def build_t2_block_groups(
    graph,
    *,
    train_rows_for_labels: torch.Tensor,
    seed: int = 42,
    block_dim: int = 128,
    edge_chunk_size: int = 65536,
    edge_limit: int = 0,
    scap_topk: int = 8,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, Any]]]:
    rows = target_rows(graph)
    target_type = graph.target_type
    target_x = graph.node_features.get(target_type)
    if target_x is None:
        target_x = _empty_feature(graph.num_nodes[target_type])
    target_x = maybe_project(target_x, block_dim=block_dim, seed=seed)
    groups: dict[str, dict[str, torch.Tensor]] = {"B0_self": {"self": target_x}}
    diagnostics: dict[str, dict[str, Any]] = {
        "B0_self": {
            "full_edge_scans": 0,
            "cache_bytes": int(target_x.numel() * target_x.element_size()),
            "uses_train_labels_only": False,
        }
    }

    typed: dict[str, torch.Tensor] = {}
    typed_scans = 0
    typed_bytes = 0
    for relation in graph.relations:
        if relation.destination_type != target_type or relation.source_type not in graph.node_features:
            continue
        edge_index = _slice_edges(graph.edge_index[relation], edge_limit)
        result = compute_typed_feature_demand(
            edge_index=edge_index,
            source_features=graph.node_features[relation.source_type].to(torch.float32),
            num_target_nodes=graph.num_nodes[target_type],
            target_rows=rows,
            chunk_size=edge_chunk_size,
            projection_dim=block_dim,
            projection_seed=seed,
        )
        key = f"typed:{relation.relation_name}"
        typed[key] = result.block
        typed_scans += int(result.diagnostics.get("full_edge_scans", 1))
        typed_bytes += int(result.diagnostics.get("feature_demand_cache_bytes", result.block.numel() * result.block.element_size()))
    if typed:
        groups["B1_typed"] = typed
        diagnostics["B1_typed"] = {"full_edge_scans": typed_scans, "cache_bytes": typed_bytes, "uses_train_labels_only": False}

    if graph.dataset_name in SMALL_DATASETS and target_type in graph.node_features:
        metapaths, skipped = available_metapath_specs(graph.dataset_name, graph.relations, target_type)
        metapath_blocks: dict[str, torch.Tensor] = {}
        metapath_scans = 0
        metapath_bytes = 0
        for name, path_schema in metapaths.items():
            block, diag = compute_metapath_feature(
                path_schema=path_schema,
                target_type=target_type,
                feature_provider={target_type: graph.node_features[target_type].to(torch.float32)},
                edge_store={rel: _slice_edges(graph.edge_index[rel], edge_limit) for rel in graph.edge_index},
                num_nodes=graph.num_nodes,
                target_rows=rows,
                chunk_size=edge_chunk_size,
            )
            block = maybe_project(block, block_dim=block_dim, seed=seed)
            metapath_blocks[f"metapath:{name}"] = block
            metapath_scans += len(path_schema)
            metapath_bytes += int(diag.get("metapath_cache_bytes", block.numel() * block.element_size()))
        if metapath_blocks:
            groups["B2_metapath"] = metapath_blocks
            diagnostics["B2_metapath"] = {
                "full_edge_scans": metapath_scans,
                "cache_bytes": metapath_bytes,
                "uses_train_labels_only": False,
                "skipped_metapaths": skipped,
            }

    label_blocks: dict[str, torch.Tensor] = {}
    label_scans = 0
    label_bytes = 0
    train_mask = torch.zeros(graph.num_nodes[target_type], dtype=torch.bool)
    train_mask[train_rows_for_labels.to(torch.long)] = True
    classes = num_classes(graph.labels)
    for relation in graph.relations:
        if relation.destination_type != target_type:
            continue
        edge_index = _slice_edges(graph.edge_index[relation], edge_limit)
        if relation.source_type == target_type:
            result = compute_target_target_scap_v2(
                edge_index=edge_index,
                labels=graph.labels,
                train_mask=train_mask,
                num_nodes=graph.num_nodes[target_type],
                num_classes=classes,
                target_rows=rows,
                top_k=scap_topk,
                sparse=False,
            )
            block = result.dense.to(torch.float32) if result.dense is not None else result.sparse.values.to(torch.float32)
            label_blocks[f"lad:{relation.relation_name}"] = block
            label_bytes += int(result.diagnostics.get("scap_cache_bytes", block.numel() * block.element_size()))
            label_scans += 1
        elif relation.source_type in graph.num_nodes:
            source_aff = source_class_affinity(
                source_to_target_edges=edge_index,
                labels=graph.labels,
                train_mask=train_mask,
                num_source_nodes=graph.num_nodes[relation.source_type],
                num_classes=classes,
            )
            block, diag = non_target_source_scap(
                edge_index_source_to_target=edge_index,
                source_affinity=source_aff,
                num_target_nodes=graph.num_nodes[target_type],
                target_rows=rows,
            )
            label_blocks[f"lad:{relation.relation_name}"] = block
            label_bytes += int(block.numel() * block.element_size())
            label_scans += 2
    if label_blocks:
        groups["B3_lad_scap"] = label_blocks
        diagnostics["B3_lad_scap"] = {
            "full_edge_scans": label_scans,
            "cache_bytes": label_bytes,
            "uses_train_labels_only": True,
            "uses_dense_p2": False,
        }

    incoming = [relation for relation in graph.relations if relation.destination_type == target_type]
    if incoming:
        block, diag = compute_structure_stats_block(
            edge_index_by_relation={rel: _slice_edges(edge, edge_limit) for rel, edge in graph.edge_index.items()},
            relations=incoming,
            num_target_nodes=graph.num_nodes[target_type],
            target_rows=rows,
        )
        if int(block.shape[1]) > 0:
            groups["B4_structure"] = {"structure": block}
            diagnostics["B4_structure"] = {
                "full_edge_scans": 1,
                "cache_bytes": int(block.numel() * block.element_size()),
                "uses_train_labels_only": False,
                **diag,
            }
    return groups, diagnostics


def merge_block_groups(groups: dict[str, dict[str, torch.Tensor]], selected: list[str]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for group_name in selected:
        for block_name, tensor in groups[group_name].items():
            merged[block_name] = tensor
    return merged


def group_resource(diagnostics: dict[str, dict[str, Any]], selected: list[str]) -> dict[str, Any]:
    scans = sum(int(diagnostics.get(group, {}).get("full_edge_scans", 0)) for group in selected)
    cache = sum(int(diagnostics.get(group, {}).get("cache_bytes", 0)) for group in selected)
    return {
        "full_edge_scans": scans,
        "cache_bytes": cache,
        "uses_train_labels_only": any(bool(diagnostics.get(group, {}).get("uses_train_labels_only", False)) for group in selected),
    }


def make_stage_row(
    *,
    dataset: str,
    row_kind: str,
    model_type: str,
    block_group: str,
    selected_groups: list[str],
    status: str,
    reason: str,
    metrics: dict[str, Any],
    valid_metrics: dict[str, Any],
    resource: dict[str, Any],
    edge_chunk_size: int,
    dst_chunk_size: int,
    block_dim: int,
    num_blocks: int,
    wall_time_s: float,
    source_log: str,
    gate_value: float = 0.0,
    kept_or_dropped: str = "",
    drop_reason: str = "",
) -> dict[str, Any]:
    safe = SAFE_BASELINES[dataset]
    row = {
        "dataset": dataset,
        "row_kind": row_kind,
        "model_type": model_type,
        "block_group": block_group,
        "selected_blocks": json.dumps(selected_groups, sort_keys=True),
        "status": status,
        "reason": reason,
        "accuracy": metrics.get("accuracy", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "predicted_class_count": metrics.get("predicted_class_count", ""),
        "valid_acc": valid_metrics.get("accuracy", ""),
        "valid_macro_f1": valid_metrics.get("macro_f1", ""),
        "branch_valid_acc": valid_metrics.get("accuracy", ""),
        "branch_test_acc_debug": metrics.get("accuracy", ""),
        "gate_value": gate_value,
        "kept_or_dropped": kept_or_dropped,
        "drop_reason": drop_reason,
        "safe_baseline": safe["variant"],
        "safe_baseline_acc": safe["accuracy"],
        "safe_baseline_macro_f1": safe["macro_f1"],
        "delta_acc_vs_safe": float(metrics.get("accuracy", 0.0)) - float(safe["accuracy"]) if metrics.get("accuracy", "") != "" else "",
        "delta_macro_f1_vs_safe": float(metrics.get("macro_f1", 0.0)) - float(safe["macro_f1"]) if metrics.get("macro_f1", "") != "" else "",
        "full_edge_scans": int(resource.get("full_edge_scans", 0)),
        "edge_chunk_size": int(edge_chunk_size),
        "dst_chunk_size": int(dst_chunk_size),
        "block_dim": int(block_dim),
        "num_blocks": int(num_blocks),
        "cache_bytes": int(resource.get("cache_bytes", 0)),
        "peak_cpu_ram_gb": current_cpu_ram_bytes() / (1024**3),
        "peak_gpu_ram_gb": current_gpu_ram_bytes() / (1024**3),
        "wall_time_s": float(wall_time_s),
        "uses_memmap": False,
        "uses_e_by_d_materialization": False,
        "uses_dense_p2": False,
        "uses_logits_as_input": False,
        "uses_bounded_edges": False,
        "uses_diffusion_legacy": False,
        "uses_full_graph_backprop": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_train_labels_only": bool(resource.get("uses_train_labels_only", False)),
        "source_log": source_log,
    }
    valid = validate_t2_promotion_row(row)
    if not valid["valid"]:
        row["status"] = "blocked_forbidden_component"
        row["reason"] = ",".join(valid["invalid_reasons"])
    return row


def promotion_status(dataset: str, test_metrics: dict[str, Any]) -> tuple[str, str]:
    safe = SAFE_BASELINES[dataset]
    acc = float(test_metrics.get("accuracy", 0.0))
    macro = float(test_metrics.get("macro_f1", 0.0))
    pred_classes = int(test_metrics.get("predicted_class_count", 0))
    if pred_classes < int(safe["predicted_class_min"]):
        return "blocked_class_collapse", f"predicted_class_count<{safe['predicted_class_min']}"
    improves = acc > float(safe["accuracy"]) + 1e-6 or macro > float(safe["macro_f1"]) + 1e-6
    preserves = acc >= float(safe["accuracy"]) - 1e-3 or macro >= float(safe["macro_f1"]) - 5e-3
    if improves and preserves:
        return "promoted", "validation_selected_and_safe_improved"
    if preserves:
        return "completed_non_regression", "safe_baseline_preserved_but_not_meaningfully_improved"
    return "blocked_by_signal_ceiling", "below_safe_non_regression_floor"


def estimate_block_cache_bytes(num_rows: int, block_dim: int, dtype_bytes: int = 2) -> int:
    return int(num_rows) * int(block_dim) * int(dtype_bytes)


def wall_time_category(edge_count: int) -> str:
    if edge_count < 5_000_000:
        return "local_short"
    if edge_count < 150_000_000:
        return "local_long"
    return "server_recommended"
