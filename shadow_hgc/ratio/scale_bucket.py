from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


BucketName = Literal["medium", "large", "ultra"]


@dataclass(frozen=True)
class ScaleBucketSpec:
    name: BucketName
    min_nodes: int
    max_nodes: int | None
    default_ratio: float
    sweep_ratios: tuple[float, ...]


SCALE_BUCKETS: dict[BucketName, ScaleBucketSpec] = {
    "medium": ScaleBucketSpec("medium", 100_000, 1_000_000, 0.005, (0.001, 0.0025, 0.005, 0.01)),
    "large": ScaleBucketSpec("large", 1_000_000, 10_000_000, 0.0025, (0.0005, 0.001, 0.0025, 0.005)),
    "ultra": ScaleBucketSpec("ultra", 10_000_000, None, 0.0001, (0.00001, 0.00005, 0.0001, 0.0005)),
}

DATASET_BUCKETS: dict[str, BucketName] = {
    "ogbn-arxiv": "medium",
    "arxiv": "medium",
    "reddit": "medium",
    "Reddit": "medium",
    "ogbn-products": "large",
    "products": "large",
    "ogbn-papers100M": "ultra",
    "papers100M": "ultra",
    "MAG240M": "ultra",
    "mag240m": "ultra",
}

T24_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "uses_logits_as_input",
    "uses_logits",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_legacy_diffusion",
    "uses_diffusion_legacy",
    "uses_old_diffusion",
    "uses_coverage_medoid",
    "uses_source_anchors",
    "uses_bounded_edges",
    "uses_bounded_edge_performance",
    "uses_e_by_d",
    "uses_e_by_d_materialization",
    "uses_dense_metapath_adjacency",
    "is_proxy",
)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def bucket_for_node_count(num_nodes: int) -> BucketName:
    nodes = int(num_nodes)
    for name, spec in SCALE_BUCKETS.items():
        if nodes >= spec.min_nodes and (spec.max_nodes is None or nodes < spec.max_nodes):
            return name
    raise ValueError(f"node count {nodes} is outside T24 medium/large/ultra buckets")


def bucket_for_dataset(dataset: str, *, num_nodes: int | None = None) -> BucketName:
    key = str(dataset)
    if key in DATASET_BUCKETS:
        return DATASET_BUCKETS[key]
    lower = key.lower()
    if lower in DATASET_BUCKETS:
        return DATASET_BUCKETS[lower]
    if num_nodes is None:
        raise ValueError(f"unknown dataset bucket for {dataset}; pass num_nodes for auto bucket")
    return bucket_for_node_count(int(num_nodes))


def ratio_preset(*, dataset: str | None = None, num_nodes: int | None = None, scale_bucket: str = "auto", preset: str = "bucket_default") -> list[float]:
    if scale_bucket == "auto":
        if dataset is None and num_nodes is None:
            raise ValueError("dataset or num_nodes is required for auto scale bucket")
        bucket = bucket_for_dataset(dataset or "", num_nodes=num_nodes)
    else:
        bucket = scale_bucket  # type: ignore[assignment]
    if bucket not in SCALE_BUCKETS:
        raise ValueError(f"unsupported scale bucket: {bucket}")
    spec = SCALE_BUCKETS[bucket]  # type: ignore[index]
    if preset == "bucket_default":
        return [float(spec.default_ratio)]
    if preset == "bucket_sweep":
        return [float(value) for value in spec.sweep_ratios]
    raise ValueError(f"unsupported ratio preset: {preset}")


def account_full_node_ratio(
    *,
    original_total_nodes: int,
    target_prototypes: int,
    shadow_nodes: int,
    other_condensed_nodes: int = 0,
    condensed_edges: int = 0,
    original_feature_bytes: int | None = None,
    condensed_feature_bytes: int | None = None,
) -> dict[str, Any]:
    total = int(target_prototypes) + int(shadow_nodes) + int(other_condensed_nodes)
    original = max(1, int(original_total_nodes))
    ratio = float(total) / float(original)
    byte_compression = ""
    if original_feature_bytes is not None and condensed_feature_bytes is not None and int(original_feature_bytes) > 0:
        byte_compression = float(condensed_feature_bytes) / float(original_feature_bytes)
    return {
        "requested_full_node_ratio": "",
        "actual_full_node_ratio": ratio,
        "target_prototypes": int(target_prototypes),
        "shadow_nodes": int(shadow_nodes),
        "other_condensed_nodes": int(other_condensed_nodes),
        "total_condensed_nodes": int(total),
        "total_condensed_edges": int(condensed_edges),
        "condensed_nodes": int(total),
        "condensed_edges": int(condensed_edges),
        "byte_size_compression": byte_compression,
    }


def validate_t24_promoted_row(row: dict[str, Any]) -> dict[str, Any]:
    forbidden = [flag for flag in T24_FORBIDDEN_FLAGS if _truthy(row.get(flag, False))]
    if _truthy(row.get("is_proxy", False)) or str(row.get("status", "")).endswith("proxy"):
        if "is_proxy" not in forbidden:
            forbidden.append("is_proxy")
    if str(row.get("promotion_status", "")).startswith("promoted") and row.get("actual_full_node_ratio", "") in {"", None}:
        forbidden.append("missing_full_node_ratio")
    return {"valid": not forbidden, "forbidden_flags": forbidden}


def fixed_bucket_main_rows() -> list[dict[str, Any]]:
    datasets = [
        ("ogbn-arxiv", 169_343),
        ("Reddit", 232_965),
        ("ogbn-products", 2_449_029),
    ]
    rows: list[dict[str, Any]] = []
    for dataset, nodes in datasets:
        bucket = bucket_for_dataset(dataset, num_nodes=nodes)
        rows.append(
            {
                "dataset": dataset,
                "nodes": nodes,
                "scale_bucket": bucket,
                "ratio_mode": "full_node",
                "main_ratio": SCALE_BUCKETS[bucket].default_ratio,
                "sweep_ratios": ",".join(str(value) for value in SCALE_BUCKETS[bucket].sweep_ratios),
            }
        )
    return rows
