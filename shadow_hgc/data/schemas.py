from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set

import torch


@dataclass(frozen=True, order=True)
class DirectedRelation:
    """A message relation represented as source --relation_name--> destination."""

    source_type: str
    relation_name: str
    destination_type: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("source_type", self.source_type),
            ("relation_name", self.relation_name),
            ("destination_type", self.destination_type),
        ):
            if not value or not isinstance(value, str):
                raise ValueError(f"{field_name} must be a non-empty string")

    def key(self) -> tuple[str, str, str]:
        return (self.source_type, self.relation_name, self.destination_type)

    def is_incoming_to(self, node_type: str) -> bool:
        return self.destination_type == node_type

    def is_target_target(self, target_type: str) -> bool:
        return self.source_type == target_type and self.destination_type == target_type

    def __str__(self) -> str:
        return f"{self.source_type}--{self.relation_name}-->{self.destination_type}"


def validate_edge_index_direction(
    relation: DirectedRelation,
    edge_index: torch.Tensor,
    *,
    num_src_nodes: int,
    num_dst_nodes: int,
) -> None:
    """Validate the fixed edge convention edge_index[0]=source, edge_index[1]=destination."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"{relation}: edge_index must have shape [2, num_edges]")
    if edge_index.dtype != torch.long:
        raise ValueError(f"{relation}: edge_index must use torch.long indices")
    if edge_index.numel() == 0:
        return

    src = edge_index[0]
    dst = edge_index[1]
    if int(src.min()) < 0 or int(src.max()) >= num_src_nodes:
        raise ValueError(
            f"{relation}: source indices in edge_index[0] exceed source node count "
            f"{num_src_nodes}"
        )
    if int(dst.min()) < 0 or int(dst.max()) >= num_dst_nodes:
        raise ValueError(
            f"{relation}: destination indices in edge_index[1] exceed destination node count "
            f"{num_dst_nodes}"
        )


def _as_relation_set(relations: Iterable[DirectedRelation]) -> Set[DirectedRelation]:
    relation_set = set(relations)
    if any(not isinstance(rel, DirectedRelation) for rel in relation_set):
        raise TypeError("relations must be DirectedRelation objects")
    return relation_set


def ensure_schema_preserved(
    *,
    exposed_node_types: Iterable[str],
    exposed_relations: Iterable[DirectedRelation],
    original_node_types: Iterable[str],
    original_relations: Iterable[DirectedRelation],
) -> bool:
    """Reject condensed graphs that expose implementation-level shadow schema."""

    exposed_type_set = set(exposed_node_types)
    original_type_set = set(original_node_types)
    extra_types = exposed_type_set - original_type_set
    if extra_types:
        raise ValueError(f"condensed graph exposes non-original node types: {sorted(extra_types)}")

    exposed_relation_set = _as_relation_set(exposed_relations)
    original_relation_set = _as_relation_set(original_relations)
    extra_relations = exposed_relation_set - original_relation_set
    if extra_relations:
        names = ", ".join(str(rel) for rel in sorted(extra_relations))
        raise ValueError(f"condensed graph exposes non-original relations: {names}")

    return True
