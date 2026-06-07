from __future__ import annotations

from dataclasses import dataclass

import torch

from shadow_hgc.data.schemas import DirectedRelation, ensure_schema_preserved
from shadow_hgc.shadows.assign import build_b1_shadow_edges


@dataclass
class RelationShadowPlan:
    shadow_features: torch.Tensor
    assignment: torch.Tensor
    skeleton_edge_index: torch.Tensor | None = None
    skeleton_edge_weight: torch.Tensor | None = None


@dataclass
class CondensedGraph:
    node_features: dict[str, torch.Tensor]
    edge_index: dict[DirectedRelation, torch.Tensor]
    edge_weight: dict[DirectedRelation, torch.Tensor]
    target_type: str
    target_indices: torch.Tensor
    target_labels: torch.Tensor
    target_weights: torch.Tensor

    @property
    def exposed_node_types(self) -> set[str]:
        return set(self.node_features)

    @property
    def exposed_relations(self) -> set[DirectedRelation]:
        return set(self.edge_index)


def materialize_condensed_graph(
    *,
    target_type: str,
    original_node_types: set[str],
    original_relations: set[DirectedRelation],
    prototype_features: torch.Tensor,
    prototype_labels: torch.Tensor,
    prototype_weights: torch.Tensor,
    relation_plans: dict[DirectedRelation, RelationShadowPlan],
) -> CondensedGraph:
    """Expose shadow pools through original source types and original edge types."""

    node_features: dict[str, torch.Tensor] = {target_type: prototype_features.clone()}
    edge_index: dict[DirectedRelation, torch.Tensor] = {}
    edge_weight: dict[DirectedRelation, torch.Tensor] = {}
    num_prototypes = prototype_features.shape[0]

    for relation, plan in relation_plans.items():
        if relation not in original_relations:
            raise ValueError(f"{relation} is not in the original relation schema")
        source_type = relation.source_type
        shadow_source_offset = 0
        if source_type == target_type:
            shadow_source_offset = node_features[target_type].shape[0]
            node_features[target_type] = torch.cat([node_features[target_type], plan.shadow_features], dim=0)
        else:
            if source_type in node_features:
                shadow_source_offset = node_features[source_type].shape[0]
                node_features[source_type] = torch.cat([node_features[source_type], plan.shadow_features], dim=0)
            else:
                node_features[source_type] = plan.shadow_features.clone()

        shadow_edge_index, shadow_edge_weight = build_b1_shadow_edges(plan.assignment)
        shadow_edge_index = shadow_edge_index.clone()
        shadow_edge_index[0] += shadow_source_offset

        pieces = []
        weights = []
        if plan.skeleton_edge_index is not None and plan.skeleton_edge_index.numel() > 0:
            pieces.append(plan.skeleton_edge_index)
            weights.append(plan.skeleton_edge_weight.to(torch.float32))
        pieces.append(shadow_edge_index)
        weights.append(shadow_edge_weight)
        edge_index[relation] = torch.cat(pieces, dim=1)
        edge_weight[relation] = torch.cat(weights, dim=0)

    for node_type in original_node_types:
        node_features.setdefault(node_type, torch.empty(0, prototype_features.shape[1]))

    ensure_schema_preserved(
        exposed_node_types=set(node_features),
        exposed_relations=set(edge_index),
        original_node_types=original_node_types,
        original_relations=original_relations,
    )
    return CondensedGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        target_type=target_type,
        target_indices=torch.arange(num_prototypes, dtype=torch.long),
        target_labels=prototype_labels.clone(),
        target_weights=prototype_weights.to(torch.float32).clone(),
    )
