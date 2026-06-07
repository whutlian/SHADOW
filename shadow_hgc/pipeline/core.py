from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from shadow_hgc.data.loaders import HeteroGraphData
from shadow_hgc.data.schemas import DirectedRelation, ensure_schema_preserved
from shadow_hgc.demand.aggregate import aggregate_relation_demand
from shadow_hgc.eval.diagnostics import feature_norm_summary, shadow_reconstruction_error
from shadow_hgc.eval.metrics import macro_f1_score
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.features.base import featureless_source_neighbor_mean
from shadow_hgc.features.degree import compute_degree_features
from shadow_hgc.features.projection import fit_standardizer, fixed_random_projection, standardize
from shadow_hgc.graph.materialize import RelationShadowPlan, materialize_condensed_graph
from shadow_hgc.models.losses import prototype_cross_entropy
from shadow_hgc.models.weighted_rel_linear import WeightedRelationLinearConv
from shadow_hgc.prototype.cluster import class_wise_prototypes
from shadow_hgc.prototype.signatures import build_target_signature
from shadow_hgc.shadows.assign import assign_nearest_shadow
from shadow_hgc.shadows.calibrate import calibrate_shadow_norm
from shadow_hgc.shadows.factorize import factorize_shadows
from shadow_hgc.skeleton.transition import compute_target_target_residual_skeleton


def _stable_type_seed(seed: int, node_type: str) -> int:
    return seed + sum(ord(ch) for ch in node_type)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def prepare_model_features(
    graph: HeteroGraphData,
    *,
    feature_dim: int,
    seed: int,
    standardization_scope: str = "train_only",
    include_degree_features: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, list[DirectedRelation]]:
    target_type = graph.target_type
    target_relations = [rel for rel in graph.relations if rel.destination_type == target_type]
    if target_type not in graph.node_features:
        raise ValueError(f"target type {target_type} must have base features")

    psi: dict[str, torch.Tensor] = {}
    target_projected = fixed_random_projection(
        graph.node_features[target_type].to(torch.float32),
        out_dim=feature_dim,
        seed=_stable_type_seed(seed, target_type),
    )
    stats = fit_standardizer(target_projected, rows=graph.train_idx)
    psi[target_type] = standardize(target_projected, stats)

    source_types = sorted({relation.source_type for relation in target_relations if relation.source_type != target_type})
    for source_type in source_types:
        if source_type in graph.node_features:
            projected = fixed_random_projection(
                graph.node_features[source_type].to(torch.float32),
                out_dim=feature_dim,
                seed=_stable_type_seed(seed, source_type),
            )
            source_rows = None
            psi[source_type] = standardize(projected, fit_standardizer(projected, rows=source_rows))
        else:
            relation = next(rel for rel in target_relations if rel.source_type == source_type)
            psi[source_type] = featureless_source_neighbor_mean(
                source_to_target_edge_index=graph.edge_index[relation],
                target_base_features=psi[target_type],
                num_source_nodes=graph.num_nodes[source_type],
            )

    _, degree_features = compute_degree_features(
        graph.edge_index,
        target_relations,
        num_target_nodes=graph.num_nodes[target_type],
    )
    phi = {node_type: features for node_type, features in psi.items()}
    if include_degree_features:
        phi[target_type] = torch.cat([psi[target_type], degree_features], dim=1)
        signature_degree = degree_features
    else:
        phi[target_type] = psi[target_type]
        signature_degree = torch.zeros(
            graph.num_nodes[target_type],
            1,
            dtype=psi[target_type].dtype,
            device=psi[target_type].device,
        )
    return psi, phi, signature_degree, target_relations


def _relation_demand(
    graph: HeteroGraphData,
    phi: dict[str, torch.Tensor],
    target_relations: list[DirectedRelation],
    *,
    edge_chunk_size: int | None,
):
    demand = {}
    alpha = {}
    for relation in target_relations:
        demand[relation], alpha[relation] = aggregate_relation_demand(
            edge_index=graph.edge_index[relation],
            source_features=phi[relation.source_type],
            num_dst_nodes=graph.num_nodes[relation.destination_type],
            edge_chunk_size=edge_chunk_size,
            return_alpha=True,
        )
    return demand, alpha


def _build_relation_plans(
    *,
    graph: HeteroGraphData,
    phi: dict[str, torch.Tensor],
    demand: dict[DirectedRelation, torch.Tensor],
    alpha: dict[DirectedRelation, torch.Tensor],
    prototype_result,
    target_relations: list[DirectedRelation],
    M_r: int,
    k_s: int,
    seed: int,
    residual_shadow: bool,
    shadow_mode: str,
    calibration_enabled: bool,
):
    plans: dict[DirectedRelation, RelationShadowPlan] = {}
    diagnostics: dict[str, dict] = {}
    signed_any = False
    for rel_index, relation in enumerate(target_relations):
        if relation.is_target_target(graph.target_type):
            skeleton = compute_target_target_residual_skeleton(
                demand=demand[relation],
                prototype_features=prototype_result.prototype_features,
                target_to_cell=prototype_result.target_to_cell,
                cell_members=prototype_result.cell_members,
                edge_index=graph.edge_index[relation],
                alpha=alpha[relation],
                k_s=k_s,
            )
            residual = skeleton.residual
            if not residual_shadow:
                shadow_features = torch.zeros(1, residual.shape[1], dtype=residual.dtype, device=residual.device)
                assignment = torch.zeros(residual.shape[0], dtype=torch.long, device=residual.device)
                gamma = 1.0
            elif shadow_mode == "real_source_centroid":
                shadow_features = factorize_shadows(
                    phi[relation.source_type],
                    num_shadows=M_r,
                    seed=seed + rel_index,
                ).to(residual.device)
                assignment = assign_nearest_shadow(residual, shadow_features)
                shadow_features, gamma = calibrate_shadow_norm(
                    residual,
                    shadow_features,
                    assignment,
                    enabled=calibration_enabled,
                )
                assignment = assign_nearest_shadow(residual, shadow_features)
            else:
                shadow_features = factorize_shadows(residual, num_shadows=M_r, seed=seed + rel_index)
                assignment = assign_nearest_shadow(residual, shadow_features)
                shadow_features, gamma = calibrate_shadow_norm(
                    residual,
                    shadow_features,
                    assignment,
                    enabled=calibration_enabled,
                )
                assignment = assign_nearest_shadow(residual, shadow_features)
            plans[relation] = RelationShadowPlan(
                shadow_features=shadow_features,
                assignment=assignment,
                skeleton_edge_index=skeleton.skeleton_edge_index,
                skeleton_edge_weight=skeleton.skeleton_edge_weight,
            )
            diagnostics[str(relation)] = {
                "SkeletonMassCoverage": skeleton.skeleton_mass_coverage,
                "ResidualEnergy": skeleton.residual_energy,
                "ShadowReconErr": shadow_reconstruction_error(residual, shadow_features, assignment),
                "gamma": gamma,
                "shadow_norms": feature_norm_summary(shadow_features),
            }
        else:
            residual = torch.stack(
                [demand[relation][members].mean(dim=0) for members in prototype_result.cell_members],
                dim=0,
            )
            factor_rows = phi[relation.source_type] if shadow_mode == "real_source_centroid" else residual
            shadow_features = factorize_shadows(
                factor_rows,
                num_shadows=M_r,
                seed=seed + rel_index,
                sample_weight=None if shadow_mode == "real_source_centroid" else prototype_result.prototype_weights,
            ).to(residual.device)
            assignment = assign_nearest_shadow(residual, shadow_features)
            shadow_features, gamma = calibrate_shadow_norm(
                residual,
                shadow_features,
                assignment,
                enabled=calibration_enabled,
            )
            assignment = assign_nearest_shadow(residual, shadow_features)
            plans[relation] = RelationShadowPlan(shadow_features=shadow_features, assignment=assignment)
            diagnostics[str(relation)] = {
                "SkeletonMassCoverage": 0.0,
                "ResidualEnergy": 1.0,
                "ShadowReconErr": shadow_reconstruction_error(residual, shadow_features, assignment),
                "gamma": gamma,
                "shadow_norms": feature_norm_summary(shadow_features),
                "real_source_norms": feature_norm_summary(phi[relation.source_type]),
            }
        signed_any = signed_any or bool(torch.any(plans[relation].shadow_features < 0.0).item())
    return plans, diagnostics, signed_any


def _train_and_infer(
    graph: HeteroGraphData,
    phi: dict[str, torch.Tensor],
    alpha: dict[DirectedRelation, torch.Tensor],
    condensed,
    relations: list[DirectedRelation],
    *,
    epochs: int,
    seed: int,
    loss_type: str,
    inference_edge_chunk_size: int | None,
) -> tuple[float | None, float | None, float, float]:
    torch.manual_seed(seed)
    num_classes = int(graph.labels[graph.train_idx].max().item()) + 1
    in_channels = {node_type: features.shape[1] for node_type, features in condensed.node_features.items()}
    model = WeightedRelationLinearConv(
        in_channels=in_channels,
        out_channels=num_classes,
        node_types=list(condensed.node_features),
        relations=relations,
        activation=None,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
    train_start = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(condensed.node_features, condensed.edge_index, condensed.edge_weight)
        logits = out[graph.target_type][condensed.target_indices]
        loss = prototype_cross_entropy(logits, condensed.target_labels, condensed.target_weights, loss_type=loss_type)
        loss.backward()
        opt.step()
    train_time = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    out = model(
        phi,
        graph.edge_index,
        {relation: alpha[relation] for relation in relations},
        edge_chunk_size=inference_edge_chunk_size,
    )
    pred = out[graph.target_type].argmax(dim=1)
    accuracy = None
    macro_f1 = None
    if graph.test_idx.numel() > 0:
        test_pred = pred[graph.test_idx]
        test_label = graph.labels[graph.test_idx]
        accuracy = float((test_pred == test_label).to(torch.float32).mean().item())
        macro_f1 = macro_f1_score(test_pred, test_label, num_classes=num_classes)
    infer_time = time.perf_counter() - infer_start
    return accuracy, macro_f1, train_time, infer_time


def run_shadow_hgc_experiment(
    graph: HeteroGraphData,
    *,
    output_path: str | Path,
    seed: int,
    epochs: int,
    M_tau: int,
    M_r: int,
    k_s: int,
    feature_dim: int,
    include_degree_features: bool = True,
    residual_shadow: bool = True,
    shadow_mode: str = "virtual_demand_shadow",
    loss_type: str = "weighted",
    calibration_enabled: bool = True,
    inference_edge_chunk_size: int | None = 500_000,
    demand_edge_chunk_size: int | None = 500_000,
) -> dict:
    start = time.perf_counter()
    if shadow_mode not in {"virtual_demand_shadow", "real_source_centroid"}:
        raise ValueError(f"unknown shadow_mode: {shadow_mode}")
    psi, phi, degree_features, target_relations = prepare_model_features(
        graph,
        feature_dim=feature_dim,
        seed=seed,
        include_degree_features=include_degree_features,
    )
    demand, alpha = _relation_demand(graph, phi, target_relations, edge_chunk_size=demand_edge_chunk_size)
    signature = build_target_signature(
        psi[graph.target_type],
        demand,
        degree_features,
        eta=0.1,
        relation_order=target_relations,
    )
    prototypes = class_wise_prototypes(
        phi_target=phi[graph.target_type],
        signatures=signature,
        labels=graph.labels,
        train_idx=graph.train_idx,
        M_tau=M_tau,
        seed=seed,
    )
    relation_plans, diagnostics, residual_signed = _build_relation_plans(
        graph=graph,
        phi=phi,
        demand=demand,
        alpha=alpha,
        prototype_result=prototypes,
        target_relations=target_relations,
        M_r=M_r,
        k_s=k_s,
        seed=seed,
        residual_shadow=residual_shadow,
        shadow_mode=shadow_mode,
        calibration_enabled=calibration_enabled,
    )
    original_node_types = {graph.target_type} | {relation.source_type for relation in target_relations}
    condensed = materialize_condensed_graph(
        target_type=graph.target_type,
        original_node_types=original_node_types,
        original_relations=set(target_relations),
        prototype_features=prototypes.prototype_features,
        prototype_labels=prototypes.prototype_labels,
        prototype_weights=prototypes.prototype_weights,
        relation_plans=relation_plans,
    )
    accuracy, macro_f1, train_time, infer_time = _train_and_infer(
        graph,
        phi,
        alpha,
        condensed,
        target_relations,
        epochs=epochs,
        seed=seed,
        loss_type=loss_type,
        inference_edge_chunk_size=inference_edge_chunk_size,
    )
    condensation_time = time.perf_counter() - start - train_time - infer_time
    schema_preserved = ensure_schema_preserved(
        exposed_node_types=set(condensed.node_features),
        exposed_relations=set(condensed.edge_index),
        original_node_types=original_node_types,
        original_relations=set(target_relations),
    )
    summary = {
        "method": "Shadow-HGC-R-1",
        "dataset": graph.dataset_name,
        "split": "processed_local",
        "target_type": graph.target_type,
        "directed_relations": [str(relation) for relation in target_relations],
        "M_tau": M_tau,
        "M_r": {str(relation): M_r for relation in target_relations},
        "k_s": k_s,
        "feature_dim": feature_dim,
        "target_input_dim": int(phi[graph.target_type].shape[1]),
        "seed": seed,
        "condensed_nodes_by_type": {k: int(v.shape[0]) for k, v in condensed.node_features.items()},
        "condensed_edges_by_relation": {str(k): int(v.shape[1]) for k, v in condensed.edge_index.items()},
        "condensation_time": max(0.0, condensation_time),
        "training_time": train_time,
        "inference_time": infer_time,
        "peak_cpu_ram": current_cpu_ram_bytes(),
        "peak_gpu_ram": current_gpu_ram_bytes(),
        "disk_bytes": 0,
        "num_full_edge_scans": 0,
        "edge_slice_cache_bytes": 0,
        "demand_edge_chunk_size": demand_edge_chunk_size,
        "inference_edge_chunk_size": inference_edge_chunk_size,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "diagnostics": diagnostics,
        "target_prototypes_by_class": {
            str(int(label.item())): int((prototypes.prototype_labels == label).sum().item())
            for label in prototypes.prototype_labels.unique()
        },
        "cluster_size_distribution": {
            "min": float(prototypes.prototype_weights.min().item()),
            "median": float(torch.median(prototypes.prototype_weights).item()),
            "max": float(prototypes.prototype_weights.max().item()),
        },
        "ablation": {
            "include_degree_features": include_degree_features,
            "residual_shadow": residual_shadow,
            "shadow_mode": shadow_mode,
            "loss_type": loss_type,
            "calibration_enabled": calibration_enabled,
        },
        "schema_preserved": bool(schema_preserved),
        "all_edge_weights_nonnegative": all(bool(torch.all(w >= 0).item()) for w in condensed.edge_weight.values()),
        "residual_shadows_signed": residual_signed,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return summary
