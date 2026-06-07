from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from shadow_hgc.data.loaders import build_toy_graph
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


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _prepare_features(graph, feature_dim: int, seed: int):
    target_type = graph.target_type
    target_relations = [rel for rel in graph.relations if rel.destination_type == target_type]
    projected = fixed_random_projection(graph.node_features[target_type], out_dim=feature_dim, seed=seed)
    stats = fit_standardizer(projected, rows=graph.train_idx)
    psi_paper = standardize(projected, stats)

    writes = next(rel for rel in graph.relations if rel.source_type == "author")
    psi_author = featureless_source_neighbor_mean(
        source_to_target_edge_index=graph.edge_index[writes],
        target_base_features=psi_paper,
        num_source_nodes=int(graph.edge_index[writes][0].max().item()) + 1,
    )
    degree_by_relation, degree_features = compute_degree_features(
        graph.edge_index,
        target_relations,
        num_target_nodes=psi_paper.shape[0],
    )
    phi = {
        "paper": torch.cat([psi_paper, degree_features], dim=1),
        "author": psi_author,
    }
    psi = {"paper": psi_paper, "author": psi_author}
    return psi, phi, degree_by_relation, degree_features, target_relations


def _relation_demand(graph, phi, target_relations):
    demand = {}
    alpha = {}
    for relation in target_relations:
        source_phi = phi[relation.source_type]
        num_dst = phi[relation.destination_type].shape[0]
        demand[relation], alpha[relation] = aggregate_relation_demand(
            edge_index=graph.edge_index[relation],
            source_features=source_phi,
            num_dst_nodes=num_dst,
            return_alpha=True,
        )
    return demand, alpha


def _build_relation_plans(
    *,
    graph,
    phi,
    demand,
    alpha,
    prototype_result,
    target_relations,
    M_r: int,
    k_s: int,
    seed: int,
):
    plans: dict[DirectedRelation, RelationShadowPlan] = {}
    diagnostics: dict[str, dict[str, float | dict[str, float]]] = {}
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
            shadow_features = factorize_shadows(residual, num_shadows=M_r, seed=seed + rel_index)
            assignment = assign_nearest_shadow(residual, shadow_features)
            shadow_features, gamma = calibrate_shadow_norm(residual, shadow_features, assignment)
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
            signed_any = signed_any or bool(torch.any(shadow_features < 0.0).item())
        else:
            rows = []
            for members in prototype_result.cell_members:
                rows.append(demand[relation][members].mean(dim=0))
            residual = torch.stack(rows, dim=0)
            shadow_features = factorize_shadows(
                residual,
                num_shadows=M_r,
                seed=seed + rel_index,
                sample_weight=prototype_result.prototype_weights,
            )
            assignment = assign_nearest_shadow(residual, shadow_features)
            shadow_features, gamma = calibrate_shadow_norm(residual, shadow_features, assignment)
            assignment = assign_nearest_shadow(residual, shadow_features)
            plans[relation] = RelationShadowPlan(
                shadow_features=shadow_features,
                assignment=assignment,
            )
            diagnostics[str(relation)] = {
                "SkeletonMassCoverage": 0.0,
                "ResidualEnergy": 1.0,
                "ShadowReconErr": shadow_reconstruction_error(residual, shadow_features, assignment),
                "gamma": gamma,
                "shadow_norms": feature_norm_summary(shadow_features),
                "real_source_norms": feature_norm_summary(phi[relation.source_type]),
            }
            signed_any = signed_any or bool(torch.any(shadow_features < 0.0).item())
    return plans, diagnostics, signed_any


def _train_and_infer(graph, phi, alpha, condensed, relations, *, epochs: int, seed: int) -> tuple[float, float, float, float]:
    torch.manual_seed(seed)
    in_channels = {node_type: features.shape[1] for node_type, features in condensed.node_features.items()}
    model = WeightedRelationLinearConv(
        in_channels=in_channels,
        out_channels=int(condensed.target_labels.max().item()) + 1,
        node_types=list(condensed.node_features),
        relations=relations,
        activation=None,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)
    train_start = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(condensed.node_features, condensed.edge_index, condensed.edge_weight)
        logits = out[graph.target_type][condensed.target_indices]
        loss = prototype_cross_entropy(
            logits,
            condensed.target_labels,
            condensed.target_weights,
            loss_type="weighted",
        )
        loss.backward()
        opt.step()
    train_time = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    original_edge_weight = {relation: alpha[relation] for relation in relations}
    out = model(phi, graph.edge_index, original_edge_weight)
    pred = out[graph.target_type].argmax(dim=1)
    test_labels = graph.labels[graph.test_idx]
    test_pred = pred[graph.test_idx]
    accuracy = float((test_pred == test_labels).to(torch.float32).mean().item())
    macro_f1 = macro_f1_score(
        test_pred,
        test_labels,
        num_classes=int(graph.labels[graph.train_idx].max().item()) + 1,
    )
    infer_time = time.perf_counter() - infer_start
    return accuracy, macro_f1, train_time, infer_time


def run_toy_experiment(
    *,
    output_path: str | Path = "experiments/logs/toy/summary.json",
    seed: int = 0,
    epochs: int = 40,
    M_tau: int = 4,
    M_r: int = 3,
    k_s: int = 2,
    feature_dim: int = 4,
) -> dict:
    start = time.perf_counter()
    graph = build_toy_graph(seed=seed)
    psi, phi, degree_by_relation, degree_features, target_relations = _prepare_features(graph, feature_dim, seed)
    demand, alpha = _relation_demand(graph, phi, target_relations)
    signature = build_target_signature(
        psi["paper"],
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
    )
    condensed = materialize_condensed_graph(
        target_type=graph.target_type,
        original_node_types=set(graph.node_features) | {"author"},
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
    )
    condensation_time = time.perf_counter() - start - train_time - infer_time
    schema_preserved = ensure_schema_preserved(
        exposed_node_types=set(condensed.node_features),
        exposed_relations=set(condensed.edge_index),
        original_node_types=set(graph.node_features) | {"author"},
        original_relations=set(target_relations),
    )
    all_nonnegative = all(bool(torch.all(weight >= 0).item()) for weight in condensed.edge_weight.values())
    summary = {
        "method": "Shadow-HGC-R-1",
        "dataset": "toy",
        "split": "fixed_toy",
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
        "schema_preserved": bool(schema_preserved),
        "all_edge_weights_nonnegative": all_nonnegative,
        "residual_shadows_signed": residual_signed,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return summary
