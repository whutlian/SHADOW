from __future__ import annotations

import gc
import time
from pathlib import Path

import torch

from shadow_hgc.data.id_index import IdIndex
from shadow_hgc.data.loaders import HeteroGraphData
from shadow_hgc.data.schemas import DirectedRelation, ensure_schema_preserved
from shadow_hgc.demand.aggregate import aggregate_relation_demand
from shadow_hgc.demand.normalize import destination_row_normalize
from shadow_hgc.eval.diagnostics import feature_norm_summary, shadow_reconstruction_error
from shadow_hgc.eval.logging import attach_run_metadata, write_json_summary
from shadow_hgc.eval.metrics import macro_f1_score
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.features.base import featureless_source_neighbor_mean
from shadow_hgc.features.degree import compute_degree_features
from shadow_hgc.features.projection import fit_standardizer, fixed_random_projection, standardize
from shadow_hgc.graph.materialize import RelationShadowPlan, materialize_condensed_graph
from shadow_hgc.models.losses import prototype_cross_entropy
from shadow_hgc.models.weighted_rel_linear import RelationMessageEncoderMLP, WeightedRelationLinearConv
from shadow_hgc.prototype.cluster import class_wise_prototypes
from shadow_hgc.prototype.signatures import build_target_signature
from shadow_hgc.shadows.assign import assign_nearest_shadow
from shadow_hgc.shadows.budgets import resolve_shadow_budgets
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


def infer_class_metadata(labels: torch.Tensor, train_idx: torch.Tensor, test_idx: torch.Tensor) -> dict:
    valid_labels = labels[labels >= 0]
    num_classes_global = 0 if valid_labels.numel() == 0 else int(valid_labels.max().item()) + 1
    train_classes = torch.unique(labels[train_idx][labels[train_idx] >= 0]).to(torch.long)
    test_classes = torch.unique(labels[test_idx][labels[test_idx] >= 0]).to(torch.long)
    return {
        "num_classes_global": num_classes_global,
        "num_classes_train": int(train_classes.numel()),
        "train_label_classes": [int(value.item()) for value in train_classes],
        "test_label_classes": [int(value.item()) for value in test_classes],
    }


def prediction_diagnostics(pred: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor, *, num_classes: int) -> dict:
    if idx.numel() == 0:
        return {
            "predicted_class_histogram": {},
            "num_predicted_classes": 0,
            "prediction_entropy": 0.0,
            "weighted_f1": None,
        }
    selected_pred = pred[idx]
    selected_labels = labels[idx]
    hist = torch.bincount(selected_pred.clamp_min(0), minlength=num_classes).to(torch.float64)
    probs = hist / hist.sum().clamp_min(1.0)
    entropy = float(-(probs[probs > 0] * torch.log(probs[probs > 0])).sum().item())
    weighted_f1 = 0.0
    total = 0.0
    for class_id in range(num_classes):
        true_mask = selected_labels == class_id
        pred_mask = selected_pred == class_id
        support = float(true_mask.sum().item())
        if support == 0:
            continue
        tp = float((true_mask & pred_mask).sum().item())
        fp = float((~true_mask & pred_mask).sum().item())
        fn = float((true_mask & ~pred_mask).sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        weighted_f1 += support * f1
        total += support
    return {
        "predicted_class_histogram": {str(i): int(v.item()) for i, v in enumerate(hist) if int(v.item()) > 0},
        "num_predicted_classes": int((hist > 0).sum().item()),
        "prediction_entropy": entropy,
        "weighted_f1": None if total == 0 else weighted_f1 / total,
    }


def prepare_model_features(
    graph: HeteroGraphData,
    *,
    feature_dim: int,
    seed: int,
    projection_type: str = "random",
    standardization_scope: str = "train_only",
    include_degree_features: bool = True,
    degree_scale: float = 0.1,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, list[DirectedRelation]]:
    target_type = graph.target_type
    target_relations = [rel for rel in graph.relations if rel.destination_type == target_type]
    if target_type not in graph.node_features:
        raise ValueError(f"target type {target_type} must have base features")

    psi: dict[str, torch.Tensor] = {}
    target_raw = graph.node_features[target_type].to(torch.float32)
    if projection_type == "raw":
        target_projected = target_raw
    elif projection_type == "random":
        target_projected = fixed_random_projection(
            target_raw,
            out_dim=feature_dim,
            seed=_stable_type_seed(seed, target_type),
        )
    else:
        raise ValueError(f"unknown projection_type: {projection_type}")
    stats = fit_standardizer(target_projected, rows=graph.train_idx)
    psi[target_type] = standardize(target_projected, stats)

    source_types = sorted({relation.source_type for relation in target_relations if relation.source_type != target_type})
    for source_type in source_types:
        if source_type in graph.node_features:
            raw = graph.node_features[source_type].to(torch.float32)
            if projection_type == "raw":
                projected = raw
            elif projection_type == "random":
                projected = fixed_random_projection(
                    raw,
                    out_dim=feature_dim,
                    seed=_stable_type_seed(seed, source_type),
                )
            else:
                raise ValueError(f"unknown projection_type: {projection_type}")
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
        phi[target_type] = torch.cat([psi[target_type], degree_scale * degree_features], dim=1)
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
    demand_dst_idx: torch.Tensor | None = None,
):
    demand = {}
    alpha = {}
    for relation in target_relations:
        edge_index = graph.edge_index[relation]
        if demand_dst_idx is None:
            demand[relation], alpha[relation] = aggregate_relation_demand(
                edge_index=edge_index,
                source_features=phi[relation.source_type],
                num_dst_nodes=graph.num_nodes[relation.destination_type],
                edge_chunk_size=edge_chunk_size,
                return_alpha=True,
            )
            continue

        rel_alpha = destination_row_normalize(edge_index, graph.num_nodes[relation.destination_type])
        dst_index = IdIndex.build(
            demand_dst_idx.to(torch.long),
            num_nodes=graph.num_nodes[relation.destination_type],
        )
        rel_demand = torch.zeros(
            demand_dst_idx.numel(),
            phi[relation.source_type].shape[1],
            dtype=phi[relation.source_type].dtype,
            device=phi[relation.source_type].device,
        )
        src, dst = edge_index[0], edge_index[1]
        chunk_size = edge_chunk_size or (edge_index.shape[1] if edge_index.shape[1] > 0 else 1)
        for start in range(0, edge_index.shape[1], chunk_size):
            end = min(start + chunk_size, edge_index.shape[1])
            local_dst = dst_index.lookup(dst[start:end]).to(rel_demand.device)
            mask = local_dst >= 0
            if not bool(mask.any()):
                continue
            chunk_src = src[start:end][mask]
            chunk_weight = rel_alpha[start:end][mask].to(rel_demand.dtype).unsqueeze(-1)
            rel_demand.index_add_(0, local_dst[mask], phi[relation.source_type][chunk_src] * chunk_weight)
        demand[relation] = rel_demand
        alpha[relation] = rel_alpha
    return demand, alpha


def _build_relation_plans(
    *,
    graph: HeteroGraphData,
    phi: dict[str, torch.Tensor],
    demand: dict[DirectedRelation, torch.Tensor],
    alpha: dict[DirectedRelation, torch.Tensor],
    prototype_result,
    target_relations: list[DirectedRelation],
    M_r: dict[DirectedRelation, int],
    k_s: int,
    seed: int,
    residual_shadow: bool,
    shadow_mode: str,
    calibration_enabled: bool,
    demand_row_by_target: torch.Tensor | None = None,
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
                demand_row_by_target=demand_row_by_target,
            )
            residual = skeleton.residual
            if shadow_mode == "private_shadow":
                shadow_features = residual.clone()
                assignment = torch.arange(residual.shape[0], dtype=torch.long, device=residual.device)
                gamma = 1.0
            elif not residual_shadow:
                shadow_features = torch.zeros(1, residual.shape[1], dtype=residual.dtype, device=residual.device)
                assignment = torch.zeros(residual.shape[0], dtype=torch.long, device=residual.device)
                gamma = 1.0
            elif shadow_mode == "real_source_centroid":
                shadow_features = factorize_shadows(
                    phi[relation.source_type],
                    num_shadows=M_r[relation],
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
                shadow_features = factorize_shadows(residual, num_shadows=M_r[relation], seed=seed + rel_index)
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
            def cell_mean(members: torch.Tensor) -> torch.Tensor:
                rows = members
                if demand_row_by_target is not None:
                    rows = demand_row_by_target[members]
                    rows = rows[rows >= 0]
                return demand[relation][rows].mean(dim=0)

            residual = torch.stack(
                [cell_mean(members) for members in prototype_result.cell_members],
                dim=0,
            )
            if shadow_mode == "private_shadow":
                shadow_features = residual.clone()
                assignment = torch.arange(residual.shape[0], dtype=torch.long, device=residual.device)
                gamma = 1.0
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
                continue
            factor_rows = phi[relation.source_type] if shadow_mode == "real_source_centroid" else residual
            shadow_features = factorize_shadows(
                factor_rows,
                num_shadows=M_r[relation],
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
    model_type: str,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
) -> tuple[float | None, float | None, float, float, dict, dict]:
    torch.manual_seed(seed)
    class_metadata = infer_class_metadata(graph.labels, graph.train_idx, graph.test_idx)
    num_classes = class_metadata["num_classes_global"]
    in_channels = {node_type: features.shape[1] for node_type, features in condensed.node_features.items()}
    if model_type == "relation_linear":
        model = WeightedRelationLinearConv(
            in_channels=in_channels,
            out_channels=num_classes,
            node_types=list(condensed.node_features),
            relations=relations,
            activation=None,
        )
    elif model_type == "relation_mlp":
        model = RelationMessageEncoderMLP(
            in_channels=in_channels,
            out_channels=num_classes,
            node_types=list(condensed.node_features),
            relations=relations,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"unknown model_type: {model_type}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_start = time.perf_counter()
    train_loss_start = None
    train_loss_end = None
    for _ in range(epochs):
        opt.zero_grad()
        out = model(condensed.node_features, condensed.edge_index, condensed.edge_weight)
        logits = out[graph.target_type][condensed.target_indices]
        loss = prototype_cross_entropy(logits, condensed.target_labels, condensed.target_weights, loss_type=loss_type)
        if train_loss_start is None:
            train_loss_start = float(loss.detach().item())
        loss.backward()
        opt.step()
        train_loss_end = float(loss.detach().item())
    train_time = time.perf_counter() - train_start
    model.eval()
    with torch.no_grad():
        condensed_out = model(condensed.node_features, condensed.edge_index, condensed.edge_weight)
        condensed_logits = condensed_out[graph.target_type][condensed.target_indices]
        prototype_pred = condensed_logits.argmax(dim=1)
        prototype_train_acc = float((prototype_pred == condensed.target_labels).to(torch.float32).mean().item())

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
    pred_diag = prediction_diagnostics(pred, graph.labels, graph.test_idx, num_classes=num_classes)
    if graph.test_idx.numel() > 0:
        test_pred = pred[graph.test_idx]
        test_label = graph.labels[graph.test_idx]
        accuracy = float((test_pred == test_label).to(torch.float32).mean().item())
        macro_f1 = macro_f1_score(test_pred, test_label, num_classes=num_classes)
    infer_time = time.perf_counter() - infer_start
    train_metrics = {
        "train_loss_start": train_loss_start,
        "train_loss_end": train_loss_end,
        "prototype_train_acc": prototype_train_acc,
        "num_optimizer_steps": int(epochs),
        "num_epochs": int(epochs),
        "learning_rate": float(lr),
        "weight_decay": float(weight_decay),
    }
    return accuracy, macro_f1, train_time, infer_time, train_metrics, pred_diag


def run_shadow_hgc_experiment(
    graph: HeteroGraphData,
    *,
    output_path: str | Path,
    method_name: str = "Shadow-HGC-R-1",
    seed: int,
    epochs: int,
    M_tau: int,
    M_r: int | dict | None,
    k_s: int,
    feature_dim: int,
    projection_type: str = "random",
    degree_scale: float = 0.1,
    signature_degree_eta: float = 0.1,
    min_proto_per_class: int = 1,
    budget_alpha: float = 0.5,
    strict_budget: bool = False,
    shadow_non_target_ratio: float = 1.0,
    shadow_target_target_ratio: float = 0.5,
    min_shadows_per_relation: int = 8,
    include_degree_features: bool = True,
    residual_shadow: bool = True,
    shadow_mode: str = "virtual_demand_shadow",
    loss_type: str = "weighted",
    calibration_enabled: bool = True,
    model_type: str = "relation_linear",
    hidden_dim: int = 128,
    dropout: float = 0.3,
    lr: float = 0.03,
    weight_decay: float = 1e-4,
    inference_edge_chunk_size: int | None = 500_000,
    demand_edge_chunk_size: int | None = 500_000,
    self_only: bool = False,
) -> dict:
    start = time.perf_counter()
    if shadow_mode not in {"virtual_demand_shadow", "real_source_centroid", "private_shadow"}:
        raise ValueError(f"unknown shadow_mode: {shadow_mode}")
    psi, phi, degree_features, target_relations = prepare_model_features(
        graph,
        feature_dim=feature_dim,
        seed=seed,
        projection_type=projection_type,
        include_degree_features=include_degree_features,
        degree_scale=degree_scale,
    )
    demand, alpha = _relation_demand(
        graph,
        phi,
        target_relations,
        edge_chunk_size=demand_edge_chunk_size,
        demand_dst_idx=graph.train_idx,
    )
    signature = build_target_signature(
        psi[graph.target_type][graph.train_idx],
        demand,
        degree_features[graph.train_idx],
        eta=signature_degree_eta,
        relation_order=target_relations,
    )
    prototypes = class_wise_prototypes(
        phi_target=phi[graph.target_type],
        signatures=signature,
        labels=graph.labels,
        train_idx=graph.train_idx,
        M_tau=M_tau,
        signature_idx=graph.train_idx,
        min_proto_per_class=min_proto_per_class,
        budget_alpha=budget_alpha,
        strict_budget=strict_budget,
        seed=seed,
    )
    resolved_M_r = resolve_shadow_budgets(
        relations=target_relations,
        target_type=graph.target_type,
        effective_M_tau=prototypes.effective_M_tau,
        requested_M_r=M_r,
        non_target_ratio=shadow_non_target_ratio,
        target_target_ratio=shadow_target_target_ratio,
        min_shadows_per_relation=min_shadows_per_relation,
    )
    demand_row_by_target = torch.full((graph.labels.numel(),), -1, dtype=torch.long)
    demand_row_by_target[graph.train_idx] = torch.arange(graph.train_idx.numel(), dtype=torch.long)
    if self_only:
        relation_plans = {}
        diagnostics = {}
        residual_signed = False
        model_relations: list[DirectedRelation] = []
        original_node_types = {graph.target_type}
        original_relations = set()
    else:
        relation_plans, diagnostics, residual_signed = _build_relation_plans(
            graph=graph,
            phi=phi,
            demand=demand,
            alpha=alpha,
            prototype_result=prototypes,
            target_relations=target_relations,
            M_r=resolved_M_r,
            k_s=k_s,
            seed=seed,
            residual_shadow=residual_shadow,
            shadow_mode=shadow_mode,
            calibration_enabled=calibration_enabled,
            demand_row_by_target=demand_row_by_target,
        )
        model_relations = target_relations
        original_node_types = {graph.target_type} | {relation.source_type for relation in target_relations}
        original_relations = set(target_relations)
    condensed = materialize_condensed_graph(
        target_type=graph.target_type,
        original_node_types=original_node_types,
        original_relations=original_relations,
        prototype_features=prototypes.prototype_features,
        prototype_labels=prototypes.prototype_labels,
        prototype_weights=prototypes.prototype_weights,
        relation_plans=relation_plans,
    )
    prototype_summary = {
        "requested_M_tau": prototypes.requested_M_tau,
        "effective_M_tau": prototypes.effective_M_tau,
        "num_classes": prototypes.num_classes,
        "min_proto_per_class": prototypes.min_proto_per_class,
        "budget_alpha": prototypes.budget_alpha,
        "budget_upshifted": prototypes.budget_upshifted,
        "class_budget": dict(prototypes.class_budget),
        "target_prototypes_by_class": {
            str(int(label.item())): int((prototypes.prototype_labels == label).sum().item())
            for label in prototypes.prototype_labels.unique()
        },
        "cluster_size_distribution": {
            "min": float(prototypes.prototype_weights.min().item()),
            "median": float(torch.median(prototypes.prototype_weights).item()),
            "max": float(prototypes.prototype_weights.max().item()),
        },
    }
    target_base_dim = int(psi[graph.target_type].shape[1])
    target_degree_dim = int(degree_features.shape[1] if include_degree_features else 0)
    target_input_dim = int(phi[graph.target_type].shape[1])
    del demand, signature, relation_plans, prototypes, psi, degree_features, demand_row_by_target
    gc.collect()
    accuracy, macro_f1, train_time, infer_time, train_metrics, pred_diag = _train_and_infer(
        graph,
        phi,
        alpha,
        condensed,
        model_relations,
        epochs=epochs,
        seed=seed,
        loss_type=loss_type,
        inference_edge_chunk_size=inference_edge_chunk_size,
        model_type=model_type,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
    )
    condensation_time = time.perf_counter() - start - train_time - infer_time
    schema_preserved = ensure_schema_preserved(
        exposed_node_types=set(condensed.node_features),
        exposed_relations=set(condensed.edge_index),
        original_node_types=original_node_types,
        original_relations=original_relations,
    )
    class_metadata = infer_class_metadata(graph.labels, graph.train_idx, graph.test_idx)
    recon_recommendations = {
        relation: "increase M_r or run b=2 ablation"
        for relation, values in diagnostics.items()
        if values.get("ShadowReconErr", 0.0) > 0.6
    }
    config_for_hash = {
        "method": method_name,
        "dataset": graph.dataset_name,
        "seed": seed,
        "epochs": epochs,
        "M_tau": M_tau,
        "M_r": M_r,
        "resolved_M_r": {str(relation): value for relation, value in resolved_M_r.items()},
        "k_s": k_s,
        "feature_dim": feature_dim,
        "projection_type": projection_type,
        "degree_scale": degree_scale,
        "signature_degree_eta": signature_degree_eta,
        "min_proto_per_class": min_proto_per_class,
        "budget_alpha": budget_alpha,
        "strict_budget": strict_budget,
        "shadow_non_target_ratio": shadow_non_target_ratio,
        "shadow_target_target_ratio": shadow_target_target_ratio,
        "min_shadows_per_relation": min_shadows_per_relation,
        "include_degree_features": include_degree_features,
        "residual_shadow": residual_shadow,
        "shadow_mode": shadow_mode,
        "loss_type": loss_type,
        "calibration_enabled": calibration_enabled,
        "model": model_type,
        "loss_type": loss_type,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "self_only": self_only,
    }
    summary = {
        "method": method_name,
        "dataset": graph.dataset_name,
        "split": "processed_local",
        "target_type": graph.target_type,
        "directed_relations": [str(relation) for relation in target_relations],
        "M_tau": M_tau,
        "requested_M_tau": prototype_summary["requested_M_tau"],
        "effective_M_tau": prototype_summary["effective_M_tau"],
        "num_classes": prototype_summary["num_classes"],
        "min_proto_per_class": prototype_summary["min_proto_per_class"],
        "budget_alpha": prototype_summary["budget_alpha"],
        "budget_upshifted": prototype_summary["budget_upshifted"],
        "class_budget": prototype_summary["class_budget"],
        "requested_M_r": M_r,
        "resolved_M_r": {str(relation): value for relation, value in resolved_M_r.items()},
        "M_r": {str(relation): value for relation, value in resolved_M_r.items()},
        "k_s": k_s,
        "feature_dim": feature_dim,
        "projection_type": projection_type,
        "standardization_scope": "train_only",
        "degree_scale": degree_scale,
        "signature_degree_eta": signature_degree_eta,
        "target_base_dim": target_base_dim,
        "target_degree_dim": target_degree_dim,
        "target_input_dim": target_input_dim,
        "model": model_type,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
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
        **class_metadata,
        **train_metrics,
        **pred_diag,
        "diagnostics": diagnostics,
        "shadow_reconstruction_recommendations": recon_recommendations,
        "target_prototypes_by_class": prototype_summary["target_prototypes_by_class"],
        "cluster_size_distribution": prototype_summary["cluster_size_distribution"],
        "ablation": {
            "include_degree_features": include_degree_features,
            "residual_shadow": residual_shadow,
            "shadow_mode": shadow_mode,
            "loss_type": loss_type,
            "calibration_enabled": calibration_enabled,
            "self_only": self_only,
        },
        "schema_preserved": bool(schema_preserved),
        "all_edge_weights_nonnegative": all(bool(torch.all(w >= 0).item()) for w in condensed.edge_weight.values()),
        "residual_shadows_signed": residual_signed,
    }
    output_path = Path(output_path)
    summary = attach_run_metadata(_jsonable(summary), config=config_for_hash)
    write_json_summary(output_path, summary, config=config_for_hash)
    return summary
