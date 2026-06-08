from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import torch

from shadow_hgc.data.id_index import IdIndex
from shadow_hgc.data.loaders import HeteroGraphData
from shadow_hgc.data.schemas import DirectedRelation, ensure_schema_preserved
from shadow_hgc.demand.aggregate import aggregate_relation_demand
from shadow_hgc.demand.normalize import destination_row_normalize
from shadow_hgc.diagnostics.rank import relation_rank_diagnostics
from shadow_hgc.diagnostics.reconstruction import reconstruction_error, row_norm_distribution
from shadow_hgc.eval.class_collapse import class_collapse_diagnostics
from shadow_hgc.eval.diagnostics import feature_norm_summary, shadow_reconstruction_error
from shadow_hgc.eval.logging import attach_run_metadata, write_json_summary
from shadow_hgc.eval.metrics import macro_f1_score
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.features.base import featureless_source_neighbor_mean
from shadow_hgc.features.degree import compute_degree_features
from shadow_hgc.features.diffusion import diffusion_target_features
from shadow_hgc.features.block_norm import FeatureBlock, fit_transform_feature_blocks
from shadow_hgc.features.metapath import metapath_target_features
from shadow_hgc.features.multiscale import fixed_block_projection
from shadow_hgc.features.projection import fit_standardizer, fixed_random_projection, standardize
from shadow_hgc.graph.materialize import RelationShadowPlan, materialize_condensed_graph
from shadow_hgc.models.losses import prototype_cross_entropy
from shadow_hgc.models.factory import build_model
from shadow_hgc.models.weighted_rel_linear import WeightedRelationLinearConv
from shadow_hgc.prototype.budgets import compute_target_budget_from_ratio, validate_budget_mode_args
from shadow_hgc.prototype.cluster import class_wise_prototypes
from shadow_hgc.prototype.signatures import build_target_signature
from shadow_hgc.shadows.assign import assign_nearest_shadow, assign_nearest_shadow_chunked
from shadow_hgc.shadows.adaptive import adaptive_assignment_b, adaptive_shadow_budgets
from shadow_hgc.shadows.assign import topb_nonnegative_assignment, topb_nonnegative_assignment_chunked
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


def _estimate_feature_bytes(features: dict[str, torch.Tensor]) -> int:
    total = 0
    for tensor in features.values():
        total += int(tensor.numel() * tensor.element_size())
    return int(total)


def _estimate_edge_bytes(
    edge_index: dict[DirectedRelation, torch.Tensor],
    edge_weight: dict[DirectedRelation, torch.Tensor] | None = None,
) -> int:
    total = 0
    for relation, index in edge_index.items():
        total += int(index.numel() * index.element_size())
        if edge_weight is not None and relation in edge_weight:
            total += int(edge_weight[relation].numel() * edge_weight[relation].element_size())
    return int(total)


def _summarize_block_stats(stats_by_name: dict) -> dict:
    summary = {}
    for name, stats in stats_by_name.items():
        summary[name] = {
            "dim": int(stats.mean.numel()),
            "mean_abs": float(stats.mean.abs().mean().item()),
            "std_mean": float(stats.std.mean().item()),
            "norm_median": float(stats.norm_median),
            "norm_p95": float(stats.norm_p95),
        }
    return summary


def _normalize_named_feature_blocks(
    features: torch.Tensor,
    names: list[str],
    *,
    prefix: str,
    target_type: str,
    train_idx: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, dict]:
    if mode == "none" or features.shape[1] == 0 or not names:
        return features, {}
    if features.shape[1] % len(names) != 0:
        blocks = [
            FeatureBlock(
                name=f"{prefix}:{'+'.join(names)}",
                tensor_or_provider=features,
                dim=int(features.shape[1]),
                node_type=target_type,
                role=prefix,
            )
        ]
    else:
        dim = features.shape[1] // len(names)
        blocks = [
            FeatureBlock(
                name=f"{prefix}:{name}",
                tensor_or_provider=features[:, start : start + dim],
                dim=int(dim),
                node_type=target_type,
                role=prefix,
            )
            for name, start in zip(names, range(0, features.shape[1], dim))
        ]
    transformed, stats = fit_transform_feature_blocks(blocks, fit_indices=train_idx, mode=mode)
    normalized = torch.cat([transformed[block.name] for block in blocks], dim=1)
    return normalized, _summarize_block_stats(stats)


def _trace_memory(stage: str) -> None:
    if not os.environ.get("SHADOW_HGC_TRACE_MEMORY"):
        return
    try:
        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        print(
            f"[mem] {stage} rss_gb={info.rss / 1e9:.3f} vms_gb={info.vms / 1e9:.3f}",
            flush=True,
        )
    except Exception as exc:
        print(f"[mem] {stage} unavailable={exc}", flush=True)


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
        **class_collapse_diagnostics(pred, labels, idx, num_classes=num_classes),
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
    feature_mode: str = "base",
    diffusion_steps: tuple[int, ...] = (1,),
    include_highpass: bool = False,
    metapath_signature: bool = False,
    metapath_model_input: bool = False,
    multiscale_dim: int = 128,
    block_norm: str = "none",
    return_metadata: bool = False,
):
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
    signature_extra: torch.Tensor | None = None
    multiscale_metadata = {
        "feature_mode": feature_mode,
        "diffusion_steps": list(diffusion_steps),
        "include_highpass": bool(include_highpass),
        "metapath_signature": bool(metapath_signature),
        "metapath_model_input": bool(metapath_model_input),
        "multiscale_dim": int(multiscale_dim),
        "block_norm": block_norm,
        "blocks": [],
        "block_stats": {},
    }
    if feature_mode in {"diffusion", "diffusion_metapath"}:
        target_target_edges = [
            graph.edge_index[relation]
            for relation in target_relations
            if relation.is_target_target(target_type)
        ]
        if target_target_edges:
            diffusion_edges = torch.cat(target_target_edges, dim=1)
            diffusion = diffusion_target_features(
                psi[target_type],
                edge_index=diffusion_edges,
                num_nodes=graph.num_nodes[target_type],
                steps=diffusion_steps,
                include_highpass=include_highpass,
            )
            diffusion_features = fixed_block_projection(
                diffusion.features,
                out_dim=max(1, int(multiscale_dim)) * max(1, len(diffusion.block_names)),
                seed=_stable_type_seed(seed, f"{target_type}_diffusion"),
            )
            diffusion_features, block_stats = _normalize_named_feature_blocks(
                diffusion_features,
                diffusion.block_names,
                prefix="diffusion",
                target_type=target_type,
                train_idx=graph.train_idx,
                mode=block_norm,
            )
            phi[target_type] = torch.cat([phi[target_type], diffusion_features], dim=1)
            signature_extra = diffusion_features
            multiscale_metadata["blocks"].extend(diffusion.block_names)
            multiscale_metadata["block_stats"].update(block_stats)
    if feature_mode in {"metapath", "diffusion_metapath"}:
        metapath = metapath_target_features(
            edge_index=graph.edge_index,
            relations=target_relations,
            target_type=target_type,
            psi_target=psi[target_type],
            num_nodes=graph.num_nodes,
        )
        if metapath.features.shape[1] > 0:
            metapath_features = fixed_block_projection(
                metapath.features,
                out_dim=max(1, int(multiscale_dim)),
                seed=_stable_type_seed(seed, f"{target_type}_metapath"),
            )
            metapath_features, block_stats = _normalize_named_feature_blocks(
                metapath_features,
                metapath.path_names[:1] or ["metapath"],
                prefix="metapath",
                target_type=target_type,
                train_idx=graph.train_idx,
                mode=block_norm,
            )
            if metapath_model_input:
                phi[target_type] = torch.cat([phi[target_type], metapath_features], dim=1)
            if metapath_signature:
                signature_extra = metapath_features if signature_extra is None else torch.cat([signature_extra, metapath_features], dim=1)
            multiscale_metadata["blocks"].extend(metapath.path_names)
            multiscale_metadata["block_stats"].update(block_stats)
    if return_metadata:
        return psi, phi, signature_degree, target_relations, signature_extra, multiscale_metadata
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
    shadow_policy: str = "fixed",
    shadow_min_per_relation: int = 8,
    shadow_max_multiplier: float = 2.0,
    adaptive_b: bool = False,
    b_max: int = 4,
    assignment_chunk_size: int | None = None,
    rank_diagnostic_k: int = 64,
    skeleton_policy: str = "fixed_k",
    skeleton_coverage: float = 0.65,
    skeleton_k_max: int = 8,
):
    plans: dict[DirectedRelation, RelationShadowPlan] = {}
    diagnostics: dict[str, dict] = {}
    rank_diagnostics: dict[str, dict] = {}
    realized_M_r: dict[DirectedRelation, int] = {}
    b_by_relation: dict[str, int] = {}
    signed_any = False

    def choose_shadow_budget(relation: DirectedRelation, residual: torch.Tensor) -> tuple[int, dict]:
        rank_diag = relation_rank_diagnostics(residual, rank_k=rank_diagnostic_k)
        rank_diagnostics[str(relation)] = rank_diag
        if shadow_policy == "fixed":
            budget = int(M_r[relation])
        elif shadow_policy == "rank_adaptive":
            budget = adaptive_shadow_budgets(
                {str(relation): rank_diag},
                effective_M_tau=prototype_result.effective_M_tau,
                shadow_min_per_relation=shadow_min_per_relation,
                shadow_max_multiplier=shadow_max_multiplier,
            )[str(relation)]
        else:
            raise ValueError("shadow_policy must be fixed or rank_adaptive")
        realized_M_r[relation] = int(budget)
        return int(budget), rank_diag

    def make_plan(
        relation: DirectedRelation,
        residual: torch.Tensor,
        shadow_features: torch.Tensor,
        assignment: torch.Tensor,
        *,
        skeleton_edge_index: torch.Tensor | None = None,
        skeleton_edge_weight: torch.Tensor | None = None,
    ) -> tuple[RelationShadowPlan, float, int]:
        b1_err = shadow_reconstruction_error(residual, shadow_features, assignment)
        b_value = adaptive_assignment_b(b1_err, b_max=b_max) if adaptive_b else 1
        if b_value <= 1 or shadow_features.shape[0] <= 1:
            plan = RelationShadowPlan(
                shadow_features=shadow_features,
                assignment=assignment,
                skeleton_edge_index=skeleton_edge_index,
                skeleton_edge_weight=skeleton_edge_weight,
            )
            return plan, b1_err, 1
        if assignment_chunk_size is not None:
            topb = topb_nonnegative_assignment_chunked(
                residual,
                shadow_features,
                b=b_value,
                chunk_size=assignment_chunk_size,
            )
        else:
            topb = topb_nonnegative_assignment(residual, shadow_features, b=b_value)
        plan = RelationShadowPlan(
            shadow_features=shadow_features,
            assignment=topb.topk_index[:, 0],
            shadow_edge_index=topb.edge_index,
            shadow_edge_weight=topb.edge_weight,
            skeleton_edge_index=skeleton_edge_index,
            skeleton_edge_weight=skeleton_edge_weight,
        )
        return plan, reconstruction_error(residual, topb.reconstruction), b_value

    for rel_index, relation in enumerate(target_relations):
        _trace_memory(f"relation_plan:start:{relation}")
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
                skeleton_policy=skeleton_policy,
                skeleton_coverage=skeleton_coverage,
                skeleton_k_max=skeleton_k_max,
            )
            residual = skeleton.residual
            num_shadows, rank_diag = choose_shadow_budget(relation, residual)
            _trace_memory(f"relation_plan:after_skeleton:{relation}")
            if shadow_mode == "private_shadow":
                shadow_features = residual.clone()
                assignment = torch.arange(residual.shape[0], dtype=torch.long, device=residual.device)
                gamma = 1.0
                realized_M_r[relation] = int(shadow_features.shape[0])
            elif not residual_shadow:
                shadow_features = torch.zeros(1, residual.shape[1], dtype=residual.dtype, device=residual.device)
                assignment = torch.zeros(residual.shape[0], dtype=torch.long, device=residual.device)
                gamma = 1.0
                realized_M_r[relation] = 1
            elif shadow_mode == "real_source_centroid":
                shadow_features = factorize_shadows(
                    phi[relation.source_type],
                    num_shadows=num_shadows,
                    seed=seed + rel_index,
                ).to(residual.device)
                assignment = (
                    assign_nearest_shadow_chunked(residual, shadow_features, chunk_size=assignment_chunk_size)
                    if assignment_chunk_size is not None
                    else assign_nearest_shadow(residual, shadow_features)
                )
                shadow_features, gamma = calibrate_shadow_norm(
                    residual,
                    shadow_features,
                    assignment,
                    enabled=calibration_enabled,
                )
                assignment = (
                    assign_nearest_shadow_chunked(residual, shadow_features, chunk_size=assignment_chunk_size)
                    if assignment_chunk_size is not None
                    else assign_nearest_shadow(residual, shadow_features)
                )
            else:
                shadow_features = factorize_shadows(residual, num_shadows=num_shadows, seed=seed + rel_index)
                _trace_memory(f"relation_plan:after_factorize:{relation}")
                assignment = (
                    assign_nearest_shadow_chunked(residual, shadow_features, chunk_size=assignment_chunk_size)
                    if assignment_chunk_size is not None
                    else assign_nearest_shadow(residual, shadow_features)
                )
                _trace_memory(f"relation_plan:after_assign:{relation}")
                shadow_features, gamma = calibrate_shadow_norm(
                    residual,
                    shadow_features,
                    assignment,
                    enabled=calibration_enabled,
                )
                assignment = (
                    assign_nearest_shadow_chunked(residual, shadow_features, chunk_size=assignment_chunk_size)
                    if assignment_chunk_size is not None
                    else assign_nearest_shadow(residual, shadow_features)
                )
                _trace_memory(f"relation_plan:after_reassign:{relation}")
            plan, recon_err, b_value = make_plan(
                relation,
                residual,
                shadow_features,
                assignment,
                skeleton_edge_index=skeleton.skeleton_edge_index,
                skeleton_edge_weight=skeleton.skeleton_edge_weight,
            )
            plans[relation] = plan
            b_by_relation[str(relation)] = b_value
            shadow_norm_dist = row_norm_distribution(shadow_features)
            diagnostics[str(relation)] = {
                "SkeletonMassCoverage": skeleton.skeleton_mass_coverage,
                "ResidualEnergy": skeleton.residual_energy,
                "ShadowReconErr": recon_err,
                "gamma": gamma,
                "realized_M_r": realized_M_r[relation],
                "adaptive_b": b_value,
                "mean_adaptive_k": skeleton.mean_adaptive_k,
                "median_adaptive_k": skeleton.median_adaptive_k,
                "max_adaptive_k": skeleton.max_adaptive_k,
                "shadow_norms": feature_norm_summary(shadow_features),
                "shadow_feature_norm_mean": shadow_norm_dist["mean"],
                "shadow_feature_norm_q995": shadow_norm_dist["q995"],
                **rank_diag,
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
            num_shadows, rank_diag = choose_shadow_budget(relation, residual)
            if shadow_mode == "private_shadow":
                shadow_features = residual.clone()
                assignment = torch.arange(residual.shape[0], dtype=torch.long, device=residual.device)
                gamma = 1.0
                realized_M_r[relation] = int(shadow_features.shape[0])
                plan, recon_err, b_value = make_plan(relation, residual, shadow_features, assignment)
                plans[relation] = plan
                b_by_relation[str(relation)] = b_value
                shadow_norm_dist = row_norm_distribution(shadow_features)
                diagnostics[str(relation)] = {
                    "SkeletonMassCoverage": 0.0,
                    "ResidualEnergy": 1.0,
                    "ShadowReconErr": recon_err,
                    "gamma": gamma,
                    "realized_M_r": realized_M_r[relation],
                    "adaptive_b": b_value,
                    "shadow_norms": feature_norm_summary(shadow_features),
                    "shadow_feature_norm_mean": shadow_norm_dist["mean"],
                    "shadow_feature_norm_q995": shadow_norm_dist["q995"],
                    "real_source_norms": feature_norm_summary(phi[relation.source_type]),
                    **rank_diag,
                }
                signed_any = signed_any or bool(torch.any(plans[relation].shadow_features < 0.0).item())
                continue
            factor_rows = phi[relation.source_type] if shadow_mode == "real_source_centroid" else residual
            shadow_features = factorize_shadows(
                factor_rows,
                num_shadows=num_shadows,
                seed=seed + rel_index,
                sample_weight=None if shadow_mode == "real_source_centroid" else prototype_result.prototype_weights,
            ).to(residual.device)
            _trace_memory(f"relation_plan:after_factorize:{relation}")
            assignment = (
                assign_nearest_shadow_chunked(residual, shadow_features, chunk_size=assignment_chunk_size)
                if assignment_chunk_size is not None
                else assign_nearest_shadow(residual, shadow_features)
            )
            _trace_memory(f"relation_plan:after_assign:{relation}")
            shadow_features, gamma = calibrate_shadow_norm(
                residual,
                shadow_features,
                assignment,
                enabled=calibration_enabled,
            )
            assignment = (
                assign_nearest_shadow_chunked(residual, shadow_features, chunk_size=assignment_chunk_size)
                if assignment_chunk_size is not None
                else assign_nearest_shadow(residual, shadow_features)
            )
            _trace_memory(f"relation_plan:after_reassign:{relation}")
            plan, recon_err, b_value = make_plan(relation, residual, shadow_features, assignment)
            plans[relation] = plan
            b_by_relation[str(relation)] = b_value
            shadow_norm_dist = row_norm_distribution(shadow_features)
            diagnostics[str(relation)] = {
                "SkeletonMassCoverage": 0.0,
                "ResidualEnergy": 1.0,
                "ShadowReconErr": recon_err,
                "gamma": gamma,
                "realized_M_r": realized_M_r[relation],
                "adaptive_b": b_value,
                "shadow_norms": feature_norm_summary(shadow_features),
                "shadow_feature_norm_mean": shadow_norm_dist["mean"],
                "shadow_feature_norm_q995": shadow_norm_dist["q995"],
                "real_source_norms": feature_norm_summary(phi[relation.source_type]),
                **rank_diag,
            }
        signed_any = signed_any or bool(torch.any(plans[relation].shadow_features < 0.0).item())
        _trace_memory(f"relation_plan:end:{relation}")
    diagnostics["rank"] = rank_diagnostics
    diagnostics["adaptive_b"] = b_by_relation
    return plans, diagnostics, signed_any, realized_M_r, b_by_relation, rank_diagnostics


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
    inference_dst_chunk_size: int | None,
    model_type: str,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    relation_gate: bool = False,
    relation_gate_init: float = 1.0,
    block_gate: bool = False,
    logit_adjustment_tau: float = 1.0,
) -> tuple[float | None, float | None, float, float, dict, dict]:
    _trace_memory("train_infer:start")
    torch.manual_seed(seed)
    class_metadata = infer_class_metadata(graph.labels, graph.train_idx, graph.test_idx)
    num_classes = class_metadata["num_classes_global"]
    in_channels = {node_type: features.shape[1] for node_type, features in condensed.node_features.items()}
    model, model_diagnostics = build_model(
        model_type=model_type,
        in_channels=in_channels,
        out_channels=num_classes,
        node_types=list(condensed.node_features),
        relations=relations,
        target_type=graph.target_type,
        hidden_dim=hidden_dim,
        dropout=dropout,
        relation_gate=relation_gate,
        relation_gate_init=relation_gate_init,
        block_gate=block_gate,
    )
    final_logits_activation = str(model_diagnostics.get("final_logits_activation", "none"))
    if final_logits_activation == "unsafe_relu_logits":
        raise ValueError("unsafe final ReLU logits are forbidden in R++")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_label_counts = torch.bincount(
        graph.labels[graph.train_idx].clamp_min(0).to(torch.long),
        minlength=num_classes,
    ).to(torch.float32)
    class_prior = train_label_counts / train_label_counts.sum().clamp_min(1.0)
    _trace_memory("train_infer:after_model")
    train_start = time.perf_counter()
    train_loss_start = None
    train_loss_end = None
    for _ in range(epochs):
        opt.zero_grad()
        out = model(condensed.node_features, condensed.edge_index, condensed.edge_weight)
        logits = out[graph.target_type][condensed.target_indices]
        loss = prototype_cross_entropy(
            logits,
            condensed.target_labels,
            condensed.target_weights,
            loss_type=loss_type,
            class_prior=class_prior,
            logit_adjustment_tau=logit_adjustment_tau,
        )
        if train_loss_start is None:
            train_loss_start = float(loss.detach().item())
        loss.backward()
        opt.step()
        train_loss_end = float(loss.detach().item())
    train_time = time.perf_counter() - train_start
    _trace_memory("train_infer:after_train")
    model.eval()
    with torch.no_grad():
        condensed_out = model(condensed.node_features, condensed.edge_index, condensed.edge_weight)
        condensed_logits = condensed_out[graph.target_type][condensed.target_indices]
        prototype_pred = condensed_logits.argmax(dim=1)
        prototype_train_acc = float((prototype_pred == condensed.target_labels).to(torch.float32).mean().item())

    infer_start = time.perf_counter()
    _trace_memory("train_infer:before_original_infer")
    with torch.no_grad():
        if inference_dst_chunk_size is not None and hasattr(model, "infer_target_chunked"):
            target_logits = model.infer_target_chunked(
                phi,
                graph.edge_index,
                {relation: alpha[relation] for relation in relations},
                dst_chunk_size=inference_dst_chunk_size,
                edge_chunk_size=inference_edge_chunk_size,
            )
        else:
            out = model(
                phi,
                graph.edge_index,
                {relation: alpha[relation] for relation in relations},
                edge_chunk_size=inference_edge_chunk_size,
            )
            target_logits = out[graph.target_type]
        pred = target_logits.argmax(dim=1)
        accuracy = None
        macro_f1 = None
        pred_diag = prediction_diagnostics(pred, graph.labels, graph.test_idx, num_classes=num_classes)
        if graph.test_idx.numel() > 0:
            test_pred = pred[graph.test_idx]
            test_label = graph.labels[graph.test_idx]
            accuracy = float((test_pred == test_label).to(torch.float32).mean().item())
            macro_f1 = macro_f1_score(test_pred, test_label, num_classes=num_classes)
    infer_time = time.perf_counter() - infer_start
    _trace_memory("train_infer:after_original_infer")
    train_metrics = {
        "train_loss_start": train_loss_start,
        "train_loss_end": train_loss_end,
        "prototype_train_acc": prototype_train_acc,
        "prototype_train_loss_start": train_loss_start,
        "prototype_train_loss_end": train_loss_end,
        "num_optimizer_steps": int(epochs),
        "num_epochs": int(epochs),
        "learning_rate": float(lr),
        "weight_decay": float(weight_decay),
        "final_logits_activation": final_logits_activation,
        "model_diagnostics": model_diagnostics,
    }
    if hasattr(model, "relation_gate_values"):
        train_metrics["relation_gate_values"] = model.relation_gate_values()
    if hasattr(model, "block_gate_values"):
        train_metrics["block_gate_values"] = model.block_gate_values()
    elif isinstance(model_diagnostics.get("block_gates"), dict):
        train_metrics["block_gate_values"] = model_diagnostics["block_gates"]
    return accuracy, macro_f1, train_time, infer_time, train_metrics, pred_diag


def run_shadow_hgc_experiment(
    graph: HeteroGraphData,
    *,
    output_path: str | Path,
    method_name: str = "Shadow-HGC-R-1",
    seed: int,
    epochs: int,
    M_tau: int | None = None,
    budget_mode: str = "count",
    ratio: float | None = None,
    ratio_base: str = "train_target",
    target_budget: int | None = None,
    max_target_budget: int | None = None,
    budget_rounding: str = "nearest",
    M_r: int | dict | None = None,
    k_s: int = 2,
    feature_dim: int = 64,
    projection_type: str = "random",
    degree_scale: float = 0.1,
    signature_degree_eta: float = 0.1,
    min_proto_per_class: int = 1,
    budget_alpha: float = 0.5,
    strict_budget: bool = False,
    shadow_non_target_ratio: float = 1.0,
    shadow_target_target_ratio: float = 0.5,
    min_shadows_per_relation: int = 8,
    shadow_policy: str = "fixed",
    shadow_min_per_relation: int = 8,
    shadow_max_multiplier: float = 2.0,
    adaptive_b: bool = False,
    b_max: int = 4,
    rank_diagnostic_k: int = 64,
    include_degree_features: bool = True,
    feature_mode: str = "base",
    diffusion_steps: tuple[int, ...] | list[int] = (1,),
    include_highpass: bool = False,
    metapath_signature: bool = False,
    metapath_model_input: bool = False,
    multiscale_dim: int = 128,
    residual_shadow: bool = True,
    shadow_mode: str = "virtual_demand_shadow",
    loss_type: str = "weighted",
    logit_adjustment_tau: float = 1.0,
    calibration_enabled: bool = True,
    model_type: str = "relation_linear",
    relation_gate: bool = False,
    relation_gate_init: float = 1.0,
    skeleton_policy: str = "fixed_k",
    skeleton_coverage: float = 0.65,
    skeleton_k_max: int = 8,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    lr: float = 0.03,
    weight_decay: float = 1e-4,
    inference_edge_chunk_size: int | None = 500_000,
    demand_edge_chunk_size: int | None = 500_000,
    ratio_mode: str = "target_only",
    shadow_total_budget: int | None = None,
    rank_adaptive_global_cap: bool = False,
    max_total_condensed_ratio: float | None = None,
    assignment_chunk_size: int | None = None,
    inference_dst_chunk_size: int | None = None,
    block_norm: str = "none",
    block_gate: bool = False,
    block_dropout: float = 0.0,
    self_only: bool = False,
) -> dict:
    start = time.perf_counter()
    _trace_memory("pipeline:start")
    if shadow_mode not in {"virtual_demand_shadow", "real_source_centroid", "private_shadow"}:
        raise ValueError(f"unknown shadow_mode: {shadow_mode}")
    if shadow_policy not in {"fixed", "rank_adaptive"}:
        raise ValueError("shadow_policy must be fixed or rank_adaptive")
    if feature_mode not in {"base", "diffusion", "metapath", "diffusion_metapath"}:
        raise ValueError("feature_mode must be base, diffusion, metapath, or diffusion_metapath")
    if skeleton_policy not in {"fixed_k", "coverage"}:
        raise ValueError("skeleton_policy must be fixed_k or coverage")
    if ratio_base not in {"train_target", "all_target"}:
        raise ValueError("ratio_base must be either train_target or all_target")
    if ratio_mode not in {"target_only", "total_nodes"}:
        raise ValueError("ratio_mode must be target_only or total_nodes")
    if block_norm not in {"none", "standardize", "l2", "standardize_l2"}:
        raise ValueError("block_norm must be none, standardize, l2, or standardize_l2")
    if M_tau is not None and target_budget is None:
        target_budget = M_tau
    if budget_mode == "ratio":
        validate_budget_mode_args(budget_mode=budget_mode, ratio=ratio, target_budget=None)
        train_labels = graph.labels[graph.train_idx]
        num_train_classes = int(torch.unique(train_labels[train_labels >= 0]).numel())
        base_count = int(graph.train_idx.numel() if ratio_base == "train_target" else graph.num_nodes[graph.target_type])
        budget_metadata = compute_target_budget_from_ratio(
            num_train_target_nodes=base_count,
            num_train_classes=num_train_classes,
            ratio=float(ratio),
            min_proto_per_class=min_proto_per_class,
            max_target_budget=max_target_budget,
            rounding=budget_rounding,
        )
        budget_metadata["ratio_base"] = ratio_base
        target_budget = int(budget_metadata["effective_target_prototypes"])
    else:
        validate_budget_mode_args(budget_mode="count", ratio=None, target_budget=target_budget)
        train_labels = graph.labels[graph.train_idx]
        num_train_classes = int(torch.unique(train_labels[train_labels >= 0]).numel())
        min_required = num_train_classes * min_proto_per_class
        target_budget = int(target_budget)
        budget_metadata = {
            "budget_mode": "count",
            "ratio": None,
            "ratio_base": ratio_base,
            "num_train_target_nodes": int(graph.train_idx.numel()),
            "num_train_classes": int(num_train_classes),
            "min_proto_per_class": int(min_proto_per_class),
            "requested_target_budget": target_budget,
            "min_required_target_budget": int(min_required),
            "effective_target_prototypes": int(max(target_budget, min_required)),
            "effective_target_ratio": float(max(target_budget, min_required) / max(1, int(graph.train_idx.numel()))),
            "budget_rounding": budget_rounding,
            "max_target_budget": None if max_target_budget is None else int(max_target_budget),
            "budget_upshifted": bool(target_budget < min_required),
        }
    M_tau = int(target_budget)
    diffusion_steps_tuple = tuple(int(step) for step in diffusion_steps)
    psi, phi, degree_features, target_relations, signature_extra, multiscale_metadata = prepare_model_features(
        graph,
        feature_dim=feature_dim,
        seed=seed,
        projection_type=projection_type,
        include_degree_features=include_degree_features,
        degree_scale=degree_scale,
        feature_mode=feature_mode,
        diffusion_steps=diffusion_steps_tuple,
        include_highpass=include_highpass,
        metapath_signature=metapath_signature,
        metapath_model_input=metapath_model_input,
        multiscale_dim=multiscale_dim,
        block_norm=block_norm,
        return_metadata=True,
    )
    _trace_memory("pipeline:after_features")
    demand, alpha = _relation_demand(
        graph,
        phi,
        target_relations,
        edge_chunk_size=demand_edge_chunk_size,
        demand_dst_idx=graph.train_idx,
    )
    _trace_memory("pipeline:after_demand")
    signature = build_target_signature(
        psi[graph.target_type][graph.train_idx],
        demand,
        degree_features[graph.train_idx],
        eta=signature_degree_eta,
        relation_order=target_relations,
        extra_blocks=None if signature_extra is None else [signature_extra[graph.train_idx]],
    )
    _trace_memory("pipeline:after_signature")
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
    _trace_memory("pipeline:after_prototypes")
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
        rank_diagnostics = {}
        b_by_relation = {}
        model_relations: list[DirectedRelation] = []
        original_node_types = {graph.target_type}
        original_relations = set()
    else:
        relation_plans, diagnostics, residual_signed, realized_M_r, b_by_relation, rank_diagnostics = _build_relation_plans(
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
            shadow_policy=shadow_policy,
            shadow_min_per_relation=shadow_min_per_relation,
            shadow_max_multiplier=shadow_max_multiplier,
            adaptive_b=adaptive_b,
            b_max=b_max,
            assignment_chunk_size=assignment_chunk_size,
            rank_diagnostic_k=rank_diagnostic_k,
            skeleton_policy=skeleton_policy,
            skeleton_coverage=skeleton_coverage,
            skeleton_k_max=skeleton_k_max,
        )
        resolved_M_r = realized_M_r
        _trace_memory("pipeline:after_relation_plans")
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
    _trace_memory("pipeline:after_materialize")
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
    class_budget_values = list(prototype_summary["class_budget"].values())
    budget_metadata.update(
        {
            "requested_target_budget": prototype_summary["requested_M_tau"],
            "effective_target_prototypes": prototype_summary["effective_M_tau"],
            "effective_target_ratio": float(prototype_summary["effective_M_tau"] / max(1, int(graph.train_idx.numel()))),
            "class_budget_min": int(min(class_budget_values)) if class_budget_values else 0,
            "class_budget_median": float(torch.median(torch.tensor(class_budget_values, dtype=torch.float32)).item()) if class_budget_values else 0.0,
            "class_budget_max": int(max(class_budget_values)) if class_budget_values else 0,
        }
    )
    target_base_dim = int(psi[graph.target_type].shape[1])
    target_degree_dim = int(degree_features.shape[1] if include_degree_features else 0)
    target_input_dim = int(phi[graph.target_type].shape[1])
    del demand, signature, relation_plans, prototypes, psi, degree_features, demand_row_by_target
    gc.collect()
    _trace_memory("pipeline:after_condense_gc")
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
        inference_dst_chunk_size=inference_dst_chunk_size,
        model_type=model_type,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        relation_gate=relation_gate,
        relation_gate_init=relation_gate_init,
        block_gate=block_gate,
        logit_adjustment_tau=logit_adjustment_tau,
    )
    condensation_time = time.perf_counter() - start - train_time - infer_time
    schema_preserved = ensure_schema_preserved(
        exposed_node_types=set(condensed.node_features),
        exposed_relations=set(condensed.edge_index),
        original_node_types=original_node_types,
        original_relations=original_relations,
    )
    class_metadata = infer_class_metadata(graph.labels, graph.train_idx, graph.test_idx)
    if train_metrics.get("relation_gate_values"):
        diagnostics["relation_gates"] = train_metrics["relation_gate_values"]
    if train_metrics.get("block_gate_values"):
        diagnostics["block_gates"] = train_metrics["block_gate_values"]
    recon_recommendations = {
        relation: "increase M_r or run b=2 ablation"
        for relation, values in diagnostics.items()
        if isinstance(values, dict) and values.get("ShadowReconErr", 0.0) > 0.6
    }
    config_for_hash = {
        "method": method_name,
        "dataset": graph.dataset_name,
        "seed": seed,
        "epochs": epochs,
        "budget_mode": budget_mode,
        "ratio": ratio,
        "ratio_base": ratio_base,
        "target_budget": M_tau,
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
        "shadow_policy": shadow_policy,
        "shadow_min_per_relation": shadow_min_per_relation,
        "shadow_max_multiplier": shadow_max_multiplier,
        "adaptive_b": adaptive_b,
        "b_max": b_max,
        "rank_diagnostic_k": rank_diagnostic_k,
        "feature_mode": feature_mode,
        "diffusion_steps": list(diffusion_steps_tuple),
        "include_highpass": include_highpass,
        "metapath_signature": metapath_signature,
        "metapath_model_input": metapath_model_input,
        "multiscale_dim": multiscale_dim,
        "include_degree_features": include_degree_features,
        "residual_shadow": residual_shadow,
        "shadow_mode": shadow_mode,
        "loss_type": loss_type,
        "logit_adjustment_tau": logit_adjustment_tau,
        "calibration_enabled": calibration_enabled,
        "model": model_type,
        "loss_type": loss_type,
        "relation_gate": relation_gate,
        "relation_gate_init": relation_gate_init,
        "skeleton_policy": skeleton_policy,
        "skeleton_coverage": skeleton_coverage,
        "skeleton_k_max": skeleton_k_max,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "self_only": self_only,
        "ratio_mode": ratio_mode,
        "shadow_total_budget": shadow_total_budget,
        "rank_adaptive_global_cap": rank_adaptive_global_cap,
        "max_total_condensed_ratio": max_total_condensed_ratio,
        "assignment_chunk_size": assignment_chunk_size,
        "inference_dst_chunk_size": inference_dst_chunk_size,
        "block_norm": block_norm,
        "block_gate": block_gate,
        "block_dropout": block_dropout,
    }
    condensed_nodes_by_type = {k: int(v.shape[0]) for k, v in condensed.node_features.items()}
    condensed_edges_by_relation = {str(k): int(v.shape[1]) for k, v in condensed.edge_index.items()}
    condensed_nodes_total = int(sum(condensed_nodes_by_type.values()))
    condensed_edges_total = int(sum(condensed_edges_by_relation.values()))
    shadow_nodes_total = max(0, condensed_nodes_total - int(prototype_summary["effective_M_tau"]))
    original_nodes_total = int(sum(graph.num_nodes.values()))
    original_edges_total = int(sum(int(index.shape[1]) for index in graph.edge_index.values()))
    original_bytes = _estimate_feature_bytes(phi) + _estimate_edge_bytes(graph.edge_index, {relation: alpha[relation] for relation in target_relations})
    condensed_bytes = _estimate_feature_bytes(condensed.node_features) + _estimate_edge_bytes(condensed.edge_index, condensed.edge_weight)
    requested_target_ratio = None if ratio is None else float(ratio)
    effective_target_ratio = float(prototype_summary["effective_M_tau"] / max(1, int(graph.train_idx.numel())))
    shadow_node_ratio = float(shadow_nodes_total / max(1, original_nodes_total))
    total_condensed_node_ratio = float(condensed_nodes_total / max(1, original_nodes_total))
    total_condensed_edge_ratio = float(condensed_edges_total / max(1, original_edges_total))
    byte_size_compression = float(condensed_bytes / max(1, original_bytes))
    shadow_feature_norm_stats = {
        relation: values.get("shadow_norms", {})
        for relation, values in diagnostics.items()
        if "shadow_norms" in values
    }
    nonnegative_weights = all(bool(torch.all(w >= 0).item()) for w in condensed.edge_weight.values())
    summary = {
        "method": method_name,
        "method_variant": method_name,
        "dataset": graph.dataset_name,
        "split": "processed_local",
        "target_type": graph.target_type,
        "directed_relations": [str(relation) for relation in target_relations],
        **budget_metadata,
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
        "shadow_budget_policy": shadow_policy,
        "legacy_shadow_budget_policy": "explicit" if M_r is not None else "ratio_based",
        "shadow_policy": shadow_policy,
        "shadow_min_per_relation": shadow_min_per_relation,
        "shadow_max_multiplier": shadow_max_multiplier,
        "adaptive_b_enabled": adaptive_b,
        "b_max": b_max,
        "b_by_relation": b_by_relation,
        "rank_diagnostic_k": rank_diagnostic_k,
        "shadow_ratio_target_target": shadow_target_target_ratio,
        "shadow_ratio_non_target": shadow_non_target_ratio,
        "min_shadow_per_relation": min_shadows_per_relation,
        "shadow_budgets_by_relation": {str(relation): value for relation, value in resolved_M_r.items()},
        "shadow_total_budget": shadow_total_budget,
        "rank_adaptive_global_cap": rank_adaptive_global_cap,
        "max_total_condensed_ratio": max_total_condensed_ratio,
        "shadow_nodes_total": shadow_nodes_total,
        "k_s": k_s,
        "feature_dim": feature_dim,
        "projection_type": projection_type,
        "standardization_scope": "train_only",
        "degree_scale": degree_scale,
        "signature_degree_eta": signature_degree_eta,
        "feature_mode": feature_mode,
        "diffusion_steps": list(diffusion_steps_tuple),
        "include_highpass": include_highpass,
        "metapath_signature": metapath_signature,
        "metapath_model_input": metapath_model_input,
        "multiscale_dim": multiscale_dim,
        "multiscale_metadata": multiscale_metadata,
        "block_norm": block_norm,
        "block_gate": block_gate,
        "block_dropout": block_dropout,
        "block_stats": multiscale_metadata.get("block_stats", {}),
        "target_base_dim": target_base_dim,
        "target_degree_dim": target_degree_dim,
        "target_input_dim": target_input_dim,
        "model": model_type,
        "model_type": model_type,
        "relation_gate": relation_gate,
        "relation_gate_init": relation_gate_init,
        "relation_gate_values": train_metrics.get("relation_gate_values", {}),
        "logit_adjustment_tau": logit_adjustment_tau,
        "skeleton_policy": skeleton_policy,
        "skeleton_coverage": skeleton_coverage,
        "skeleton_k_max": skeleton_k_max,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "seed": seed,
        "condensed_nodes_by_type": condensed_nodes_by_type,
        "condensed_edges_by_relation": condensed_edges_by_relation,
        "condensed_nodes_total": condensed_nodes_total,
        "condensed_edges_total": condensed_edges_total,
        "ratio_mode": ratio_mode,
        "requested_target_ratio": requested_target_ratio,
        "effective_target_ratio": effective_target_ratio,
        "shadow_node_ratio": shadow_node_ratio,
        "total_condensed_node_ratio": total_condensed_node_ratio,
        "total_condensed_edge_ratio": total_condensed_edge_ratio,
        "byte_size_compression": byte_size_compression,
        "condensed_node_ratio_to_train_target": float(condensed_nodes_total / max(1, int(graph.train_idx.numel()))),
        "condensed_node_ratio_to_all_task_nodes": float(condensed_nodes_total / max(1, int(graph.num_nodes[graph.target_type]))),
        "condensation_time": max(0.0, condensation_time),
        "training_time": train_time,
        "inference_time": infer_time,
        "peak_cpu_ram": current_cpu_ram_bytes(),
        "peak_gpu_ram": current_gpu_ram_bytes(),
        "disk_bytes": 0,
        "num_full_edge_scans": 0,
        "full_edge_scans": 0,
        "edge_slice_cache_bytes": 0,
        "cache_all_targets": False,
        "demand_edge_chunk_size": demand_edge_chunk_size,
        "inference_edge_chunk_size": inference_edge_chunk_size,
        "inference_dst_chunk_size": inference_dst_chunk_size,
        "assignment_chunk_size": assignment_chunk_size,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        **class_metadata,
        **train_metrics,
        **pred_diag,
        "predicted_classes": pred_diag["num_predicted_classes"],
        "diagnostics": diagnostics,
        "rank_diagnostics_by_relation": rank_diagnostics,
        "shadow_recon_err_by_relation": {relation: values.get("ShadowReconErr") for relation, values in diagnostics.items() if isinstance(values, dict) and "ShadowReconErr" in values},
        "skeleton_coverage_by_relation": {relation: values.get("SkeletonMassCoverage") for relation, values in diagnostics.items() if isinstance(values, dict) and "SkeletonMassCoverage" in values},
        "residual_energy_by_relation": {relation: values.get("ResidualEnergy") for relation, values in diagnostics.items() if isinstance(values, dict) and "ResidualEnergy" in values},
        "shadow_feature_norm_stats": shadow_feature_norm_stats,
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
        "nonnegative_weights": bool(nonnegative_weights),
        "all_edge_weights_nonnegative": bool(nonnegative_weights),
        "uses_custom_weighted_layer": True,
        "uses_library_auto_normalization": False,
        "alpha_normalization": "destination_row",
        "residual_shadows_signed": residual_signed,
        "status": "completed",
    }
    if not summary["schema_preserved"]:
        raise ValueError("schema_preserved is false")
    if not summary["nonnegative_weights"]:
        raise ValueError("condensed graph contains negative edge weights")
    output_path = Path(output_path)
    summary = attach_run_metadata(_jsonable(summary), config=config_for_hash)
    write_json_summary(output_path, summary, config=config_for_hash)
    return summary
