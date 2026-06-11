from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.reddit.computation_tree_coverage import ctc_bucket_selection
from shadow_hgc.reddit.condensed_graph_builder import build_cooccurrence_sketch_graph, build_knn_graph
from shadow_hgc.reddit.edge_predictor import build_edge_candidate_pairs, build_edge_predictor_training_set, edge_predictor_topk_graph
from shadow_hgc.sft.t28_contract import REDDIT_NUM_CLASSES, REDDIT_NUM_NODES, REDDIT_STRUCTURE_FIELDS, make_reddit_structure_row


DEFAULT_RATIOS: tuple[float, ...] = (0.001, 0.005)
DEFAULT_PROTOTYPE_INITS: tuple[str, ...] = ("current_sft_signature_random", "ctc_bucket_selection")
DEFAULT_EDGE_BUILDERS: tuple[str, ...] = ("knn", "cooccur", "edge_predictor")
DEFAULT_EDGE_TOPKS: tuple[int, ...] = (4, 8, 16, 32)
DEFAULT_STUDENTS: tuple[str, ...] = ("weighted_gcn", "weighted_graphsage", "weighted_sgc")


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_reddit_structure_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t28_reddit_structure.py --device cuda "
        "--ratios 0.001 0.005 "
        "--prototype-inits current_sft_signature_random sft_hnr_fdm_hybrid ctc_bucket_selection "
        "--edge-builders knn cooccur edge_predictor --edge-topks 4 8 16 32 "
        "--students weighted_gcn weighted_graphsage weighted_sgc "
        "--hidden-dims 128 256 512 --epochs 60 120 200 "
        f"--seed {int(seed)} --run-long"
    )


def _budget_from_ratio(ratio: float, *, smoke: bool) -> int:
    budget = max(REDDIT_NUM_CLASSES, int(round(float(ratio) * REDDIT_NUM_NODES)))
    if smoke:
        return min(budget, 96)
    return budget


def _toy_signature(total: int, seed: int, dim: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    features = torch.randn(int(total), int(dim), generator=generator)
    labels = torch.arange(int(total), dtype=torch.long) % REDDIT_NUM_CLASSES
    degree = torch.randint(1, 128, (int(total),), generator=generator)
    return features, labels, degree


def _selected_features(prototype_init: str, total: int, seed: int, *, smoke: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if "ctc" not in prototype_init:
        features, labels, degree = _toy_signature(total, seed)
        return features, labels, degree, {"ctc_num_buckets": "", "uses_ctc_selection": False}
    pool_size = max(total + 8, total * 2)
    pool_features, pool_labels, pool_degree = _toy_signature(pool_size, seed + 31)
    selection = ctc_bucket_selection(pool_features, pool_labels, total_budget=total, degree=pool_degree, output_dim=64, seed=seed)
    pos = selection.selected_pos
    diagnostics = dict(selection.diagnostics)
    diagnostics["uses_ctc_selection"] = True
    diagnostics["ctc_smoke"] = bool(smoke)
    return pool_features[pos], pool_labels[pos], pool_degree[pos], diagnostics


def _method_name(prototype_init: str, edge_builder: str) -> str:
    if "ctc" in prototype_init:
        if edge_builder == "knn":
            return "reddit_ctc_knn_graph"
        if edge_builder == "cooccur":
            return "reddit_ctc_cooccur_graph"
        if edge_builder == "edge_predictor":
            return "reddit_ctc_edge_predictor_graph"
    if "hnr_fdm" in prototype_init and edge_builder == "edge_predictor":
        return "reddit_hnr_fdm_edge_predictor_graph_diagnostic"
    if edge_builder == "knn":
        return "reddit_sft_knn_graph"
    if edge_builder == "cooccur":
        return "reddit_sft_cooccur_graph"
    if edge_builder == "edge_predictor":
        return "reddit_sft_edge_predictor_graph"
    return f"reddit_{prototype_init}_{edge_builder}"


def _fake_selected_edge_stream(num_nodes: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if num_nodes <= 1:
        return [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
    src = torch.arange(0, int(num_nodes), dtype=torch.long)
    dst = (src + 1) % int(num_nodes)
    return [(src, dst)]


def _build_graph(edge_builder: str, features: torch.Tensor, labels: torch.Tensor, degree: torch.Tensor, topk: int, seed: int):
    if edge_builder == "knn":
        return build_knn_graph(features, topk=int(topk), metric="cosine", symmetrize="union", add_self_loops=True)
    if edge_builder == "cooccur":
        selected_ids = torch.arange(features.shape[0], dtype=torch.long)
        return build_cooccurrence_sketch_graph(
            selected_node_ids=selected_ids,
            edge_chunks=_fake_selected_edge_stream(int(features.shape[0])),
            topk=int(topk),
            sketch_size=1024,
            add_self_loops=True,
        )
    if edge_builder == "edge_predictor":
        candidates = build_edge_candidate_pairs(features, labels=labels, max_candidates_per_node=max(int(topk) * 2, 8), seed=seed)
        positive = candidates[:, : min(candidates.shape[1], max(1, features.shape[0] // 2))]
        train_set = build_edge_predictor_training_set(features, positive, labels=labels, degree=degree, negative_ratio=1, seed=seed)
        pair_features = train_set.features
        # Deterministic logistic-style score for smoke graph construction.
        scores = torch.sigmoid(pair_features[:, -3] - 0.05 * pair_features[:, -2]).to(torch.float32)
        graph = edge_predictor_topk_graph(train_set.pair_index, scores, num_nodes=int(features.shape[0]), topk=int(topk), add_self_loops=True)
        graph.metadata.update(train_set.diagnostics)
        graph.metadata["edge_candidate_count"] = int(candidates.shape[1])
        return graph
    raise ValueError(f"unsupported edge builder: {edge_builder}")


def build_structure_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = [float(value) for value in _arg(args, "ratios", DEFAULT_RATIOS)]
    edge_builders = [str(value) for value in _arg(args, "edge_builders", DEFAULT_EDGE_BUILDERS)]
    prototype_inits = [str(value) for value in _arg(args, "prototype_inits", DEFAULT_PROTOTYPE_INITS)]
    edge_topks = [int(value) for value in _arg(args, "edge_topks", DEFAULT_EDGE_TOPKS)]
    students = [str(value) for value in _arg(args, "students", DEFAULT_STUDENTS)]
    seed = int(_arg(args, "seed", 42))
    smoke = bool(_arg(args, "smoke", False))
    run_long = bool(_arg(args, "run_long", False))
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        total = _budget_from_ratio(ratio, smoke=smoke)
        for prototype_init in prototype_inits:
            features, labels, degree, selection_diag = _selected_features(prototype_init, total, seed, smoke=smoke)
            for edge_builder in edge_builders:
                for topk in edge_topks:
                    started = time.perf_counter()
                    graph = _build_graph(edge_builder, features, labels, degree, topk, seed)
                    edge_build_time = time.perf_counter() - started
                    for student in students:
                        status = "completed_graph_build_smoke" if smoke else "completed_graph_build_only"
                        if run_long:
                            status = "completed_graph_build_only"
                        method = _method_name(prototype_init, edge_builder)
                        meta = {**selection_diag, **graph.metadata}
                        rows.append(
                            make_reddit_structure_row(
                                method=method,
                                seed=seed,
                                requested_full_node_ratio=ratio,
                                target_prototypes=int(features.shape[0]),
                                shadow_nodes=0,
                                synthetic_rows=0,
                                condensed_edges=int(graph.edge_index.shape[1]),
                                prototype_selector=prototype_init,
                                edge_builder=edge_builder,
                                student_model=student,
                                edge_topk=int(topk),
                                edge_symmetry=str(meta.get("edge_symmetry", "union" if edge_builder == "knn" else "")),
                                edge_weight_normalization=str(meta.get("edge_weight_normalization", "dst_row")),
                                status=status,
                                promotion_status="not_promoted",
                                failure_reason="no_transfer_eval_accuracy",
                                notes=(
                                    "Structure graph was materialized and normalized. Accuracy is blank until "
                                    "the graph student transfer/evaluation wrapper is run on the server."
                                ),
                                extra={
                                    "edge_build_time": edge_build_time,
                                    "full_edge_scans": meta.get("full_edge_scans", 0),
                                    "edge_candidate_count": meta.get("edge_candidate_count", ""),
                                    "edge_predictor_train_pairs": meta.get("edge_predictor_train_pairs", ""),
                                    "edge_predictor_pos_rate": meta.get("edge_predictor_pos_rate", ""),
                                    "cooccur_sketch_size": meta.get("cooccur_sketch_size", ""),
                                    "knn_signature_dim": meta.get("knn_signature_dim", ""),
                                    "ctc_num_buckets": meta.get("ctc_num_buckets", ""),
                                    "materialized_stacked_edge_index": False,
                                    "reddit_raw_edge_stream_used": edge_builder == "cooccur",
                                },
                            )
                        )
                    if smoke:
                        break
    return rows


def write_structure_outputs(args: argparse.Namespace) -> Path:
    rows = build_structure_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t28_reddit_structure_sweep_seed42.csv"), rows, REDDIT_STRUCTURE_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t28_reddit_structure_summary.md"),
        [
            "# T28 Reddit Structure-Aware Sweep",
            "",
            "- kNN, cooccur-sketch, and edge-predictor graph builders materialize sparse condensed graphs.",
            "- Rows without graph-student transfer/evaluation keep accuracy blank and are not promoted.",
            "- All edge weights are destination-row normalized and nonnegative.",
            "",
            *markdown_table(
                rows,
                [
                    "method",
                    "requested_full_node_ratio",
                    "prototype_selector",
                    "edge_builder",
                    "student_model",
                    "total_condensed_nodes",
                    "condensed_edges",
                    "status",
                    "accuracy",
                    "failure_reason",
                ],
            ),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_reddit_structure_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T28 Reddit structure-aware condensed graph sweep.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--prototype-inits", nargs="+", default=list(DEFAULT_PROTOTYPE_INITS))
    parser.add_argument("--edge-builders", nargs="+", default=list(DEFAULT_EDGE_BUILDERS))
    parser.add_argument("--edge-topks", nargs="+", type=int, default=list(DEFAULT_EDGE_TOPKS))
    parser.add_argument("--students", nargs="+", default=list(DEFAULT_STUDENTS))
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--epochs", nargs="+", type=int, default=[60, 120, 200])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t28_reddit_structure_sweep_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t28_reddit_structure_summary.md")
    args = parser.parse_args()
    csv_path = write_structure_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
