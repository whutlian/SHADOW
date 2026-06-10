from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t24_reddit_condense import _read_preprop_manifest, _train_eval
from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.data.edge_stream import MemmapEdgeStream
from shadow_hgc.data.reddit_stream import load_reddit_raw_memmap_labels_and_splits, load_reddit_raw_memmap_manifest
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.ratio.scale_bucket import account_full_node_ratio
from shadow_hgc.sft.coreset import select_classwise_coreset_rows
from shadow_hgc.sft.fdm_lite import T25_METHODS, select_fdm_lite_rows
from shadow_hgc.sft.hnr import HNRStats, compute_streaming_hnr_stats
from shadow_hgc.sft.signature_cache import write_or_load_sft_signature_cache_from_memmap
from shadow_hgc.sft.t25_contract import T25_OUTPUT_FIELDS, make_t25_row
from shadow_hgc.train.lazy_sft_memmap import load_manifest_block_store


REDDIT_T24_REFERENCE = {
    0.005: {"accuracy": 0.9244564924689873, "macro_f1": 0.8862562817528249},
}

CURRENT_METHODS: list[tuple[str, str]] = [
    ("current_sft_signature_random", "random"),
    ("current_sft_signature_medoid", "medoid"),
    ("current_sft_signature_kcenter", "kcenter"),
    ("current_sft_signature_shadow_b1", "hybrid"),
]

FIELDS = T25_OUTPUT_FIELDS + ["promotion_reason", "signature_cache_bytes", "preprop_cache_bytes", "prediction_entropy"]


def _load_train_signature(signature_dir: str | Path, metadata: dict[str, Any]) -> torch.Tensor:
    train_meta = metadata["arrays"]["train_signature"]
    array = np.memmap(
        Path(signature_dir) / train_meta["path"],
        mode="r",
        dtype=np.dtype(train_meta["dtype"]),
        shape=tuple(int(value) for value in train_meta["shape"]),
    )
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())


def _compute_hnr(args: argparse.Namespace, labels: torch.Tensor, train_rows: torch.Tensor) -> tuple[HNRStats | None, torch.Tensor | None]:
    if not bool(args.enable_hnr):
        return None, None
    manifest = load_reddit_raw_memmap_manifest(args.memmap_root)
    root = Path(args.memmap_root)

    def stream_factory() -> MemmapEdgeStream:
        return MemmapEdgeStream(
            root / manifest["src_path"],
            root / manifest["dst_path"],
            chunk_size=int(args.edge_chunk_size),
            edge_limit=args.edge_limit,
        )

    stats = compute_streaming_hnr_stats(
        edge_stream_factory=stream_factory,
        num_nodes=int(manifest["num_nodes"]),
        labels=labels,
        train_rows=train_rows,
        target_rows=train_rows,
        num_classes=int(manifest["num_classes"]),
    )
    hnr_features = torch.stack(
        [
            torch.log1p(stats.degree.to(torch.float32)),
            stats.homophily,
            stats.quality,
            stats.label_max_affinity,
            stats.missing_label_ratio,
            stats.node_weight,
        ],
        dim=1,
    )
    return stats, hnr_features


def _budget_for_ratio(num_nodes: int, num_classes: int, ratio: float, target_fraction: float, *, shadow_materialized: bool) -> tuple[int, int, int]:
    total = max(1, int(round(int(num_nodes) * float(ratio))))
    if shadow_materialized:
        target = max(int(num_classes), int(round(total * float(target_fraction))))
    else:
        target = max(int(num_classes), total)
    shadow = max(0, total - target)
    return total, target, shadow


def _promotion(method: str, ratio: float, metrics: dict[str, Any] | None) -> tuple[str, str]:
    if "shadow" in method or method.endswith("_b2"):
        return "not_promoted", "shadow_materialization_not_trained"
    if metrics is None:
        return "not_promoted", "not_trained"
    acc = float(metrics.get("accuracy", 0.0))
    macro = float(metrics.get("macro_f1", 0.0))
    ref = REDDIT_T24_REFERENCE.get(float(ratio))
    if ref is not None and (acc < float(ref["accuracy"]) or macro < float(ref["macro_f1"])):
        return "not_promoted", "no_regression_gate_not_met"
    if method == "sft_hnr_fdm_hybrid" and float(ratio) == 0.005 and acc >= 0.928 and macro >= 0.890:
        return "promoted", "passed_reddit_0p50_t25_gate"
    if method == "sft_hnr_fdm_shadow_b2":
        return "not_promoted", "b2_ablation_only"
    return "not_promoted", "acceptance_gate_not_met"


def run_reddit_t25(args: argparse.Namespace) -> list[dict[str, Any]]:
    labels, train_rows, _valid_rows, test_rows = load_reddit_raw_memmap_labels_and_splits(args.memmap_root)
    preprop_manifest = _read_preprop_manifest(args.manifest_dir)
    selected_blocks = json.loads(args.selected_blocks)
    store = load_manifest_block_store(args.manifest_dir).subset(selected_blocks)
    signature_started = time.perf_counter()
    signature_cache = write_or_load_sft_signature_cache_from_memmap(
        manifest_dir=args.manifest_dir,
        splits={"train": train_rows},
        train_rows=train_rows,
        out_dir=args.signature_dir,
        selected_blocks=selected_blocks,
        batch_size=args.signature_batch_size,
    )
    signature = _load_train_signature(args.signature_dir, signature_cache.metadata)
    signature_time = float(time.perf_counter() - signature_started)
    hnr_started = time.perf_counter()
    hnr_stats, hnr_features = _compute_hnr(args, labels, train_rows)
    hnr_time = float(time.perf_counter() - hnr_started)
    t25_signature = signature if hnr_features is None else torch.cat([signature, hnr_features], dim=1)
    num_nodes = int(preprop_manifest.get("blocks", [{}])[0].get("shape", [labels.numel()])[0])
    num_classes = int(labels.max().item()) + 1
    rows: list[dict[str, Any]] = []
    methods: list[str] = [method for method, _ in CURRENT_METHODS] + list(T25_METHODS)
    requested_methods = set(args.methods or methods)
    for ratio in [float(value) for value in args.ratios]:
        for method in methods:
            if method not in requested_methods:
                continue
            shadow_method = "shadow" in method or method.endswith("_b2")
            shadow_materialized = False
            _total, target_budget, shadow_nodes = _budget_for_ratio(num_nodes, num_classes, ratio, args.target_fraction, shadow_materialized=shadow_materialized)
            condensation_started = time.perf_counter()
            metrics: dict[str, Any] | None
            if method.startswith("current_"):
                mode = dict(CURRENT_METHODS)[method]
                selected = select_classwise_coreset_rows(signature, labels, train_rows, target_budget, mode=mode, seed=int(args.seed))
                fdm_diag = {"fdm_signature_dim": int(signature.shape[1]), "fdm_num_subclasses": "", "fdm_candidate_pool_size": "", "fdm_mode": "none"}
                strata = None
            else:
                selection = select_fdm_lite_rows(
                    t25_signature,
                    labels,
                    train_rows,
                    total_budget=target_budget,
                    method=method,
                    node_weight=None if hnr_stats is None else hnr_stats.node_weight,
                    stratum=None if hnr_stats is None else hnr_stats.stratum,
                    fdm_signature_dim=int(args.fdm_signature_dim),
                    scale_bucket=args.scale_bucket,
                    candidate_rho=int(args.fdm_candidate_rho),
                    candidate_max=int(args.fdm_candidate_max),
                    fdm_k_min=int(args.fdm_k_min),
                    fdm_k_max=int(args.fdm_k_max),
                    seed=int(args.seed),
                )
                selected = selection.selected_rows
                fdm_diag = selection.diagnostics
                strata = None
            del strata
            condensation_time = float(time.perf_counter() - condensation_started) + signature_time / max(1, len(args.ratios) * max(1, len(requested_methods)))
            if args.train:
                metrics, train_s, infer_s = _train_eval(
                    store=store,
                    labels=labels,
                    full_train_rows=train_rows,
                    selected_rows=selected,
                    test_rows=test_rows,
                    epochs=int(args.epochs),
                    hidden_dim=int(args.hidden_dim),
                    device=args.device,
                    batch_size=int(args.batch_size),
                    eval_batch_size=int(args.eval_batch_size),
                    seed=int(args.seed),
                )
                status = "completed_streaming"
                reason = "trained T25 selection over full Reddit streaming-preprop memmap blocks"
            else:
                metrics = None
                train_s = ""
                infer_s = ""
                status = "ready_not_trained"
                reason = "use --train to run local condensed training"
            shadow_b = 2 if method.endswith("_b2") else (1 if shadow_method else "")
            condensed_edges = int(selected.numel())
            promotion_status, promotion_reason = _promotion(method, ratio, metrics)
            row_status = status if not shadow_method else f"{status}_diagnostic"
            row = make_t25_row(
                dataset="Reddit",
                method=method,
                requested_full_node_ratio=ratio,
                original_total_nodes=num_nodes,
                target_prototypes=int(selected.numel()),
                shadow_nodes=int(shadow_nodes),
                total_condensed_edges=condensed_edges,
                seed=int(args.seed),
                accuracy="" if metrics is None else metrics.get("accuracy", ""),
                macro_f1="" if metrics is None else metrics.get("macro_f1", ""),
                predicted_classes="" if metrics is None else metrics.get("predicted_class_count", ""),
                status=row_status,
                promotion_status=promotion_status,
                notes=reason if not shadow_method else f"{reason}; shadow graph materialization is not trained in this SFT-row runner",
                promotion_reason=promotion_reason,
                precompute_time=signature_time + hnr_time,
                condensation_time=condensation_time,
                training_time=train_s,
                inference_time=infer_s,
                peak_cpu_ram=current_cpu_ram_bytes() / (1024**3),
                peak_gpu_ram=current_gpu_ram_bytes() / (1024**3),
                cache_bytes=int(preprop_manifest.get("total_cache_bytes", 0)) + int(signature_cache.metadata["cache_bytes"]),
                preprop_cache_bytes=int(preprop_manifest.get("total_cache_bytes", 0)),
                signature_cache_bytes=int(signature_cache.metadata["cache_bytes"]),
                full_edge_scans=int(preprop_manifest.get("full_edge_scans", 0)),
                hnr_edge_scans=0 if hnr_stats is None else hnr_stats.hnr_edge_scans,
                hnr_cache_bytes=0 if hnr_stats is None else hnr_stats.hnr_cache_bytes,
                hnr_hist_mode=args.hnr_hist_mode if args.enable_hnr else "none",
                shadow_b=shadow_b,
                prediction_entropy="" if metrics is None else metrics.get("prediction_entropy", ""),
                **fdm_diag,
            )
            if promotion_status != "promoted":
                row["failure_reason"] = promotion_reason
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T25 Reddit HNR-FDM-lite ratio sweep.")
    parser.add_argument("--manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--selected-blocks", default='["X0","X1","X2","X3","Xres1","Y1","Y2","Y3","structure"]')
    parser.add_argument("--signature-dir", default="experiments/sft_signatures/Reddit/t25_hnr_fdm")
    parser.add_argument("--signature-batch-size", type=int, default=32768)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.001, 0.0025, 0.005, 0.01])
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--enable-hnr", action="store_true", default=True)
    parser.add_argument("--enable-fdm", action="store_true", default=True)
    parser.add_argument("--fdm-mode", default="lite", choices=["lite", "full"])
    parser.add_argument("--ratio-mode", default="full_node", choices=["full_node"])
    parser.add_argument("--scale-bucket", default="medium", choices=["medium", "large", "ultra"])
    parser.add_argument("--target-shadow-split", default="0.70,0.30")
    parser.add_argument("--fdm-signature-dim", type=int, default=128, choices=[64, 128])
    parser.add_argument("--fdm-k-min", type=int, default=2)
    parser.add_argument("--fdm-k-max", type=int, default=32)
    parser.add_argument("--fdm-candidate-rho", type=int, default=16)
    parser.add_argument("--fdm-candidate-max", type=int, default=1024)
    parser.add_argument("--hnr-hist-mode", default="topk", choices=["auto", "full", "topk", "none"])
    parser.add_argument("--shadow-b", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--edge-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--edge-limit", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv", default="experiments/tables/t25_reddit_hnr_fdm_ratio_sweep_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t25_reddit_hnr_fdm_ratio_sweep.md")
    args = parser.parse_args()
    target_fraction, _shadow_fraction = [float(value) for value in str(args.target_shadow_split).split(",", maxsplit=1)]
    args.target_fraction = target_fraction
    if args.fdm_mode != "lite":
        raise ValueError("promoted T25 runner only supports --fdm-mode lite")
    rows = run_reddit_t25(args)
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(
        args.report,
        [
            "# T25 Reddit HNR-FDM-lite",
            "",
            f"- Train mode: `{bool(args.train)}`",
            f"- HNR enabled: `{bool(args.enable_hnr)}`",
            "- Rows use full-node ratio accounting and remain not_promoted unless the no-regression and T25 gates pass.",
            "",
            *markdown_table(rows, ["requested_full_node_ratio", "method", "status", "actual_full_node_ratio", "accuracy", "macro_f1", "predicted_classes", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
