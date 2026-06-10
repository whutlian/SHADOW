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

from scripts.run_t24_products_sft_recovery import _train_eval
from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.sft.coreset import select_classwise_coreset_rows
from shadow_hgc.sft.fdm_lite import select_fdm_lite_rows
from shadow_hgc.sft.products_recovery import PRODUCTS_FULLGRAPH_TEACHER
from shadow_hgc.sft.signature_cache import write_or_load_sft_signature_cache_from_memmap
from shadow_hgc.sft.t25_contract import T25_OUTPUT_FIELDS, make_t25_row
from shadow_hgc.train.lazy_sft_memmap import load_manifest_block_store, load_products_labels_and_splits


FIELDS = T25_OUTPUT_FIELDS + [
    "ladder_stage",
    "fullgraph_acc",
    "full_to_current_gap",
    "promotion_reason",
    "products_diag_memmap_row_order_matches_node_id",
    "products_diag_masks_aligned",
    "products_diag_all_classes_have_floor",
    "products_diag_predicted_class_collapse",
]


def _load_train_signature(signature_dir: str | Path, metadata: dict[str, Any]) -> torch.Tensor:
    train_meta = metadata["arrays"]["train_signature"]
    array = np.memmap(
        Path(signature_dir) / train_meta["path"],
        mode="r",
        dtype=np.dtype(train_meta["dtype"]),
        shape=tuple(int(value) for value in train_meta["shape"]),
    )
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())


def _make_products_row(
    *,
    ratio: float,
    method: str,
    ladder_stage: str,
    num_nodes: int,
    target_prototypes: int,
    shadow_nodes: int,
    metrics: dict[str, Any] | None,
    status: str,
    seed: int,
    timing: dict[str, Any],
    diagnostics: dict[str, Any],
    selected_rows: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    promotion_status: str = "not_promoted",
    promotion_reason: str = "diagnostic_only",
) -> dict[str, Any]:
    acc = "" if metrics is None else metrics.get("accuracy", "")
    macro = "" if metrics is None else metrics.get("macro_f1", "")
    predicted = "" if metrics is None else metrics.get("predicted_class_count", "")
    row = make_t25_row(
        dataset="ogbn-products",
        method=method,
        requested_full_node_ratio=float(ratio),
        original_total_nodes=int(num_nodes),
        target_prototypes=int(target_prototypes),
        shadow_nodes=int(shadow_nodes),
        total_condensed_edges=int(target_prototypes) * (2 if method.endswith("b2") else 1),
        seed=seed,
        accuracy=acc,
        macro_f1=macro,
        predicted_classes=predicted,
        status=status,
        promotion_status=promotion_status,
        promotion_reason=promotion_reason,
        notes=ladder_stage,
        **timing,
        **diagnostics,
    )
    full = float(PRODUCTS_FULLGRAPH_TEACHER["accuracy"])
    row["ladder_stage"] = ladder_stage
    row["fullgraph_acc"] = full
    row["full_to_current_gap"] = "" if acc == "" else full - float(acc)
    row.setdefault("products_diag_memmap_row_order_matches_node_id", "")
    row.setdefault("products_diag_masks_aligned", "")
    if selected_rows is not None and labels is not None and int(selected_rows.numel()) > 0:
        y = labels[selected_rows.to(torch.long)].to(torch.long)
        counts = torch.bincount(y, minlength=int(labels.max().item()) + 1)
        row["products_diag_all_classes_have_floor"] = bool(torch.all(counts > 0).item())
    else:
        row.setdefault("products_diag_all_classes_have_floor", "")
    row["products_diag_predicted_class_collapse"] = "" if predicted == "" else int(predicted) < 40
    if promotion_status != "promoted":
        row["failure_reason"] = promotion_reason
    return row


def run_products_t25(args: argparse.Namespace) -> list[dict[str, Any]]:
    labels, train_rows, valid_rows, test_rows = load_products_labels_and_splits(args.products_root)
    store = load_manifest_block_store(args.manifest_dir).subset(json.loads(args.selected_blocks))
    signature_cache = write_or_load_sft_signature_cache_from_memmap(
        manifest_dir=args.manifest_dir,
        splits={"train": train_rows},
        train_rows=train_rows,
        out_dir=args.signature_dir,
        selected_blocks=json.loads(args.selected_blocks),
        batch_size=args.signature_batch_size,
    )
    signature_started = time.perf_counter()
    signature = _load_train_signature(args.signature_dir, signature_cache.metadata)
    signature_time = float(time.perf_counter() - signature_started)
    num_nodes = 2_449_029
    num_classes = int(labels.max().item()) + 1
    split_union = torch.cat([train_rows, valid_rows, test_rows]).to(torch.long)
    split_unique = torch.unique(split_union)
    masks_aligned = bool(
        int(split_unique.numel()) == int(split_union.numel())
        and int(split_union.min().item()) >= 0
        and int(split_union.max().item()) < int(labels.numel())
    )
    store_row_order = bool(int(store.num_rows) == int(labels.numel()))
    rows: list[dict[str, Any]] = []
    for ratio in [float(value) for value in args.ratios]:
        total = max(1, int(round(num_nodes * ratio)))
        target = max(num_classes, total)
        shadow = 0
        common_timing = {
            "precompute_time": signature_time,
            "peak_cpu_ram": current_cpu_ram_bytes() / (1024**3),
            "peak_gpu_ram": current_gpu_ram_bytes() / (1024**3),
            "cache_bytes": int(signature_cache.metadata["cache_bytes"]),
            "full_edge_scans": "",
            "hnr_edge_scans": 0,
            "hnr_cache_bytes": 0,
            "hnr_hist_mode": "none",
        }
        rows.append(
            _make_products_row(
                ratio=ratio,
                method="P0_identity_replay",
                ladder_stage="P0",
                num_nodes=num_nodes,
                target_prototypes=target,
                shadow_nodes=0,
                metrics={"accuracy": PRODUCTS_FULLGRAPH_TEACHER["accuracy"], "macro_f1": PRODUCTS_FULLGRAPH_TEACHER["macro_f1"], "predicted_class_count": num_classes},
                status="completed_fullgraph_replay",
                seed=int(args.seed),
                timing=common_timing,
                diagnostics={
                    "fdm_mode": "none",
                    "fdm_signature_dim": "",
                    "fdm_num_subclasses": "",
                    "fdm_candidate_pool_size": "",
                    "shadow_b": "",
                    "products_diag_memmap_row_order_matches_node_id": store_row_order,
                    "products_diag_masks_aligned": masks_aligned,
                },
                selected_rows=train_rows,
                labels=labels,
                promotion_reason="identity_reference_only",
            )
        )
        selections: list[tuple[str, str, torch.Tensor, dict[str, Any], int]] = []
        selected_p1 = select_classwise_coreset_rows(signature, labels, train_rows, target, mode="medoid", seed=int(args.seed))
        selections.append(("P1_selected_real_prototypes_replay", "P1", selected_p1, {"fdm_mode": "none", "fdm_signature_dim": int(signature.shape[1]), "fdm_num_subclasses": "", "fdm_candidate_pool_size": ""}, 0))
        fdm = select_fdm_lite_rows(
            signature,
            labels,
            train_rows,
            total_budget=target,
            method="sft_hnr_fdm_hybrid",
            fdm_signature_dim=int(args.fdm_signature_dim),
            scale_bucket="large",
            candidate_rho=int(args.fdm_candidate_rho),
            candidate_max=int(args.fdm_candidate_max),
            seed=int(args.seed),
        )
        selections.append(("P2_hnr_fdm_prototype_oracle", "P2", fdm.selected_rows, fdm.diagnostics, 0))
        selections.append(("P3_hnr_fdm_shadow_b1", "P3", fdm.selected_rows, fdm.diagnostics, 1))
        selections.append(("P3_hnr_fdm_shadow_b2", "P3", fdm.selected_rows, fdm.diagnostics, 2))
        for method, stage, selected, diag, shadow_b in selections:
            started = time.perf_counter()
            if args.train:
                metrics, train_s, infer_s = _train_eval(store, labels, train_rows, valid_rows, test_rows, selected, epochs=args.epochs, hidden_dim=args.hidden_dim, device=args.device)
                status = "completed_streaming"
            else:
                metrics = None
                train_s = ""
                infer_s = ""
                status = "ready_not_trained"
            timing = dict(common_timing)
            timing.update({"condensation_time": time.perf_counter() - started, "training_time": train_s, "inference_time": infer_s})
            diagnostics = dict(diag)
            diagnostics["shadow_b"] = shadow_b if shadow_b else ""
            diagnostics["products_diag_memmap_row_order_matches_node_id"] = store_row_order
            diagnostics["products_diag_masks_aligned"] = masks_aligned
            method_shadow = 0
            promotion_reason = "products_t25_gate_not_met"
            if shadow_b:
                promotion_reason = "shadow_materialization_not_trained"
            promote = False
            if metrics is not None and stage == "P2" and ratio == 0.0025 and float(metrics["accuracy"]) >= 0.68:
                promote = True
                promotion_reason = "passed_products_P2_0p25_gate"
            rows.append(
                _make_products_row(
                    ratio=ratio,
                    method=method,
                    ladder_stage=stage,
                    num_nodes=num_nodes,
                    target_prototypes=int(selected.numel()),
                    shadow_nodes=int(method_shadow),
                    metrics=metrics,
                    status=status,
                    seed=int(args.seed),
                    timing=timing,
                    diagnostics=diagnostics,
                    selected_rows=selected,
                    labels=labels,
                    promotion_status="promoted" if promote else "not_promoted",
                    promotion_reason=promotion_reason,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T25 products recovery ladder.")
    parser.add_argument("--products-root", default="dataset/ogbn_products")
    parser.add_argument("--manifest-dir", default="experiments/preprop/t22_ogbn_products_seed42")
    parser.add_argument("--selected-blocks", default='["X0","X1","X2","X3","Xres1","Xres2","structure","Y1","Y2","Y3"]')
    parser.add_argument("--signature-dir", default="experiments/sft_signatures/ogbn-products/t25_hnr_fdm")
    parser.add_argument("--signature-batch-size", type=int, default=32768)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0005, 0.001, 0.0025, 0.005])
    parser.add_argument("--target-shadow-split", default="0.70,0.30")
    parser.add_argument("--fdm-signature-dim", type=int, default=128, choices=[64, 128])
    parser.add_argument("--fdm-candidate-rho", type=int, default=16)
    parser.add_argument("--fdm-candidate-max", type=int, default=2048)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv", default="experiments/tables/t25_products_recovery_ladder_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t25_products_recovery_notes.md")
    args = parser.parse_args()
    args.target_fraction = float(str(args.target_shadow_split).split(",", maxsplit=1)[0])
    rows = run_products_t25(args)
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(
        args.report,
        [
            "# T25 Products Recovery Ladder",
            "",
            f"- Train mode: `{bool(args.train)}`",
            "- Rows are diagnostic unless the explicit products recovery gates pass.",
            "",
            *markdown_table(rows, ["requested_full_node_ratio", "ladder_stage", "method", "status", "accuracy", "macro_f1", "predicted_classes", "actual_full_node_ratio", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
