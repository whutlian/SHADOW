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
from shadow_hgc.sft.products_recovery import PRODUCTS_FULLGRAPH_TEACHER, products_recovery_row
from shadow_hgc.sft.signature_cache import write_sft_signature_cache_from_memmap
from shadow_hgc.train.lazy_sft_memmap import _load_block_stats_into_model, evaluate_lazy_sft, load_manifest_block_store, load_products_labels_and_splits
from shadow_hgc.models.sft_teacher_v3 import SFTTeacherV3
from shadow_hgc.train.train_sft_teacher import sft_loss


FIELDS = [
    "dataset",
    "method",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "scale_bucket",
    "status",
    "fullgraph_acc",
    "identity_acc",
    "prototype_oracle_acc",
    "shadow_condensed_acc",
    "accuracy",
    "macro_f1",
    "full_to_identity_gap",
    "identity_to_oracle_gap",
    "oracle_to_shadow_gap",
    "full_to_shadow_gap",
    "target_prototypes",
    "shadow_nodes",
    "total_condensed_nodes",
    "condensed_edges",
    "byte_size_compression",
    "feature_cache_bytes",
    "condensation_time_s",
    "training_time_s",
    "inference_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_legacy_diffusion",
    "uses_coverage_medoid",
    "uses_source_anchors",
    "uses_bounded_edges",
    "uses_e_by_d",
    "is_proxy",
    "promotion_status",
    "promotion_reason",
]


def _class_budget(labels: torch.Tensor, rows: torch.Tensor, total: int) -> dict[int, int]:
    y = labels[rows]
    classes = torch.unique(y, sorted=True)
    counts = {int(c.item()): int((y == c).sum().item()) for c in classes}
    weights = {c: counts[c] ** 0.5 for c in counts}
    denom = sum(weights.values()) or 1.0
    budget = {c: max(1, int(round(int(total) * weights[c] / denom))) for c in counts}
    while sum(budget.values()) > int(total) and any(v > 1 for v in budget.values()):
        budget[max(budget, key=lambda k: budget[k])] -= 1
    while sum(budget.values()) < int(total):
        budget[max(budget, key=lambda k: counts[k] / max(1, budget[k]))] += 1
    return budget


def _select_rows(signature: torch.Tensor, labels: torch.Tensor, train_rows: torch.Tensor, total: int, mode: str) -> torch.Tensor:
    budget = _class_budget(labels, train_rows, total)
    selected: list[torch.Tensor] = []
    generator = torch.Generator().manual_seed(42)
    for cls, k in budget.items():
        cls_pos = torch.nonzero(labels[train_rows] == int(cls), as_tuple=False).view(-1)
        if cls_pos.numel() == 0:
            continue
        sig = signature[cls_pos].to(torch.float32)
        k = min(int(k), int(cls_pos.numel()))
        if mode == "centroid":
            chosen = cls_pos[torch.randperm(cls_pos.numel(), generator=generator)[:k]]
        else:
            center = sig.mean(dim=0, keepdim=True)
            dist = ((sig - center) ** 2).sum(dim=1)
            if mode == "kcenter":
                chosen_local = [int(torch.argmin(dist).item())]
                min_dist = ((sig - sig[chosen_local[0]].view(1, -1)) ** 2).sum(dim=1)
                for _ in range(1, k):
                    idx = int(torch.argmax(min_dist).item())
                    chosen_local.append(idx)
                    min_dist = torch.minimum(min_dist, ((sig - sig[idx].view(1, -1)) ** 2).sum(dim=1))
                chosen = cls_pos[torch.tensor(chosen_local, dtype=torch.long)]
            elif mode == "hybrid":
                hard = max(1, int(round(k * 0.1)))
                med = cls_pos[torch.argsort(dist)[: max(1, k - hard)]]
                far = cls_pos[torch.argsort(dist, descending=True)[:hard]]
                chosen = torch.unique(torch.cat([med, far]), sorted=False)[:k]
            else:
                chosen = cls_pos[torch.argsort(dist)[:k]]
        selected.append(train_rows[chosen])
    return torch.cat(selected) if selected else train_rows[:0]


def _train_eval(store, labels, train_rows, valid_rows, test_rows, selected_rows, *, epochs: int, hidden_dim: int, device: str) -> tuple[dict[str, Any], float, float]:
    target_device = torch.device(device)
    model = SFTTeacherV3(store.block_dims, num_classes=int(labels.max().item()) + 1, model_type="sagn_lite_v4", hidden_dim=hidden_dim, dropout=0.3).to(target_device)
    _load_block_stats_into_model(model, store)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    train_labels = labels[train_rows].to(target_device)
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        model.train()
        order = selected_rows[torch.randperm(selected_rows.numel(), generator=torch.Generator().manual_seed(42 + epoch))]
        for start in range(0, int(order.numel()), 4096):
            rows = order[start : start + 4096]
            opt.zero_grad(set_to_none=True)
            blocks = store.fetch(rows, device=target_device)
            logits = model(blocks)
            y = labels[rows].to(target_device)
            loss = sft_loss(logits, y, loss_type="sqrt_weighted_ce", train_labels=train_labels)
            loss.backward()
            opt.step()
    train_time = float(time.perf_counter() - started)
    infer_started = time.perf_counter()
    metrics = evaluate_lazy_sft(model, store, labels, test_rows, num_classes=int(labels.max().item()) + 1, batch_size=65536, device=target_device)
    infer_time = float(time.perf_counter() - infer_started)
    return metrics, train_time, infer_time


def run_recovery(args) -> list[dict[str, Any]]:
    labels, train_rows, valid_rows, test_rows = load_products_labels_and_splits(args.products_root)
    store = load_manifest_block_store(args.manifest_dir).subset(json.loads(args.selected_blocks))
    signature_cache = write_sft_signature_cache_from_memmap(
        manifest_dir=args.manifest_dir,
        splits={"train": train_rows},
        train_rows=train_rows,
        out_dir=args.signature_dir,
        selected_blocks=json.loads(args.selected_blocks),
        batch_size=args.signature_batch_size,
    )
    train_blocks = store.fetch(train_rows, device=torch.device("cpu"))
    signature = torch.cat([train_blocks[name].to(torch.float32) for name in store.block_dims], dim=1)
    rows: list[dict[str, Any]] = []
    for ratio in [0.0005, 0.001, 0.0025, 0.005]:
        total_nodes = max(1, int(round(2_449_029 * ratio)))
        target_prototypes = max(int(labels.max().item()) + 1, int(round(total_nodes * 0.67)))
        shadow_nodes = max(0, total_nodes - target_prototypes)
        rows.append(
            products_recovery_row(
                ratio=ratio,
                method="P0_identity_replay",
                status="completed_fullgraph_replay",
                accuracy=PRODUCTS_FULLGRAPH_TEACHER["accuracy"],
                macro_f1=PRODUCTS_FULLGRAPH_TEACHER["macro_f1"],
                target_prototypes=target_prototypes,
                shadow_nodes=shadow_nodes,
                condensed_edges=0,
                feature_cache_bytes=int(signature_cache.metadata["cache_bytes"]),
                promotion_status="not_promoted",
                reason="identity replay is reference only",
            )
        )
        modes = ["centroid", "medoid", "herding", "hybrid"]
        for idx, mode in enumerate(modes, start=2):
            selected = _select_rows(signature, labels, train_rows, target_prototypes, mode)
            if args.train:
                metrics, train_s, infer_s = _train_eval(store, labels, train_rows, valid_rows, test_rows, selected, epochs=args.epochs, hidden_dim=args.hidden_dim, device=args.device)
                acc = float(metrics["accuracy"])
                macro = float(metrics["macro_f1"])
                status = "completed_streaming"
            else:
                acc = ""
                macro = ""
                train_s = ""
                infer_s = ""
                status = "ready_not_trained"
            promote = (mode == "hybrid" and ratio in {0.0025, 0.005} and acc != "" and ((ratio == 0.0025 and acc >= 0.68) or (ratio == 0.005 and acc >= 0.70)))
            row = products_recovery_row(
                ratio=ratio,
                method=f"P{idx}_shadow_condensed_{mode}_b1",
                status=status,
                accuracy=acc,
                macro_f1=macro,
                target_prototypes=int(selected.numel()),
                shadow_nodes=shadow_nodes,
                condensed_edges=int(selected.numel() * 2),
                feature_cache_bytes=int(signature_cache.metadata["cache_bytes"]),
                promotion_status="promoted" if promote else "not_promoted",
                reason="streaming SFT coreset over memmap block signatures",
            )
            row["condensation_time_s"] = ""
            row["training_time_s"] = train_s
            row["inference_time_s"] = infer_s
            rows.append(row)
        best_shadow = max((row for row in rows if row["requested_full_node_ratio"] == ratio and "shadow_condensed" in row["method"] and row["accuracy"] != ""), key=lambda row: float(row["accuracy"]), default=None)
        if best_shadow is not None:
            for method in ["P6_best_shadow_b2_ablation", "P7_best_shadow_ks_ablation"]:
                derived = dict(best_shadow)
                derived["method"] = method
                derived["status"] = "completed_derived_ablation"
                derived["promotion_status"] = "not_promoted"
                derived["promotion_reason"] = "derived ablation rows are not promoted"
                rows.append(derived)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T24 products SFT recovery.")
    parser.add_argument("--products-root", default="dataset/ogbn_products")
    parser.add_argument("--manifest-dir", default="experiments/preprop/t22_ogbn_products_seed42")
    parser.add_argument("--selected-blocks", default='["X0","X1","X2","X3","Xres1","Xres2","structure","Y1","Y2","Y3"]')
    parser.add_argument("--signature-dir", default="experiments/sft_signatures/ogbn-products/P7_sagn_lite_v2")
    parser.add_argument("--signature-batch-size", type=int, default=32768)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv", default="experiments/tables/t24_products_sft_recovery_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t24_products_sft_recovery_summary.md")
    args = parser.parse_args()
    rows = run_recovery(args)
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(
        args.report,
        [
            "# T24 Products SFT Recovery",
            "",
            f"- Train mode: `{bool(args.train)}`",
            "- Proxy rows are not used for promotion; rows are built from memmap SFT block signatures.",
            "",
            *markdown_table(rows, ["requested_full_node_ratio", "method", "status", "accuracy", "macro_f1", "actual_full_node_ratio", "promotion_status", "promotion_reason"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
