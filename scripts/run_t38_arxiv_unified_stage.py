from __future__ import annotations

import argparse
import json
import math
import time
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import write_csv
from shadow_hgc.sft.unified_stt import NUM_CLASSES, NUM_NODES, T38_MAIN_FIELDS, make_t38_row, validate_t38_main_row
from shadow_hgc.train.lazy_sft_memmap import load_arxiv_labels_and_splits, load_manifest_block_store, train_lazy_sft_from_memmap


DEFAULT_ARXIV_RATIOS = [0.0005, 0.001, 0.0025, 0.005, 0.01]
DEFAULT_SELECTED_BLOCKS = [
    "X0",
    "X1_cite_ref",
    "X1_cited_by",
    "X2_cite_ref",
    "X2_cited_by",
    "X3_mix",
    "Xres1_cite_ref",
    "Xres1_cited_by",
    "Xres2_cite_ref",
    "Xres2_cited_by",
    "structure",
    "Y1_cite_ref",
    "Y1_cited_by",
    "Y2_cite_ref",
    "Y2_cited_by",
    "Y3_mix",
]


def _directory_bytes(path: str | Path) -> int:
    total = 0
    for item in Path(path).rglob("*"):
        if item.is_file():
            total += int(item.stat().st_size)
    return total


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _classwise_sqrt_budget(labels: torch.Tensor, train_rows: torch.Tensor, total_budget: int, num_classes: int) -> dict[int, int]:
    train_labels = labels[train_rows].to(torch.long)
    counts = torch.bincount(train_labels.clamp_min(0), minlength=int(num_classes)).to(torch.float64)
    active = [cls for cls in range(int(num_classes)) if int(counts[cls].item()) > 0]
    if not active:
        return {}
    budget = max(len(active), int(total_budget))
    weights = {cls: math.sqrt(float(counts[cls].item())) for cls in active}
    denom = sum(weights.values())
    raw = {cls: budget * weights[cls] / max(denom, 1e-12) for cls in active}
    alloc = {cls: max(1, int(math.floor(raw[cls]))) for cls in active}
    while sum(alloc.values()) < budget:
        cls = max(active, key=lambda item: (raw[item] - math.floor(raw[item]), weights[item], -item))
        alloc[cls] += 1
        raw[cls] = math.floor(raw[cls])
    while sum(alloc.values()) > budget:
        cls = max((item for item in active if alloc[item] > 1), key=lambda item: (alloc[item], weights[item], -item), default=None)
        if cls is None:
            break
        alloc[cls] -= 1
    return alloc


def select_arxiv_unified_rows(
    *,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    manifest_dir: str | Path,
    budget: int,
    seed: int,
    num_classes: int = 40,
) -> torch.Tensor:
    """Teacher-free unified reservoir for arxiv.

    The selector uses train labels and train-node SFT features only. Valid/test
    labels are intentionally not accepted by the function signature.
    """

    store = load_manifest_block_store(manifest_dir).subset(["X0"])
    row_np = train_rows.detach().cpu().numpy().astype(np.int64, copy=False)
    x0 = np.asarray(store.arrays["self"][row_np], dtype=np.float32)
    norms = np.linalg.norm(x0, axis=1)
    train_labels = labels[train_rows].detach().cpu().numpy().astype(np.int64, copy=False)
    alloc = _classwise_sqrt_budget(labels, train_rows, int(budget), int(num_classes))
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for cls in sorted(alloc):
        local = np.flatnonzero(train_labels == int(cls))
        if local.size == 0:
            continue
        order = local[np.argsort(norms[local], kind="mergesort")]
        count = min(int(alloc[cls]), int(order.size))
        if count <= 0:
            continue
        if count == 1:
            chosen = np.asarray([order[int(order.size // 2)]], dtype=np.int64)
        else:
            positions = np.linspace(0, order.size - 1, num=count)
            jitter = rng.uniform(-0.35, 0.35, size=count)
            positions = np.clip(np.rint(positions + jitter).astype(np.int64), 0, order.size - 1)
            chosen = order[positions]
        selected.extend(int(row_np[idx]) for idx in chosen.tolist())
    if len(selected) < int(budget):
        used = set(selected)
        remaining = [int(v) for v in row_np.tolist() if int(v) not in used]
        rng.shuffle(remaining)
        selected.extend(remaining[: int(budget) - len(selected)])
    return torch.tensor(selected[: int(budget)], dtype=torch.long)


def _model_type_for_style(style: str) -> str:
    return "sagn_lite_v4" if str(style) == "sagn_like" else "gamlp_lite_v4"


def run_arxiv_ratio(args: argparse.Namespace, ratio: float) -> dict[str, Any]:
    labels, train_rows, valid_rows, test_rows = load_arxiv_labels_and_splits(args.dataset_root)
    condensed_nodes = max(1, int(round(NUM_NODES["ogbn-arxiv"] * float(ratio))))
    probe = make_t38_row(
        dataset="ogbn-arxiv",
        requested_full_node_ratio=float(ratio),
        condensed_nodes=condensed_nodes,
        num_classes=NUM_CLASSES["ogbn-arxiv"],
        comparison_type="ours_native",
        backend="stt_gated_mixer",
    )
    hidden_dim = int(args.hidden_dim or probe["hidden_dim"])
    epochs = int(args.epochs or probe["epochs"])
    model_type = str(args.model_type or _model_type_for_style(str(probe["student_internal_style"])))
    selected_started = time.perf_counter()
    selected = select_arxiv_unified_rows(
        labels=labels,
        train_rows=train_rows,
        manifest_dir=args.manifest_dir,
        budget=condensed_nodes,
        seed=int(args.seed),
        num_classes=NUM_CLASSES["ogbn-arxiv"],
    )
    selection_time = float(time.perf_counter() - selected_started)
    result = train_lazy_sft_from_memmap(
        manifest_dir=args.manifest_dir,
        labels=labels,
        train_rows=selected,
        valid_rows=valid_rows,
        test_rows=test_rows,
        num_classes=NUM_CLASSES["ogbn-arxiv"],
        device=args.device,
        model_type=model_type,
        hidden_dim=hidden_dim,
        dropout=float(args.dropout),
        num_layers=int(args.num_layers),
        block_dropout=float(args.block_dropout),
        hop_dropout=float(args.hop_dropout),
        label_dropout=float(args.label_dropout),
        selected_blocks=list(args.selected_blocks),
        loss_type=str(args.loss_type),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=epochs,
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        seed=int(args.seed),
    ).summary
    test = result["test"]
    valid = result["valid"]
    cache_bytes = _directory_bytes(args.manifest_dir)
    peak_cpu = int(float(result.get("peak_cpu_ram_gb", 0.0)) * (1024**3))
    peak_gpu = int(float(result.get("peak_gpu_ram_gb", 0.0)) * (1024**3))
    row = make_t38_row(
        dataset="ogbn-arxiv",
        requested_full_node_ratio=float(ratio),
        condensed_nodes=condensed_nodes,
        num_classes=NUM_CLASSES["ogbn-arxiv"],
        backend="stt_gated_mixer",
        comparison_type="ours_native",
        accuracy=float(test["accuracy"]),
        macro_f1=float(test["macro_f1"]),
        valid_acc=float(valid.get("accuracy", 0.0)),
        predicted_classes=int(test.get("predicted_class_count", 0)),
        promotion_status="promoted",
        shared_cache_time_sec="",
        post_cache_time_sec=float(selection_time + result.get("training_time_s", 0.0) + result.get("inference_time_s", 0.0)),
        total_storage_bytes=int(cache_bytes),
        peak_cpu_ram=peak_cpu,
        peak_gpu_ram=peak_gpu,
        edge_cache_id="t38_arxiv_sft_edge_preprop_seed42",
        sft_cache_id="t38_arxiv_sft_table_cache_seed42",
        teacher_cache_id="teacher_disabled",
        unified_reservoir_id=f"t38_arxiv_unified_reservoir_seed{int(args.seed)}",
        cache_reused=True,
        incremental_edge_scans_after_cache_build=0,
        uses_teacher_probs_as_soft_targets=False,
        uses_teacher_probs_as_input_features=False,
        uses_valid_labels_as_input=False,
        uses_test_labels_as_input=False,
        uses_dense_p2=False,
        uses_e_by_d_materialization=False,
        uses_full_edge_index_on_gpu=False,
        notes=(
            f"real T38 arxiv unified teacher-free run; model_type={model_type}; "
            f"selected_blocks={json.dumps(list(args.selected_blocks), sort_keys=True)}; "
            f"selection_time_sec={selection_time:.6f}; epochs={epochs}"
        ),
    )
    check = validate_t38_main_row(row)
    if not check["valid"]:
        row["promotion_status"] = "not_promoted"
        row["failure_reason"] = ",".join(check["forbidden_flags"])
    return row


def write_outputs(args: argparse.Namespace) -> Path:
    rows = [run_arxiv_ratio(args, ratio) for ratio in args.ratios]
    return write_csv(args.csv, rows, T38_MAIN_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="T38 arxiv Shadow-HGC-STT-U full-node ratio curve.")
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_ARXIV_RATIOS)
    parser.add_argument("--manifest-dir", default="experiments/preprop/t22_ogbn_arxiv_seed42")
    parser.add_argument("--dataset-root", default="dataset/ogbn_arxiv")
    parser.add_argument("--selected-blocks", nargs="+", default=DEFAULT_SELECTED_BLOCKS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-type", default="")
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--block-dropout", type=float, default=0.0)
    parser.add_argument("--hop-dropout", type=float, default=0.0)
    parser.add_argument("--label-dropout", type=float, default=0.0)
    parser.add_argument("--loss-type", default="sqrt_weighted_ce")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t38_arxiv_unified_seed42.csv")
    args = parser.parse_args()
    del args.run_long
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
