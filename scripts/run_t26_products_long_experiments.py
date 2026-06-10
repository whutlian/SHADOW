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
from shadow_hgc.models.sft_teacher_v3 import SFTTeacherV3
from shadow_hgc.sft.coreset import select_classwise_coreset_rows
from shadow_hgc.sft.products_recovery_t26 import mixed_class_budget, nearest_prototype_oracle
from shadow_hgc.sft.signature_cache import write_or_load_sft_signature_cache_from_memmap
from shadow_hgc.sft.t26_contract import T26_PRODUCTS_METHODS
from shadow_hgc.sft.uca import coverage_gap_metrics
from shadow_hgc.train.lazy_sft_memmap import _load_block_stats_into_model, evaluate_lazy_sft, load_manifest_block_store, load_products_labels_and_splits
from shadow_hgc.train.train_sft_teacher import sft_loss


PRODUCTS_NODES = 2_449_029

FIELDS = [
    "dataset",
    "method",
    "seed",
    "requested_full_node_ratio",
    "target_prototypes",
    "shadow_nodes",
    "total_condensed_edges",
    "accuracy",
    "macro_f1",
    "predicted_class_count",
    "status",
    "p0a_alltrain_acc",
    "p0b_self_fit_acc",
    "p0d_prototype_oracle_acc",
    "p0d_centroid_oracle_acc",
    "training_time",
    "inference_time",
    "condensation_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "cache_bytes",
    "train_class_counts_json",
    "selected_class_counts_json",
    "predicted_class_counts_json",
    "coverage_gap_l1",
    "coverage_gap_l2",
    "uca_num_domains",
    "uca_domain_seed",
    "uca_scope",
    "trainer_recipe",
    "trainer_balanced_batches",
    "trainer_label_smoothing",
    "trainer_mixup_alpha",
    "notes",
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


def _budgeted_hybrid(signature: torch.Tensor, labels: torch.Tensor, train_rows: torch.Tensor, total: int, *, seed: int) -> torch.Tensor:
    return select_classwise_coreset_rows(signature, labels, train_rows, total, mode="hybrid", seed=int(seed))


def _budgeted_random(signature: torch.Tensor, labels: torch.Tensor, train_rows: torch.Tensor, total: int, *, seed: int) -> torch.Tensor:
    return select_classwise_coreset_rows(signature, labels, train_rows, total, mode="random", seed=int(seed))


def _budgeted_mode(signature: torch.Tensor, labels: torch.Tensor, train_rows: torch.Tensor, total: int, *, mode: str, seed: int) -> torch.Tensor:
    return select_classwise_coreset_rows(signature, labels, train_rows, total, mode=mode, seed=int(seed))


def _class_counts_json(labels: torch.Tensor, rows: torch.Tensor, *, num_classes: int) -> str:
    if rows.numel() == 0:
        return "{}"
    y = labels[rows.to(torch.long)].to(torch.long).cpu()
    hist = torch.bincount(y.clamp_min(0), minlength=int(num_classes))
    return json.dumps({str(idx): int(value) for idx, value in enumerate(hist.tolist()) if int(value) > 0}, sort_keys=True)


def _merge_selected(primary: torch.Tensor, fallback: torch.Tensor, total: int) -> torch.Tensor:
    seen: set[int] = set()
    merged: list[int] = []
    for tensor in (primary.to(torch.long).cpu(), fallback.to(torch.long).cpu()):
        for value in tensor.tolist():
            row = int(value)
            if row in seen:
                continue
            seen.add(row)
            merged.append(row)
            if len(merged) >= int(total):
                return torch.tensor(merged, dtype=torch.long)
    return torch.tensor(merged, dtype=torch.long)


def _balanced_order(selected_rows: torch.Tensor, labels: torch.Tensor, *, seed: int) -> torch.Tensor:
    selected_rows = selected_rows.to(torch.long).cpu()
    if selected_rows.numel() == 0:
        return selected_rows
    generator = torch.Generator().manual_seed(int(seed))
    y = labels[selected_rows].to(torch.long).cpu()
    classes = torch.unique(y, sorted=True)
    groups = [selected_rows[y == cls] for cls in classes if torch.any(y == cls)]
    pointers = [0 for _ in groups]
    shuffled = [group[torch.randperm(group.numel(), generator=generator)] for group in groups]
    out: list[int] = []
    while len(out) < int(selected_rows.numel()):
        for group_idx, group in enumerate(shuffled):
            if len(out) >= int(selected_rows.numel()):
                break
            if group.numel() == 0:
                continue
            pos = pointers[group_idx] % int(group.numel())
            out.append(int(group[pos].item()))
            pointers[group_idx] += 1
    return torch.tensor(out, dtype=torch.long)


def _train_eval_variant(
    store,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    test_rows: torch.Tensor,
    selected_rows: torch.Tensor,
    *,
    epochs: int,
    hidden_dim: int,
    device: str,
    balanced_batches: bool = False,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
) -> tuple[dict[str, Any], float, float]:
    target_device = torch.device(device)
    torch.manual_seed(42)
    model = SFTTeacherV3(store.block_dims, num_classes=int(labels.max().item()) + 1, model_type="sagn_lite_v4", hidden_dim=hidden_dim, dropout=0.3).to(target_device)
    _load_block_stats_into_model(model, store)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    train_labels = labels[train_rows].to(target_device)
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        model.train()
        if bool(balanced_batches):
            order = _balanced_order(selected_rows, labels, seed=42 + epoch)
        else:
            order = selected_rows[torch.randperm(selected_rows.numel(), generator=torch.Generator().manual_seed(42 + epoch))]
        for start in range(0, int(order.numel()), 4096):
            rows = order[start : start + 4096]
            opt.zero_grad(set_to_none=True)
            blocks = store.fetch(rows, device=target_device)
            y = labels[rows].to(target_device)
            if float(mixup_alpha) > 0.0 and int(rows.numel()) > 1:
                rng = np.random.default_rng(10_000 + int(epoch) * 1000 + int(start))
                lam = float(rng.beta(float(mixup_alpha), float(mixup_alpha)))
                perm = torch.randperm(int(rows.numel()), device=target_device)
                mixed_blocks = {name: lam * value + (1.0 - lam) * value[perm] for name, value in blocks.items()}
                logits = model(mixed_blocks)
                loss = lam * sft_loss(logits, y, loss_type="sqrt_weighted_ce", train_labels=train_labels, label_smoothing=float(label_smoothing))
                loss = loss + (1.0 - lam) * sft_loss(logits, y[perm], loss_type="sqrt_weighted_ce", train_labels=train_labels, label_smoothing=float(label_smoothing))
            else:
                logits = model(blocks)
                loss = sft_loss(logits, y, loss_type="sqrt_weighted_ce", train_labels=train_labels, label_smoothing=float(label_smoothing))
            loss.backward()
            opt.step()
    train_time = float(time.perf_counter() - started)
    infer_started = time.perf_counter()
    metrics = evaluate_lazy_sft(model, store, labels, test_rows, num_classes=int(labels.max().item()) + 1, batch_size=65536, device=target_device)
    infer_time = float(time.perf_counter() - infer_started)
    return metrics, train_time, infer_time


def _select_train_target_uca_light(
    signature: torch.Tensor,
    train_rows: torch.Tensor,
    *,
    budget: int,
    num_domains: int,
    seed: int,
    chunk_size: int = 32768,
    compact_dim: int = 32,
) -> tuple[torch.Tensor, dict[str, Any]]:
    signature = signature.to(torch.float32).cpu()
    train_rows = train_rows.to(torch.long).cpu()
    n = int(signature.shape[0])
    if n == 0:
        return train_rows[:0], {
            "coverage_gap_l1": 0.0,
            "coverage_gap_l2": 0.0,
            "selected_coverage_gap_l1": 0.0,
            "selected_coverage_gap_l2": 0.0,
            "uca_num_domains": 0,
            "uca_domain_seed": int(seed),
            "uca_uses_valid_test_labels": False,
        }
    k = max(1, min(int(num_domains), n))
    budget = max(1, min(int(budget), n))
    compact = signature[:, : min(int(compact_dim), int(signature.shape[1]))].contiguous()
    generator = torch.Generator().manual_seed(int(seed))
    center_idx = torch.randperm(n, generator=generator)[:k]
    centers = compact[center_idx].contiguous()
    domains = torch.empty(n, dtype=torch.long)
    nearest_dist = torch.empty(n, dtype=torch.float32)
    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        dist = torch.cdist(compact[start:end], centers, p=2)
        values, indices = torch.min(dist, dim=1)
        domains[start:end] = indices.to(torch.long)
        nearest_dist[start:end] = values.to(torch.float32)
    all_hist = torch.bincount(domains, minlength=k)
    weights = all_hist.to(torch.float64) / all_hist.sum().clamp_min(1).to(torch.float64)
    raw_quota = weights * int(budget)
    quota = torch.floor(raw_quota).to(torch.long)
    remainder = int(budget) - int(quota.sum().item())
    if remainder > 0:
        order = torch.argsort(raw_quota - quota.to(raw_quota.dtype), descending=True)
        quota[order[:remainder]] += 1
    selected_pos: list[int] = []
    for domain in range(k):
        take = int(quota[domain].item())
        if take <= 0:
            continue
        candidates = torch.nonzero(domains == domain, as_tuple=False).view(-1)
        if candidates.numel() == 0:
            continue
        local_order = candidates[torch.argsort(nearest_dist[candidates])[:take]]
        selected_pos.extend(int(value) for value in local_order.tolist())
    if len(selected_pos) < budget:
        seen = set(selected_pos)
        for value in torch.argsort(nearest_dist).tolist():
            row = int(value)
            if row in seen:
                continue
            selected_pos.append(row)
            seen.add(row)
            if len(selected_pos) >= budget:
                break
    selected_pos_tensor = torch.tensor(selected_pos[:budget], dtype=torch.long)
    selected_hist = torch.bincount(domains[selected_pos_tensor], minlength=k) if selected_pos_tensor.numel() else torch.zeros(k, dtype=torch.long)
    selected_gap = coverage_gap_metrics(all_hist, selected_hist)
    stats: dict[str, Any] = {
        "coverage_gap_l1": 0.0,
        "coverage_gap_l2": 0.0,
        "domains_total": int(k),
        "domains_without_train_support": 0,
        "domains_without_unlabeled_support": int((all_hist <= 0).sum().item()),
        "selected_coverage_gap_l1": selected_gap["coverage_gap_l1"],
        "selected_coverage_gap_l2": selected_gap["coverage_gap_l2"],
        "domain_hist_all": [int(value) for value in all_hist.tolist()],
        "domain_hist_selected": [int(value) for value in selected_hist.tolist()],
        "uca_num_domains": int(k),
        "uca_domain_seed": int(seed),
        "uca_uses_valid_test_labels": False,
    }
    return train_rows[selected_pos_tensor], stats


def _select_method_rows(
    method: str,
    signature: torch.Tensor,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    total: int,
    *,
    seed: int,
    uca_domains: int,
    uca_cache: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any], str]:
    if method == "products_cb_random":
        return _budgeted_mode(signature, labels, train_rows, total, mode="random", seed=seed), {}, "class-wise random coreset"
    if method == "products_cb_kcenter":
        return _budgeted_mode(signature, labels, train_rows, total, mode="kcenter", seed=seed), {}, "class-wise k-center coreset"
    if method == "products_cb_herding":
        return _budgeted_mode(signature, labels, train_rows, total, mode="medoid", seed=seed), {}, "class-wise medoid/herding-style coreset"
    if method == "products_cb_hybrid":
        return _budgeted_mode(signature, labels, train_rows, total, mode="hybrid", seed=seed), {}, "class-wise hybrid medoid/far coreset"
    if uca_cache is not None and "uca_selection" in uca_cache:
        uca_selected, stats = uca_cache["uca_selection"]
    else:
        uca_selected, stats = _select_train_target_uca_light(
            signature,
            train_rows=train_rows,
            budget=int(total),
            num_domains=int(uca_domains),
            seed=int(seed),
        )
        if uca_cache is not None:
            uca_cache["uca_selection"] = (uca_selected, stats)
    if method == "products_uca_kmeans_labeled_nearest":
        return uca_selected, stats, "train-target UCA domains with nearest labeled rows"
    hybrid = _budgeted_mode(signature, labels, train_rows, total, mode="hybrid", seed=seed)
    if method in {"products_uca_hybrid", "products_uca_hybrid_mixup", "products_uca_hybrid_balanced_trainer"}:
        return _merge_selected(uca_selected, hybrid, total), stats, "train-target UCA primary selection with hybrid class-wise fill"
    raise ValueError(f"unsupported products method: {method}")


def run_products_long(args: argparse.Namespace) -> list[dict[str, Any]]:
    labels, train_rows, valid_rows, test_rows = load_products_labels_and_splits(args.products_root)
    store = load_manifest_block_store(args.manifest_dir).subset(json.loads(args.selected_blocks))
    signature_cache = write_or_load_sft_signature_cache_from_memmap(
        manifest_dir=args.manifest_dir,
        splits={"train": train_rows},
        train_rows=train_rows,
        out_dir=args.signature_dir,
        selected_blocks=json.loads(args.selected_blocks),
        batch_size=int(args.signature_batch_size),
    )
    signature = _load_train_signature(args.signature_dir, signature_cache.metadata)
    rows: list[dict[str, Any]] = []
    num_classes = int(labels.max().item()) + 1
    common = {
        "dataset": "ogbn-products",
        "seed": int(args.seed),
        "shadow_nodes": 0,
        "peak_cpu_ram": current_cpu_ram_bytes() / (1024**3),
        "peak_gpu_ram": current_gpu_ram_bytes() / (1024**3),
        "cache_bytes": int(signature_cache.metadata["cache_bytes"]),
        "train_class_counts_json": _class_counts_json(labels, train_rows, num_classes=num_classes),
    }

    if bool(args.run_p0a):
        started = time.perf_counter()
        metrics, train_s, infer_s = _train_eval(
            store,
            labels,
            train_rows,
            valid_rows,
            test_rows,
            train_rows,
            epochs=int(args.p0a_epochs),
            hidden_dim=int(args.hidden_dim),
            device=args.device,
        )
        rows.append(
            {
                **common,
                "method": "P0a_alltrain_condensed_trainer_parity",
                "requested_full_node_ratio": float(train_rows.numel()) / PRODUCTS_NODES,
                "target_prototypes": int(train_rows.numel()),
                "total_condensed_edges": int(train_rows.numel()),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "predicted_class_count": int(metrics["predicted_class_count"]),
                "selected_class_counts_json": _class_counts_json(labels, train_rows, num_classes=num_classes),
                "predicted_class_counts_json": metrics.get("predicted_class_counts_json", ""),
                "status": "completed_long",
                "p0a_alltrain_acc": float(metrics["accuracy"]),
                "training_time": train_s,
                "inference_time": infer_s,
                "condensation_time": time.perf_counter() - started,
                "notes": f"all-train condensed trainer parity, epochs={int(args.p0a_epochs)}",
            }
        )

    for ratio in [float(value) for value in args.ratios]:
        total = max(num_classes, int(round(PRODUCTS_NODES * ratio)))
        cond_started = time.perf_counter()
        selected = _budgeted_hybrid(signature, labels, train_rows, total, seed=int(args.seed))
        condensation_time = time.perf_counter() - cond_started
        if bool(args.run_p0b):
            metrics, train_s, infer_s = _train_eval(
                store,
                labels,
                train_rows,
                selected,
                selected,
                selected,
                epochs=int(args.p0b_epochs),
                hidden_dim=int(args.hidden_dim),
                device=args.device,
            )
            rows.append(
                {
                    **common,
                    "method": "P0b_selected_prototype_self_fit",
                    "requested_full_node_ratio": ratio,
                    "target_prototypes": int(selected.numel()),
                    "total_condensed_edges": int(selected.numel()),
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "predicted_class_count": int(metrics["predicted_class_count"]),
                    "selected_class_counts_json": _class_counts_json(labels, selected, num_classes=num_classes),
                    "predicted_class_counts_json": metrics.get("predicted_class_counts_json", ""),
                    "status": "completed_long",
                    "p0b_self_fit_acc": float(metrics["accuracy"]),
                    "training_time": train_s,
                    "inference_time": infer_s,
                    "condensation_time": condensation_time,
                    "notes": f"selected prototype self-fit, epochs={int(args.p0b_epochs)}",
                }
            )
        if bool(args.run_p0c):
            random_selected = _budgeted_random(signature, labels, train_rows, total, seed=int(args.seed))
            metrics, train_s, infer_s = _train_eval(
                store,
                labels,
                train_rows,
                valid_rows,
                test_rows,
                random_selected,
                epochs=int(args.p0c_epochs),
                hidden_dim=int(args.hidden_dim),
                device=args.device,
            )
            rows.append(
                {
                    **common,
                    "method": "P0c_same_budget_random_subset",
                    "requested_full_node_ratio": ratio,
                    "target_prototypes": int(random_selected.numel()),
                    "total_condensed_edges": int(random_selected.numel()),
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "predicted_class_count": int(metrics["predicted_class_count"]),
                    "selected_class_counts_json": _class_counts_json(labels, random_selected, num_classes=num_classes),
                    "predicted_class_counts_json": metrics.get("predicted_class_counts_json", ""),
                    "status": "completed_long",
                    "training_time": train_s,
                    "inference_time": infer_s,
                    "condensation_time": condensation_time,
                    "notes": f"same-budget random subset, epochs={int(args.p0c_epochs)}",
                }
            )
        if bool(args.run_p0d):
            selected_pos = torch.searchsorted(train_rows, selected)
            oracle = nearest_prototype_oracle(
                signature,
                labels[train_rows],
                selected_pos,
                signature,
                labels[train_rows],
                metric="euclidean",
            )
            rows.append(
                {
                    **common,
                    "method": "P0d_nearest_prototype_oracle",
                    "requested_full_node_ratio": ratio,
                    "target_prototypes": int(selected.numel()),
                    "total_condensed_edges": int(selected.numel()),
                    "accuracy": oracle["prototype_oracle_acc"],
                    "macro_f1": "",
                    "predicted_class_count": "",
                    "status": "completed_long_train_signature_oracle",
                    "p0d_prototype_oracle_acc": oracle["prototype_oracle_acc"],
                    "p0d_centroid_oracle_acc": oracle["centroid_oracle_acc"],
                    "condensation_time": condensation_time,
                    "notes": "nearest prototype oracle on train SFT signatures; no valid/test labels used for selection",
                }
            )
        if bool(args.run_methods):
            uca_cache: dict[str, Any] = {}
            for method in T26_PRODUCTS_METHODS:
                selected_method, selection_stats, selection_note = _select_method_rows(
                    method,
                    signature,
                    labels,
                    train_rows,
                    total,
                    seed=int(args.seed),
                    uca_domains=int(args.uca_domains),
                    uca_cache=uca_cache,
                )
                balanced = method == "products_uca_hybrid_balanced_trainer"
                smoothing = 0.05 if balanced else 0.0
                mixup = 0.4 if method == "products_uca_hybrid_mixup" else 0.0
                metrics, train_s, infer_s = _train_eval_variant(
                    store,
                    labels,
                    train_rows,
                    test_rows,
                    selected_method,
                    epochs=int(args.method_epochs),
                    hidden_dim=int(args.hidden_dim),
                    device=args.device,
                    balanced_batches=balanced,
                    label_smoothing=smoothing,
                    mixup_alpha=mixup,
                )
                rows.append(
                    {
                        **common,
                        "method": method,
                        "requested_full_node_ratio": ratio,
                        "target_prototypes": int(selected_method.numel()),
                        "total_condensed_edges": int(selected_method.numel()),
                        "accuracy": float(metrics["accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                        "predicted_class_count": int(metrics["predicted_class_count"]),
                        "status": "completed_long",
                        "training_time": train_s,
                        "inference_time": infer_s,
                        "condensation_time": condensation_time,
                        "selected_class_counts_json": _class_counts_json(labels, selected_method, num_classes=num_classes),
                        "predicted_class_counts_json": metrics.get("predicted_class_counts_json", ""),
                        "coverage_gap_l1": selection_stats.get("selected_coverage_gap_l1", selection_stats.get("coverage_gap_l1", "")),
                        "coverage_gap_l2": selection_stats.get("selected_coverage_gap_l2", selection_stats.get("coverage_gap_l2", "")),
                        "uca_num_domains": selection_stats.get("uca_num_domains", ""),
                        "uca_domain_seed": selection_stats.get("uca_domain_seed", ""),
                        "uca_scope": "train_target_signature" if method.startswith("products_uca") else "",
                        "trainer_recipe": "balanced_sqrt_weighted_ce" if balanced else ("mixup_sqrt_weighted_ce" if mixup > 0.0 else "standard_sqrt_weighted_ce"),
                        "trainer_balanced_batches": balanced,
                        "trainer_label_smoothing": smoothing,
                        "trainer_mixup_alpha": mixup,
                        "notes": f"{selection_note}; epochs={int(args.method_epochs)}; no valid/test labels used for selection",
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real T26 products long diagnostics.")
    parser.add_argument("--products-root", default="dataset/ogbn_products")
    parser.add_argument("--manifest-dir", default="experiments/preprop/t22_ogbn_products_seed42")
    parser.add_argument("--selected-blocks", default='["X0","X1","X2","X3","Xres1","Xres2","structure","Y1","Y2","Y3"]')
    parser.add_argument("--signature-dir", default="experiments/sft_signatures/ogbn-products/t26_long")
    parser.add_argument("--signature-batch-size", type=int, default=32768)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0025, 0.005])
    parser.add_argument("--run-p0a", action="store_true", default=True)
    parser.add_argument("--run-p0b", action="store_true", default=True)
    parser.add_argument("--run-p0c", action="store_true", default=True)
    parser.add_argument("--run-p0d", action="store_true")
    parser.add_argument("--run-methods", action="store_true", default=True)
    parser.add_argument("--uca-domains", type=int, default=256)
    parser.add_argument("--p0a-epochs", type=int, default=20)
    parser.add_argument("--p0b-epochs", type=int, default=80)
    parser.add_argument("--p0c-epochs", type=int, default=20)
    parser.add_argument("--method-epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv", default="experiments/tables/t26_products_long_experiments_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t26_products_long_experiments.md")
    args = parser.parse_args()
    rows = run_products_long(args)
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(
        args.report,
        [
            "# T26 Products Long Experiments",
            "",
            f"- Device: `{args.device}`",
            f"- P0a epochs: `{int(args.p0a_epochs)}`",
            f"- P0b epochs: `{int(args.p0b_epochs)}`",
            f"- P0c epochs: `{int(args.p0c_epochs)}`",
            f"- Method epochs: `{int(args.method_epochs)}`",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "status", "accuracy", "macro_f1", "predicted_class_count", "training_time", "inference_time", "notes"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
