from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.data.reddit_stream import load_reddit_raw_memmap_labels_and_splits
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.sft.t31_contract import (
    REDDIT_TTC_001_GATE,
    REDDIT_TTC_005_GATE,
    T31_REQUIRED_FIELDS,
    apply_t31_promotion_guard,
    make_t31_row,
    ratio_budget,
)
from shadow_hgc.sft.teacher_transport import (
    build_ttc_condensed_table,
    teacher_probability_diagnostics,
    train_soft_label_condensed_student,
)
from shadow_hgc.train.lazy_sft_memmap import (
    _build_lazy_model,
    _iter_batches,
    _load_block_stats_into_model,
    evaluate_lazy_sft,
    load_manifest_block_store,
)
from shadow_hgc.train.train_sft_teacher import sft_loss


DEFAULT_BLOCKS = ["X0", "X1", "X2", "X3", "Xres1", "Y1", "Y2", "Y3", "structure"]


def build_reddit_ttc_server_command() -> str:
    return (
        "python scripts/run_t31_reddit_ttc.py --device cuda --ratios 0.0005 0.001 0.0025 0.005 0.01 "
        "--ttc-modes ttc_confidence_balanced ttc_uncertainty_balanced ttc_margin_boundary "
        "ttc_coverage_plus_boundary ttc_coverage_plus_boundary_plus_mixup --candidate-nodes all "
        "--temperatures 2 4 --lambda-hard 0.25 0.5 1.0 --lambda-prior 0.02 0.05 0.10 "
        "--students table_head_mlp --hidden-dims 128 256 512 --epochs 60 120 200 --seed 42 --run-long"
    )


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _selected_blocks(args: argparse.Namespace) -> list[str]:
    value = _arg(args, "selected_blocks", json.dumps(DEFAULT_BLOCKS))
    if isinstance(value, str):
        return [str(v) for v in json.loads(value)]
    return [str(v) for v in value]


def _torch_device(device: str) -> torch.device:
    return torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")


def _metrics_summary(summary: dict[str, Any]) -> dict[str, Any]:
    test = summary.get("test", {})
    valid = summary.get("valid", {})
    return {
        "teacher_accuracy": test.get("accuracy", ""),
        "teacher_macro_f1": test.get("macro_f1", ""),
        "teacher_valid_acc": valid.get("accuracy", ""),
        "teacher_predicted_classes": test.get("predicted_class_count", ""),
    }


def _train_teacher_and_cache(args: argparse.Namespace, cache_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    labels, train_rows, valid_rows, test_rows = load_reddit_raw_memmap_labels_and_splits(_arg(args, "memmap_root", "dataset/Reddit/processed/raw_memmap"))
    device = _torch_device(str(_arg(args, "device", "cuda")))
    store = load_manifest_block_store(_arg(args, "manifest_dir", "experiments/preprop/t24_reddit_streaming_seed42")).subset(_selected_blocks(args))
    model = _build_lazy_model(
        store.block_dims,
        num_classes=int(labels.max().item()) + 1,
        model_type=str(_arg(args, "teacher_model_type", "sagn_lite_v4")),
        hidden_dim=int(_arg(args, "teacher_hidden_dim", 128)),
        dropout=float(_arg(args, "teacher_dropout", 0.3)),
        num_layers=int(_arg(args, "teacher_num_layers", 2)),
        block_dropout=float(_arg(args, "teacher_block_dropout", 0.0)),
        hop_dropout=float(_arg(args, "teacher_hop_dropout", 0.0)),
        label_dropout=float(_arg(args, "teacher_label_dropout", 0.05)),
        attention_heads=int(_arg(args, "teacher_attention_heads", 1)),
        activation=str(_arg(args, "teacher_activation", "relu")),
        norm=str(_arg(args, "teacher_norm", "none")),
    ).to(device)
    _load_block_stats_into_model(model, store)
    torch.manual_seed(int(_arg(args, "seed", 42)))
    labels = labels.to(torch.long).cpu()
    train_rows = train_rows.to(torch.long).cpu()
    valid_rows = valid_rows.to(torch.long).cpu()
    test_rows = test_rows.to(torch.long).cpu()
    train_labels = labels[train_rows].to(torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=float(_arg(args, "teacher_lr", 0.003)), weight_decay=float(_arg(args, "teacher_weight_decay", 1e-4)))
    best_score = -1.0
    best_valid: dict[str, Any] | None = None
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    batch_size = int(_arg(args, "teacher_batch_size", 16384))
    eval_batch_size = int(_arg(args, "teacher_eval_batch_size", 65536))
    for epoch in range(int(_arg(args, "teacher_epochs", 30))):
        model.train()
        for batch_rows in _iter_batches(train_rows, batch_size=batch_size, shuffle=True, seed=int(_arg(args, "seed", 42)) + epoch):
            blocks = store.fetch(batch_rows, device=device)
            y = labels[batch_rows].to(device=device, dtype=torch.long)
            logits = model(blocks)
            loss = sft_loss(
                logits,
                y,
                loss_type=str(_arg(args, "teacher_loss_type", "sqrt_weighted_ce")),
                train_labels=train_labels.to(device),
                label_smoothing=float(_arg(args, "teacher_label_smoothing", 0.0)),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        valid = evaluate_lazy_sft(model, store, labels, valid_rows, num_classes=int(labels.max().item()) + 1, batch_size=eval_batch_size, device=device)
        score = float(valid["accuracy"]) + 0.05 * float(valid["macro_f1"])
        if score > best_score:
            best_score = score
            best_valid = dict(valid)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    test = evaluate_lazy_sft(model, store, labels, test_rows, num_classes=int(labels.max().item()) + 1, batch_size=eval_batch_size, device=device)
    logits_path = cache_dir / "teacher_logits.npy"
    probs_path = cache_dir / "teacher_probs.npy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logits_mm = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(store.num_rows, int(labels.max().item()) + 1))
    probs_mm = np.lib.format.open_memmap(probs_path, mode="w+", dtype=np.float32, shape=(store.num_rows, int(labels.max().item()) + 1))
    model.eval()
    with torch.no_grad():
        for batch_rows in _iter_batches(torch.arange(store.num_rows), batch_size=eval_batch_size, shuffle=False, seed=0):
            blocks = store.fetch(batch_rows, device=device)
            logits = model(blocks).detach().cpu().to(torch.float32)
            probs = torch.softmax(logits, dim=1)
            rows_np = batch_rows.numpy()
            logits_mm[rows_np] = logits.numpy()
            probs_mm[rows_np] = probs.numpy()
    logits_mm.flush()
    probs_mm.flush()
    probs = torch.from_numpy(np.asarray(np.load(probs_path, mmap_mode="r"), dtype=np.float32))
    diag = teacher_probability_diagnostics(probs)
    metadata = {
        "dataset": "Reddit",
        "teacher_method": str(_arg(args, "teacher_model_type", "sagn_lite_v4")),
        "teacher_seed_list": [int(_arg(args, "seed", 42))],
        **_metrics_summary({"test": test, "valid": best_valid or {}}),
        **diag,
        "teacher_temperature": 1.0,
        "logits_path": str(logits_path),
        "probs_path": str(probs_path),
        "teacher_cache_bytes": int(logits_path.stat().st_size + probs_path.stat().st_size),
        "training_time": float(time.perf_counter() - started),
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "uses_valid_labels_for_hyperparam_selection": True,
        "uses_teacher_logits": True,
        "uses_kd": False,
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def load_or_train_teacher_cache(args: argparse.Namespace) -> dict[str, Any] | None:
    cache_dir = Path(_arg(args, "teacher_cache_dir", "experiments/cache/t31_reddit_ttc_teacher_seed42"))
    metadata_path = cache_dir / "metadata.json"
    probs_path = cache_dir / "teacher_probs.npy"
    if metadata_path.exists() and probs_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    if not bool(_arg(args, "run_long", False)):
        return None
    return _train_teacher_and_cache(args, cache_dir)


def _concat_features(args: argparse.Namespace) -> torch.Tensor:
    store = load_manifest_block_store(_arg(args, "manifest_dir", "experiments/preprop/t24_reddit_streaming_seed42")).subset(_selected_blocks(args))
    arrays = [np.asarray(store.arrays[key], dtype=np.float32) for key in store.arrays]
    return torch.from_numpy(np.concatenate(arrays, axis=1).astype(np.float32, copy=False))


def _split_concat_blocks(z: torch.Tensor, block_dims: dict[str, int]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    start = 0
    for key, dim in block_dims.items():
        stop = start + int(dim)
        out[key] = z[:, start:stop].contiguous()
        start = stop
    if start != int(z.shape[1]):
        raise ValueError(f"concatenated feature dim {z.shape[1]} does not match block dims {start}")
    return out


def _train_blockwise_soft_student(
    *,
    args: argparse.Namespace,
    table,
    labels: torch.Tensor,
    valid_idx: torch.Tensor,
    test_idx: torch.Tensor,
    hidden_dim: int,
    epochs: int,
    temperature: float,
    lambda_hard: float,
    lambda_prior: float,
    target_prior: torch.Tensor,
) -> dict[str, Any]:
    device = _torch_device(str(_arg(args, "device", "cuda")))
    store = load_manifest_block_store(_arg(args, "manifest_dir", "experiments/preprop/t24_reddit_streaming_seed42")).subset(_selected_blocks(args))
    model = _build_lazy_model(
        store.block_dims,
        num_classes=int(labels.max().item()) + 1,
        model_type=str(_arg(args, "student_model_type", "sagn_lite_v4")),
        hidden_dim=int(hidden_dim),
        dropout=float(_arg(args, "dropout", 0.1)),
        num_layers=2,
        block_dropout=0.0,
        hop_dropout=0.0,
        label_dropout=0.05,
        attention_heads=1,
        activation="relu",
        norm="none",
    ).to(device)
    _load_block_stats_into_model(model, store)
    table_blocks_cpu = _split_concat_blocks(table.z_syn.float().cpu(), store.block_dims)
    y_soft_cpu = table.y_syn_soft.float().cpu()
    y_hard_cpu = table.y_syn_hard.long().cpu()
    hard_mask_cpu = table.hard_anchor_mask.bool().cpu()
    opt = torch.optim.AdamW(model.parameters(), lr=float(_arg(args, "student_lr", 0.003)), weight_decay=float(_arg(args, "weight_decay", 5e-4)))
    rows = torch.arange(table.z_syn.shape[0], dtype=torch.long)
    batch_size = int(_arg(args, "student_batch_size", max(512, table.z_syn.shape[0])))
    prior = target_prior.float().to(device)
    for epoch in range(int(epochs)):
        model.train()
        order = rows[torch.randperm(rows.numel(), generator=torch.Generator().manual_seed(int(_arg(args, "seed", 42)) + epoch))]
        for start in range(0, order.numel(), batch_size):
            idx = order[start : start + batch_size]
            blocks = {key: value[idx].to(device) for key, value in table_blocks_cpu.items()}
            y_soft = y_soft_cpu[idx].to(device)
            logits = model(blocks)
            loss = F.kl_div(F.log_softmax(logits / float(temperature), dim=1), y_soft, reduction="batchmean") * (float(temperature) ** 2)
            hard_mask = hard_mask_cpu[idx].to(device)
            if hard_mask.any():
                y_hard = y_hard_cpu[idx].to(device)
                loss = loss + float(lambda_hard) * F.cross_entropy(logits[hard_mask], y_hard[hard_mask])
            mean_pred = F.softmax(logits, dim=1).mean(dim=0).clamp_min(1e-12)
            loss = loss + float(lambda_prior) * F.kl_div(mean_pred.log(), prior, reduction="sum")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    valid = evaluate_lazy_sft(model, store, labels, valid_idx, num_classes=int(labels.max().item()) + 1, batch_size=int(_arg(args, "teacher_eval_batch_size", 65536)), device=device)
    test = evaluate_lazy_sft(model, store, labels, test_idx, num_classes=int(labels.max().item()) + 1, batch_size=int(_arg(args, "teacher_eval_batch_size", 65536)), device=device)
    return {
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "valid_acc": valid["accuracy"],
        "predicted_classes": test["predicted_class_count"],
        "prediction_entropy": test.get("prediction_entropy", ""),
    }


def _promotion_for_ratio(ratio: float, accuracy: float) -> tuple[str, str]:
    if abs(float(ratio) - 0.001) < 1e-12 and float(accuracy) >= REDDIT_TTC_001_GATE:
        return "promoted", ""
    if abs(float(ratio) - 0.005) < 1e-12 and float(accuracy) >= REDDIT_TTC_005_GATE:
        return "promoted", ""
    return "not_promoted", "ttc_gate_not_met"


def build_ttc_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    metadata = load_or_train_teacher_cache(args)
    ratios = [float(v) for v in _arg(args, "ratios", [0.001, 0.005])]
    modes = [str(v) for v in _arg(args, "ttc_modes", ["ttc_coverage_plus_boundary_plus_mixup"])]
    if metadata is None:
        return [
            make_t31_row(
                dataset="Reddit",
                method=f"reddit_{modes[0]}",
                seed=int(_arg(args, "seed", 42)),
                requested_full_node_ratio=ratio,
                total_condensed_nodes=ratio_budget("Reddit", ratio),
                status="blocked",
                failure_reason="missing_reddit_teacher_cache",
                promotion_track="sota_chase",
                promotion_status="not_promoted",
                uses_teacher_logits=True,
                next_action=build_reddit_ttc_server_command(),
            )
            for ratio in ratios
        ]
    labels, train_idx, valid_idx, test_idx = load_reddit_raw_memmap_labels_and_splits(_arg(args, "memmap_root", "dataset/Reddit/processed/raw_memmap"))
    features = _concat_features(args)
    probs = torch.from_numpy(np.asarray(np.load(metadata["probs_path"], mmap_mode="r"), dtype=np.float32))
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        budget = ratio_budget("Reddit", ratio)
        for mode in modes:
            started = time.perf_counter()
            table = build_ttc_condensed_table(
                features=features,
                teacher_probs=probs,
                labels=labels,
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
                num_rows=budget,
                mode=mode,
                seed=int(_arg(args, "seed", 42)),
                mixup_alpha=float(_arg(args, "mixup_alpha", 0.4)),
            )
            best: dict[str, Any] | None = None
            for hidden_dim in [int(v) for v in _arg(args, "hidden_dims", [128])]:
                for epochs in [int(v) for v in _arg(args, "epochs", [120])]:
                    for temperature in [float(v) for v in _arg(args, "temperatures", [2.0])]:
                        for lambda_hard in [float(v) for v in _arg(args, "lambda_hard", [0.5])]:
                            for lambda_prior in [float(v) for v in _arg(args, "lambda_prior", [0.05])]:
                                if str(_arg(args, "student_backend", "blockwise_sft")) == "concat_mlp":
                                    result = train_soft_label_condensed_student(
                                        z_syn=table.z_syn,
                                        y_syn_soft=table.y_syn_soft,
                                        train_anchor_hard=table.y_syn_hard,
                                        hard_anchor_mask=table.hard_anchor_mask,
                                        eval_features=features[test_idx],
                                        eval_labels=labels[test_idx],
                                        valid_features=features[valid_idx],
                                        valid_labels=labels[valid_idx],
                                        hidden_dim=hidden_dim,
                                        epochs=epochs,
                                        weight_decay=float(_arg(args, "weight_decay", 5e-4)),
                                        dropout=float(_arg(args, "dropout", 0.1)),
                                        temperature=temperature,
                                        lambda_hard=lambda_hard,
                                        lambda_prior=lambda_prior,
                                        target_prior=probs.mean(dim=0),
                                        device=str(_arg(args, "device", "cuda")),
                                        seed=int(_arg(args, "seed", 42)),
                                    )
                                else:
                                    result = _train_blockwise_soft_student(
                                        args=args,
                                        table=table,
                                        labels=labels,
                                        valid_idx=valid_idx,
                                        test_idx=test_idx,
                                        hidden_dim=hidden_dim,
                                        epochs=epochs,
                                        temperature=temperature,
                                        lambda_hard=lambda_hard,
                                        lambda_prior=lambda_prior,
                                        target_prior=probs.mean(dim=0),
                                    )
                                score = float(result.get("valid_acc", 0.0)) + 0.05 * float(result.get("macro_f1", 0.0))
                                payload = {
                                    **result,
                                    "hidden_dim": hidden_dim,
                                    "epochs": epochs,
                                    "temperature": temperature,
                                    "lambda_hard": lambda_hard,
                                    "lambda_prior": lambda_prior,
                                    "score": score,
                                }
                                best = payload if best is None or score > float(best["score"]) else best
            assert best is not None
            promotion_status, failure = _promotion_for_ratio(ratio, float(best["accuracy"]))
            row = make_t31_row(
                dataset="Reddit",
                method=f"reddit_{mode}",
                seed=int(_arg(args, "seed", 42)),
                requested_full_node_ratio=ratio,
                total_condensed_nodes=budget,
                syn_rows=budget,
                shadow_nodes=0,
                condensed_edges=0,
                accuracy=best["accuracy"],
                macro_f1=best["macro_f1"],
                valid_acc=best["valid_acc"],
                predicted_classes=best["predicted_classes"],
                status="completed_long",
                failure_reason=failure,
                promotion_track="sota_chase",
                promotion_status=promotion_status,
                teacher_method=metadata.get("teacher_method", ""),
                teacher_accuracy=metadata.get("teacher_accuracy", ""),
                teacher_macro_f1=metadata.get("teacher_macro_f1", ""),
                teacher_valid_acc=metadata.get("teacher_valid_acc", ""),
                teacher_temperature=best["temperature"],
                teacher_entropy_mean=metadata.get("teacher_entropy_mean", ""),
                teacher_margin_mean=metadata.get("teacher_margin_mean", ""),
                teacher_disagreement_mean=metadata.get("teacher_disagreement_mean", 0.0),
                teacher_cache_bytes=metadata.get("teacher_cache_bytes", ""),
                teacher_logits_cache_path=metadata.get("logits_path", ""),
                uses_teacher_logits=True,
                uses_kd=False,
                soft_label_source="teacher_probs_cache",
                candidate_nodes="all",
                candidate_bucket_counts_json=json.dumps(table.diagnostics["candidate_bucket_counts"], sort_keys=True),
                selected_bucket_counts_json=json.dumps(table.diagnostics["selected_bucket_counts"], sort_keys=True),
                soft_class_mass_coverage=table.diagnostics["soft_class_mass_coverage"],
                entropy_bucket_coverage=table.diagnostics["entropy_bucket_coverage"],
                margin_bucket_coverage=table.diagnostics["margin_bucket_coverage"],
                degree_bucket_coverage=table.diagnostics["degree_bucket_coverage"],
                hard_anchor_count=table.diagnostics["hard_anchor_count"],
                soft_only_count=table.diagnostics["soft_only_count"],
                mixup_row_count=table.diagnostics["mixup_row_count"],
                target_prior_type="teacher_soft_prior",
                student_model=str(_arg(args, "student_backend", "blockwise_sft")),
                hidden_dim=best["hidden_dim"],
                epochs=best["epochs"],
                dropout=float(_arg(args, "dropout", 0.1)),
                weight_decay=float(_arg(args, "weight_decay", 5e-4)),
                label_smoothing=0.0,
                uses_valid_labels_for_hyperparam_selection=True,
                condensation_time=float(time.perf_counter() - started),
                training_time="included_in_condensation_time",
                peak_cpu_ram=current_cpu_ram_bytes(),
                peak_gpu_ram=current_gpu_ram_bytes(),
                cache_bytes=metadata.get("teacher_cache_bytes", ""),
                full_edge_scans=13,
                notes="TTC uses all-node SFT features and teacher soft targets; no teacher logits are input features.",
                source_table=metadata.get("probs_path", ""),
            )
            rows.append(apply_t31_promotion_guard(row))
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_ttc_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t31_reddit_ttc_seed42.csv"), rows, T31_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t31_reddit_ttc_notes.md"),
        [
            "# T31 Reddit TTC",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "accuracy", "macro_f1", "valid_acc", "teacher_accuracy", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_reddit_ttc_server_command()}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T31 Reddit Teacher-Transport Condensation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--selected-blocks", default=json.dumps(DEFAULT_BLOCKS))
    parser.add_argument("--teacher-cache-dir", default="experiments/cache/t31_reddit_ttc_teacher_seed42")
    parser.add_argument("--teacher-model-type", default="sagn_lite_v4")
    parser.add_argument("--teacher-hidden-dim", type=int, default=128)
    parser.add_argument("--teacher-dropout", type=float, default=0.3)
    parser.add_argument("--teacher-num-layers", type=int, default=2)
    parser.add_argument("--teacher-block-dropout", type=float, default=0.0)
    parser.add_argument("--teacher-hop-dropout", type=float, default=0.0)
    parser.add_argument("--teacher-label-dropout", type=float, default=0.05)
    parser.add_argument("--teacher-attention-heads", type=int, default=1)
    parser.add_argument("--teacher-activation", default="relu")
    parser.add_argument("--teacher-norm", default="none")
    parser.add_argument("--teacher-loss-type", default="sqrt_weighted_ce")
    parser.add_argument("--teacher-lr", type=float, default=0.003)
    parser.add_argument("--teacher-weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-label-smoothing", type=float, default=0.0)
    parser.add_argument("--teacher-epochs", type=int, default=30)
    parser.add_argument("--teacher-batch-size", type=int, default=16384)
    parser.add_argument("--teacher-eval-batch-size", type=int, default=65536)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.001, 0.005])
    parser.add_argument("--ttc-modes", nargs="+", default=["ttc_coverage_plus_boundary_plus_mixup"])
    parser.add_argument("--candidate-nodes", default="all")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[2.0])
    parser.add_argument("--lambda-hard", nargs="+", type=float, default=[0.5])
    parser.add_argument("--lambda-prior", nargs="+", type=float, default=[0.05])
    parser.add_argument("--students", nargs="+", default=["table_head_mlp"])
    parser.add_argument("--student-backend", default="blockwise_sft", choices=["blockwise_sft", "concat_mlp"])
    parser.add_argument("--student-model-type", default="sagn_lite_v4")
    parser.add_argument("--student-lr", type=float, default=0.003)
    parser.add_argument("--student-batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128])
    parser.add_argument("--epochs", nargs="+", type=int, default=[120])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t31_reddit_ttc_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t31_reddit_ttc_notes.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
