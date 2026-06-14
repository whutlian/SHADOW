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

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.data.reddit_stream import load_reddit_raw_memmap_labels_and_splits
from shadow_hgc.sft.unified_objective import select_unified_prefixes_from_memmap
from shadow_hgc.sft.unified_stt import NUM_CLASSES, NUM_NODES, T38_MAIN_FIELDS, make_t38_row, validate_t38_main_row
from shadow_hgc.train.lazy_sft_memmap import load_arxiv_labels_and_splits, load_products_labels_and_splits, train_lazy_sft_from_memmap


DEFAULT_RATIOS: dict[str, list[float]] = {
    "ogbn-arxiv": [0.0005, 0.001, 0.0025, 0.005, 0.01],
    "Reddit": [0.0005, 0.001, 0.002, 0.0025, 0.005, 0.01],
    "ogbn-products": [0.0002, 0.0004, 0.0008, 0.0025, 0.005],
    "ogbn-papers100M": [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01],
}

ALIASES = {
    "all": "all",
    "arxiv": "ogbn-arxiv",
    "ogbn-arxiv": "ogbn-arxiv",
    "reddit": "Reddit",
    "Reddit": "Reddit",
    "products": "ogbn-products",
    "product": "ogbn-products",
    "ogbn-products": "ogbn-products",
    "papers100m": "ogbn-papers100M",
    "papers100M": "ogbn-papers100M",
    "papers": "ogbn-papers100M",
    "ogbn-papers100M": "ogbn-papers100M",
}

MEDIUM_BLOCKS: dict[str, list[str]] = {
    "ogbn-arxiv": [
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
    ],
    "Reddit": ["X0", "X1", "X2", "X3", "Xres1", "Y1", "Y2", "Y3", "structure"],
    "ogbn-products": ["X0", "X1", "X2", "X3", "Xres1", "Xres2", "Y1", "Y2", "Y3", "structure"],
}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _directory_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for item in root.rglob("*"):
        if item.is_file():
            total += int(item.stat().st_size)
    return total


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def _canonical_datasets(values: list[str]) -> list[str]:
    out = [ALIASES[str(value)] for value in values]
    if out == ["all"] or "all" in out:
        return ["ogbn-arxiv", "Reddit", "ogbn-products", "ogbn-papers100M"]
    return out


def _condensed_nodes(dataset: str, ratio: float) -> int:
    return max(1, int(round(NUM_NODES[dataset] * float(ratio))))


def _manifest_shared_cache_time(manifest_dir: str | Path) -> float | str:
    manifest = _read_json(Path(manifest_dir) / "manifest.json")
    return manifest.get("wall_time_s", "")


def _teacher_metadata(cache_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(cache_dir) / "metadata.json")


def _blocked_row(dataset: str, ratio: float, reason: str, *, seed: int) -> dict[str, Any]:
    del seed
    return make_t38_row(
        dataset=dataset,
        requested_full_node_ratio=float(ratio),
        condensed_nodes=_condensed_nodes(dataset, ratio),
        num_classes=NUM_CLASSES[dataset],
        backend="stt_gated_mixer",
        comparison_type="t39_unified_e2e",
        promotion_status="blocked",
        failure_reason=str(reason),
        cache_reused=False,
        notes="T39 real unified e2e row did not complete; failure is reported rather than hidden",
    )


def _labels_for_medium(dataset: str, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dataset == "ogbn-arxiv":
        return load_arxiv_labels_and_splits(args.arxiv_dataset_root)
    if dataset == "ogbn-products":
        return load_products_labels_and_splits(args.products_dataset_root)
    if dataset == "Reddit":
        return load_reddit_raw_memmap_labels_and_splits(args.reddit_memmap_root)
    raise ValueError(f"not a medium memmap dataset: {dataset}")


def _manifest_for_medium(dataset: str, args: argparse.Namespace) -> str:
    if dataset == "ogbn-arxiv":
        return str(args.arxiv_manifest_dir)
    if dataset == "ogbn-products":
        return str(args.products_manifest_dir)
    if dataset == "Reddit":
        return str(args.reddit_manifest_dir)
    raise ValueError(f"not a medium memmap dataset: {dataset}")


def _teacher_for_medium(dataset: str, args: argparse.Namespace) -> tuple[str | None, float | None, int]:
    if dataset != "Reddit" or not bool(args.use_reddit_teacher_cache):
        return None, None, 0
    meta = _teacher_metadata(args.reddit_teacher_cache_dir)
    path = meta.get("probs_path", "")
    if not path:
        path = str(Path(args.reddit_teacher_cache_dir) / "teacher_probs.npy")
    if not Path(path).exists():
        return None, None, 0
    return str(path), _f(meta.get("teacher_valid_acc"), None), int(_f(meta.get("teacher_cache_bytes"), 0))


def _exception_reason(exc: BaseException) -> str:
    text = str(exc)
    lowered = text.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "OOM:" + text[:240]
    if "timed out" in lowered or "timeout" in lowered:
        return "OOT:" + text[:240]
    return type(exc).__name__ + ":" + text[:240]


def run_medium_dataset(args: argparse.Namespace, dataset: str, ratios: list[float]) -> list[dict[str, Any]]:
    print(json.dumps({"event": "dataset_start", "dataset": dataset, "ratios": ratios}, sort_keys=True), flush=True)
    labels, train_rows, valid_rows, test_rows = _labels_for_medium(dataset, args)
    manifest_dir = _manifest_for_medium(dataset, args)
    teacher_probs_path, teacher_valid_acc, teacher_bytes = _teacher_for_medium(dataset, args)
    budgets = [_condensed_nodes(dataset, ratio) for ratio in ratios]
    max_budget = max(budgets)
    max_probe = make_t38_row(
        dataset=dataset,
        requested_full_node_ratio=max(ratios),
        condensed_nodes=max_budget,
        num_classes=NUM_CLASSES[dataset],
        teacher_valid_acc=teacher_valid_acc,
        uses_teacher_probs_as_soft_targets=bool(teacher_probs_path),
    )
    selection_started = time.perf_counter()
    print(json.dumps({"event": "selection_start", "dataset": dataset, "max_budget": max_budget}, sort_keys=True), flush=True)
    prefixes = select_unified_prefixes_from_memmap(
        labels=labels,
        train_rows=train_rows,
        manifest_dir=manifest_dir,
        budgets=budgets,
        num_classes=NUM_CLASSES[dataset],
        seed=int(args.seed),
        selection_weights={
            "coverage": _f(max_probe["coverage_weight"]),
            "hard": _f(max_probe["hard_weight"]),
            "soft": _f(max_probe["soft_weight"]),
            "boundary": _f(max_probe["boundary_weight"]),
            "rare": _f(max_probe["rare_weight"]),
            "diversity": _f(max_probe["diversity_weight"]),
        },
        teacher_probs_path=teacher_probs_path,
    )
    selection_time = float(time.perf_counter() - selection_started)
    print(json.dumps({"event": "selection_done", "dataset": dataset, "selection_time_sec": selection_time}, sort_keys=True), flush=True)
    teacher_probs = np.load(teacher_probs_path, mmap_mode="r") if teacher_probs_path else None
    rows: list[dict[str, Any]] = []
    cache_bytes = _directory_bytes(manifest_dir) + int(teacher_bytes)
    shared_cache_time_sec = _manifest_shared_cache_time(manifest_dir)
    for ratio, budget in zip(ratios, budgets):
        try:
            print(json.dumps({"event": "ratio_start", "dataset": dataset, "ratio": ratio, "budget": budget}, sort_keys=True), flush=True)
            probe = make_t38_row(
                dataset=dataset,
                requested_full_node_ratio=float(ratio),
                condensed_nodes=budget,
                num_classes=NUM_CLASSES[dataset],
                teacher_valid_acc=teacher_valid_acc,
                uses_teacher_probs_as_soft_targets=bool(teacher_probs_path),
            )
            epochs = int(args.epochs_override or probe["epochs"])
            if int(args.epochs_cap) > 0:
                epochs = min(epochs, int(args.epochs_cap))
            result = train_lazy_sft_from_memmap(
                manifest_dir=manifest_dir,
                labels=labels,
                train_rows=prefixes[budget],
                valid_rows=valid_rows,
                test_rows=test_rows,
                num_classes=NUM_CLASSES[dataset],
                device=args.device,
                model_type="stt_gated_mixer",
                hidden_dim=int(args.hidden_dim_override or probe["hidden_dim"]),
                dropout=float(args.dropout),
                num_layers=int(args.num_layers),
                block_dropout=0.0,
                hop_dropout=0.0,
                label_dropout=float(args.label_dropout),
                selected_blocks=MEDIUM_BLOCKS[dataset],
                loss_type=str(args.loss_type),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                epochs=epochs,
                batch_size=int(args.batch_size),
                eval_batch_size=int(args.eval_batch_size),
                seed=int(args.seed),
                eval_every=int(args.eval_every),
                teacher_probs=teacher_probs,
                lambda_hard=_f(probe["alpha_hard"], 1.0),
                lambda_soft=_f(probe["alpha_soft"], 0.0) if teacher_probs is not None else 0.0,
                lambda_prior=_f(probe["alpha_prior"], 0.0) if teacher_probs is not None else 0.0,
                soft_temperature=_f(probe["soft_temperature"], 2.0),
            ).summary
            test = result["test"]
            valid = result["valid"]
            peak_cpu = int(float(result.get("peak_cpu_ram_gb", 0.0)) * (1024**3))
            peak_gpu = int(float(result.get("peak_gpu_ram_gb", 0.0)) * (1024**3))
            row = make_t38_row(
                dataset=dataset,
                requested_full_node_ratio=float(ratio),
                condensed_nodes=budget,
                num_classes=NUM_CLASSES[dataset],
                backend="stt_gated_mixer",
                comparison_type="t39_unified_e2e",
                accuracy=float(test["accuracy"]),
                macro_f1=float(test["macro_f1"]),
                valid_acc=float(valid.get("accuracy", 0.0)),
                predicted_classes=int(test.get("predicted_class_count", 0)),
                promotion_status="promoted",
                teacher_valid_acc=teacher_valid_acc,
                shared_cache_time_sec=shared_cache_time_sec,
                post_cache_time_sec=float(selection_time + result.get("training_time_s", 0.0) + result.get("inference_time_s", 0.0)),
                total_storage_bytes=int(cache_bytes),
                storage=int(cache_bytes),
                peak_cpu_ram=peak_cpu,
                peak_gpu_ram=peak_gpu,
                edge_cache_id=f"t39_{dataset}_graph_signal_cache_seed{int(args.seed)}",
                sft_cache_id=f"t39_{dataset}_sft_table_cache_seed{int(args.seed)}",
                teacher_cache_id=f"t39_{dataset}_teacher_topk{probe['teacher_cache_k']}_seed{int(args.seed)}" if teacher_probs_path else "teacher_disabled",
                unified_reservoir_id=f"t39_{dataset}_unified_objective_reservoir_seed{int(args.seed)}",
                cache_reused=True,
                incremental_edge_scans_after_cache_build=0,
                uses_teacher_probs_as_soft_targets=bool(teacher_probs_path),
                uses_teacher_probs_as_input_features=False,
                uses_valid_labels_as_input=False,
                uses_test_labels_as_input=False,
                uses_dense_p2=False,
                uses_e_by_d_materialization=False,
                uses_full_edge_index_on_gpu=False,
                notes=(
                    "real T39 unified objective + nested prefix + STT-GatedMixer memmap run; "
                    f"selection_time_sec={selection_time:.6f}; teacher_cache_k={probe['teacher_cache_k']}; "
                    f"budget_phase={probe['budget_phase']}"
                ),
            )
            check = validate_t38_main_row(row)
            if not check["valid"]:
                row["promotion_status"] = "not_promoted"
                row["failure_reason"] = ",".join(check["forbidden_flags"])
            rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "ratio_done",
                        "dataset": dataset,
                        "ratio": ratio,
                        "accuracy": row.get("accuracy", ""),
                        "macro_f1": row.get("macro_f1", ""),
                        "post_cache_time_sec": row.get("post_cache_time_sec", ""),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except BaseException as exc:
            rows.append(_blocked_row(dataset, ratio, _exception_reason(exc), seed=int(args.seed)))
            print(json.dumps({"event": "ratio_blocked", "dataset": dataset, "ratio": ratio, "reason": rows[-1]["failure_reason"]}, sort_keys=True), flush=True)
            if bool(args.fail_fast):
                raise
    return rows


def _papers_shared_cache_time(ctx: Any) -> float:
    parts = [
        ctx.manifest.get("wall_time_s", ""),
        ctx.graph.get("edge_cache_time", ctx.graph.get("wall_time_s", "")),
        ctx.sft.get("sft_cache_time", ctx.sft.get("wall_time_s", "")),
        ctx.teacher.get("teacher_cache_time", ctx.teacher.get("wall_time_s", "")),
        ctx.bank.get("selection_bank_time", ""),
    ]
    return float(sum(_f(value) for value in parts if value not in {"", None}))


def run_papers100m(args: argparse.Namespace, ratios: list[float]) -> list[dict[str, Any]]:
    from scripts.t37_papers100m_common import ensure_t37_bank
    from shadow_hgc.ultra.papers100m_condensed import materialize_condensed_table, train_and_eval_condensed_table
    from shadow_hgc.ultra.papers100m_memmap import directory_bytes
    from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext

    rows: list[dict[str, Any]] = []
    print(json.dumps({"event": "dataset_start", "dataset": "ogbn-papers100M", "ratios": ratios}, sort_keys=True), flush=True)
    try:
        print(json.dumps({"event": "papers_bank_start", "max_ratio": max(ratios), "method": str(args.papers_selection_method)}, sort_keys=True), flush=True)
        policy, _bank = ensure_t37_bank(
            args.papers_cache_root,
            method=str(args.papers_selection_method),
            seed=int(args.seed),
            max_ratio=max(ratios),
            teacher_weight_eta=float(args.papers_teacher_weight_eta),
            force=bool(args.force_rebuild_bank),
        )
        ctx = Papers100MCacheContext(args.papers_cache_root, selection_policy=policy, seed=int(args.seed))
        print(json.dumps({"event": "papers_bank_done", "policy": policy}, sort_keys=True), flush=True)
    except BaseException as exc:
        return [_blocked_row("ogbn-papers100M", ratio, _exception_reason(exc), seed=int(args.seed)) for ratio in ratios]

    teacher_valid_acc = _f(ctx.teacher.get("valid_acc", ctx.teacher.get("teacher_valid_acc", "")), None)
    storage_bytes = int(directory_bytes(args.papers_cache_root))
    for ratio in ratios:
        try:
            print(json.dumps({"event": "ratio_start", "dataset": "ogbn-papers100M", "ratio": ratio}, sort_keys=True), flush=True)
            budget = _condensed_nodes("ogbn-papers100M", ratio)
            probe = make_t38_row(
                dataset="ogbn-papers100M",
                requested_full_node_ratio=float(ratio),
                condensed_nodes=budget,
                num_classes=NUM_CLASSES["ogbn-papers100M"],
                teacher_valid_acc=teacher_valid_acc,
                num_teacher_nodes=int(ctx.manifest.get("target_universe_size", 1_546_782)),
                uses_teacher_probs_as_soft_targets=True,
            )
            epochs = int(args.papers_epochs_override or probe["epochs"])
            if int(args.epochs_cap) > 0:
                epochs = min(epochs, int(args.epochs_cap))
            materialized = materialize_condensed_table(ctx, float(ratio), policy=policy, seed=int(args.seed))
            result = train_and_eval_condensed_table(
                ctx,
                float(ratio),
                student=str(args.papers_student),
                hidden_dim=int(args.hidden_dim_override or probe["hidden_dim"]),
                epochs=epochs,
                temperature=_f(probe["soft_temperature"], 2.0),
                lambda_hard=_f(probe["alpha_hard"], 1.0),
                lambda_soft=_f(probe["alpha_soft"], 0.0),
                lambda_prior=_f(probe["alpha_prior"], 0.0),
                device=args.device,
            )
            ids = ctx.cache_ids()
            row = make_t38_row(
                dataset="ogbn-papers100M",
                requested_full_node_ratio=float(ratio),
                condensed_nodes=int(materialized.get("condensed_nodes", budget)),
                num_classes=NUM_CLASSES["ogbn-papers100M"],
                backend="stt_gated_mixer",
                comparison_type="t39_unified_e2e",
                accuracy=result.get("accuracy", ""),
                macro_f1=result.get("macro_f1", ""),
                valid_acc=result.get("valid_acc", ""),
                predicted_classes=result.get("predicted_classes", ""),
                promotion_status="promoted",
                teacher_valid_acc=teacher_valid_acc,
                num_teacher_nodes=int(ctx.manifest.get("target_universe_size", 1_546_782)),
                shared_cache_time_sec=_papers_shared_cache_time(ctx),
                post_cache_time_sec=float(_f(materialized.get("condensed_materialize_time")) + _f(result.get("student_train_time")) + _f(result.get("eval_time"))),
                total_storage_bytes=storage_bytes,
                storage=storage_bytes,
                peak_cpu_ram="",
                peak_gpu_ram=result.get("peak_gpu_ram", ""),
                edge_cache_id=ids["edge_slice_cache_id"],
                sft_cache_id=ids["sft_cache_id"],
                teacher_cache_id=ids["teacher_cache_id"],
                unified_reservoir_id=ids["selection_bank_id"],
                cache_reused=True,
                incremental_edge_scans_after_cache_build=0,
                uses_teacher_probs_as_soft_targets=True,
                uses_teacher_probs_as_input_features=False,
                uses_valid_labels_as_input=False,
                uses_test_labels_as_input=False,
                uses_dense_p2=False,
                uses_e_by_d_materialization=False,
                uses_full_edge_index_on_gpu=False,
                notes=f"real T39 papers100M one-cache unified schedule; bank_policy={policy}; teacher_cache_k={probe['teacher_cache_k']}",
            )
            check = validate_t38_main_row(row)
            if not check["valid"]:
                row["promotion_status"] = "not_promoted"
                row["failure_reason"] = ",".join(check["forbidden_flags"])
            rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "ratio_done",
                        "dataset": "ogbn-papers100M",
                        "ratio": ratio,
                        "accuracy": row.get("accuracy", ""),
                        "macro_f1": row.get("macro_f1", ""),
                        "post_cache_time_sec": row.get("post_cache_time_sec", ""),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except BaseException as exc:
            rows.append(_blocked_row("ogbn-papers100M", ratio, _exception_reason(exc), seed=int(args.seed)))
            print(json.dumps({"event": "ratio_blocked", "dataset": "ogbn-papers100M", "ratio": ratio, "reason": rows[-1]["failure_reason"]}, sort_keys=True), flush=True)
            if bool(args.fail_fast):
                raise
    return rows


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets = _canonical_datasets([str(value) for value in args.datasets])
    for dataset in datasets:
        ratios = [float(value) for value in (args.ratios if args.ratios else DEFAULT_RATIOS[dataset])]
        if dataset == "ogbn-papers100M":
            rows.extend(run_papers100m(args, ratios))
        else:
            rows.extend(run_medium_dataset(args, dataset, ratios))
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_rows(args)
    csv_path = write_csv(args.csv, rows, T38_MAIN_FIELDS)
    ensure_report(
        args.summary,
        [
            "# T39 Unified E2E Stage",
            "",
            "This table is produced by real unified objective execution. Medium datasets use the same nested prefix selector and STT-GatedMixer lazy memmap trainer; papers100M uses the one-cache ultra path with the same schedule fields.",
            "",
            *markdown_table(
                rows,
                [
                    "dataset",
                    "requested_full_node_ratio",
                    "accuracy",
                    "macro_f1",
                    "valid_acc",
                    "budget_phase",
                    "teacher_cache_k",
                    "student_capacity",
                    "shared_cache_time_sec",
                    "post_cache_time_sec",
                    "storage",
                    "promotion_status",
                    "failure_reason",
                ],
            ),
            "",
            f"- CSV: `{csv_path}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T39 single unified end-to-end Shadow-HGC-STT-U runner.")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--ratios", nargs="+", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arxiv-manifest-dir", default="experiments/preprop/t22_ogbn_arxiv_seed42")
    parser.add_argument("--products-manifest-dir", default="experiments/preprop/t22_ogbn_products_seed42")
    parser.add_argument("--reddit-manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--arxiv-dataset-root", default="dataset/ogbn_arxiv")
    parser.add_argument("--products-dataset-root", default="dataset/ogbn_products")
    parser.add_argument("--reddit-memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--use-reddit-teacher-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reddit-teacher-cache-dir", default="experiments/cache/t31_reddit_ttc_teacher_seed42")
    parser.add_argument("--papers-cache-root", default="caches/papers100m/stt_v1")
    parser.add_argument("--papers-selection-method", default="scr_full_stochastic_coverage_plus_teacher_weight")
    parser.add_argument("--papers-teacher-weight-eta", type=float, default=0.10)
    parser.add_argument("--papers-student", default="papers100m_sagn_table")
    parser.add_argument("--force-rebuild-bank", action="store_true")
    parser.add_argument("--hidden-dim-override", type=int, default=0)
    parser.add_argument("--epochs-override", type=int, default=0)
    parser.add_argument("--papers-epochs-override", type=int, default=0)
    parser.add_argument("--epochs-cap", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--label-dropout", type=float, default=0.0)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--loss-type", default="sqrt_weighted_ce")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t39_unified_e2e_main_curve_seed42.csv")
    parser.add_argument("--summary", default="experiments/summaries/t39_unified_e2e_stage_summary.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
