from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.codebook_assignment import build_codebook_assignment
from shadow_hgc.sft.qoc_condense import build_qoc_table
from shadow_hgc.sft.qoc_transfer_eval import train_qoc_table_head
from shadow_hgc.sft.quotient_operator import _finalize_rows, build_quotient_operator
from shadow_hgc.sft.t30_contract import T30_REQUIRED_FIELDS, make_t30_row, num_classes, ratio_budget
from shadow_hgc.train.lazy_sft_memmap import load_manifest_block_store


DEFAULT_RATIOS = (0.001, 0.005)
DEFAULT_ASSIGNMENTS = (
    "qoc_class_conditional_online_kmeans",
    "qoc_sft_ctc_assignment",
    "qoc_sft_bonsai_assignment",
    "qoc_hybrid_assignment",
)
DEFAULT_TOPKS = (8, 16, 32)
DEFAULT_STUDENTS = ("operator_sft_table_head",)

CONTROL_REFERENCES = {
    0.0005: {"reddit_random_frozen_init": (0.8547654525, 0.7978628905)},
    0.001: {
        "current_sft_signature_random": (0.8983896738, 0.8433886103),
        "sft_hnr_fdm_hybrid": (0.9215841158, 0.8848907779),
        "reddit_random_frozen_init": (0.8983178644, 0.8428406957),
    },
    0.0025: {
        "current_sft_signature_random": (0.9163958853, 0.8805489216),
        "sft_hnr_fdm_hybrid": (0.9140441269, 0.8730428740),
    },
    0.005: {
        "current_sft_signature_random": (0.9244564925, 0.8862562818),
        "sft_hnr_fdm_hybrid": (0.9217097822, 0.8817167426),
        "reddit_random_frozen_init": (0.9212609734, 0.8824960392),
    },
    0.01: {
        "current_sft_signature_random": (0.9245283019, 0.8892093666),
        "sft_hnr_fdm_hybrid": (0.9236127318, 0.8881558607),
    },
}


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_reddit_qoc_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t30_reddit_qoc.py --device cuda --ratios 0.001 0.005 "
        "--assignment-modes qoc_class_conditional_online_kmeans qoc_sft_ctc_assignment qoc_sft_bonsai_assignment qoc_hybrid_assignment "
        "--operator-topks 8 16 32 --quotient-build-modes original_dest_normalized code_row_normalized_fallback "
        "--students operator_sft_table_head --hidden-dims 128 256 512 --epochs 60 120 200 "
        "--manifest-dir experiments/preprop/t24_reddit_streaming_seed42 --memmap-root dataset/Reddit/processed/raw_memmap "
        f"--seed {int(seed)} --run-long"
    )


def build_reddit_qoc_pltc_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t30_reddit_qoc.py --device cuda --ratios 0.001 0.005 "
        "--assignment-modes qoc_pltc_confidence_balanced qoc_pltc_uncertainty_balanced qoc_pltc_class_mass_balanced "
        "--teacher sft_fullgraph --enable-pltc --promotion-track sota_chase "
        "--operator-topks 8 16 32 --students operator_sft_table_head --hidden-dims 128 256 512 --epochs 60 120 200 "
        f"--seed {int(seed)} --run-long"
    )


def _method_for_mode(mode: str) -> str:
    mapping = {
        "qoc_class_conditional_online_kmeans": "reddit_qoc_hard_online_kmeans",
        "qoc_sft_ctc_assignment": "reddit_qoc_hard_ctc_assignment",
        "qoc_sft_bonsai_assignment": "reddit_qoc_hard_bonsai_assignment",
        "qoc_hybrid_assignment": "reddit_qoc_hard_hybrid_assignment",
        "qoc_pltc_confidence_balanced": "reddit_qoc_pltc_confidence_balanced",
        "qoc_pltc_uncertainty_balanced": "reddit_qoc_pltc_uncertainty_balanced",
        "qoc_pltc_class_mass_balanced": "reddit_qoc_pltc_class_mass_balanced",
    }
    return mapping.get(mode, f"reddit_{mode}")


def _track_for_mode(mode: str, explicit_track: str, enable_pltc: bool) -> str:
    if explicit_track:
        return explicit_track
    if enable_pltc or mode.startswith("qoc_pltc"):
        return "sota_chase"
    return "safe_main"


def _operator_smoke_diag(num_codewords: int, topk: int, mode: str) -> dict[str, Any]:
    edge_src = torch.arange(int(num_codewords), dtype=torch.long)
    edge_dst = (edge_src + 1) % int(num_codewords)
    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    assignments = torch.arange(int(num_codewords), dtype=torch.long)
    result = build_quotient_operator(edge_index=edge_index, assignments=assignments, num_codewords=int(num_codewords), topk=int(topk), mode=mode)
    return result.diagnostics


def _load_transfer_cache(cache_dir: str | Path) -> dict[str, torch.Tensor] | None:
    root = Path(cache_dir)
    required = ["input_syn.npy", "labels_syn.npy", "code_weights.npy", "input_real.npy", "labels_real.npy"]
    if not root.exists() or not all((root / name).exists() for name in required):
        return None
    return {
        "input_syn": torch.from_numpy(np.load(root / "input_syn.npy")).to(torch.float32),
        "labels_syn": torch.from_numpy(np.load(root / "labels_syn.npy")).to(torch.long),
        "code_weights": torch.from_numpy(np.load(root / "code_weights.npy")).to(torch.float32),
        "input_real": torch.from_numpy(np.load(root / "input_real.npy")).to(torch.float32),
        "labels_real": torch.from_numpy(np.load(root / "labels_real.npy")).to(torch.long),
    }


def _block_key(name: str) -> str:
    return "self" if str(name) == "X0" else str(name).lower()


def _existing_path(value: str | Path) -> Path | None:
    if not str(value):
        return None
    path = Path(value)
    return path if path.exists() else None


def _load_npy(root: Path, manifest: dict[str, Any], key: str, default: str, *, mmap: bool = True) -> np.ndarray:
    path = root / str(manifest.get(key, default))
    return np.load(path, mmap_mode="r" if mmap else None)


def _load_reddit_memmap(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    return {
        "root": base,
        "manifest": manifest,
        "src": _load_npy(base, manifest, "src_path", "src.npy"),
        "dst": _load_npy(base, manifest, "dst_path", "dst.npy"),
        "labels": _load_npy(base, manifest, "label_path", "y.int64.npy", mmap=False).astype(np.int64, copy=False),
        "train_idx": _load_npy(base, manifest, "train_idx_path", "train_idx.npy", mmap=False).astype(np.int64, copy=False),
        "valid_idx": _load_npy(base, manifest, "valid_idx_path", "valid_idx.npy", mmap=False).astype(np.int64, copy=False),
        "test_idx": _load_npy(base, manifest, "test_idx_path", "test_idx.npy", mmap=False).astype(np.int64, copy=False),
        "num_nodes": int(manifest.get("num_nodes", 0)),
        "num_edges": int(manifest.get("num_edges", 0)),
        "num_classes": int(manifest.get("num_classes", num_classes("Reddit"))),
    }


def _concat_store_blocks(store: Any, block_names: list[str], rows: np.ndarray | None = None) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for name in block_names:
        key = _block_key(name)
        if key not in store.arrays:
            raise ValueError(f"required preprop block is missing: {name}")
        raw = store.arrays[key] if rows is None else store.arrays[key][rows]
        arrays.append(np.asarray(raw, dtype=np.float32))
    return np.concatenate(arrays, axis=1) if len(arrays) > 1 else arrays[0]


def _mean_by_assignment(values: np.ndarray, assignments: torch.Tensor, num_codewords: int) -> torch.Tensor:
    x = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    a = assignments.to(torch.long).cpu()
    sums = torch.zeros(int(num_codewords), int(x.shape[1]), dtype=torch.float32)
    counts = torch.bincount(a, minlength=int(num_codewords)).to(torch.float32)
    sums.index_add_(0, a, x)
    nonzero = counts > 0
    sums[nonzero] = sums[nonzero] / counts[nonzero].unsqueeze(1)
    if bool((~nonzero).any()):
        sums[~nonzero] = x.mean(dim=0)
    return sums


def _train_onehot(
    *,
    labels: np.ndarray,
    train_idx: np.ndarray,
    rows: np.ndarray,
    classes: int,
) -> np.ndarray:
    out = np.zeros((int(rows.shape[0]), int(classes)), dtype=np.float32)
    is_train = np.zeros(int(labels.shape[0]), dtype=bool)
    is_train[train_idx] = True
    selected = is_train[rows]
    if bool(selected.any()):
        selected_rows = rows[selected]
        y = labels[selected_rows]
        valid = (y >= 0) & (y < int(classes))
        out[np.flatnonzero(selected)[valid], y[valid]] = 1.0
    return out


def _stream_reddit_quotient_operator(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    assignments: torch.Tensor,
    num_codewords: int,
    topk: int,
    mode: str,
    chunk_size: int,
    eps: float = 1e-12,
) -> Any:
    started = time.perf_counter()
    if mode not in {"original_dest_normalized", "code_row_normalized_fallback"}:
        raise ValueError(f"unsupported quotient build mode: {mode}")
    assign = assignments.to(torch.long).cpu().numpy()
    nodes = int(assign.shape[0])
    codewords = int(num_codewords)
    valid_assign = (assign >= 0) & (assign < codewords)
    code_dest_mass = np.bincount(assign[valid_assign], minlength=codewords).astype(np.float64)
    rows: list[defaultdict[int, float]] = [defaultdict(float) for _ in range(codewords)]
    full_edge_scans = 1
    deg_in = np.ones(nodes, dtype=np.float64)
    edges = int(min(len(src), len(dst)))
    step = max(1, int(chunk_size))
    if mode == "original_dest_normalized":
        full_edge_scans = 2
        deg_in = np.zeros(nodes, dtype=np.float64)
        for start in range(0, edges, step):
            d = np.asarray(dst[start : start + step], dtype=np.int64)
            valid = (d >= 0) & (d < nodes)
            if bool(valid.any()):
                deg_in += np.bincount(d[valid], minlength=nodes).astype(np.float64)
    for start in range(0, edges, step):
        s = np.asarray(src[start : start + step], dtype=np.int64)
        d = np.asarray(dst[start : start + step], dtype=np.int64)
        valid = (s >= 0) & (s < nodes) & (d >= 0) & (d < nodes)
        if not bool(valid.any()):
            continue
        s = s[valid]
        d = d[valid]
        cu = assign[s]
        cv = assign[d]
        valid_code = (cu >= 0) & (cu < codewords) & (cv >= 0) & (cv < codewords)
        if not bool(valid_code.any()):
            continue
        cu = cu[valid_code].astype(np.int64, copy=False)
        cv = cv[valid_code].astype(np.int64, copy=False)
        d = d[valid_code]
        if mode == "original_dest_normalized":
            contrib = 1.0 / np.maximum(deg_in[d] * code_dest_mass[cv], float(eps))
        else:
            contrib = np.ones_like(cu, dtype=np.float64)
        pair = cv * codewords + cu
        unique_pair, inverse = np.unique(pair, return_inverse=True)
        summed = np.bincount(inverse, weights=contrib).astype(np.float64, copy=False)
        for pair_id, value in zip(unique_pair.tolist(), summed.tolist()):
            rows[int(pair_id // codewords)][int(pair_id % codewords)] += float(value)
    if mode == "code_row_normalized_fallback":
        for row in rows:
            denom = sum(max(0.0, value) for value in row.values())
            if denom > 0.0:
                for src_code in list(row):
                    row[src_code] = max(0.0, row[src_code]) / denom
    edge_index, edge_weight, diagnostics = _finalize_rows(rows, topk=int(topk))
    diagnostics.update(
        {
            "operator_topk": int(topk),
            "quotient_build_mode": mode,
            "operator_mode": "sparse_codeword_quotient",
            "operator_build_time": float(time.perf_counter() - started),
            "full_edge_scans": int(full_edge_scans),
        }
    )
    return edge_index, edge_weight, diagnostics


def _build_reddit_transfer_from_preprop(
    *,
    args: argparse.Namespace,
    num_codewords: int,
    mode: str,
    quotient_mode: str,
    topk: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    started = time.perf_counter()
    manifest_dir = Path(_arg(args, "manifest_dir", "experiments/preprop/t24_reddit_streaming_seed42"))
    memmap_root = Path(_arg(args, "memmap_root", "dataset/Reddit/processed/raw_memmap"))
    store = load_manifest_block_store(manifest_dir)
    mem = _load_reddit_memmap(memmap_root)
    classes = int(mem["num_classes"])
    labels = np.asarray(mem["labels"], dtype=np.int64)
    train_idx = np.asarray(mem["train_idx"], dtype=np.int64)
    test_idx = np.asarray(mem["test_idx"], dtype=np.int64)
    assignment_blocks = [str(v) for v in _arg(args, "assignment_blocks", ["X0", "X1", "Y1", "structure"])]
    table_blocks = [str(v) for v in _arg(args, "selected_blocks", ["X0", "X1", "X2", "Y1", "Y2", "structure"])]
    required_table = ["X0", "X1", "X2", "Y1", "Y2"]
    missing_table = [name for name in required_table if name not in table_blocks]
    if missing_table:
        raise ValueError(f"selected_blocks must include {missing_table} for real transfer table")
    assignment_features = _concat_store_blocks(store, assignment_blocks)
    assignment_result = build_codebook_assignment(
        features=torch.from_numpy(assignment_features),
        labels=torch.from_numpy(labels).to(torch.long),
        train_idx=torch.from_numpy(train_idx).to(torch.long),
        num_codewords=int(num_codewords),
        num_classes=classes,
        mode=mode,
        seed=int(_arg(args, "seed", 42)),
        chunk_size=int(_arg(args, "assignment_chunk_size", 4096)),
        refine_steps=int(_arg(args, "assignment_refine_steps", 1)),
    )
    edge_index, edge_weight, operator_diag = _stream_reddit_quotient_operator(
        src=mem["src"],
        dst=mem["dst"],
        assignments=assignment_result.assignments,
        num_codewords=int(num_codewords),
        topk=int(topk),
        mode=quotient_mode,
        chunk_size=int(_arg(args, "edge_chunk_size", 2_000_000)),
    )
    x0_all = _concat_store_blocks(store, ["X0"])
    z0 = _mean_by_assignment(x0_all, assignment_result.assignments, int(num_codewords))
    train_mass = assignment_result.codebook_train_label_mass.to(torch.float32)
    node_mass = assignment_result.codebook_node_mass.to(torch.float32).clamp_min(1.0)
    y0 = train_mass / node_mass.unsqueeze(1)
    structure = None
    if "structure" in table_blocks:
        structure_all = _concat_store_blocks(store, ["structure"])
        structure = _mean_by_assignment(structure_all, assignment_result.assignments, int(num_codewords))
    input_syn, table_diag = build_qoc_table(z0=z0, y0=y0, edge_index=edge_index, edge_weight=edge_weight, structure=structure)
    labels_syn = train_mass.argmax(dim=1).to(torch.long)
    code_weights = train_mass.sum(dim=1).to(torch.float32)
    eval_rows = test_idx
    real_blocks = [
        _concat_store_blocks(store, ["X0"], eval_rows),
        _concat_store_blocks(store, ["X1"], eval_rows),
        _concat_store_blocks(store, ["X2"], eval_rows),
        _train_onehot(labels=labels, train_idx=train_idx, rows=eval_rows, classes=classes),
        _concat_store_blocks(store, ["Y1"], eval_rows),
        _concat_store_blocks(store, ["Y2"], eval_rows),
    ]
    if "structure" in table_blocks:
        real_blocks.append(_concat_store_blocks(store, ["structure"], eval_rows))
    input_real = torch.from_numpy(np.concatenate(real_blocks, axis=1)).to(torch.float32)
    labels_real = torch.from_numpy(labels[eval_rows]).to(torch.long)
    diagnostics = {
        **assignment_result.diagnostics,
        **operator_diag,
        **table_diag,
        "precompute_time": float(time.perf_counter() - started),
        "cache_bytes": int(sum(Path(path).stat().st_size for path in [manifest_dir / "manifest.json", memmap_root / "manifest.json"] if path.exists())),
        "uses_processed_data_pt": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_dense_adjacency": False,
        "uses_e_by_d_materialization": False,
        "uses_exact_pairwise": False,
    }
    cache = {
        "input_syn": input_syn,
        "labels_syn": labels_syn,
        "code_weights": code_weights,
        "input_real": input_real,
        "labels_real": labels_real,
    }
    return cache, diagnostics, str(manifest_dir)


def build_reddit_control_rows(ratios: list[float], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        for method, (acc, macro) in CONTROL_REFERENCES.get(float(ratio), {}).items():
            budget = ratio_budget("Reddit", ratio)
            rows.append(
                make_t30_row(
                    dataset="Reddit",
                    method=method,
                    seed=seed,
                    requested_full_node_ratio=ratio,
                    num_codewords=budget,
                    accuracy=acc,
                    macro_f1=macro,
                    predicted_classes=41,
                    status="completed_reference",
                    promotion_status="not_promoted",
                    promotion_track="safe_main",
                    failure_reason="control_reference_not_new_t30_method",
                    notes="T30 control reference carried from prior stages; not a QOC promotion candidate.",
                    transfer_eval_type="reference",
                    student_model="reference",
                )
            )
    return rows


def build_reddit_qoc_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = [float(v) for v in _arg(args, "ratios", DEFAULT_RATIOS)]
    modes = [str(v) for v in _arg(args, "assignment_modes", DEFAULT_ASSIGNMENTS)]
    topks = [int(v) for v in _arg(args, "operator_topks", DEFAULT_TOPKS)]
    quotient_modes = [str(v) for v in _arg(args, "quotient_build_modes", ["code_row_normalized_fallback"])]
    students = [str(v) for v in _arg(args, "students", DEFAULT_STUDENTS)]
    hidden_dims = [int(v) for v in _arg(args, "hidden_dims", [128])]
    epochs_list = [int(v) for v in _arg(args, "epochs", [60])]
    seed = int(_arg(args, "seed", 42))
    explicit_track = str(_arg(args, "promotion_track", ""))
    cache = _load_transfer_cache(_arg(args, "sft_cache_dir", ""))
    direct_cache: dict[tuple[float, str, str, int], tuple[dict[str, torch.Tensor], dict[str, Any], str]] = {}
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        budget = ratio_budget("Reddit", ratio)
        for mode in modes:
            track = _track_for_mode(mode, explicit_track, bool(_arg(args, "enable_pltc", False)))
            labeled = int(round(budget * (0.5 if track == "sota_chase" else 0.7)))
            for qmode in quotient_modes:
                for topk in topks:
                    diag_started = time.perf_counter()
                    diag = _operator_smoke_diag(budget, topk, qmode)
                    diag["operator_build_time"] = float(time.perf_counter() - diag_started)
                    transfer_cache = cache
                    source_table = str(Path(_arg(args, "sft_cache_dir", ""))) if cache is not None else ""
                    if transfer_cache is None and bool(_arg(args, "run_long", False)) and not bool(_arg(args, "smoke", False)):
                        manifest_path = _existing_path(_arg(args, "manifest_dir", "experiments/preprop/t24_reddit_streaming_seed42"))
                        memmap_path = _existing_path(_arg(args, "memmap_root", "dataset/Reddit/processed/raw_memmap"))
                        if manifest_path is not None and memmap_path is not None:
                            direct_key = (float(ratio), mode, qmode, int(topk))
                            if direct_key not in direct_cache:
                                direct_cache[direct_key] = _build_reddit_transfer_from_preprop(
                                    args=args,
                                    num_codewords=budget,
                                    mode=mode,
                                    quotient_mode=qmode,
                                    topk=topk,
                                )
                            transfer_cache, diag, source_table = direct_cache[direct_key]
                    for student in students:
                        base_kwargs = dict(
                            dataset="Reddit",
                            method=_method_for_mode(mode),
                            seed=seed,
                            requested_full_node_ratio=ratio,
                            num_codewords=budget,
                            num_labeled_codewords=labeled,
                            num_unlabeled_codewords=max(0, budget - labeled),
                            total_condensed_edges=int(diag["operator_edges_after_topk"]),
                            promotion_status="not_promoted",
                            promotion_track=track,
                            assignment_mode=mode,
                            operator_mode="sparse_codeword_quotient",
                            quotient_build_mode=qmode,
                            student_model=student,
                            uses_teacher_logits=track == "sota_chase",
                        )
                        if transfer_cache is None:
                            rows.append(
                                make_t30_row(
                                    **base_kwargs,
                                    status="completed_operator_smoke" if bool(_arg(args, "smoke", False)) else "blocked",
                                    failure_reason="no_transfer_eval_accuracy" if bool(_arg(args, "smoke", False)) else "missing_reddit_sft_transfer_cache",
                                    transfer_eval_type="operator_smoke",
                                    notes="QOC operator path ran, but real transfer cache is missing so no accuracy is reported.",
                                    next_action=build_reddit_qoc_server_command(seed),
                                    extra=diag,
                                )
                            )
                            continue
                        for hidden_dim in hidden_dims:
                            for epochs in epochs_list:
                                result = train_qoc_table_head(
                                    input_syn=transfer_cache["input_syn"],
                                    labels_syn=transfer_cache["labels_syn"],
                                    code_weights=transfer_cache["code_weights"],
                                    input_real=transfer_cache["input_real"],
                                    labels_real=transfer_cache["labels_real"],
                                    num_classes=41,
                                    hidden_dim=hidden_dim,
                                    epochs=epochs,
                                    seed=seed,
                                )
                                rows.append(
                                    make_t30_row(
                                        **base_kwargs,
                                        status="completed_transfer_eval",
                                        failure_reason="",
                                        accuracy=result.metrics["accuracy"],
                                        macro_f1=result.metrics["macro_f1"],
                                        predicted_classes=result.metrics["predicted_classes"],
                                        transfer_eval_type="real_transfer_eval",
                                        extra={**diag, **result.metrics, "student_hidden_dim": int(hidden_dim), "student_epochs": int(epochs)},
                                        source_table=source_table,
                                    )
                                )
                    if bool(_arg(args, "smoke", False)):
                        break
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    ratios = [float(v) for v in _arg(args, "ratios", DEFAULT_RATIOS)]
    seeds_arg = _arg(args, "seeds", None)
    seeds = [int(v) for v in (seeds_arg if seeds_arg else [int(_arg(args, "seed", 42))])]
    rows: list[dict[str, Any]] = []
    original_seed = int(_arg(args, "seed", 42))
    for seed in seeds:
        args.seed = seed
        if seed == 42:
            rows.extend(build_reddit_control_rows(ratios, seed))
        rows.extend(build_reddit_qoc_rows(args))
    args.seed = original_seed
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t30_reddit_qoc_seed42.csv"), rows, T30_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t30_reddit_qoc_notes.md"),
        [
            "# T30 Reddit QOC",
            "",
            "- Control references are carried separately from QOC rows.",
            "- QOC rows use a prebuilt transfer cache when supplied; otherwise `--run-long` builds real transfer tables from Reddit preprop memmaps.",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "num_codewords", "operator_topk", "operator_row_sum_error", "transfer_eval_type", "accuracy", "macro_f1", "status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- QOC-hard command: `{build_reddit_qoc_server_command(seed=int(_arg(args, 'seed', 42)))}`",
            f"- QOC-soft command: `{build_reddit_qoc_pltc_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T30 Reddit Shadow-QOC experiments.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--assignment-modes", nargs="+", default=list(DEFAULT_ASSIGNMENTS))
    parser.add_argument("--operator-topks", nargs="+", type=int, default=list(DEFAULT_TOPKS))
    parser.add_argument("--quotient-build-modes", nargs="+", default=["code_row_normalized_fallback"])
    parser.add_argument("--students", nargs="+", default=list(DEFAULT_STUDENTS))
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--epochs", nargs="+", type=int, default=[60, 120, 200])
    parser.add_argument("--teacher", default="")
    parser.add_argument("--enable-pltc", action="store_true")
    parser.add_argument("--promotion-track", default="")
    parser.add_argument("--sft-cache-dir", default="")
    parser.add_argument("--manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--selected-blocks", nargs="+", default=["X0", "X1", "X2", "Y1", "Y2", "structure"])
    parser.add_argument("--assignment-blocks", nargs="+", default=["X0", "X1", "Y1", "structure"])
    parser.add_argument("--assignment-chunk-size", type=int, default=4096)
    parser.add_argument("--assignment-refine-steps", type=int, default=1)
    parser.add_argument("--edge-chunk-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t30_reddit_qoc_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t30_reddit_qoc_notes.md")
    args = parser.parse_args()
    csv_path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
