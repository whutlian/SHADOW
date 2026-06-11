from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from scripts.run_t31_arxiv_actual_cns import _load_arxiv_arrays, train_raw_x_mlp_logits
from shadow_hgc.sft.arxiv_actual_cns import run_actual_cns_grid
from shadow_hgc.sft.arxiv_base_predictors_v3 import load_validated_base_logits
from shadow_hgc.sft.arxiv_cns_forensic_v4 import (
    arxiv_teacher_gate_reason,
    checksum_file,
    checksum_tensor,
    edge_direction_checksums,
    reject_historical_lad_logits,
)
from shadow_hgc.sft.t32_arxiv_cns import cns_grid_plan_v2, transform_arxiv_edge_index
from shadow_hgc.sft.t33_contract import T33_REQUIRED_FIELDS, apply_t33_promotion_guard, make_t33_row


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _find_logits(base_dir: Path, predictor: str) -> Path | None:
    for suffix in (".pt", ".npy"):
        path = base_dir / f"{predictor}_logits{suffix}"
        if path.exists():
            return path
    path = base_dir / predictor
    return path if path.exists() else None


def _base_method(predictor: str) -> str:
    return f"arxiv_{predictor}_base_v4" if predictor in {"raw_x_mlp", "mlp_on_sft"} else f"arxiv_{predictor}_base"


def _cns_method(predictor: str) -> str:
    return f"arxiv_{predictor}_cns_forensic_v4" if predictor in {"raw_x_mlp", "mlp_on_sft"} else f"arxiv_{predictor}_cns_forensic"


def _blocked(args: argparse.Namespace, predictor: str, method: str, reason: str, **fields: Any) -> dict[str, Any]:
    return make_t33_row(
        dataset="ogbn-arxiv",
        method=method,
        seed=int(_arg(args, "seed", 42)),
        status="blocked",
        failure_reason=reason,
        promotion_track="safe_main",
        promotion_status="not_promoted",
        base_predictor=predictor,
        **fields,
    )


def build_arxiv_forensic_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    started = time.perf_counter()
    base_dir = Path(_arg(args, "base_logits_dir", "experiments/logits/t33_arxiv"))
    t31_dir = Path(_arg(args, "fallback_t31_logits_dir", "experiments/logits/t31_arxiv"))
    features, labels, train_idx, valid_idx, test_idx, edge_index = _load_arxiv_arrays(_arg(args, "dataset_root", "dataset/ogbn_arxiv"))
    edge_checksums = edge_direction_checksums(edge_index)
    common = {
        "feature_checksum": checksum_tensor(features[: min(4096, features.shape[0])]),
        "mask_checksum": ";".join([checksum_tensor(train_idx), checksum_tensor(valid_idx), checksum_tensor(test_idx)]),
        "edge_checksum": edge_checksums["edge_checksum_undirected_sym"],
        "notes": json.dumps(
            {
                "num_nodes": int(labels.numel()),
                "num_edges": int(edge_index.shape[1]),
                "num_classes": int(labels.max().item()) + 1,
                "train_count": int(train_idx.numel()),
                "valid_count": int(valid_idx.numel()),
                "test_count": int(test_idx.numel()),
                "feature_shape": list(features.shape),
                **edge_checksums,
            },
            sort_keys=True,
        ),
    }
    plan = cns_grid_plan_v2(
        graph_directions=[str(v) for v in _arg(args, "graph_directions", ["cite_ref", "cited_by", "undirected_sym"])],
        correction_alphas=[float(v) for v in _arg(args, "correction_alphas", [0.2])],
        smoothing_alphas=[float(v) for v in _arg(args, "smoothing_alphas", [0.4])],
        correction_steps=[int(v) for v in _arg(args, "correction_steps", [10])],
        smoothing_steps=[int(v) for v in _arg(args, "smoothing_steps", [20])],
        autoscale=[str(v) for v in _arg(args, "autoscale", ["off"])],
        normalization_modes=[str(v) for v in _arg(args, "normalization_modes", ["dst_row"])],
        self_loop_modes=[str(v) for v in _arg(args, "self_loop_modes", ["none"])],
    )
    rows: list[dict[str, Any]] = []
    for predictor in [str(v) for v in _arg(args, "base_predictors", ["raw_x_mlp"])]:
        logits_path = _find_logits(base_dir, predictor) or _find_logits(t31_dir, predictor)
        if logits_path is None and bool(_arg(args, "train_base_logits_if_missing", False)) and predictor == "raw_x_mlp":
            logits_path = train_raw_x_mlp_logits(args, base_dir / "raw_x_mlp_logits.pt")
        if logits_path is None:
            rows.append(_blocked(args, predictor, _base_method(predictor), "missing_base_logits", **common))
            rows.append(_blocked(args, predictor, _cns_method(predictor), "missing_base_logits", **common))
            continue
        historical = reject_historical_lad_logits(logits_path)
        if historical:
            rows.append(_blocked(args, predictor, _base_method(predictor), historical, logits_cache_path=str(logits_path), **common))
            rows.append(_blocked(args, predictor, _cns_method(predictor), historical, logits_cache_path=str(logits_path), **common))
            continue
        cache = load_validated_base_logits(logits_path)
        logits_hash = checksum_file(logits_path)
        base_acc = cache.metadata.get("test_acc", cache.metadata.get("accuracy", ""))
        base_valid = cache.metadata.get("valid_acc", "")
        base_macro = cache.metadata.get("macro_f1", "")
        rows.append(
            make_t33_row(
                dataset="ogbn-arxiv",
                method=_base_method(predictor),
                seed=int(_arg(args, "seed", 42)),
                status="completed_long",
                failure_reason="raw_feature_or_split_mismatch" if predictor == "raw_x_mlp" and float(base_acc or 0.0) < 0.55 else "",
                promotion_track="safe_main",
                promotion_status="not_promoted",
                base_predictor=predictor,
                base_accuracy=base_acc,
                base_valid_acc=base_valid,
                base_macro_f1=base_macro,
                predicted_classes=cache.metadata.get("predicted_classes", ""),
                logits_cache_path=str(logits_path),
                logits_cache_hash=logits_hash,
                hidden_dim=cache.metadata.get("hidden_dim", ""),
                epochs=cache.metadata.get("epochs", ""),
                **common,
            )
        )
        best_payload: dict[str, Any] | None = None
        for item in plan:
            if item["normalization_mode"] != "dst_row":
                continue
            self_loop = "target_all" if item["self_loop_mode"] == "add_self_loop" else item["self_loop_mode"]
            try:
                result = run_actual_cns_grid(
                    logits=cache.logits,
                    labels=labels,
                    train_idx=train_idx,
                    valid_idx=valid_idx,
                    test_idx=test_idx,
                    edge_index=transform_arxiv_edge_index(edge_index, graph_direction=item["graph_direction"], self_loop_mode=self_loop),
                    num_classes=int(cache.logits.shape[1]),
                    correction_alphas=[float(item["correction_alpha"])],
                    smoothing_alphas=[float(item["smoothing_alpha"])],
                    correction_steps=[int(item["correction_steps"])],
                    smoothing_steps=[int(item["smoothing_steps"])],
                    autoscale=str(item["autoscale"]) == "on",
                )
            except (RuntimeError, IndexError, ValueError):
                continue
            payload = {**item, **result.best_row}
            if best_payload is None or float(payload.get("valid_acc", 0.0)) > float(best_payload.get("valid_acc", 0.0)):
                best_payload = payload
        if best_payload is None:
            rows.append(_blocked(args, predictor, _cns_method(predictor), "cns_grid_no_improvement", logits_cache_path=str(logits_path), logits_cache_hash=logits_hash, **common))
            continue
        failure = arxiv_teacher_gate_reason(base_predictor=predictor, cns_accuracy=best_payload.get("accuracy", ""))
        rows.append(
            apply_t33_promotion_guard(
                make_t33_row(
                    dataset="ogbn-arxiv",
                    method=_cns_method(predictor),
                    seed=int(_arg(args, "seed", 42)),
                    accuracy=best_payload.get("accuracy", ""),
                    macro_f1=best_payload.get("macro_f1", ""),
                    valid_acc=best_payload.get("valid_acc", ""),
                    predicted_classes=best_payload.get("predicted_classes", ""),
                    status="completed_long",
                    failure_reason=failure,
                    promotion_track="safe_main",
                    promotion_status="not_promoted" if failure else "promoted",
                    base_predictor=predictor,
                    base_accuracy=base_acc,
                    base_valid_acc=base_valid,
                    base_macro_f1=base_macro,
                    cns_accuracy=best_payload.get("accuracy", ""),
                    cns_valid_acc=best_payload.get("valid_acc", ""),
                    cns_macro_f1=best_payload.get("macro_f1", ""),
                    graph_direction=best_payload.get("graph_direction", ""),
                    normalization_mode=best_payload.get("normalization_mode", "dst_row"),
                    self_loop_mode=best_payload.get("self_loop_mode", ""),
                    correction_alpha=best_payload.get("correction_alpha", ""),
                    smoothing_alpha=best_payload.get("smoothing_alpha", ""),
                    correction_steps=best_payload.get("correction_steps", ""),
                    smoothing_steps=best_payload.get("smoothing_steps", ""),
                    autoscale=best_payload.get("autoscale", ""),
                    logits_cache_path=str(logits_path),
                    logits_cache_hash=logits_hash,
                    precompute_time=float(time.perf_counter() - started),
                    **common,
                )
            )
        )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_arxiv_forensic_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t33_arxiv_cns_forensic.csv"), rows, T33_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t33_arxiv_teacher_repair_notes.md"),
        ["# T33 Arxiv C&S Forensic", "", *markdown_table(rows, ["method", "base_predictor", "base_accuracy", "cns_accuracy", "cns_valid_acc", "graph_direction", "status", "failure_reason"]), "", f"- CSV: `{csv_path}`"],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T33 arxiv actual C&S forensic repair.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-root", default="dataset/ogbn_arxiv")
    parser.add_argument("--base-logits-dir", default="experiments/logits/t33_arxiv")
    parser.add_argument("--fallback-t31-logits-dir", default="experiments/logits/t31_arxiv")
    parser.add_argument("--base-predictors", nargs="+", default=["raw_x_mlp", "mlp_on_sft", "sagn_lite_v5", "gamlp_lite_v5"])
    parser.add_argument("--train-base-logits-if-missing", action="store_true")
    parser.add_argument("--enable-cns", action="store_true")
    parser.add_argument("--graph-directions", nargs="+", default=["cite_ref", "cited_by", "undirected_sym"])
    parser.add_argument("--normalization-modes", nargs="+", default=["dst_row"])
    parser.add_argument("--self-loop-modes", nargs="+", default=["none"])
    parser.add_argument("--correction-alphas", nargs="+", type=float, default=[0.2])
    parser.add_argument("--smoothing-alphas", nargs="+", type=float, default=[0.4])
    parser.add_argument("--correction-steps", nargs="+", type=int, default=[10])
    parser.add_argument("--smoothing-steps", nargs="+", type=int, default=[20])
    parser.add_argument("--autoscale", nargs="+", default=["off"])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t33_arxiv_cns_forensic.csv")
    parser.add_argument("--report", default="experiments/summaries/t33_arxiv_teacher_repair_notes.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
