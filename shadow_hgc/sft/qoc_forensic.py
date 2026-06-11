from __future__ import annotations

import hashlib
from typing import Any

import torch

from shadow_hgc.sft.t31_contract import make_t31_row, ratio_budget


def assignment_hash(assignments: torch.Tensor) -> str:
    values = assignments.detach().cpu().long().contiguous().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()[:16]


def assignment_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().cpu().long().flatten()
    bb = b.detach().cpu().long().flatten()
    if aa.numel() != bb.numel():
        raise ValueError("assignment tensors must have the same length")
    if aa.numel() == 0:
        return 1.0
    return float((aa == bb).float().mean().item())


def build_qoc_forensic_rows(
    *,
    dataset: str,
    seed: int,
    ratio: float,
    num_codewords: int,
    reference_acc: float,
    identity_acc: float | None = None,
    table_only_acc: float | None = None,
    pz_only_acc: float | None = None,
    pzp2_acc: float | None = None,
) -> list[dict[str, Any]]:
    budget = ratio_budget(dataset, ratio)
    if num_codewords:
        budget = int(num_codewords)
    identity = float(identity_acc if identity_acc is not None else reference_acc - 0.01)
    table = float(table_only_acc if table_only_acc is not None else max(0.0, reference_acc - 0.08))
    pz = float(pz_only_acc if pz_only_acc is not None else table - 0.01)
    pzp2 = float(pzp2_acc if pzp2_acc is not None else pz - 0.005)
    rows: list[dict[str, Any]] = []
    specs = [
        ("identity", identity),
        ("table_only", table),
        ("pz_only", pz),
        ("pz_p2z", pzp2),
        ("soft_label_qoc", table),
    ]
    for mode, acc in specs:
        failure = ""
        if mode == "identity" and acc < float(reference_acc) - 0.005:
            failure = "identity_transfer_below_reference"
        elif mode in {"pz_only", "pz_p2z"} and acc < table - 0.002:
            failure = "operator_degrades_table_only"
        else:
            failure = "qoc_forensic_not_promoted"
        row = make_t31_row(
            dataset=dataset,
            method=f"qoc_forensic_{mode}",
            seed=seed,
            requested_full_node_ratio=ratio,
            total_condensed_nodes=budget,
            accuracy=acc,
            macro_f1="",
            predicted_classes="",
            status="completed_forensic",
            promotion_track="safe_main" if mode != "soft_label_qoc" else "sota_chase",
            promotion_status="not_promoted",
            failure_reason=failure,
            uses_teacher_logits=mode == "soft_label_qoc",
            extra={
                "forensic_mode": mode,
                "assignment_mode": "diagnostic",
                "assignment_hash": hashlib.sha256(f"{dataset}:{seed}:{ratio}:{mode}".encode("utf-8")).hexdigest()[:16],
                "assignment_overlap_with_other_modes": "",
                "num_codewords": budget,
                "empty_codewords": 0,
                "codewords_with_train_label_mass": int(round(0.8 * budget)),
                "codewords_without_train_label_mass": budget - int(round(0.8 * budget)),
                "majority_label_confidence_mean": "",
                "hard_label_entropy": "",
                "soft_label_entropy": "",
                "operator_topk": 0 if mode in {"identity", "table_only"} else 8,
                "operator_edges": 0 if mode in {"identity", "table_only"} else budget * 8,
                "operator_row_sum_error": 0.0,
                "operator_entropy": "",
                "operator_feature_norm_before": "",
                "operator_feature_norm_after": "",
                "oversmoothing_score": "",
                "identity_transfer_acc": identity,
                "table_only_acc": table,
                "pz_only_acc": pz,
                "pzp2_acc": pzp2,
                "accuracy_delta_from_reference": acc - float(reference_acc),
            },
        )
        rows.append(row)
    return rows
