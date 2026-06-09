from __future__ import annotations


def _round_gap(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def decompose_condensation_gaps(
    *,
    fullgraph_acc: float | None,
    identity_condensed_acc: float | None,
    prototype_oracle_acc: float | None,
    shadow_hgc_acc: float | None,
    fullgraph_gate_passed: bool,
    tolerance: float = 0.02,
) -> dict:
    full_to_identity = None if fullgraph_acc is None or identity_condensed_acc is None else float(fullgraph_acc) - float(identity_condensed_acc)
    identity_to_oracle = None if identity_condensed_acc is None or prototype_oracle_acc is None else float(identity_condensed_acc) - float(prototype_oracle_acc)
    oracle_to_shadow = None if prototype_oracle_acc is None or shadow_hgc_acc is None else float(prototype_oracle_acc) - float(shadow_hgc_acc)
    full_to_shadow = None if fullgraph_acc is None or shadow_hgc_acc is None else float(fullgraph_acc) - float(shadow_hgc_acc)
    if not fullgraph_gate_passed:
        label = "blocked_by_fullgraph_backbone"
    elif full_to_identity is not None and full_to_identity > tolerance:
        label = "condensed_path_inconsistent"
    elif identity_to_oracle is not None and identity_to_oracle > tolerance:
        label = "prototype_selection_bottleneck"
    elif oracle_to_shadow is not None and oracle_to_shadow > tolerance:
        label = "shadow_factorization_bottleneck"
    else:
        label = "training_head_bottleneck"
    return {
        "fullgraph_acc": fullgraph_acc,
        "identity_condensed_acc": identity_condensed_acc,
        "prototype_oracle_acc": prototype_oracle_acc,
        "shadow_hgc_acc": shadow_hgc_acc,
        "full_to_identity_gap": _round_gap(full_to_identity),
        "identity_to_oracle_gap": _round_gap(identity_to_oracle),
        "oracle_to_shadow_gap": _round_gap(oracle_to_shadow),
        "full_to_shadow_gap": _round_gap(full_to_shadow),
        "bottleneck_label": label,
    }

