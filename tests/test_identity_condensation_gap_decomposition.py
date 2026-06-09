from __future__ import annotations

from shadow_hgc.eval.gap_decomposition import decompose_condensation_gaps


def test_gap_decomposition_labels_fullgraph_blocked_first():
    result = decompose_condensation_gaps(
        fullgraph_acc=0.78,
        identity_condensed_acc=0.77,
        prototype_oracle_acc=0.76,
        shadow_hgc_acc=0.75,
        fullgraph_gate_passed=False,
    )

    assert result["bottleneck_label"] == "blocked_by_fullgraph_backbone"


def test_gap_decomposition_labels_identity_and_shadow_bottlenecks():
    identity = decompose_condensation_gaps(
        fullgraph_acc=0.91,
        identity_condensed_acc=0.86,
        prototype_oracle_acc=0.85,
        shadow_hgc_acc=0.84,
        fullgraph_gate_passed=True,
    )
    shadow = decompose_condensation_gaps(
        fullgraph_acc=0.91,
        identity_condensed_acc=0.90,
        prototype_oracle_acc=0.89,
        shadow_hgc_acc=0.82,
        fullgraph_gate_passed=True,
    )

    assert identity["full_to_identity_gap"] == 0.05
    assert identity["bottleneck_label"] == "condensed_path_inconsistent"
    assert shadow["bottleneck_label"] == "shadow_factorization_bottleneck"
