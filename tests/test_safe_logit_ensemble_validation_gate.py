import torch

from shadow_hgc.logits.ensemble import evaluate_ensemble_promotion


def test_safe_logit_ensemble_requires_validation_improvement_and_test_nonregression():
    row = evaluate_ensemble_promotion(
        valid_acc=0.61,
        test_acc=0.59,
        macro_f1=0.5,
        predicted_class_count=3,
        best_component_valid_acc=0.60,
        best_component_test_acc=0.591,
        epsilon=0.0005,
        tolerance=0.001,
        component_forbidden_flags=[False, False],
        component_bounded_edges=[False, False],
    )

    assert row["promotion_status"] == "promoted"

    blocked = evaluate_ensemble_promotion(
        valid_acc=0.61,
        test_acc=0.58,
        macro_f1=0.5,
        predicted_class_count=3,
        best_component_valid_acc=0.60,
        best_component_test_acc=0.591,
        epsilon=0.0005,
        tolerance=0.001,
        component_forbidden_flags=[False, False],
        component_bounded_edges=[False, False],
    )

    assert blocked["promotion_status"] == "blocked"
    assert "test_regression" in blocked["promotion_reason"]
