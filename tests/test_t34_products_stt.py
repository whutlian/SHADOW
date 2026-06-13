from __future__ import annotations

import torch

from shadow_hgc.sft.products_stt import class_coverage_loss, products_promotion_status, zero_predicted_classes


def test_t34_products_balanced_gate_requires_macro_and_class_coverage() -> None:
    assert products_promotion_status(method="products_stt_balanced", ratio=0.005, accuracy=0.771, macro_f1=0.421, predicted_classes=40) == ("promoted", "")
    assert products_promotion_status(method="products_stt_balanced", ratio=0.005, accuracy=0.79, macro_f1=0.40, predicted_classes=41) == (
        "not_promoted",
        "products_balanced_macro_or_class_gate_not_met",
    )
    assert products_promotion_status(method="products_stt_official", ratio=0.005, accuracy=0.799, macro_f1=0.39, predicted_classes=31) == (
        "not_promoted",
        "products_official_accuracy_gate_not_met",
    )


def test_t34_products_zero_predicted_classes_and_coverage_loss() -> None:
    pred = torch.tensor([0, 0, 2, 2])
    assert zero_predicted_classes(pred, num_classes=4) == 2
    probs = torch.tensor([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]], dtype=torch.float32)
    target = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32)
    assert class_coverage_loss(probs, target).item() > 0.0
