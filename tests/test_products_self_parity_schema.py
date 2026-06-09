import pytest
import torch

from scripts.debug_products_self_parity import validate_products_self_parity_inputs


def test_products_self_parity_schema_accepts_squeezed_labels_and_ogb_evaluator():
    diagnostics = validate_products_self_parity_inputs(
        num_nodes=5,
        features=torch.randn(5, 3),
        labels=torch.tensor([[0], [1], [2], [1], [0]]),
        train_idx=torch.tensor([0, 1]),
        valid_idx=torch.tensor([2]),
        test_idx=torch.tensor([3, 4]),
        output_dim=3,
        num_classes_from_dataset=3,
        uses_ogb_evaluator=True,
        uses_bounded_edges=False,
    )

    assert diagnostics["label_shape"] == [5]
    assert diagnostics["label_min"] == 0
    assert diagnostics["label_max"] == 2
    assert diagnostics["output_dim"] == 3
    assert diagnostics["uses_ogb_evaluator"] is True
    assert diagnostics["uses_bounded_edges"] is False


def test_products_self_parity_schema_rejects_wrong_output_dim():
    with pytest.raises(ValueError, match="output_dim"):
        validate_products_self_parity_inputs(
            num_nodes=4,
            features=torch.randn(4, 2),
            labels=torch.tensor([0, 1, 2, 0]),
            train_idx=torch.tensor([0, 1]),
            valid_idx=torch.tensor([2]),
            test_idx=torch.tensor([3]),
            output_dim=2,
            num_classes_from_dataset=3,
            uses_ogb_evaluator=True,
            uses_bounded_edges=False,
        )


def test_products_self_parity_schema_rejects_overlapping_splits():
    with pytest.raises(ValueError, match="overlap"):
        validate_products_self_parity_inputs(
            num_nodes=4,
            features=torch.randn(4, 2),
            labels=torch.tensor([0, 1, 2, 0]),
            train_idx=torch.tensor([0, 1]),
            valid_idx=torch.tensor([1, 2]),
            test_idx=torch.tensor([3]),
            output_dim=3,
            num_classes_from_dataset=3,
            uses_ogb_evaluator=True,
            uses_bounded_edges=False,
        )
