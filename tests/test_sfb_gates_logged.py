from __future__ import annotations

import torch

from shadow_hgc.fullgraph.sfb_model import BlockGatedResidualTableModel


def test_sfb_gates_are_positive_and_logged_by_block_name():
    model = BlockGatedResidualTableModel({"self": 3, "scap:PAP": 4}, num_classes=2, hidden_dim=4)
    blocks = {"self": torch.randn(3, 3), "scap:PAP": torch.randn(3, 4)}
    model.fit_block_stats(blocks, source="train_target_rows")
    model.freeze_block_stats()
    _ = model(blocks)

    gates = model.block_gate_values()

    assert set(gates) == {"self", "scap:PAP"}
    assert all(value > 0.0 for value in gates.values())
    assert model.diagnostics()["block_gates"] == gates
