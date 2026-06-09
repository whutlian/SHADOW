from __future__ import annotations

import torch

from shadow_hgc.features.scap import prior_center_scap


def test_scap_prior_centering_preserves_mass_and_uses_train_prior():
    block = torch.tensor([[0.6, 0.4], [0.0, 0.0]], dtype=torch.float32)
    train_labels = torch.tensor([0, 0, 1, -1], dtype=torch.long)

    centered, meta = prior_center_scap(block, train_labels=train_labels, num_classes=2)

    assert torch.allclose(centered[0], torch.tensor([-1.0 / 15.0, 1.0 / 15.0]), atol=1e-6)
    assert torch.allclose(centered[1], torch.zeros(2), atol=1e-6)
    assert meta["prior_centering"] is True
    assert meta["train_class_prior"] == [2.0 / 3.0, 1.0 / 3.0]
