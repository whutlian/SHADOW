from __future__ import annotations

import torch

from shadow_hgc.features.scap_v2 import prior_center_dense


def test_scap_v2_prior_centering_uses_train_class_prior():
    dense = torch.tensor([[0.6, 0.4], [0.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, -1], dtype=torch.long)

    centered, diagnostics = prior_center_dense(dense, train_labels=labels, num_classes=2)

    assert torch.allclose(centered[0], torch.tensor([-1.0 / 15.0, 1.0 / 15.0]), atol=1e-6)
    assert diagnostics["class_prior"] == [2.0 / 3.0, 1.0 / 3.0]
