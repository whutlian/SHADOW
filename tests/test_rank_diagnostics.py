import math

import torch

from shadow_hgc.diagnostics.rank import relation_rank_diagnostics


def test_rank_diagnostics_are_finite_on_tiny_matrix():
    matrix = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

    diag = relation_rank_diagnostics(matrix)

    assert math.isfinite(diag["stable_rank"])
    assert math.isfinite(diag["entropy_effective_rank"])
    assert diag["stable_rank"] > 0.0
    assert diag["entropy_effective_rank"] > 0.0
    assert diag["relation_demand_norm_q995"] >= diag["relation_demand_norm_median"]
