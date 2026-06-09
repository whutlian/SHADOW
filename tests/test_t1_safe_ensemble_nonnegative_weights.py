from shadow_hgc.logits.ensemble import nonnegative_grid_weights


def test_t1_safe_ensemble_grid_weights_are_nonnegative_simplex():
    weights = nonnegative_grid_weights(num_models=3, step=0.5)

    assert weights
    for row in weights:
        assert min(row) >= 0.0
        assert abs(sum(row) - 1.0) < 1e-9
