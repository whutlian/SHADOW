from shadow_hgc.baselines.target_coreset import run_target_coreset_baselines
from shadow_hgc.data.loaders import build_toy_graph


def test_target_coreset_baselines_return_named_methods():
    rows = run_target_coreset_baselines(
        build_toy_graph(seed=0),
        seed=0,
        epochs=2,
        M_tau=4,
        feature_dim=4,
    )

    methods = {row["method"] for row in rows}
    assert {"Random-HG", "Herding-HG", "K-Center-HG"} <= methods
    assert all(row["status"] == "completed" for row in rows)
