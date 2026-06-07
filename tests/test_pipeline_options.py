import json

from shadow_hgc.data.loaders import build_toy_graph
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


def test_core_pipeline_logs_ablation_options_and_real_resource_fields(tmp_path):
    graph = build_toy_graph(seed=0)

    main = run_shadow_hgc_experiment(
        graph,
        output_path=tmp_path / "main.json",
        seed=0,
        epochs=2,
        M_tau=4,
        M_r=3,
        k_s=2,
        feature_dim=4,
    )
    mean_only = run_shadow_hgc_experiment(
        graph,
        output_path=tmp_path / "mean_only.json",
        seed=0,
        epochs=2,
        M_tau=4,
        M_r=3,
        k_s=2,
        feature_dim=4,
        include_degree_features=False,
        residual_shadow=False,
        shadow_mode="real_source_centroid",
        loss_type="unweighted",
        calibration_enabled=False,
    )

    on_disk = json.loads((tmp_path / "mean_only.json").read_text())
    assert on_disk["ablation"]["include_degree_features"] is False
    assert on_disk["ablation"]["residual_shadow"] is False
    assert on_disk["ablation"]["shadow_mode"] == "real_source_centroid"
    assert on_disk["ablation"]["loss_type"] == "unweighted"
    assert on_disk["peak_cpu_ram"] > 0
    assert on_disk["peak_gpu_ram"] >= 0
    assert main["target_input_dim"] > mean_only["target_input_dim"]
