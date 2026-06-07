from shadow_hgc.data.loaders import build_toy_graph
from shadow_hgc.pipeline.ablation import run_ablation_suite, write_skeleton_coverage_figure


def test_ablation_suite_runs_core_mechanism_rows(tmp_path):
    rows = run_ablation_suite(
        build_toy_graph(seed=0),
        log_dir=tmp_path,
        seed=0,
        epochs=1,
        M_tau=4,
        M_r=3,
        k_s=2,
        feature_dim=4,
        k_s_values=[0, 2],
    )

    completed = {row["ablation"] for row in rows if row["status"] == "completed"}
    assert "mean_only_demand" in completed
    assert "residual_shadow_off" in completed
    assert "real_source_centroid" in completed
    assert "target_target_skeleton" in completed
    assert all("shadow_recon_err_mean" in row for row in rows)

    csv_path = tmp_path / "skeleton_coverage_vs_accuracy.csv"
    svg_path = tmp_path / "skeleton_coverage_vs_accuracy.svg"
    write_skeleton_coverage_figure(rows, csv_path=csv_path, svg_path=svg_path)

    assert csv_path.exists()
    assert svg_path.exists()
    assert "Skeleton Coverage vs Accuracy" in svg_path.read_text()
