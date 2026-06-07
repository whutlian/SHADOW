import json

from shadow_hgc.pipeline.toy import run_toy_experiment


def test_toy_pipeline_runs_end_to_end_and_logs_required_diagnostics(tmp_path):
    summary_path = tmp_path / "toy_summary.json"

    summary = run_toy_experiment(
        output_path=summary_path,
        seed=5,
        epochs=5,
        M_tau=4,
        M_r=3,
        k_s=2,
    )

    on_disk = json.loads(summary_path.read_text())
    assert on_disk["method"] == "Shadow-HGC-R-1"
    assert on_disk["dataset"] == "toy"
    assert "paper--cite_ref-->paper" in on_disk["diagnostics"]
    assert "ResidualEnergy" in on_disk["diagnostics"]["paper--cite_ref-->paper"]
    assert on_disk["schema_preserved"] is True
    assert on_disk["all_edge_weights_nonnegative"] is True
    assert on_disk["residual_shadows_signed"] is True
    assert on_disk["peak_cpu_ram"] > 0
    assert on_disk["peak_gpu_ram"] >= 0
    assert on_disk["macro_f1"] is not None
    assert on_disk["num_full_edge_scans"] == 0
    assert summary["accuracy"] >= 0.0
