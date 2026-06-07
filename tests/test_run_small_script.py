import subprocess
import sys


def test_run_small_script_writes_main_and_ablation_tables(tmp_path):
    output = tmp_path / "small_main.csv"
    ablation = tmp_path / "small_ablation.csv"
    figure_csv = tmp_path / "fig.csv"
    figure_svg = tmp_path / "fig.svg"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_small.py",
            "--datasets",
            "acm",
            "--seeds",
            "0",
            "--epochs",
            "1",
            "--M-tau-values",
            "4",
            "--M-r",
            "2",
            "--feature-dim",
            "4",
            "--output",
            str(output),
            "--ablation-output",
            str(ablation),
            "--figure-csv",
            str(figure_csv),
            "--figure-svg",
            str(figure_svg),
        ],
        cwd=".",
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "Shadow-HGC-R-1" in output.read_text()
    assert "pending_real_dataset_run" not in ablation.read_text()
    assert figure_svg.exists()
