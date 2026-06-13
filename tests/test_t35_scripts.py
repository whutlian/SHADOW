from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.run_t35_papers100m_stt_stage import _server_commands
from t35_fixtures import make_toy_papers100m_root


def test_t35_summary_server_commands_use_parser_argument_names():
    commands = "\n".join(_server_commands(Path("caches/papers100m/stt_v1")))

    assert "--hidden-dim " not in commands
    assert "--soft-temperature" not in commands
    assert "--hidden-dims" in commands
    assert "--temperatures" in commands


def test_t35_stage_script_runs_toy_one_cache_ratio_flow(tmp_path):
    data_root = make_toy_papers100m_root(tmp_path)
    cache_root = tmp_path / "cache"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_t35_papers100m_stt_stage.py",
            "--data-root",
            str(data_root),
            "--cache-root",
            str(cache_root),
            "--stages",
            "manifest",
            "edge_cache",
            "sft_cache",
            "teacher_cache",
            "selection_bank",
            "ratios",
            "summarize",
            "--ratios",
            "0.25",
            "0.75",
            "--teacher-cache-mode",
            "topk2_tail",
            "--max-ratio",
            "0.75",
            "--build-cache-once",
            "--reuse-cache",
            "--allow-toy",
            "--tables-dir",
            str(tmp_path / "tables"),
            "--summaries-dir",
            str(tmp_path / "summaries"),
        ],
        cwd=".",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "status=completed" in result.stdout
    assert (tmp_path / "tables" / "t35_papers100m_ratio_curve.csv").exists()
    assert (tmp_path / "tables" / "t35_papers100m_cache_reuse_audit.csv").exists()
    assert (tmp_path / "summaries" / "t35_papers100m_one_cache_stt_summary.md").exists()
