from pathlib import Path

from shadow_hgc.sft.stc_contract import T25_T26_DIAGNOSTIC_METHODS


def test_products_required_methods_and_server_command_are_declared():
    import scripts.run_t27_stc_products as products_runner
    from scripts.run_t27_stc_products import REQUIRED_PRODUCTS_METHODS, build_products_server_command

    assert "products_uca_mixup_frozen" in REQUIRED_PRODUCTS_METHODS
    assert "products_uca_mixup_trainable_delta_rho005" in REQUIRED_PRODUCTS_METHODS
    assert "products_uca_mixup_outer_plus_coverage_balanced" in REQUIRED_PRODUCTS_METHODS
    command = build_products_server_command(seed=42)
    assert "scripts/run_t27_stc_products.py" in command
    assert "--ratios 0.0025 0.005" in command
    assert "--methods frozen_init trainable_delta gradient_matching outer_loop outer_loop_plus_coverage" in command
    assert "--run-long" in command
    assert hasattr(products_runner, "run_products_long")


def test_reddit_required_methods_and_seed_commands_are_declared():
    import scripts.run_t27_stc_reddit as reddit_runner
    from scripts.run_t27_stc_reddit import REQUIRED_REDDIT_METHODS, build_reddit_server_command

    assert "reddit_random_frozen_init" in REQUIRED_REDDIT_METHODS
    assert "reddit_random_gm_plus_moment" in REQUIRED_REDDIT_METHODS
    assert "reddit_medoid_trainable_delta" in REQUIRED_REDDIT_METHODS
    command = build_reddit_server_command(seeds=[1, 2, 3, 4, 5])
    assert "--seeds 1 2 3 4 5" in command
    assert "--ratios 0.005 0.01" in command
    assert "--run-long" in command
    assert hasattr(reddit_runner, "run_reddit_long")


def test_arxiv_required_variants_and_gate_command_are_declared():
    import scripts.run_t27_arxiv_teacher_pivot as arxiv_runner
    from scripts.run_t27_arxiv_teacher_pivot import REQUIRED_ARXIV_VARIANTS, build_arxiv_server_command

    assert "arxiv_timeaware_sft_v5_h512" in REQUIRED_ARXIV_VARIANTS
    assert "arxiv_correct_smooth_no_logits" in REQUIRED_ARXIV_VARIANTS
    command = build_arxiv_server_command(seed=42)
    assert "--variants year_features temporal_decay temporal_decay_year residual_no_logits" in command
    assert "--temporal-decay-gammas 0.05 0.10" in command
    assert "--run-long" in command
    assert hasattr(arxiv_runner, "run_arxiv_long")


def test_t27_stage_declares_required_outputs_and_demotes_hnr_fdm():
    from scripts.run_t27_stage import REQUIRED_OUTPUTS, build_next_server_commands

    assert "sft_hnr_fdm_hybrid" in T25_T26_DIAGNOSTIC_METHODS
    assert "experiments/tables/t27_stc_products_seed42.csv" in REQUIRED_OUTPUTS
    assert "experiments/tables/t27_stc_reddit_seed42.csv" in REQUIRED_OUTPUTS
    assert "experiments/tables/t27_arxiv_teacher_pivot_seed42.csv" in REQUIRED_OUTPUTS
    assert "experiments/summaries/t27_sft_stc_stage_summary.md" in REQUIRED_OUTPUTS
    commands = build_next_server_commands()
    assert "run_t27_stc_products.py" in "\n".join(commands)
    assert "run_t27_stc_reddit.py" in "\n".join(commands)
    assert "run_t27_arxiv_teacher_pivot.py" in "\n".join(commands)


def test_t27_stage_summary_paths_are_t27_only():
    from scripts.run_t27_stage import REQUIRED_OUTPUTS

    for output in REQUIRED_OUTPUTS:
        path = Path(output)
        assert "t27" in path.name
