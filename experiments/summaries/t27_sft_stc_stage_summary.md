# T27 SFT-STC Stage Summary

## Files Changed

- `docs/superpowers/plans/2026-06-11-t27-sft-stc.md`
- `shadow_hgc/sft/stc.py`
- `shadow_hgc/sft/stc_contract.py`
- `shadow_hgc/sft/stc_init.py`
- `shadow_hgc/sft/stc_losses.py`
- `shadow_hgc/sft/stc_trainer.py`
- `shadow_hgc/sft/timeaware_arxiv.py`
- `scripts/run_t27_stc_products.py`
- `scripts/run_t27_stc_reddit.py`
- `scripts/run_t27_arxiv_teacher_pivot.py`
- `scripts/run_t27_stage.py`
- `tests/test_t27_stc_core.py`
- `tests/test_t27_scripts.py`

## Method Names And Flags

- New main family: `sft_stc_frozen_init`, `sft_stc_trainable_delta`, `sft_stc_gradient_matching`, `sft_stc_outer_loop`, `sft_stc_outer_loop_plus_coverage`, `sft_stc_gm_plus_coverage`.
- Structure-free accounting: `ratio_mode=full_node`, `target_prototypes=syn_rows`, `shadow_nodes=0`, `condensed_edges=0`.
- Forbidden promoted flags: `uses_logits_as_input`, `uses_teacher_logits`, `uses_kd`, `uses_dense_p2`, `uses_e_by_d_materialization`, `uses_full_edge_index_on_gpu`, `uses_valid_labels`, `uses_test_labels`.
- T25/T26 HNR/FDM methods demoted to diagnostic: sft_hnr_random, sft_hnr_fdm_herding, sft_hnr_fdm_kcenter, sft_hnr_fdm_hybrid, sft_hnr_fdm_shadow_b1, sft_hnr_fdm_shadow_b2.

## Tests

- Verification result: `full pytest: 357 passed in 83.77s; targeted T27/non-regression: 32 passed`
- Added tests: `tests/test_t27_stc_core.py`, `tests/test_t27_scripts.py`.

## Requirement Checklist

| requirement_check | requirement_status | notes |
|---|---|---|
| t27_schema | completed | Every generated row is written with the T27 required field list. |
| stc_structure_free_ratio | completed | Rows use ratio_mode=full_node with shadow_nodes=0 and condensed_edges=0. |
| hnr_fdm_demoted | completed | T25/T26 HNR/FDM methods are diagnostic/non-main and not promoted by default. |
| forbidden_promoted_flags | completed | No promoted row may use logits, KD, dense P2, E-by-d, full edge GPU, valid labels, or test labels. |
| products_required_rows | completed | Products required STC method grid is present for 0.25% and 0.50%. |
| reddit_required_rows | completed | Reddit required STC method grid is present for 0.50% and 1.00%. |
| arxiv_teacher_pivot_rows | completed | Arxiv teacher-pivot rows are present and condensation remains gate-controlled. |
| no_fabricated_full_results | completed | Smoke/server-ready rows do not claim full dataset metrics or promotion. |
| performance_regression_guard | completed | No T27 row is promoted below dataset gates; smoke rows are explicitly not promoted. |

## Experiments And Outputs

| dataset | method | requested_full_node_ratio | seed | status | accuracy | macro_f1 | predicted_classes | promotion_status | failure_reason | source_table |
|---|---|---|---|---|---|---|---|---|---|---|
| ogbn-products | products_uca_mixup_frozen | 0.0025 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_trainable_delta_rho005 | 0.0025 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_trainable_delta_rho010 | 0.0025 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_gm | 0.0025 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_outer | 0.0025 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_outer_plus_coverage_official | 0.0025 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_outer_plus_coverage_balanced | 0.0025 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_random_trainable_delta | 0.0025 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_cb_random_trainable_delta | 0.0025 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_frozen | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_trainable_delta_rho005 | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_trainable_delta_rho010 | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_gm | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_outer | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_outer_plus_coverage_official | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_outer_plus_coverage_balanced | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_random_trainable_delta | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_cb_random_trainable_delta | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_products_run | experiments/tables/t27_stc_products_seed42.csv |
| Reddit | reddit_random_frozen_init | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho005 | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho010 | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_outer | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm_plus_moment | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_kcenter_trainable_delta | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_medoid_trainable_delta | 0.005 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_frozen_init | 0.01 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho005 | 0.01 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho010 | 0.01 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm | 0.01 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_outer | 0.01 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm_plus_moment | 0.01 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_kcenter_trainable_delta | 0.01 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_medoid_trainable_delta | 0.01 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_not_full_reddit_run | experiments/tables/t27_stc_reddit_seed42.csv |
| ogbn-arxiv | arxiv_timeaware_sft_v5_h512 | 0.0 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_teacher_pivot_not_full_run | experiments/tables/t27_arxiv_teacher_pivot_seed42.csv |
| ogbn-arxiv | arxiv_timeaware_sft_v5_h768 | 0.0 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_teacher_pivot_not_full_run | experiments/tables/t27_arxiv_teacher_pivot_seed42.csv |
| ogbn-arxiv | arxiv_timeaware_sft_v5_decay_gamma005 | 0.0 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_teacher_pivot_not_full_run | experiments/tables/t27_arxiv_teacher_pivot_seed42.csv |
| ogbn-arxiv | arxiv_timeaware_sft_v5_decay_gamma010 | 0.0 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_teacher_pivot_not_full_run | experiments/tables/t27_arxiv_teacher_pivot_seed42.csv |
| ogbn-arxiv | arxiv_correct_smooth_no_logits | 0.0 | 42 | completed_smoke |  |  |  | not_promoted | local_smoke_teacher_pivot_not_full_run | experiments/tables/t27_arxiv_teacher_pivot_seed42.csv |
| ogbn-arxiv | arxiv_gnn_teacher_upper_bound | 0.0 | 42 | completed_smoke |  |  |  | upper_bound_diagnostic | upper_bound_diagnostic_not_promoted | experiments/tables/t27_arxiv_teacher_pivot_seed42.csv |
| ogbn-arxiv | arxiv_t26_best_teacher_reference | 0.0 | 42 | completed_reference | 0.706828796576343 | 0.5045803241909133 | 39 | not_promoted | arxiv_teacher_below_0.715 | experiments/tables/t27_arxiv_teacher_pivot_seed42.csv |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |

## Promotion Decision

- Promoted rows: `0`.
- Forbidden promoted rows: `0`.
- T27 remains implemented and smoke/server-ready, but no full Products/Reddit STC row is promoted from local smoke output.
- Arxiv STC remains blocked until teacher A1 accuracy >= 0.715.

## CSV Paths

- `experiments/tables/t27_stc_products_seed42.csv`
- `experiments/tables/t27_stc_reddit_seed42.csv`
- `experiments/tables/t27_arxiv_teacher_pivot_seed42.csv`
- `experiments/tables/t27_stage_summary_seed42.csv`

## Next Server Commands

```powershell
python scripts/run_t27_stc_products.py --device cuda --ratios 0.0025 0.005 --init products_uca_hybrid_mixup --methods frozen_init trainable_delta gradient_matching outer_loop outer_loop_plus_coverage --products-coverage-track official balanced --delta-rhos 0.05 0.10 0.20 --stc-outer-steps 1000 --seed 42
python scripts/run_t27_stc_reddit.py --device cuda --ratios 0.005 0.01 --init current_sft_signature_random --methods frozen_init trainable_delta gradient_matching outer_loop --delta-rhos 0.05 0.10 --seeds 1 2 3 4 5
python scripts/run_t27_arxiv_teacher_pivot.py --device cuda --variants year_features temporal_decay temporal_decay_year residual_no_logits --hidden-dims 512 768 --temporal-decay-gammas 0.05 0.10 --seed 42
python scripts/run_t27_stage.py
```

- Stage CSV: `experiments\tables\t27_stage_summary_seed42.csv`
