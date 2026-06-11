# T27 Products STC Tiny-Ratio Notes

- This run covers additional ogbn-products T27 SFT-STC long rows at 0.02%, 0.04%, and 0.08% full-node ratios.
- All local rows are real long runs with `accuracy`, `macro_f1`, and `predicted_classes` filled.
- No row is promoted; no teacher logits, KD, dense P2, E-by-d, full edge GPU, valid labels, or test labels are used.

| requested_full_node_ratio | method | status | stc_objective | stc_delta_rho | promotion_status | failure_reason |
|---|---|---|---|---|---|---|
| 0.0002 | products_uca_mixup_frozen | completed_long | frozen_init |  | not_promoted | products_gate_not_met |
| 0.0002 | products_uca_mixup_trainable_delta_rho005 | completed_long | trainable_delta | 0.05 | not_promoted | products_gate_not_met |
| 0.0002 | products_uca_mixup_trainable_delta_rho010 | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |
| 0.0002 | products_uca_mixup_gm | completed_long | gradient_matching | 0.1 | not_promoted | products_gate_not_met |
| 0.0002 | products_uca_mixup_outer | completed_long | outer_loop | 0.1 | not_promoted | products_gate_not_met |
| 0.0002 | products_uca_mixup_outer_plus_coverage_official | completed_long | outer_loop_plus_coverage | 0.1 | not_promoted | products_gate_not_met |
| 0.0002 | products_uca_mixup_outer_plus_coverage_balanced | completed_long | outer_loop_plus_coverage | 0.1 | not_promoted | products_gate_not_met |
| 0.0002 | products_random_trainable_delta | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |
| 0.0002 | products_cb_random_trainable_delta | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |
| 0.0004 | products_uca_mixup_frozen | completed_long | frozen_init |  | not_promoted | products_gate_not_met |
| 0.0004 | products_uca_mixup_trainable_delta_rho005 | completed_long | trainable_delta | 0.05 | not_promoted | products_gate_not_met |
| 0.0004 | products_uca_mixup_trainable_delta_rho010 | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |
| 0.0004 | products_uca_mixup_gm | completed_long | gradient_matching | 0.1 | not_promoted | products_gate_not_met |
| 0.0004 | products_uca_mixup_outer | completed_long | outer_loop | 0.1 | not_promoted | products_gate_not_met |
| 0.0004 | products_uca_mixup_outer_plus_coverage_official | completed_long | outer_loop_plus_coverage | 0.1 | not_promoted | products_gate_not_met |
| 0.0004 | products_uca_mixup_outer_plus_coverage_balanced | completed_long | outer_loop_plus_coverage | 0.1 | not_promoted | products_gate_not_met |
| 0.0004 | products_random_trainable_delta | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |
| 0.0004 | products_cb_random_trainable_delta | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |
| 0.0008 | products_uca_mixup_frozen | completed_long | frozen_init |  | not_promoted | products_gate_not_met |
| 0.0008 | products_uca_mixup_trainable_delta_rho005 | completed_long | trainable_delta | 0.05 | not_promoted | products_gate_not_met |
| 0.0008 | products_uca_mixup_trainable_delta_rho010 | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |
| 0.0008 | products_uca_mixup_gm | completed_long | gradient_matching | 0.1 | not_promoted | products_gate_not_met |
| 0.0008 | products_uca_mixup_outer | completed_long | outer_loop | 0.1 | not_promoted | products_gate_not_met |
| 0.0008 | products_uca_mixup_outer_plus_coverage_official | completed_long | outer_loop_plus_coverage | 0.1 | not_promoted | products_gate_not_met |
| 0.0008 | products_uca_mixup_outer_plus_coverage_balanced | completed_long | outer_loop_plus_coverage | 0.1 | not_promoted | products_gate_not_met |
| 0.0008 | products_random_trainable_delta | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |
| 0.0008 | products_cb_random_trainable_delta | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |

## Best Rows By Ratio

| ratio_percent | requested_full_node_ratio | best_method | syn_rows | accuracy | macro_f1 | predicted_classes | promotion_status |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.0002 | products_uca_mixup_trainable_delta_rho010 | 490 | 0.6858000868 | 0.3094500395 | 22 | not_promoted |
| 0.04 | 0.0004 | products_uca_mixup_frozen | 980 | 0.7000873439 | 0.3283601128 | 27 | not_promoted |
| 0.08 | 0.0008 | products_uca_mixup_frozen | 1959 | 0.7204511699 | 0.3483658099 | 27 | not_promoted |

- CSV: `experiments/tables/t27_stc_products_0p02_0p04_0p08_percent_seed42.csv`
- Full command: `C:\Users\slian\anaconda3\envs\pytorch\python.exe scripts/run_t27_stc_products.py --run-long --device cuda --stc-device cuda --ratios 0.0002 0.0004 0.0008 --stc-outer-steps 40 --stc-real-subset-size 2048 --stc-real-batch-size 2048 --gm-real-batch-size 2048 --final-epochs 80 --final-hidden-dim 128 --seed 42 --csv experiments/tables/t27_stc_products_0p02_0p04_0p08_percent_seed42.csv --report experiments/summaries/t27_products_stc_0p02_0p04_0p08_percent_notes.md`
