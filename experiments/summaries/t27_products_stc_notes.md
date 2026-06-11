# T27 Products STC Notes

- Required Products STC rows are declared at 0.25% and 0.50% full-node ratios.
- Local rows are smoke/server-ready unless a full run fills `accuracy`, `macro_f1`, and `predicted_classes`.
- No row is promoted from smoke output; no teacher logits, KD, dense P2, E-by-d, full edge GPU, valid labels, or test labels are used.

| requested_full_node_ratio | method | status | stc_objective | stc_delta_rho | promotion_status | failure_reason |
|---|---|---|---|---|---|---|
| 0.0025 | products_uca_mixup_frozen | completed_long | frozen_init |  | not_promoted | products_gate_not_met |
| 0.0025 | products_uca_mixup_trainable_delta_rho005 | completed_long | trainable_delta | 0.05 | not_promoted | products_gate_not_met |
| 0.0025 | products_uca_mixup_trainable_delta_rho010 | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |
| 0.0025 | products_uca_mixup_gm | completed_long | gradient_matching | 0.1 | not_promoted | products_gate_not_met |
| 0.0025 | products_uca_mixup_outer | completed_long | outer_loop | 0.1 | not_promoted | products_gate_not_met |
| 0.0025 | products_uca_mixup_outer_plus_coverage_official | completed_long | outer_loop_plus_coverage | 0.1 | not_promoted | products_gate_not_met |
| 0.0025 | products_uca_mixup_outer_plus_coverage_balanced | completed_long | outer_loop_plus_coverage | 0.1 | not_promoted | products_gate_not_met |
| 0.0025 | products_random_trainable_delta | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |
| 0.0025 | products_cb_random_trainable_delta | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |
| 0.005 | products_uca_mixup_frozen | completed_long | frozen_init |  | not_promoted | products_gate_not_met |
| 0.005 | products_uca_mixup_trainable_delta_rho005 | completed_long | trainable_delta | 0.05 | not_promoted | products_gate_not_met |
| 0.005 | products_uca_mixup_trainable_delta_rho010 | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |
| 0.005 | products_uca_mixup_gm | completed_long | gradient_matching | 0.1 | not_promoted | products_gate_not_met |
| 0.005 | products_uca_mixup_outer | completed_long | outer_loop | 0.1 | not_promoted | products_gate_not_met |
| 0.005 | products_uca_mixup_outer_plus_coverage_official | completed_long | outer_loop_plus_coverage | 0.1 | not_promoted | products_gate_not_met |
| 0.005 | products_uca_mixup_outer_plus_coverage_balanced | completed_long | outer_loop_plus_coverage | 0.1 | not_promoted | products_gate_not_met |
| 0.005 | products_random_trainable_delta | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |
| 0.005 | products_cb_random_trainable_delta | completed_long | trainable_delta | 0.1 | not_promoted | products_gate_not_met |

- CSV: `experiments\tables\t27_stc_products_seed42.csv`
- Full server command: `python scripts/run_t27_stc_products.py --device cuda --ratios 0.0025 0.005 --init products_uca_hybrid_mixup --methods frozen_init trainable_delta gradient_matching outer_loop outer_loop_plus_coverage --products-coverage-track official balanced --delta-rhos 0.05 0.10 0.20 --stc-outer-steps 1000 --run-long --seed 42`
