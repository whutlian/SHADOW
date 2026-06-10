# T26 Products Recovery Notes

- Product rows are blocked from promotion until P0a all-train condensed-trainer parity and P0b selected-prototype self-fit are run and pass.
- T25 replay rows are used only as source diagnostics; they are not relabeled as T26 promoted results.
- UCA leakage flag remains false for all generated rows.

## Diagnostics

| requested_full_node_ratio | method | status | p0a_passed | p0b_passed | p0f_normalization_parity | failure_reason |
|---|---|---|---|---|---|---|
| 0.0025 | P0a_alltrain_condensed_trainer_parity | ready_not_run | False | False | True | P0a_condensed_trainer_parity_not_rerun |
| 0.0025 | P0b_selected_prototype_self_fit | ready_not_run | False | False | True | P0b_self_fit_not_rerun |
| 0.0025 | P0c_same_budget_random_subset | ready_not_run | False | False | True | P0c_random_subset_not_rerun |
| 0.0025 | P0d_nearest_prototype_oracle | ready_not_run | False | False | True | P0d_oracle_not_rerun |
| 0.0025 | P0e_per_class_collapse_report | ready_not_run | False | False | True | per_class_report_schema_written_waiting_for_real_selection_and_predictions |
| 0.0025 | P0f_feature_normalization_parity | completed_diagnostic | False | False | True | normalization_parity_from_existing_manifest |
| 0.005 | P0a_alltrain_condensed_trainer_parity | ready_not_run | False | False | True | P0a_condensed_trainer_parity_not_rerun |
| 0.005 | P0b_selected_prototype_self_fit | ready_not_run | False | False | True | P0b_self_fit_not_rerun |
| 0.005 | P0c_same_budget_random_subset | ready_not_run | False | False | True | P0c_random_subset_not_rerun |
| 0.005 | P0d_nearest_prototype_oracle | ready_not_run | False | False | True | P0d_oracle_not_rerun |
| 0.005 | P0e_per_class_collapse_report | ready_not_run | False | False | True | per_class_report_schema_written_waiting_for_real_selection_and_predictions |
| 0.005 | P0f_feature_normalization_parity | completed_diagnostic | False | False | True | normalization_parity_from_existing_manifest |

## UCA Sweep

| requested_full_node_ratio | method | status | accuracy | macro_f1 | predicted_class_count | promotion_status | failure_reason |
|---|---|---|---|---|---|---|---|
| 0.0025 | products_cb_random | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.0025 | products_cb_kcenter | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.0025 | products_cb_herding | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.0025 | products_cb_hybrid | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.0025 | products_uca_kmeans_labeled_nearest | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.0025 | products_uca_hybrid | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.0025 | products_uca_hybrid_mixup | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.0025 | products_uca_hybrid_balanced_trainer | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.005 | products_cb_random | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.005 | products_cb_kcenter | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.005 | products_cb_herding | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.005 | products_cb_hybrid | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.005 | products_uca_kmeans_labeled_nearest | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.005 | products_uca_hybrid | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.005 | products_uca_hybrid_mixup | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |
| 0.005 | products_uca_hybrid_balanced_trainer | blocked_by_P0_gate |  |  |  | not_promoted | blocked_by_P0a_P0b_gate |

- Diagnostics CSV: `experiments\tables\t26_products_recovery_diagnostics_seed42.csv`
- UCA CSV: `experiments\tables\t26_products_uca_sweep_seed42.csv`
- Per-class CSV: `experiments\tables\t26_products_per_class_report_seed42.csv`
