# T26 Products Recovery Notes

- Product rows are blocked from promotion until P0a all-train condensed-trainer parity and P0b selected-prototype self-fit are run and pass.
- T25 replay rows are used only as source diagnostics; they are not relabeled as T26 promoted results.
- UCA leakage flag remains false for all generated rows.

## Diagnostics

| requested_full_node_ratio | method | status | p0a_passed | p0b_passed | p0f_normalization_parity | failure_reason |
|---|---|---|---|---|---|---|
| 0.0025 | P0a_alltrain_condensed_trainer_parity | completed_long | True | True | True |  |
| 0.0025 | P0b_selected_prototype_self_fit | completed_long | True | True | True |  |
| 0.0025 | P0c_same_budget_random_subset | completed_long | True | True | True |  |
| 0.0025 | P0d_nearest_prototype_oracle | ready_not_run | True | True | True | P0d_oracle_not_rerun |
| 0.0025 | P0e_per_class_collapse_report | completed_long_class_collapse_report | True | True | True |  |
| 0.0025 | P0f_feature_normalization_parity | completed_diagnostic | True | True | True | normalization_parity_from_existing_manifest |
| 0.005 | P0a_alltrain_condensed_trainer_parity | completed_long | True | True | True |  |
| 0.005 | P0b_selected_prototype_self_fit | completed_long | True | True | True |  |
| 0.005 | P0c_same_budget_random_subset | completed_long | True | True | True |  |
| 0.005 | P0d_nearest_prototype_oracle | ready_not_run | True | True | True | P0d_oracle_not_rerun |
| 0.005 | P0e_per_class_collapse_report | completed_long_class_collapse_report | True | True | True |  |
| 0.005 | P0f_feature_normalization_parity | completed_diagnostic | True | True | True | normalization_parity_from_existing_manifest |

## UCA Sweep

| requested_full_node_ratio | method | status | accuracy | macro_f1 | predicted_class_count | promotion_status | failure_reason |
|---|---|---|---|---|---|---|---|
| 0.0025 | products_cb_random | completed_long | 0.6923746018577637 | 0.3701378071453978 | 42 | not_promoted |  |
| 0.0025 | products_cb_kcenter | completed_long | 0.5488558762382568 | 0.32516153150902893 | 42 | not_promoted |  |
| 0.0025 | products_cb_herding | completed_long | 0.5919164643478284 | 0.3246630006894027 | 42 | not_promoted |  |
| 0.0025 | products_cb_hybrid | completed_long | 0.6206635877151008 | 0.3257283168095639 | 42 | not_promoted |  |
| 0.0025 | products_uca_kmeans_labeled_nearest | completed_long | 0.6887335405548167 | 0.3521801234630918 | 30 | not_promoted |  |
| 0.0025 | products_uca_hybrid | completed_long | 0.6887335405548167 | 0.3521801234630918 | 30 | not_promoted |  |
| 0.0025 | products_uca_hybrid_mixup | completed_long | 0.746393166842213 | 0.3791035690285768 | 30 | not_promoted |  |
| 0.0025 | products_uca_hybrid_balanced_trainer | completed_long | 0.6404702743809451 | 0.344996145725478 | 30 | not_promoted |  |
| 0.005 | products_cb_random | completed_long | 0.708108252213759 | 0.3708838171007395 | 42 | not_promoted |  |
| 0.005 | products_cb_kcenter | completed_long | 0.6156158964995113 | 0.3508466147700994 | 42 | not_promoted |  |
| 0.005 | products_cb_herding | completed_long | 0.6271585759464929 | 0.33918849346898744 | 42 | not_promoted |  |
| 0.005 | products_cb_hybrid | completed_long | 0.6708919786850157 | 0.3441880882005497 | 42 | not_promoted |  |
| 0.005 | products_uca_kmeans_labeled_nearest | completed_long | 0.7110999954362474 | 0.3640408887290348 | 31 | not_promoted |  |
| 0.005 | products_uca_hybrid | completed_long | 0.7110999954362474 | 0.3640408887290348 | 31 | not_promoted |  |
| 0.005 | products_uca_hybrid_mixup | completed_long | 0.767075099939406 | 0.3891223434748316 | 31 | not_promoted |  |
| 0.005 | products_uca_hybrid_balanced_trainer | completed_long | 0.6712656641773881 | 0.34492475606820144 | 31 | not_promoted |  |

- Diagnostics CSV: `experiments\tables\t26_products_recovery_diagnostics_seed42.csv`
- UCA CSV: `experiments\tables\t26_products_uca_sweep_seed42.csv`
- Per-class CSV: `experiments\tables\t26_products_per_class_report_seed42.csv`
