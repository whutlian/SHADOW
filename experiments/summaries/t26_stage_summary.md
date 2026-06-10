# T26 Stage Summary

## Scope

- T26 implements the contract, diagnostics, balanced trainer utilities, UCA utilities, and stage outputs requested by the attachment.
- Rows without fresh training are explicitly marked `ready_not_run`, `blocked_by_P0_gate`, or `blocked_by_teacher_gate`; no fabricated performance rows are promoted.
- T25 rows remain available as historical inputs but are not treated as default T26 promoted rows.

## Requirement Checklist

| requirement_check | requirement_status | notes |
|---|---|---|
| method_ids | completed | T26 product, Reddit, arxiv, and ultra method rows are present. |
| full_node_ratio | completed | Rows use full_node accounting from target_prototypes + shadow_nodes over original nodes. |
| forbidden_promoted_flags | completed | No promoted row may use logits, KD, dense P2, E x d, all-target ultra cache, or new exposed schema. |
| products_P0a | blocked | P0a all-train condensed-trainer parity is declared but not rerun; products performance rows remain blocked. |
| products_P0b | blocked | P0b selected-prototype self-fit is declared but not rerun; products performance rows remain blocked. |
| products_per_class_report | blocked | Per-class report schema is generated, but real collapse diagnostics require rerun P0 selection and predictions. |
| products_UCA | blocked | UCA sweep rows are generated but blocked by P0 gates; leakage flag is false. |
| reddit_seed_sweep | blocked | Seeds 1..5 are declared; missing actual runs stay not_promoted. |
| reddit_no_regression | completed | No Reddit row is promoted below the T24 0.50 reference. |
| arxiv_teacher_first | blocked | Arxiv condensation remains blocked until A1 >= 0.715. |
| ultra_contract | completed | Ultra rows are dry-run contract regressions with forbidden paths disabled. |
| machine_readable_outputs | completed | Every table also has a JSON sidecar from write_csv. |
| no_fabricated_results | completed | Rows without fresh experiments are explicitly ready_not_run/blocked and not promoted. |

## Aggregated Rows

| dataset | method | requested_full_node_ratio | seed | status | accuracy | macro_f1 | promotion_status | failure_reason | source_table |
|---|---|---|---|---|---|---|---|---|---|
| ogbn-products | P0a_alltrain_condensed_trainer_parity | 0.0025 | 42 | ready_not_run |  |  | not_promoted | P0a_condensed_trainer_parity_not_rerun | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0b_selected_prototype_self_fit | 0.0025 | 42 | ready_not_run |  |  | not_promoted | P0b_self_fit_not_rerun | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0c_same_budget_random_subset | 0.0025 | 42 | ready_not_run |  |  | not_promoted | P0c_random_subset_not_rerun | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0d_nearest_prototype_oracle | 0.0025 | 42 | ready_not_run |  |  | not_promoted | P0d_oracle_not_rerun | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0e_per_class_collapse_report | 0.0025 | 42 | ready_not_run |  |  | not_promoted | per_class_report_schema_written_waiting_for_real_selection_and_predictions | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0f_feature_normalization_parity | 0.0025 | 42 | completed_diagnostic |  |  | not_promoted | normalization_parity_from_existing_manifest | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0a_alltrain_condensed_trainer_parity | 0.005 | 42 | ready_not_run |  |  | not_promoted | P0a_condensed_trainer_parity_not_rerun | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0b_selected_prototype_self_fit | 0.005 | 42 | ready_not_run |  |  | not_promoted | P0b_self_fit_not_rerun | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0c_same_budget_random_subset | 0.005 | 42 | ready_not_run |  |  | not_promoted | P0c_random_subset_not_rerun | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0d_nearest_prototype_oracle | 0.005 | 42 | ready_not_run |  |  | not_promoted | P0d_oracle_not_rerun | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0e_per_class_collapse_report | 0.005 | 42 | ready_not_run |  |  | not_promoted | per_class_report_schema_written_waiting_for_real_selection_and_predictions | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0f_feature_normalization_parity | 0.005 | 42 | completed_diagnostic |  |  | not_promoted | normalization_parity_from_existing_manifest | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | products_cb_random | 0.0025 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_cb_kcenter | 0.0025 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_cb_herding | 0.0025 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_cb_hybrid | 0.0025 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_kmeans_labeled_nearest | 0.0025 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_hybrid | 0.0025 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_hybrid_mixup | 0.0025 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_hybrid_balanced_trainer | 0.0025 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_cb_random | 0.005 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_cb_kcenter | 0.005 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_cb_herding | 0.005 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_cb_hybrid | 0.005 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_kmeans_labeled_nearest | 0.005 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_hybrid | 0.005 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_hybrid_mixup | 0.005 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_hybrid_balanced_trainer | 0.005 | 42 | blocked_by_P0_gate |  |  | not_promoted | blocked_by_P0a_P0b_gate | experiments\tables\t26_products_uca_sweep_seed42.csv |
| Reddit | reddit_current_sft_signature_random | 0.005 | 1 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.005 | 2 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.005 | 3 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.005 | 4 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.005 | 5 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.005 | 1 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.005 | 2 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.005 | 3 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.005 | 4 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.005 | 5 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.005 | 1 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.005 | 2 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.005 | 3 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.005 | 4 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.005 | 5 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.005 | 1 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.005 | 2 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.005 | 3 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.005 | 4 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.005 | 5 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_tuned_balanced_trainer | 0.005 | 1 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_tuned_balanced_trainer | 0.005 | 2 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_tuned_balanced_trainer | 0.005 | 3 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_tuned_balanced_trainer | 0.005 | 4 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_tuned_balanced_trainer | 0.005 | 5 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_signature_mixup | 0.005 | 1 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_signature_mixup | 0.005 | 2 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_signature_mixup | 0.005 | 3 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_signature_mixup | 0.005 | 4 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_signature_mixup | 0.005 | 5 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_true_shadow_b1 | 0.005 | 1 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_true_shadow_b1 | 0.005 | 2 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_true_shadow_b1 | 0.005 | 3 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_true_shadow_b1 | 0.005 | 4 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_true_shadow_b1 | 0.005 | 5 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.01 | 1 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.01 | 2 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.01 | 3 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.01 | 4 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.01 | 5 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.01 | 1 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.01 | 2 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.01 | 3 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.01 | 4 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.01 | 5 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.01 | 1 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.01 | 2 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.01 | 3 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.01 | 4 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.01 | 5 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.01 | 1 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.01 | 2 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.01 | 3 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.01 | 4 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.01 | 5 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_tuned_balanced_trainer | 0.01 | 1 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_tuned_balanced_trainer | 0.01 | 2 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_tuned_balanced_trainer | 0.01 | 3 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_tuned_balanced_trainer | 0.01 | 4 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_tuned_balanced_trainer | 0.01 | 5 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_signature_mixup | 0.01 | 1 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_signature_mixup | 0.01 | 2 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_signature_mixup | 0.01 | 3 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_signature_mixup | 0.01 | 4 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_signature_mixup | 0.01 | 5 | ready_not_run |  |  | not_promoted | seed_not_run | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_true_shadow_b1 | 0.01 | 1 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_true_shadow_b1 | 0.01 | 2 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_true_shadow_b1 | 0.01 | 3 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_true_shadow_b1 | 0.01 | 4 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_true_shadow_b1 | 0.01 | 5 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| ogbn-arxiv | arxiv_teacher_sweep | 0.0 | 42 | completed_replay | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t26_arxiv_teacher_sweep_seed42.csv |
| ogbn-arxiv | arxiv_teacher_sweep | 0.0 | 42 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t26_arxiv_teacher_sweep_seed42.csv |
| ogbn-arxiv | arxiv_teacher_sweep | 0.0 | 42 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t26_arxiv_teacher_sweep_seed42.csv |
| ogbn-arxiv | arxiv_teacher_sweep | 0.0 | 42 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t26_arxiv_teacher_sweep_seed42.csv |
| ogbn-arxiv | arxiv_teacher_sweep | 0.0 | 42 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t26_arxiv_teacher_sweep_seed42.csv |
| ogbn-arxiv | arxiv_teacher_sweep | 0.0 | 42 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t26_arxiv_teacher_sweep_seed42.csv |
| ogbn-arxiv | arxiv_teacher_sweep | 0.0 | 42 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t26_arxiv_teacher_sweep_seed42.csv |
| ogbn-arxiv | arxiv_teacher_sweep | 0.0 | 42 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t26_arxiv_teacher_sweep_seed42.csv |
| ogbn-arxiv | arxiv_teacher_sweep | 0.0 | 42 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t26_arxiv_teacher_sweep_seed42.csv |
| ogbn-papers100M | t26_ultra_contract_regression | 0.0001 | 42 | completed_ultra_dryrun |  |  | not_promoted | ultra_performance_not_run | experiments\tables\t26_ultra_contract_regression_seed42.csv |
| MAG240M | t26_ultra_contract_regression | 0.0001 | 42 | completed_ultra_dryrun |  |  | not_promoted | ultra_performance_not_run | experiments\tables\t26_ultra_contract_regression_seed42.csv |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | blocked |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | blocked |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | blocked |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | blocked |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | blocked |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | blocked |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |

## Safety Summary

- Promoted rows: `0`
- Forbidden promoted rows: `0`
- All promoted rows safe: `True`
- Full-node ratio is preserved as `(target_prototypes + shadow_nodes) / original_num_nodes`.
- No logits input, KD, dense P2, legacy diffusion, full edge backprop, E x d materialization, full edge_index GPU path, source anchors, new exposed schema, or exact all-target ultra cache is promoted.

## Required Follow-Up Experiments

- Run products P0a/P0b with the condensed trainer path before unblocking product UCA rows.
- Run Reddit seeds 1..5 for the compact trainer/mixup grid before computing promoted mean/std rows.
- Improve arxiv teacher beyond A1 accuracy >= 0.715 before running condensation rows.

- Stage CSV: `experiments\tables\t26_stage_summary_seed42.csv`
