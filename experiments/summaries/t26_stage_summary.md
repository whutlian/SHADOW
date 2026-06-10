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
| products_P0a | completed | P0a all-train condensed-trainer parity has a passing long run. |
| products_P0b | completed | P0b selected-prototype self-fit has passing long runs for requested ratios. |
| products_per_class_report | completed | Per-class collapse report is built from real selected class counts and test prediction counts. |
| products_UCA | completed | Products UCA/CB method-level rows include real trained long-experiment metrics and keep valid/test-label selection disabled. |
| reddit_seed_sweep | completed | Required current/HNR-FDM seed sweeps have real rows for seeds 1..5; tuned/mixup/true-shadow rows remain separate diagnostics. |
| reddit_no_regression | completed | No Reddit row is promoted below the T24 0.50 reference. |
| arxiv_teacher_first | blocked | Arxiv condensation remains blocked until A1 >= 0.715. |
| ultra_contract | completed | Ultra rows are dry-run contract regressions with forbidden paths disabled. |
| machine_readable_outputs | completed | Every table also has a JSON sidecar from write_csv. |
| no_fabricated_results | completed | Rows without fresh experiments are explicitly ready_not_run/blocked and not promoted. |

## Aggregated Rows

| dataset | method | requested_full_node_ratio | seed | status | accuracy | macro_f1 | promotion_status | failure_reason | source_table |
|---|---|---|---|---|---|---|---|---|---|
| ogbn-products | P0a_alltrain_condensed_trainer_parity | 0.0025 | 42 | completed_long | 0.7567198999047035 | 0.40133336132566916 | not_promoted |  | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0b_selected_prototype_self_fit | 0.0025 | 42 | completed_long | 0.9858970154148902 | 0.8774285379174651 | not_promoted |  | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0c_same_budget_random_subset | 0.0025 | 42 | completed_long | 0.6782802876158278 | 0.36722622784014924 | not_promoted |  | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0d_nearest_prototype_oracle | 0.0025 | 42 | ready_not_run |  |  | not_promoted | P0d_oracle_not_rerun | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0e_per_class_collapse_report | 0.0025 | 42 | completed_long_class_collapse_report | 0.6404702743809451 | 0.344996145725478 | not_promoted |  | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0f_feature_normalization_parity | 0.0025 | 42 | completed_diagnostic |  |  | not_promoted | normalization_parity_from_existing_manifest | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0a_alltrain_condensed_trainer_parity | 0.005 | 42 | completed_long | 0.7567198999047035 | 0.40133336132566916 | not_promoted |  | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0b_selected_prototype_self_fit | 0.005 | 42 | completed_long | 0.9897917181197003 | 0.8573811287452714 | not_promoted |  | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0c_same_budget_random_subset | 0.005 | 42 | completed_long | 0.7213923873894025 | 0.3795675686670795 | not_promoted |  | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0d_nearest_prototype_oracle | 0.005 | 42 | ready_not_run |  |  | not_promoted | P0d_oracle_not_rerun | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0e_per_class_collapse_report | 0.005 | 42 | completed_long_class_collapse_report | 0.6712656641773881 | 0.34492475606820144 | not_promoted |  | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | P0f_feature_normalization_parity | 0.005 | 42 | completed_diagnostic |  |  | not_promoted | normalization_parity_from_existing_manifest | experiments\tables\t26_products_recovery_diagnostics_seed42.csv |
| ogbn-products | products_cb_random | 0.0025 | 42 | completed_long | 0.6923746018577637 | 0.3701378071453978 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_cb_kcenter | 0.0025 | 42 | completed_long | 0.5488558762382568 | 0.32516153150902893 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_cb_herding | 0.0025 | 42 | completed_long | 0.5919164643478284 | 0.3246630006894027 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_cb_hybrid | 0.0025 | 42 | completed_long | 0.6206635877151008 | 0.3257283168095639 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_kmeans_labeled_nearest | 0.0025 | 42 | completed_long | 0.6887335405548167 | 0.3521801234630918 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_hybrid | 0.0025 | 42 | completed_long | 0.6887335405548167 | 0.3521801234630918 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_hybrid_mixup | 0.0025 | 42 | completed_long | 0.746393166842213 | 0.3791035690285768 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_hybrid_balanced_trainer | 0.0025 | 42 | completed_long | 0.6404702743809451 | 0.344996145725478 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_cb_random | 0.005 | 42 | completed_long | 0.708108252213759 | 0.3708838171007395 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_cb_kcenter | 0.005 | 42 | completed_long | 0.6156158964995113 | 0.3508466147700994 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_cb_herding | 0.005 | 42 | completed_long | 0.6271585759464929 | 0.33918849346898744 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_cb_hybrid | 0.005 | 42 | completed_long | 0.6708919786850157 | 0.3441880882005497 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_kmeans_labeled_nearest | 0.005 | 42 | completed_long | 0.7110999954362474 | 0.3640408887290348 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_hybrid | 0.005 | 42 | completed_long | 0.7110999954362474 | 0.3640408887290348 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_hybrid_mixup | 0.005 | 42 | completed_long | 0.767075099939406 | 0.3891223434748316 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| ogbn-products | products_uca_hybrid_balanced_trainer | 0.005 | 42 | completed_long | 0.6712656641773881 | 0.34492475606820144 | not_promoted |  | experiments\tables\t26_products_uca_sweep_seed42.csv |
| Reddit | reddit_current_sft_signature_random | 0.005 | 1 | completed_reuse_existing_t25_seed | 0.9208480692242788 | 0.8829126848375343 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.005 | 2 | completed_reuse_existing_t25_seed | 0.9213507351489147 | 0.8835794982373684 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.005 | 3 | completed_reuse_existing_t25_seed | 0.923307541784105 | 0.8835912065477536 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.005 | 4 | completed_reuse_existing_t25_seed | 0.9187296913990269 | 0.8777996574376501 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.005 | 5 | completed_reuse_existing_t25_seed | 0.9215123063389764 | 0.8843630606147639 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.005 | 1 | completed_reuse_existing_t25_seed | 0.9221585910992227 | 0.8839761339149043 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.005 | 2 | completed_reuse_existing_t25_seed | 0.9184245013733551 | 0.8797743290336197 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.005 | 3 | completed_reuse_existing_t25_seed | 0.919860689729458 | 0.8809236788981895 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.005 | 4 | completed_reuse_existing_t25_seed | 0.9154803152433442 | 0.8658734518737182 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.005 | 5 | completed_reuse_existing_t25_seed | 0.9185681202089654 | 0.8820962505757506 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.005 | 1 | completed_reuse_existing_t25_seed | 0.9196632138304939 | 0.8759414475529922 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.005 | 2 | completed_reuse_existing_t25_seed | 0.9131106044557744 | 0.8593391719798166 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.005 | 3 | completed_reuse_existing_t25_seed | 0.9151930775721235 | 0.8645093369682986 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.005 | 4 | completed_reuse_existing_t25_seed | 0.912338653214369 | 0.8547320825491888 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.005 | 5 | completed_reuse_existing_t25_seed | 0.9122668437965639 | 0.8533289899784489 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.005 | 1 | completed_reuse_existing_t25_seed | 0.9184604060822577 | 0.8828385972639313 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.005 | 2 | completed_reuse_existing_t25_seed | 0.9200043085650683 | 0.8837836726405526 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.005 | 3 | completed_reuse_existing_t25_seed | 0.9171319318528625 | 0.8687583379547701 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.005 | 4 | completed_reuse_existing_t25_seed | 0.9224099240615407 | 0.8817908054811194 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.005 | 5 | completed_reuse_existing_t25_seed | 0.9210994021865968 | 0.8813321326300385 | not_promoted | no_regression_gate_not_met | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
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
| Reddit | reddit_current_sft_signature_random | 0.01 | 1 | completed_reuse_existing_t25_seed | 0.9223022099348329 | 0.8854147170852387 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.01 | 2 | completed_reuse_existing_t25_seed | 0.923307541784105 | 0.8856540529538315 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.01 | 3 | completed_reuse_existing_t25_seed | 0.9217097822379405 | 0.8822859554632233 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.01 | 4 | completed_reuse_existing_t25_seed | 0.923307541784105 | 0.8845150316819438 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_random | 0.01 | 5 | completed_reuse_existing_t25_seed | 0.9237024935820333 | 0.884814524942957 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.01 | 1 | completed_reuse_existing_t25_seed | 0.9168267418271906 | 0.8762340233044532 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.01 | 2 | completed_reuse_existing_t25_seed | 0.9206326409708633 | 0.8840635480957215 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.01 | 3 | completed_reuse_existing_t25_seed | 0.919232357323663 | 0.8808767182806599 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.01 | 4 | completed_reuse_existing_t25_seed | 0.9168985512449958 | 0.8769557062247202 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_medoid | 0.01 | 5 | completed_reuse_existing_t25_seed | 0.9187296913990269 | 0.8765074245540784 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.01 | 1 | completed_reuse_existing_t25_seed | 0.9213148304400122 | 0.8788545897748362 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.01 | 2 | completed_reuse_existing_t25_seed | 0.9176345977774986 | 0.8696098031561748 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.01 | 3 | completed_reuse_existing_t25_seed | 0.9201479274006786 | 0.8746115410908344 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.01 | 4 | completed_reuse_existing_t25_seed | 0.9139005080516309 | 0.8598709597239272 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_current_sft_signature_kcenter | 0.01 | 5 | completed_reuse_existing_t25_seed | 0.9131644615191282 | 0.8551079632786572 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.01 | 1 | completed_reuse_existing_t25_seed | 0.9199324991472632 | 0.8802594463134891 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.01 | 2 | completed_reuse_existing_t25_seed | 0.9267184891298493 | 0.8903536405436407 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.01 | 3 | completed_reuse_existing_t25_seed | 0.9236665888731307 | 0.8835800227996524 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.01 | 4 | completed_reuse_existing_t25_seed | 0.9205069744897043 | 0.8825427264106005 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
| Reddit | reddit_sft_hnr_fdm_hybrid | 0.01 | 5 | completed_reuse_existing_t25_seed | 0.9218174963646483 | 0.8817047248153992 | not_promoted | t26_trainer_recipe_not_rerun | experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv |
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
| ogbn-arxiv | arxiv_teacher_sweep | 0.0 | 42 | completed_long | 0.6991955229101084 | 0.505342975533102 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t26_arxiv_teacher_sweep_seed42.csv |
| ogbn-arxiv | arxiv_teacher_sweep | 0.0 | 42 | completed_long | 0.6985577021994527 | 0.5104670152698261 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t26_arxiv_teacher_sweep_seed42.csv |
| ogbn-arxiv | arxiv_teacher_sweep | 0.0 | 42 | completed_long | 0.7038660165010391 | 0.5146734981276115 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t26_arxiv_teacher_sweep_seed42.csv |
| ogbn-arxiv | arxiv_teacher_sweep | 0.0 | 42 | completed_long | 0.706828796576343 | 0.5045803241909133 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t26_arxiv_teacher_sweep_seed42.csv |
| ogbn-arxiv | arxiv_teacher_sweep | 0.0 | 42 | completed_long | 0.700862086702467 | 0.5070398443656902 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t26_arxiv_teacher_sweep_seed42.csv |
| ogbn-papers100M | t26_ultra_contract_regression | 0.0001 | 42 | completed_ultra_dryrun |  |  | not_promoted | ultra_performance_not_run | experiments\tables\t26_ultra_contract_regression_seed42.csv |
| MAG240M | t26_ultra_contract_regression | 0.0001 | 42 | completed_ultra_dryrun |  |  | not_promoted | ultra_performance_not_run | experiments\tables\t26_ultra_contract_regression_seed42.csv |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  | not_promoted |  |  |
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

- arxiv_teacher_first: Arxiv condensation remains blocked until A1 >= 0.715.

- Stage CSV: `experiments\tables\t26_stage_summary_seed42.csv`
