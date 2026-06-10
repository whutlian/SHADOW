# T26 Reddit Trainer Mixup Notes

- Required seeds 1..5 and ratios 0.50%/1.00% are declared.
- Missing seed runs are marked ready_not_run; no seed42 replay is promoted as a seed sweep.
- True shadow rows remain diagnostic until a schema-preserving shadow graph is materialized and trained.

| requested_full_node_ratio | seed | method | status | accuracy | macro_f1 | promotion_status | failure_reason |
|---|---|---|---|---|---|---|---|
| 0.005 | 1 | reddit_current_sft_signature_random | completed_reuse_existing_t25_seed | 0.9208480692242788 | 0.8829126848375343 | not_promoted | no_regression_gate_not_met |
| 0.005 | 2 | reddit_current_sft_signature_random | completed_reuse_existing_t25_seed | 0.9213507351489147 | 0.8835794982373684 | not_promoted | no_regression_gate_not_met |
| 0.005 | 3 | reddit_current_sft_signature_random | completed_reuse_existing_t25_seed | 0.923307541784105 | 0.8835912065477536 | not_promoted | no_regression_gate_not_met |
| 0.005 | 4 | reddit_current_sft_signature_random | completed_reuse_existing_t25_seed | 0.9187296913990269 | 0.8777996574376501 | not_promoted | no_regression_gate_not_met |
| 0.005 | 5 | reddit_current_sft_signature_random | completed_reuse_existing_t25_seed | 0.9215123063389764 | 0.8843630606147639 | not_promoted | no_regression_gate_not_met |
| 0.005 | 1 | reddit_current_sft_signature_medoid | completed_reuse_existing_t25_seed | 0.9221585910992227 | 0.8839761339149043 | not_promoted | no_regression_gate_not_met |
| 0.005 | 2 | reddit_current_sft_signature_medoid | completed_reuse_existing_t25_seed | 0.9184245013733551 | 0.8797743290336197 | not_promoted | no_regression_gate_not_met |
| 0.005 | 3 | reddit_current_sft_signature_medoid | completed_reuse_existing_t25_seed | 0.919860689729458 | 0.8809236788981895 | not_promoted | no_regression_gate_not_met |
| 0.005 | 4 | reddit_current_sft_signature_medoid | completed_reuse_existing_t25_seed | 0.9154803152433442 | 0.8658734518737182 | not_promoted | no_regression_gate_not_met |
| 0.005 | 5 | reddit_current_sft_signature_medoid | completed_reuse_existing_t25_seed | 0.9185681202089654 | 0.8820962505757506 | not_promoted | no_regression_gate_not_met |
| 0.005 | 1 | reddit_current_sft_signature_kcenter | completed_reuse_existing_t25_seed | 0.9196632138304939 | 0.8759414475529922 | not_promoted | no_regression_gate_not_met |
| 0.005 | 2 | reddit_current_sft_signature_kcenter | completed_reuse_existing_t25_seed | 0.9131106044557744 | 0.8593391719798166 | not_promoted | no_regression_gate_not_met |
| 0.005 | 3 | reddit_current_sft_signature_kcenter | completed_reuse_existing_t25_seed | 0.9151930775721235 | 0.8645093369682986 | not_promoted | no_regression_gate_not_met |
| 0.005 | 4 | reddit_current_sft_signature_kcenter | completed_reuse_existing_t25_seed | 0.912338653214369 | 0.8547320825491888 | not_promoted | no_regression_gate_not_met |
| 0.005 | 5 | reddit_current_sft_signature_kcenter | completed_reuse_existing_t25_seed | 0.9122668437965639 | 0.8533289899784489 | not_promoted | no_regression_gate_not_met |
| 0.005 | 1 | reddit_sft_hnr_fdm_hybrid | completed_reuse_existing_t25_seed | 0.9184604060822577 | 0.8828385972639313 | not_promoted | no_regression_gate_not_met |
| 0.005 | 2 | reddit_sft_hnr_fdm_hybrid | completed_reuse_existing_t25_seed | 0.9200043085650683 | 0.8837836726405526 | not_promoted | no_regression_gate_not_met |
| 0.005 | 3 | reddit_sft_hnr_fdm_hybrid | completed_reuse_existing_t25_seed | 0.9171319318528625 | 0.8687583379547701 | not_promoted | no_regression_gate_not_met |
| 0.005 | 4 | reddit_sft_hnr_fdm_hybrid | completed_reuse_existing_t25_seed | 0.9224099240615407 | 0.8817908054811194 | not_promoted | no_regression_gate_not_met |
| 0.005 | 5 | reddit_sft_hnr_fdm_hybrid | completed_reuse_existing_t25_seed | 0.9210994021865968 | 0.8813321326300385 | not_promoted | no_regression_gate_not_met |
| 0.005 | 1 | reddit_tuned_balanced_trainer | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 2 | reddit_tuned_balanced_trainer | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 3 | reddit_tuned_balanced_trainer | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 4 | reddit_tuned_balanced_trainer | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 5 | reddit_tuned_balanced_trainer | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 1 | reddit_sft_signature_mixup | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 2 | reddit_sft_signature_mixup | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 3 | reddit_sft_signature_mixup | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 4 | reddit_sft_signature_mixup | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 5 | reddit_sft_signature_mixup | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 1 | reddit_true_shadow_b1 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized |
| 0.005 | 2 | reddit_true_shadow_b1 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized |
| 0.005 | 3 | reddit_true_shadow_b1 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized |
| 0.005 | 4 | reddit_true_shadow_b1 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized |
| 0.005 | 5 | reddit_true_shadow_b1 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized |
| 0.01 | 1 | reddit_current_sft_signature_random | completed_reuse_existing_t25_seed | 0.9223022099348329 | 0.8854147170852387 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 2 | reddit_current_sft_signature_random | completed_reuse_existing_t25_seed | 0.923307541784105 | 0.8856540529538315 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 3 | reddit_current_sft_signature_random | completed_reuse_existing_t25_seed | 0.9217097822379405 | 0.8822859554632233 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 4 | reddit_current_sft_signature_random | completed_reuse_existing_t25_seed | 0.923307541784105 | 0.8845150316819438 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 5 | reddit_current_sft_signature_random | completed_reuse_existing_t25_seed | 0.9237024935820333 | 0.884814524942957 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 1 | reddit_current_sft_signature_medoid | completed_reuse_existing_t25_seed | 0.9168267418271906 | 0.8762340233044532 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 2 | reddit_current_sft_signature_medoid | completed_reuse_existing_t25_seed | 0.9206326409708633 | 0.8840635480957215 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 3 | reddit_current_sft_signature_medoid | completed_reuse_existing_t25_seed | 0.919232357323663 | 0.8808767182806599 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 4 | reddit_current_sft_signature_medoid | completed_reuse_existing_t25_seed | 0.9168985512449958 | 0.8769557062247202 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 5 | reddit_current_sft_signature_medoid | completed_reuse_existing_t25_seed | 0.9187296913990269 | 0.8765074245540784 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 1 | reddit_current_sft_signature_kcenter | completed_reuse_existing_t25_seed | 0.9213148304400122 | 0.8788545897748362 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 2 | reddit_current_sft_signature_kcenter | completed_reuse_existing_t25_seed | 0.9176345977774986 | 0.8696098031561748 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 3 | reddit_current_sft_signature_kcenter | completed_reuse_existing_t25_seed | 0.9201479274006786 | 0.8746115410908344 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 4 | reddit_current_sft_signature_kcenter | completed_reuse_existing_t25_seed | 0.9139005080516309 | 0.8598709597239272 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 5 | reddit_current_sft_signature_kcenter | completed_reuse_existing_t25_seed | 0.9131644615191282 | 0.8551079632786572 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 1 | reddit_sft_hnr_fdm_hybrid | completed_reuse_existing_t25_seed | 0.9199324991472632 | 0.8802594463134891 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 2 | reddit_sft_hnr_fdm_hybrid | completed_reuse_existing_t25_seed | 0.9267184891298493 | 0.8903536405436407 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 3 | reddit_sft_hnr_fdm_hybrid | completed_reuse_existing_t25_seed | 0.9236665888731307 | 0.8835800227996524 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 4 | reddit_sft_hnr_fdm_hybrid | completed_reuse_existing_t25_seed | 0.9205069744897043 | 0.8825427264106005 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 5 | reddit_sft_hnr_fdm_hybrid | completed_reuse_existing_t25_seed | 0.9218174963646483 | 0.8817047248153992 | not_promoted | t26_trainer_recipe_not_rerun |
| 0.01 | 1 | reddit_tuned_balanced_trainer | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 2 | reddit_tuned_balanced_trainer | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 3 | reddit_tuned_balanced_trainer | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 4 | reddit_tuned_balanced_trainer | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 5 | reddit_tuned_balanced_trainer | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 1 | reddit_sft_signature_mixup | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 2 | reddit_sft_signature_mixup | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 3 | reddit_sft_signature_mixup | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 4 | reddit_sft_signature_mixup | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 5 | reddit_sft_signature_mixup | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 1 | reddit_true_shadow_b1 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized |
| 0.01 | 2 | reddit_true_shadow_b1 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized |
| 0.01 | 3 | reddit_true_shadow_b1 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized |
| 0.01 | 4 | reddit_true_shadow_b1 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized |
| 0.01 | 5 | reddit_true_shadow_b1 | diagnostic_shadow_not_trained |  |  | not_promoted | true_shadow_graph_not_materialized |

- CSV: `experiments\tables\t26_reddit_seed_trainer_mixup_sweep.csv`
