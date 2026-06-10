# T26 Reddit Trainer Mixup Notes

- Required seeds 1..5 and ratios 0.50%/1.00% are declared.
- Missing seed runs are marked ready_not_run; no seed42 replay is promoted as a seed sweep.
- True shadow rows remain diagnostic until a schema-preserving shadow graph is materialized and trained.

| requested_full_node_ratio | seed | method | status | accuracy | macro_f1 | promotion_status | failure_reason |
|---|---|---|---|---|---|---|---|
| 0.005 | 1 | reddit_current_sft_signature_random | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 2 | reddit_current_sft_signature_random | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 3 | reddit_current_sft_signature_random | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 4 | reddit_current_sft_signature_random | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 5 | reddit_current_sft_signature_random | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 1 | reddit_current_sft_signature_medoid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 2 | reddit_current_sft_signature_medoid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 3 | reddit_current_sft_signature_medoid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 4 | reddit_current_sft_signature_medoid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 5 | reddit_current_sft_signature_medoid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 1 | reddit_current_sft_signature_kcenter | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 2 | reddit_current_sft_signature_kcenter | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 3 | reddit_current_sft_signature_kcenter | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 4 | reddit_current_sft_signature_kcenter | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 5 | reddit_current_sft_signature_kcenter | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 1 | reddit_sft_hnr_fdm_hybrid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 2 | reddit_sft_hnr_fdm_hybrid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 3 | reddit_sft_hnr_fdm_hybrid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 4 | reddit_sft_hnr_fdm_hybrid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.005 | 5 | reddit_sft_hnr_fdm_hybrid | ready_not_run |  |  | not_promoted | seed_not_run |
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
| 0.01 | 1 | reddit_current_sft_signature_random | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 2 | reddit_current_sft_signature_random | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 3 | reddit_current_sft_signature_random | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 4 | reddit_current_sft_signature_random | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 5 | reddit_current_sft_signature_random | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 1 | reddit_current_sft_signature_medoid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 2 | reddit_current_sft_signature_medoid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 3 | reddit_current_sft_signature_medoid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 4 | reddit_current_sft_signature_medoid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 5 | reddit_current_sft_signature_medoid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 1 | reddit_current_sft_signature_kcenter | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 2 | reddit_current_sft_signature_kcenter | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 3 | reddit_current_sft_signature_kcenter | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 4 | reddit_current_sft_signature_kcenter | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 5 | reddit_current_sft_signature_kcenter | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 1 | reddit_sft_hnr_fdm_hybrid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 2 | reddit_sft_hnr_fdm_hybrid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 3 | reddit_sft_hnr_fdm_hybrid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 4 | reddit_sft_hnr_fdm_hybrid | ready_not_run |  |  | not_promoted | seed_not_run |
| 0.01 | 5 | reddit_sft_hnr_fdm_hybrid | ready_not_run |  |  | not_promoted | seed_not_run |
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
