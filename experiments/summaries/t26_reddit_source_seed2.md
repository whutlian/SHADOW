# T25 Reddit HNR-FDM-lite

- Train mode: `True`
- HNR enabled: `True`
- Rows use full-node ratio accounting and remain not_promoted unless the no-regression and T25 gates pass.

| requested_full_node_ratio | method | status | actual_full_node_ratio | accuracy | macro_f1 | predicted_classes | promotion_status | failure_reason |
|---|---|---|---|---|---|---|---|---|
| 0.005 | current_sft_signature_random | completed_streaming | 0.005000751185800442 | 0.9213507351489147 | 0.8835794982373684 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | current_sft_signature_medoid | completed_streaming | 0.005000751185800442 | 0.9184245013733551 | 0.8797743290336197 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | current_sft_signature_kcenter | completed_streaming | 0.005000751185800442 | 0.9131106044557744 | 0.8593391719798166 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | sft_hnr_fdm_hybrid | completed_streaming | 0.005000751185800442 | 0.9200043085650683 | 0.8837836726405526 | 41 | not_promoted | no_regression_gate_not_met |
| 0.01 | current_sft_signature_random | completed_streaming | 0.010001502371600884 | 0.923307541784105 | 0.8856540529538315 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | current_sft_signature_medoid | completed_streaming | 0.010001502371600884 | 0.9206326409708633 | 0.8840635480957215 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | current_sft_signature_kcenter | completed_streaming | 0.010001502371600884 | 0.9176345977774986 | 0.8696098031561748 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | sft_hnr_fdm_hybrid | completed_streaming | 0.010001502371600884 | 0.9267184891298493 | 0.8903536405436407 | 41 | not_promoted | acceptance_gate_not_met |

- CSV: `experiments\tables\t26_reddit_source_seed2.csv`
