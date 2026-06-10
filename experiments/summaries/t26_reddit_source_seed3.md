# T25 Reddit HNR-FDM-lite

- Train mode: `True`
- HNR enabled: `True`
- Rows use full-node ratio accounting and remain not_promoted unless the no-regression and T25 gates pass.

| requested_full_node_ratio | method | status | actual_full_node_ratio | accuracy | macro_f1 | predicted_classes | promotion_status | failure_reason |
|---|---|---|---|---|---|---|---|---|
| 0.005 | current_sft_signature_random | completed_streaming | 0.005000751185800442 | 0.923307541784105 | 0.8835912065477536 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | current_sft_signature_medoid | completed_streaming | 0.005000751185800442 | 0.919860689729458 | 0.8809236788981895 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | current_sft_signature_kcenter | completed_streaming | 0.005000751185800442 | 0.9151930775721235 | 0.8645093369682986 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | sft_hnr_fdm_hybrid | completed_streaming | 0.005000751185800442 | 0.9171319318528625 | 0.8687583379547701 | 41 | not_promoted | no_regression_gate_not_met |
| 0.01 | current_sft_signature_random | completed_streaming | 0.010001502371600884 | 0.9217097822379405 | 0.8822859554632233 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | current_sft_signature_medoid | completed_streaming | 0.010001502371600884 | 0.919232357323663 | 0.8808767182806599 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | current_sft_signature_kcenter | completed_streaming | 0.010001502371600884 | 0.9201479274006786 | 0.8746115410908344 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | sft_hnr_fdm_hybrid | completed_streaming | 0.010001502371600884 | 0.9236665888731307 | 0.8835800227996524 | 41 | not_promoted | acceptance_gate_not_met |

- CSV: `experiments\tables\t26_reddit_source_seed3.csv`
