# T25 Reddit HNR-FDM-lite

- Train mode: `True`
- HNR enabled: `True`
- Rows use full-node ratio accounting and remain not_promoted unless the no-regression and T25 gates pass.

| requested_full_node_ratio | method | status | actual_full_node_ratio | accuracy | macro_f1 | predicted_classes | promotion_status | failure_reason |
|---|---|---|---|---|---|---|---|---|
| 0.005 | current_sft_signature_random | completed_streaming | 0.005000751185800442 | 0.9187296913990269 | 0.8777996574376501 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | current_sft_signature_medoid | completed_streaming | 0.005000751185800442 | 0.9154803152433442 | 0.8658734518737182 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | current_sft_signature_kcenter | completed_streaming | 0.005000751185800442 | 0.912338653214369 | 0.8547320825491888 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | sft_hnr_fdm_hybrid | completed_streaming | 0.005000751185800442 | 0.9224099240615407 | 0.8817908054811194 | 41 | not_promoted | no_regression_gate_not_met |
| 0.01 | current_sft_signature_random | completed_streaming | 0.010001502371600884 | 0.923307541784105 | 0.8845150316819438 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | current_sft_signature_medoid | completed_streaming | 0.010001502371600884 | 0.9168985512449958 | 0.8769557062247202 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | current_sft_signature_kcenter | completed_streaming | 0.010001502371600884 | 0.9139005080516309 | 0.8598709597239272 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | sft_hnr_fdm_hybrid | completed_streaming | 0.010001502371600884 | 0.9205069744897043 | 0.8825427264106005 | 41 | not_promoted | acceptance_gate_not_met |

- CSV: `experiments\tables\t26_reddit_source_seed4.csv`
