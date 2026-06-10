# T25 Reddit HNR-FDM-lite

- Train mode: `True`
- HNR enabled: `True`
- Rows use full-node ratio accounting and remain not_promoted unless the no-regression and T25 gates pass.

| requested_full_node_ratio | method | status | actual_full_node_ratio | accuracy | macro_f1 | predicted_classes | promotion_status | failure_reason |
|---|---|---|---|---|---|---|---|---|
| 0.005 | current_sft_signature_random | completed_streaming | 0.005000751185800442 | 0.9215123063389764 | 0.8843630606147639 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | current_sft_signature_medoid | completed_streaming | 0.005000751185800442 | 0.9185681202089654 | 0.8820962505757506 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | current_sft_signature_kcenter | completed_streaming | 0.005000751185800442 | 0.9122668437965639 | 0.8533289899784489 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | sft_hnr_fdm_hybrid | completed_streaming | 0.005000751185800442 | 0.9210994021865968 | 0.8813321326300385 | 41 | not_promoted | no_regression_gate_not_met |
| 0.01 | current_sft_signature_random | completed_streaming | 0.010001502371600884 | 0.9237024935820333 | 0.884814524942957 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | current_sft_signature_medoid | completed_streaming | 0.010001502371600884 | 0.9187296913990269 | 0.8765074245540784 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | current_sft_signature_kcenter | completed_streaming | 0.010001502371600884 | 0.9131644615191282 | 0.8551079632786572 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | sft_hnr_fdm_hybrid | completed_streaming | 0.010001502371600884 | 0.9218174963646483 | 0.8817047248153992 | 41 | not_promoted | acceptance_gate_not_met |

- CSV: `experiments\tables\t26_reddit_source_seed5.csv`
