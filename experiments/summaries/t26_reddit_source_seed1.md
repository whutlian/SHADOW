# T25 Reddit HNR-FDM-lite

- Train mode: `True`
- HNR enabled: `True`
- Rows use full-node ratio accounting and remain not_promoted unless the no-regression and T25 gates pass.

| requested_full_node_ratio | method | status | actual_full_node_ratio | accuracy | macro_f1 | predicted_classes | promotion_status | failure_reason |
|---|---|---|---|---|---|---|---|---|
| 0.005 | current_sft_signature_random | completed_streaming | 0.005000751185800442 | 0.9208480692242788 | 0.8829126848375343 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | current_sft_signature_medoid | completed_streaming | 0.005000751185800442 | 0.9221585910992227 | 0.8839761339149043 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | current_sft_signature_kcenter | completed_streaming | 0.005000751185800442 | 0.9196632138304939 | 0.8759414475529922 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | sft_hnr_fdm_hybrid | completed_streaming | 0.005000751185800442 | 0.9184604060822577 | 0.8828385972639313 | 41 | not_promoted | no_regression_gate_not_met |
| 0.01 | current_sft_signature_random | completed_streaming | 0.010001502371600884 | 0.9223022099348329 | 0.8854147170852387 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | current_sft_signature_medoid | completed_streaming | 0.010001502371600884 | 0.9168267418271906 | 0.8762340233044532 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | current_sft_signature_kcenter | completed_streaming | 0.010001502371600884 | 0.9213148304400122 | 0.8788545897748362 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | sft_hnr_fdm_hybrid | completed_streaming | 0.010001502371600884 | 0.9199324991472632 | 0.8802594463134891 | 41 | not_promoted | acceptance_gate_not_met |

- CSV: `experiments\tables\t26_reddit_source_seed1.csv`
