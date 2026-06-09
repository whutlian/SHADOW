# Small Non-Regression Repair Seed 42

| dataset | variant | accuracy | macro_f1 | predicted_class_count | historical_baseline | baseline_accuracy | delta_vs_baseline | status | promotion_reason | blocked_reason | source_log |
|---|---|---|---|---|---|---|---|---|---|---|---|
| acm | SFB-v2 B3_scap_v2 retained | 0.9154863357543945 | 0.9165802995363871 | 3 | SFB-v2 B3_scap_v2 | 0.9154863357543945 | 0.0 | promoted | non-regression gate passed; historical strong path preserved |  | experiments\logs\t0s_sfb_v2_fullgraph_seed42\acm_B3_scap_v2_seed42.json |
| dblp | DBLP_safe_base_plus_repaired_typed_demand | 0.8369718194007874 | 0.8299370408058167 | 4 | R+ current-best relation-linear | 0.837 | -2.8180599212612734e-05 | promoted | non-regression gate passed; historical strong path preserved |  | experiments\logs\small_rpp_nonregression_seed42\dblp_current_best_r0p065_seed42.json |
| imdb | IMDB_clean_S1_reused_by_safe_path | 0.42410993576049805 | 0.35393159091472626 | 5 | clean S1 MAM/MDM/MKM | 0.4241 | 9.935760498069879e-06 | promoted | non-regression gate passed; historical strong path preserved |  | experiments\logs\sota_clean_small_seed42\imdb_S1_clean_MAM_MDM_MKM_r0p05_seed42.json |

- CSV: `experiments\tables\small_nonregression_repair_seed42.csv`
