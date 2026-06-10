# T2.2 ogbn-products SFT Boost

Rows are explicit T2.2 opt-in runs. Promoted rows are checked against no logits/KD/dense-P2/bounded-edge/E-by-d flags.

| variant | status | model_type | hidden_dim | epochs | accuracy | macro_f1 | predicted_class_count | reason |
|---|---|---|---|---|---|---|---|---|
| P0_current_best_replay | completed | gamlp_lite | 512 | 100 | 0.7078990425608346 | 0.3448782881249601 | 40 | lazy_memmap_sft_completed |
| P1_h768_e200 | completed | gamlp_lite_v2 | 768 | 200 | 0.7152963886256823 | 0.346756300991601 | 40 | lazy_memmap_sft_completed |
| P1_h1024_e200 | completed | gamlp_lite_v2 | 1024 | 200 | 0.7080169771599993 | 0.3426601578227876 | 40 | lazy_memmap_sft_completed |
| P3_add_LabelReuse_Y1Y2Y3 | promoted_short | gamlp_lite_v2 | 768 | 200 | 0.7345730473803381 | 0.37625175014369194 | 42 | lazy_memmap_sft_completed |
| P4_two_stage_sqrt_to_ce | completed | gamlp_lite_v2 | 768 | 200 | 0.7047852980288655 | 0.33810563611323824 | 39 | lazy_memmap_sft_completed |
| P5_P3_plus_P4_plus_h1024 | promoted_short | gamlp_lite_v2 | 1024 | 200 | 0.7210765395548578 | 0.3753645975286824 | 42 | lazy_memmap_sft_completed |
| P6_gamlp_recursive_v2 | promoted_short | gamlp_recursive_v2 | 768 | 200 | 0.7393098611851027 | 0.379451800413963 | 42 | lazy_memmap_sft_completed |
| P7_sagn_lite_v2 | promoted_short | sagn_lite_v2 | 768 | 200 | 0.7555780580193042 | 0.4046991170720907 | 42 | lazy_memmap_sft_completed |

- CSV: `experiments\tables\t22_products_sft_boost_seed42.csv`
