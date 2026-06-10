# T2.2 ogbn-arxiv SFT Boost

Rows are explicit T2.2 opt-in runs. Promoted rows are checked against no logits/KD/dense-P2/bounded-edge/E-by-d flags.

| variant | status | model_type | hidden_dim | epochs | accuracy | macro_f1 | predicted_class_count | reason |
|---|---|---|---|---|---|---|---|---|
| A0_current_best_replay | completed | gamlp_lite | 512 | 100 | 0.6530461082649219 | 0.4173269426774673 | 39 | lazy_memmap_sft_completed |
| A1_add_X3_Xres2 | completed | gamlp_lite_v2 | 512 | 150 | 0.6729008497417854 | 0.42834986600271846 | 38 | lazy_memmap_sft_completed |
| A2_add_LabelReuse_Y1Y2Y3 | promoted_short | gamlp_lite_v2 | 512 | 150 | 0.6947513527971524 | 0.48676255326594575 | 39 | lazy_memmap_sft_completed |
| A3_true_sagn_lite_v2 | promoted_short | sagn_lite_v2 | 512 | 150 | 0.7016645063061951 | 0.5048992808650066 | 39 | lazy_memmap_sft_completed |
| A4_gamlp_recursive_v2 | promoted_short | gamlp_recursive_v2 | 512 | 150 | 0.6894636133572002 | 0.48908222402081003 | 39 | lazy_memmap_sft_completed |
| A5_two_stage_sqrt_to_ce | completed | gamlp_lite_v2 | 512 | 200 | 0.6684155299055614 | 0.4394560702948403 | 39 | lazy_memmap_sft_completed |
| A6_A4_plus_A5 | promoted_short | gamlp_recursive_v2 | 512 | 200 | 0.6822624117852808 | 0.4636449392454682 | 39 | lazy_memmap_sft_completed |
| A7_A4_plus_LabelReuse_plus_two_stage | promoted_short | gamlp_recursive_v2 | 512 | 200 | 0.6892578647408596 | 0.48908966374168283 | 39 | lazy_memmap_sft_completed |

- CSV: `experiments\tables\t22_arxiv_sft_boost_seed42.csv`
