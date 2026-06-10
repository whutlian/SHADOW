# T2.2 ACM SFT Tune

Rows are explicit T2.2 opt-in runs. Promoted rows are checked against no logits/KD/dense-P2/bounded-edge/E-by-d flags.

| variant | status | model_type | hidden_dim | epochs | accuracy | macro_f1 | predicted_class_count | reason |
|---|---|---|---|---|---|---|---|---|
| ACM_H512_D0p2_CE | completed | gamlp_lite | 512 | 80 | 0.9135977625846863 | 0.9139602979024252 | 3 | acm_quick_tune |
| ACM_H512_D0p3_CE | completed | gamlp_lite | 512 | 80 | 0.9159584641456604 | 0.9164112210273743 | 3 | acm_quick_tune |
| ACM_H1024_D0p2_CE | completed | gamlp_lite | 1024 | 80 | 0.9135977625846863 | 0.9139001170794169 | 3 | acm_quick_tune |
| ACM_H1024_D0p3_CE | completed | gamlp_lite | 1024 | 80 | 0.9140698909759521 | 0.9143547813097636 | 3 | acm_quick_tune |
| ACM_H512_D0p3_class_balanced | completed | gamlp_lite | 512 | 80 | 0.9126534461975098 | 0.9129384954770406 | 3 | acm_quick_tune |
| ACM_H512_D0p3_two_stage | completed | gamlp_recursive_v2 | 512 | 80 | 0.899433434009552 | 0.9007725119590759 | 3 | acm_quick_tune |

- CSV: `experiments\tables\t22_acm_sft_tune_seed42.csv`
