# T23 ACM SFT Tune

| variant | accuracy | macro_f1 | valid_acc | gate_093 | run_condensed_sweep |
|---|---|---|---|---|---|
| ACM_H512_D0p2_CE | 0.9135977625846863 | 0.9139602979024252 | 0.939226508140564 | False | False |
| ACM_H512_D0p3_CE | 0.9159584641456604 | 0.9164112210273743 | 0.939226508140564 | False | False |
| ACM_H1024_D0p2_CE | 0.9135977625846863 | 0.9139001170794169 | 0.939226508140564 | False | False |
| ACM_H1024_D0p3_CE | 0.9140698909759521 | 0.9143547813097636 | 0.939226508140564 | False | False |
| ACM_H512_D0p3_class_balanced | 0.9126534461975098 | 0.9129384954770406 | 0.950276255607605 | False | False |
| ACM_H512_D0p3_two_stage | 0.899433434009552 | 0.9007725119590759 | 0.9171270728111267 | False | False |

- Best ACM tune row: `ACM_H512_D0p3_CE` accuracy `0.9159584641456604`.
- Condensed ACM sweep gated on >=0.93: `False`.
- CSV: `experiments\tables\t23_acm_sft_tune_seed42.csv`
