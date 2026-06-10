# T25 Arxiv SFT-v4 Teacher Gate

- Condensation remains blocked until A1 accuracy >= 0.715.

| variant | status | accuracy | macro_f1 | predicted_classes | teacher_gate_A1 | teacher_gate_A2 | teacher_gate_A3 | failure_reason |
|---|---|---|---|---|---|---|---|---|
| A0_current_A3_true_sagn_lite_v3_replay | completed_replay | 0.7016645063061951 | 0.5048992808650066 | 39 | False | False | False | A1_teacher_gate_not_met |
| A1_filter_bank_v4_only | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | False | False | A1_teacher_gate_not_met |
| A2_LabelReuse_v3_only | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | False | False | A1_teacher_gate_not_met |
| A3_filter_bank_v4_plus_LabelReuse_v3 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | False | False | A1_teacher_gate_not_met |
| A4_A3_sagn_lite_v4_h768 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | False | False | A1_teacher_gate_not_met |
| A5_A3_sagn_lite_v4_h1024 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | False | False | A1_teacher_gate_not_met |
| A6_A3_gamlp_lite_v4_h768 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | False | False | A1_teacher_gate_not_met |
| A7_A3_gamlp_lite_v4_h1024 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | False | False | A1_teacher_gate_not_met |
| A8_best_v4_two_stage | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | False | False | A1_teacher_gate_not_met |

- CSV: `experiments\tables\t25_arxiv_sft_v4_teacher_seed42.csv`
