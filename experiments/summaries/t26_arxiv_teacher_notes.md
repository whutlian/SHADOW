# T26 Arxiv Teacher Notes

- Teacher-first gate A1 is accuracy >= 0.715.
- Condensation rows are blocked while A1 is not met.

| variant | status | accuracy | macro_f1 | predicted_class_count | teacher_gate_A1 | condensation_status | failure_reason |
|---|---|---|---|---|---|---|---|
| A0_current_A3_true_sagn_lite_v3_replay | completed_replay | 0.7016645063061951 | 0.5048992808650066 | 39 | False | blocked_by_teacher_gate | A1_teacher_gate_not_met |
| A1_filter_bank_v4_only | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | blocked_by_teacher_gate | A1_teacher_gate_not_met |
| A2_LabelReuse_v3_only | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | blocked_by_teacher_gate | A1_teacher_gate_not_met |
| A3_filter_bank_v4_plus_LabelReuse_v3 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | blocked_by_teacher_gate | A1_teacher_gate_not_met |
| A4_A3_sagn_lite_v4_h768 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | blocked_by_teacher_gate | A1_teacher_gate_not_met |
| A5_A3_sagn_lite_v4_h1024 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | blocked_by_teacher_gate | A1_teacher_gate_not_met |
| A6_A3_gamlp_lite_v4_h768 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | blocked_by_teacher_gate | A1_teacher_gate_not_met |
| A7_A3_gamlp_lite_v4_h1024 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | blocked_by_teacher_gate | A1_teacher_gate_not_met |
| A8_best_v4_two_stage | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | False | blocked_by_teacher_gate | A1_teacher_gate_not_met |

- CSV: `experiments\tables\t26_arxiv_teacher_sweep_seed42.csv`
