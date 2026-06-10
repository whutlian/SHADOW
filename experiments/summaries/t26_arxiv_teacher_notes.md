# T26 Arxiv Teacher Notes

- Teacher-first gate A1 is accuracy >= 0.715.
- Condensation rows are blocked while A1 is not met.

| variant | status | accuracy | macro_f1 | predicted_class_count | teacher_gate_A1 | condensation_status | failure_reason |
|---|---|---|---|---|---|---|---|
| A1_real_sagn_lite_v4_h768_e300 | completed_long | 0.6991955229101084 | 0.505342975533102 | 39 | False | blocked_by_teacher_gate | A1_teacher_gate_not_met |
| A2_real_sagn_lite_v4_h512_dropout0p5_labeldrop0p2_lr0p001_e400 | completed_long | 0.6985577021994527 | 0.5104670152698261 | 39 | False | blocked_by_teacher_gate | A1_teacher_gate_not_met |
| A3_real_sagn_lite_v4_h512_train_plus_valid_e300 | completed_long | 0.7038660165010391 | 0.5146734981276115 | 40 | False | blocked_by_teacher_gate | A1_teacher_gate_not_met |
| A4_real_sagn_lite_v4_h512_all_filterbank_labelreuse_e300 | completed_long | 0.706828796576343 | 0.5045803241909133 | 39 | False | blocked_by_teacher_gate | A1_teacher_gate_not_met |
| A5_real_sagn_lite_v4_h768_all_filterbank_labelreuse_e300 | completed_long | 0.700862086702467 | 0.5070398443656902 | 39 | False | blocked_by_teacher_gate | A1_teacher_gate_not_met |

- CSV: `experiments\tables\t26_arxiv_teacher_sweep_seed42.csv`
