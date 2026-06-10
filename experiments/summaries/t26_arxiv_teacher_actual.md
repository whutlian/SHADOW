# T26 Arxiv Actual Teacher Runs

- Rows are real lazy-memmap SFT teacher training runs.
- A1 gate is `accuracy >= 0.715`.

| variant | status | accuracy | macro_f1 | predicted_class_count | valid_acc | teacher_gate_A1 | model_type | hidden_dim | epochs |
|---|---|---|---|---|---|---|---|---|---|
| A1_real_sagn_lite_v4_h768_e300 | completed_long | 0.6991955229101084 | 0.505342975533102 | 39 | 0.7209973489043257 | False | sagn_lite_v4 | 768 | 300 |
| A2_real_sagn_lite_v4_h512_dropout0p5_labeldrop0p2_lr0p001_e400 | completed_long | 0.6985577021994527 | 0.5104670152698261 | 39 | 0.7231115138091883 | False | sagn_lite_v4 | 512 | 400 |
| A3_real_sagn_lite_v4_h512_train_plus_valid_e300 | completed_long | 0.7038660165010391 | 0.5146734981276115 | 40 | 0.9662404778683849 | False | sagn_lite_v4 | 512 | 300 |
| A4_real_sagn_lite_v4_h512_all_filterbank_labelreuse_e300 | completed_long | 0.706828796576343 | 0.5045803241909133 | 39 | 0.7182120205376019 | False | sagn_lite_v4 | 512 | 300 |
| A5_real_sagn_lite_v4_h768_all_filterbank_labelreuse_e300 | completed_long | 0.700862086702467 | 0.5070398443656902 | 39 | 0.7197556964998826 | False | sagn_lite_v4 | 768 | 300 |

- CSV: `experiments\tables\t26_arxiv_teacher_actual_seed42.csv`
