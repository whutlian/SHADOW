# T24 Arxiv SFT-v4

The v4 block/head interfaces are implemented. This local run keeps the previous A3 fullgraph measurement as A0 and marks unrerun v4 matrix rows as ready_not_rerun.

| variant | status | accuracy | macro_f1 | predicted_class_count | promotion_status | promotion_reason |
|---|---|---|---|---|---|---|
| A0_current_A3_true_sagn_lite_v3_replay | completed_replay | 0.7016645063061951 | 0.5048992808650066 | 39 | not_promoted | acceptance_gate_not_met |
| A1_filter_bank_v4_only | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | not_promoted | acceptance_gate_not_met |
| A2_LabelReuse_v3_only | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | not_promoted | acceptance_gate_not_met |
| A3_filter_bank_v4_plus_LabelReuse_v3 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | not_promoted | acceptance_gate_not_met |
| A4_A3_sagn_lite_v4_h768 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | not_promoted | acceptance_gate_not_met |
| A5_A3_sagn_lite_v4_h1024 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | not_promoted | acceptance_gate_not_met |
| A6_A3_gamlp_lite_v4_h768 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | not_promoted | acceptance_gate_not_met |
| A7_A3_gamlp_lite_v4_h1024 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | not_promoted | acceptance_gate_not_met |
| A8_best_v4_two_stage | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | 39 | not_promoted | acceptance_gate_not_met |

- CSV: `experiments\tables\t24_arxiv_sft_v4_seed42.csv`
