# T2-SFT-NL Condensation Recovery Gate

Recovery rows are diagnostic only. Identity replay is materialized for promoted fullgraph rows; prototype/shadow SFT block-signature compression is left as `eligible_not_run` unless launched separately.

| dataset | recovery_row | fullgraph_accuracy | status | reason |
|---|---|---|---|---|
| acm | identity_condensed_sft_replay | 0.9206798672676086 | completed_diagnostic | identity replay of the validation-selected T2 SFT table teacher; diagnostic only |
| acm | prototype_oracle_sft_block_signature | 0.9206798672676086 | eligible_not_run | fullgraph T2 gate passed; compressed SFT block-signature recovery must be launched separately |
| acm | shadow_condensed_sft_block_signature | 0.9206798672676086 | eligible_not_run | fullgraph T2 gate passed; compressed SFT block-signature recovery must be launched separately |
| dblp | identity_condensed_sft_replay | 0.9426056146621704 | completed_diagnostic | identity replay of the validation-selected T2 SFT table teacher; diagnostic only |
| dblp | prototype_oracle_sft_block_signature | 0.9426056146621704 | eligible_not_run | fullgraph T2 gate passed; compressed SFT block-signature recovery must be launched separately |
| dblp | shadow_condensed_sft_block_signature | 0.9426056146621704 | eligible_not_run | fullgraph T2 gate passed; compressed SFT block-signature recovery must be launched separately |
| imdb | identity_condensed_sft_replay | 0.47158026695251465 | completed_diagnostic | identity replay of the validation-selected T2 SFT table teacher; diagnostic only |
| imdb | prototype_oracle_sft_block_signature | 0.47158026695251465 | eligible_not_run | fullgraph T2 gate passed; compressed SFT block-signature recovery must be launched separately |
| imdb | shadow_condensed_sft_block_signature | 0.47158026695251465 | eligible_not_run | fullgraph T2 gate passed; compressed SFT block-signature recovery must be launched separately |
| ogbn-arxiv | recovery_gate | 0.6105796098709106 | blocked_by_t2_fullgraph_gate | predicted_class_count<35 |
| ogbn-products | recovery_gate |  | blocked_by_t2_fullgraph_gate | products full T2 SFT skipped locally; use --run-products-full after dry-run |

- CSV: `experiments\tables\t2_condensation_recovery_seed42.csv`
