# Fullgraph Parity Seed 42

| Dataset | Variant | Acc | Gate | Passed | Blocked | Status | Reason |
|---|---|---:|---:|---|---|---|---|
| acm | fullgraph_sehgnn_lite_current | 0.8970727324485779 | 0.9 | False | True | completed | completed |
| acm | fullgraph_sehgnn_lite_tuned | 0.9027384519577026 | 0.9 | True | False | completed | completed |
| acm | fullgraph_han_style_optional |  | 0.9 | False | True | skipped_optional_not_implemented | HAN-style optional backbone is not implemented in this sprint; row is skipped and excluded from best summaries |
| dblp | fullgraph_dblp_APA_only | 0.7919014096260071 | 0.9 | False | True | completed | completed |
| dblp | fullgraph_dblp_full_schema_sehgnn_lite | 0.8066901564598083 | 0.9 | False | True | completed | completed |
| imdb | fullgraph_imdb_sehgnn_lite_MAM_MDM_MKM | 0.4244222342967987 | 0.55 | False | True | completed | completed |
| imdb | fullgraph_imdb_han_style_optional |  | 0.55 | False | True | skipped_optional_not_implemented | HAN-style optional backbone is not implemented in this sprint; row is skipped and excluded from best summaries |
| ogbn-arxiv | fullgraph_lad_table_teacher | 0.6615641117095947 | 0.68 | False | True | completed_existing_diagnostic | completed_existing_diagnostic |
| ogbn-products | fullgraph_lad_table_teacher | 0.6884398460388184 | 0.7 | False | True | completed_existing_diagnostic | completed_existing_diagnostic |
| ogbn-products | fullgraph_lad_table_teacher_balanced_softmax |  | 0.7 | False | True | skipped_resource_guard | products calibration teacher rows are guarded; no P2/diffusion path is run |
| ogbn-products | fullgraph_lad_table_teacher_logit_adjusted |  | 0.7 | False | True | skipped_resource_guard | products calibration teacher rows are guarded; no P2/diffusion path is run |

Rows below gate set `blocked_by_fullgraph_backbone=true`; downstream promoted condensation is blocked for those datasets.

- CSV: `experiments\tables\fullgraph_parity_seed42.csv`
