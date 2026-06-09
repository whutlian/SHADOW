# Fullgraph Backbone Audit Seed 42

Small datasets use full-target-row SeHGNNLite with schema-default target feature blocks. Medium rows use the existing no-diffusion FullDemandTable-MLP diagnostics as the table-teacher audit.

| Dataset | Variant | Acc | Macro-F1 | Gate | Passed | Blocked | Status |
|---|---|---:|---:|---:|---|---|---|
| acm | fullgraph_sehgnn_lite | 0.9117091298103333 | 0.912367602189382 | 0.9 | True | False | completed |
| dblp | fullgraph_sehgnn_lite | 0.7880281805992126 | 0.7809299379587173 | 0.88 | False | True | completed |
| imdb | fullgraph_sehgnn_lite | 0.40818238258361816 | 0.36262465715408326 | 0.55 | False | True | completed |
| ogbn-arxiv | fullgraph_no_diffusion_lad_table_teacher | 0.6615641117095947 | 0.4024657105095685 | 0.65 | True | False | completed_existing_diagnostic |
| ogbn-products | fullgraph_no_diffusion_lad_table_teacher | 0.6884398460388184 | 0.33906127453008866 | 0.7 | False | True | completed_existing_diagnostic |

- CSV: `experiments\tables\fullgraph_backbone_audit_seed42.csv`
