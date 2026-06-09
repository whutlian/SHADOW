# ogbn-arxiv No-Diffusion Recovery Seed 42

| Dataset | Variant | Ratio | Acc | Macro-F1 | Status | Promoted | Reason |
|---|---|---:|---:|---:|---|---|---|
| ogbn-arxiv | LAD_reference | 0.06 | 0.590210497379303 | 0.40769136818125845 | diagnostic_existing | False | existing no-diffusion LAD_reference diagnostic retained; no P2/diffusion path run |
| ogbn-arxiv | LAD_reference_with_fixed_block_stats | 0.06 |  |  | skipped_blocked_by_fullgraph_backbone | False | ogbn-arxiv fullgraph teacher gate did not pass; no-diffusion LAD_reference retained as diagnostic |
| ogbn-arxiv | stronger_table_head | 0.06 |  |  | skipped_blocked_by_fullgraph_backbone | False | ogbn-arxiv fullgraph teacher gate did not pass; no-diffusion LAD_reference retained as diagnostic |
| ogbn-arxiv | LAD_reference | 0.12 | 0.5967738628387451 | 0.4154518236406147 | diagnostic_existing | False | existing no-diffusion LAD_reference diagnostic retained; no P2/diffusion path run |
| ogbn-arxiv | LAD_reference_with_fixed_block_stats | 0.12 |  |  | skipped_blocked_by_fullgraph_backbone | False | ogbn-arxiv fullgraph teacher gate did not pass; no-diffusion LAD_reference retained as diagnostic |
| ogbn-arxiv | stronger_table_head | 0.12 |  |  | skipped_blocked_by_fullgraph_backbone | False | ogbn-arxiv fullgraph teacher gate did not pass; no-diffusion LAD_reference retained as diagnostic |

- CSV: `experiments\tables\arxiv_no_diffusion_recovery_seed42.csv`
