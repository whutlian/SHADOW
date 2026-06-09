# T2-SFT-NL Preprop Manifest Summary

Every completed manifest records `uses_logits=false` per block and `uses_logits_as_input=false` at run level.

| dataset | status | num_blocks | total_cache_bytes | full_edge_scans | uses_logits_as_input | uses_e_by_d_materialization | uses_dense_p2 | reason |
|---|---|---|---|---|---|---|---|---|
| acm | completed | 8 | 6195200 | 10 | False | False | False | completed |
| dblp | completed | 5 | 5192960 | 11 | False | False | False | completed |
| imdb | completed | 6 | 7575552 | 8 | False | False | False | completed |
| ogbn-arxiv | completed | 3 | 65027712 | 2 | False | False | False | completed |
| ogbn-products | blocked_resource_guard | 0 | 0 | 0 | False | False | False | products full preprop skipped locally; dry-run covers resources |

- CSV: `experiments\tables\t2_preprop_manifest_index_seed42.csv`
