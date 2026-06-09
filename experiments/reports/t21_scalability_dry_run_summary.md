# T2.1 Scalability Dry Run

Dry-run estimates cache bytes and full edge scans for true chunked/memmap preprop. Ultra-scale rows are train-target-only by policy.

| dataset | cache_mode | total_cache_bytes | full_edge_scans | wall_time_category | server_recommended |
|---|---|---|---|---|---|
| ogbn-arxiv | all_target_rows | 143602864 | 6 | local_short | False |
| ogbn-products | all_target_rows | 2111062998 | 6 | local_long | False |
| ogbn-papers100M | train_target_only | 123498671072 | 6 | server_recommended | True |
| MAG240M | train_target_only | 130761289284 | 6 | server_recommended | True |

- CSV: `experiments\tables\t21_scalability_dry_run_seed42.csv`
