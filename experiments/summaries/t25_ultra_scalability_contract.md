# T25 Ultra Scalability Contract

- Dry-run planner uses train-target-only cache estimates and ultra-safe guards.

| dataset | requested_full_node_ratio | planned_total_condensed_nodes | planned_target_prototypes | planned_shadow_nodes | estimated_cache_bytes | resource_gate_S1 | resource_gate_S2 | resource_gate_S3 |
|---|---|---|---|---|---|---|---|---|
| ogbn-papers100M | 0.0001 | 11106 | 7774 | 3332 | 1313410752 | True | True | True |
| MAG240M | 0.0001 | 12175 | 8522 | 3653 | 1125740704 | True | True | True |

- CSV: `experiments\tables\t25_ultra_dryrun_seed42.csv`
