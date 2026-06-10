# T24 Reddit Streaming Preprop

| dataset | status | blocks | edge_limit | full_edge_scans | cache_bytes | peak_cpu_ram_gb | reason |
|---|---|---|---|---|---|---|---|
| Reddit | completed_streaming_smoke | X0,X1,structure | 1000000 | 3 | 61036830 | 0.5213279724121094 | streaming raw-memmap preprop completed without processed data.pt or stacked edge_index |

## Blocks

| block | kind | shape | edge_scans | cache_bytes |
|---|---|---|---|---|
| X0 | self | 232965x64 | 0 | 29819520 |
| X1 | hop_block | 232965x64 | 2 | 29819520 |
| structure | structure | 232965x3 | 1 | 1397790 |

- CSV: `experiments\tables\t24_reddit_streaming_preprop_smoke_seed42.csv`
