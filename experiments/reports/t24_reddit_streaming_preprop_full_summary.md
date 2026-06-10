# T24 Reddit Streaming Preprop

| dataset | status | blocks | edge_limit | full_edge_scans | cache_bytes | peak_cpu_ram_gb | reason |
|---|---|---|---|---|---|---|---|
| Reddit | completed_streaming_full | X0,X1,X2,X3,Xres1,Y1,Y2,Y3,structure |  | 13 | 207804780 | 0.5306587219238281 | streaming raw-memmap preprop completed without processed data.pt or stacked edge_index |

## Blocks

| block | kind | shape | edge_scans | cache_bytes |
|---|---|---|---|---|
| X0 | self | 232965x64 | 0 | 29819520 |
| X1 | hop_block | 232965x64 | 2 | 29819520 |
| X2 | hop_block | 232965x64 | 2 | 29819520 |
| X3 | hop_block | 232965x64 | 2 | 29819520 |
| Xres1 | residual | 232965x64 | 0 | 29819520 |
| Y1 | label_reuse | 232965x41 | 2 | 19103130 |
| Y2 | label_reuse | 232965x41 | 2 | 19103130 |
| Y3 | label_reuse | 232965x41 | 2 | 19103130 |
| structure | structure | 232965x3 | 1 | 1397790 |

- CSV: `experiments\tables\t24_reddit_streaming_preprop_full_seed42.csv`
