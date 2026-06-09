# LAD V2 Full-Graph Condensed Node Ratio Sweep

Single seed 42. All rows are no-diffusion LAD V2 (`compiled_plus_lad`).

| Dataset | Requested full node ratio | Actual full node ratio | Acc | Macro-F1 | Condensed nodes | Target prototypes | Shadow nodes | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| acm | 1.2000% | 1.1972% | 0.8116 | 0.8086 | 131 | 52 | 79 | completed |
| acm | 2.4000% | 2.4036% | 0.8055 | 0.8086 | 263 | 105 | 158 | completed |
| acm | 4.8000% | 4.7889% | 0.7984 | 0.7987 | 524 | 209 | 315 | completed |
| acm | 9.6000% | 9.5138% | 0.8546 | 0.8567 | 1041 | 411 | 630 | completed |
| dblp | 1.2000% | 1.2018% | 0.4732 | 0.4759 | 314 | 157 | 157 | completed |
| dblp | 2.4000% | 2.3997% | 0.5222 | 0.5020 | 627 | 314 | 313 | completed |
| dblp | 4.8000% | 4.7152% | 0.4958 | 0.4713 | 1232 | 605 | 627 | completed |
| dblp | 9.6000% | 9.1167% | 0.4570 | 0.4747 | 2382 | 1128 | 1254 | completed |
| imdb | 1.2000% | 1.1998% | 0.2552 | 0.1314 | 257 | 128 | 129 | completed |
| imdb | 2.4000% | 2.3950% | 0.2761 | 0.1005 | 513 | 256 | 257 | completed |
| imdb | 4.8000% | 4.7479% | 0.0765 | 0.0388 | 1017 | 503 | 514 | completed |
| imdb | 9.6000% | 8.7302% | 0.1246 | 0.0840 | 1870 | 842 | 1028 | completed |
| ogbn-arxiv | 0.0500% | 0.0502% | 0.5949 | 0.3285 | 85 | 57 | 28 | completed |
| ogbn-arxiv | 0.2500% | 0.2498% | 0.6012 | 0.3961 | 423 | 282 | 141 | completed |
| ogbn-arxiv | 0.5000% | 0.5002% | 0.5977 | 0.4110 | 847 | 565 | 282 | completed |
| ogbn-products | 0.0500% | 0.0500% | 0.5894 | 0.3270 | 1225 | 817 | 408 | completed |
| ogbn-products | 0.2500% | 0.2495% | 0.5733 | 0.3033 | 6111 | 4070 | 2041 | completed |
| ogbn-products | 0.5000% | 0.4971% | 0.5884 | 0.3366 | 12175 | 8093 | 4082 | completed |

- CSV: `experiments\tables\lad_full_node_ratio_seed42.csv`
- Report: `experiments\reports\lad_full_node_ratio_seed42.md`
