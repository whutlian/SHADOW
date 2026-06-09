# Identity Condensation Audit Seed 42

| Dataset | Ratio | Fullgraph | Identity | Oracle | Shadow | Full->Shadow | Compatible | Bottleneck | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| acm | 0.12 | 0.9027384519577026 | 0.9027384519577026 | None | 0.8937677145004272 | 0.008971 | False | training_head_bottleneck | completed |
| dblp | 0.096 | 0.8066901564598083 | 0.8066901564598083 | None | 0.7845070362091064 | 0.022183 | False | blocked_by_fullgraph_backbone | completed |
| imdb | 0.05 | 0.4244222342967987 | 0.4244222342967987 | None | 0.42410993576049805 | 0.000312 | False | blocked_by_fullgraph_backbone | completed |
| ogbn-arxiv | 0.12 | 0.6615641117095947 | 0.6615641117095947 | 0.6143036484718323 | 0.5967738628387451 | 0.06479 | True | blocked_by_fullgraph_backbone | completed |
| ogbn-products | 0.12 | 0.6884398460388184 | 0.6884398460388184 | 0.6576123833656311 | 0.6586742401123047 | 0.029766 | True | blocked_by_fullgraph_backbone | completed |

Identity rows use exact/full-demand diagnostics when available; otherwise they are explicit proxies or missing-input rows.
Rows with schema/config mismatch are retained for diagnosis but excluded from promoted best-row conclusions.

- CSV: `experiments\tables\identity_condensation_audit_seed42.csv`
