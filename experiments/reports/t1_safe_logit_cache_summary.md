# T1.1 Safe Logit Cache Summary

Safe-row cache generation now attempts ACM, DBLP, IMDB, ogbn-arxiv, and ogbn-products historical entries with replayable all-target logits.

- Available cache rows: `5`
- Blocked cache rows: `1`
- CSV: `experiments/tables/t1_safe_logit_cache_index_seed42.csv`

## Blocked Rows

- ogbn-products / R++ base shadow-fusion: products R++ base 500-epoch cache generation was dropped as too slow on local machine; rerun with --run-products-rpp-base to force it
