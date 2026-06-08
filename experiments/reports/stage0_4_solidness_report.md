# Stage 0-4 Solidness Report

Stage 5 status: **UNBLOCKED**

| Gate | Status | Reason |
| --- | --- | --- |
| Gate 0: tests | PASS | Full current test suite passed in this run. |
| Gate 1: toy | PASS | Toy main/private/self/full all have 1.0 accuracy and macro-F1. |
| Gate 2: small datasets | WARN | Shadow matches/beats best classical baseline on 3/3 datasets. Self-only gap >3 points on: imdb. |
| Gate 3: medium datasets | WARN | Products is monotonic/near-monotonic; arxiv still needs comparison against self/private/full graph. |
| Gate 4: I/O dry run | PASS | Ratio-aware dry-run logs contain memory/disk/scan fields and cache_all_targets=false. |

## Required Interpretation

1. Target-ratio matched baseline status: Shadow matches/beats best classical baseline on 3/3 datasets. Self-only gap >3 points on: imdb.
2. Total-node matched comparison is available when `baseline_match_mode=total_condensed_nodes` appears in `small_ratio_main.csv`.
3. IMDB is treated as a visible failure case if it remains below self-only and K-Center.
4. relation_linear vs relation_mlp should be read from rows with `model=relation_linear` and `model=relation_mlp`.
5. Products ratio scaling: Products is monotonic/near-monotonic; arxiv still needs comparison against self/private/full graph.
6. Arxiv remains a diagnostic target when below self-only/private/full graph by large margins.
7. Ratio dry-run status: Ratio-aware dry-run logs contain memory/disk/scan fields and cache_all_targets=false.
8. Ratio logs use `r...` names; count-mode compatibility uses `count...` names.
