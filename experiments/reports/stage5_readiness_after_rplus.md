# Stage 5 Readiness After R+

## Decision

Proceed to paper100M feasibility with a conservative Shadow-HGC-R+ configuration selected from the arxiv rescue signal, but require chunked/memmap diffusion materialization before ultra-scale execution.

## Evidence

- ogbn-arxiv improved from best base `0.4664` to `0.5369` with diffusion/coverage/logit adjustment.
- ogbn-products base remains strong at `0.5891`, but in-memory diffusion R+ rows are OOM; this is a scalability result, not a hidden failure.
- IMDB R+ improved from base `0.3507` to full R+ `0.3810`, supporting rank-adaptive/meta-path rescue.
- DBLP remains rank-saturated with low reconstruction error, so no extra small/medium ratio sweep is needed before Stage 5.

## Selected Stage-5 Configuration

- Variant: Shadow-HGC-R+.
- Ratio: 0.06 as a practical starting point; it produced the best arxiv R+ result with logit adjustment while avoiding the largest 12% budget.
- Feature mode: diffusion with steps 1 and 2 plus high-pass, implemented out-of-core for paper100M.
- Shadow policy: fixed initially; rank-adaptive remains enabled for hetero/high-rank rescue, but medium adaptive did not beat fixed in this run.
- Skeleton policy: coverage at 0.65, k_max 8.
- Loss: sqrt_weighted_logit_adjusted for class coverage monitoring.

## Stage-5 Gate

Before full paper100M training, run a dry-run that estimates diffusion feature memmap bytes, edge scans, peak RAM, and edge-slice cache bytes. Do not use the current in-memory medium diffusion path directly on paper100M.
