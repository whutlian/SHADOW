# T2.2-SFT-NL-OGB Stage Summary

## Required Answers

1. Did arxiv improve beyond 0.6544? True; best=0.7016645063061951.
2. Did arxiv reach 0.68 / 0.70 / 0.74 gates? True / True / False.
3. Which arxiv blocks were useful? Best variant `A3_true_sagn_lite_v2` selected `["X0", "X1_cite_ref", "X1_cited_by", "X2_cite_ref", "X2_cited_by", "X3_mix", "Xres1_cite_ref", "Xres1_cited_by", "structure", "Y1_cite_ref", "Y1_cited_by", "Y2_cite_ref", "Y2_cited_by", "Y3_mix"]`.
4. Did products improve beyond 0.7030? True; best=0.7555780580193042.
5. Did products reach 0.72 / 0.74 gates? True / True.
6. Did products macro-F1 reach 0.36? True.
7. Which products configuration is best? `P7_sagn_lite_v2`.
8. Did DBLP identity replay match fullgraph? True.
9. Did DBLP prototype oracle recover within 3 points? True.
10. Did DBLP shadow condensed reach 0.90? True.
11. Did ACM reach 0.93? False.
12. Was IMDB only diagnostic? True.
13. Did any promoted row use forbidden signals? False.
14. Cache footprint rows written: 8.
15. Ready for next condensation ratio sweep: DBLP.
16. Ready for paper100M/MAG240M scaling: dry-run only; server recommended for both ultra-scale rows.

## Best Rows

| dataset | best_variant | accuracy | macro_f1 | status |
|---|---|---|---|---|
| ogbn-arxiv | A3_true_sagn_lite_v2 | 0.7016645063061951 | 0.5048992808650066 | promoted_short |
| ogbn-products | P7_sagn_lite_v2 | 0.7555780580193042 | 0.4046991170720907 | promoted_short |
| acm | ACM_H512_D0p3_CE | 0.9159584641456604 | 0.9164112210273743 | completed |
| dblp | 0.05 | 0.924647867679596 | 0.920296385884285 | promoted |
| imdb | diagnostic_only | 0.47158026695251465 | 0.3900045245885849 | diagnostic_only |

## Stage Changes

- Added T2.2 filter-bank API with X3/Xres2/Y1-Y3/structure blocks, fp16 memmap manifests, block index, and train-row block stats.
- Added train-label-only LabelReuse blocks and support/entropy/max-affinity features.
- Added SFTTeacherV3 with SAGN-lite-v2, GAMLP-lite-v2, recursive GAMLP-v2, residual gated v2, and two-stage training support.
- Added T2.2 promotion validation, block budget dry-run, lazy selected-block training, and DBLP SFT recovery diagnostics.

- Stage CSV: `experiments\tables\t22_stage_summary_seed42.csv`
