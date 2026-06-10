# T24 Mid/Large SFT-Condense Stage Summary

## Main Results

| dataset | best | accuracy | macro_f1 | status |
|---|---|---|---|---|
| ogbn-arxiv | A0_current_A3_true_sagn_lite_v3_replay | 0.7016645063061951 | 0.5048992808650066 | completed_replay |
| ogbn-products | P3_shadow_condensed_medoid_b1 | 0.174593814714352 | 0.11342605750338396 | completed_streaming |
| Reddit | sagn_lite_v4_x0_smoke | 0.32427337765693665 | 0.0982786856773423 | completed_x0_sft_smoke |

## Required Answers

1. Did arxiv improve beyond 0.7017? `False`; best=`0.7016645063061951`.
2. Did arxiv pass 0.715 / 0.725 / 0.740? `False` / `False` / `False`.
3. Which arxiv blocks were selected? `["X0", "X1_cite_ref", "X1_cited_by", "X2_cite_ref", "X2_cited_by", "X3_mix", "Xres1_cite_ref", "Xres1_cited_by", "structure", "Y1_cite_ref", "Y1_cited_by", "Y2_cite_ref", "Y2_cited_by", "Y3_mix"]`.
4. Did products run full streaming SFT recovery rather than proxy rows? `True`.
5. Products 0.25% shadow-condensed accuracy: `0.174593814714352`.
6. Products 0.50% shadow-condensed accuracy: `0.2740235263710349`.
7. Did Reddit fullgraph SFT complete? `True`.
8. Did Reddit condensation complete at 0.50% full-node ratio? `False`.
9. Fixed bucket ratios: arxiv=0.50%, Reddit=0.50%, products=0.25%.
10. All T24 main ratios are reported as full-node ratios.
11. Any promoted row used forbidden components? `False`.
12. Fullgraph-to-condensed gap: arxiv=`not_run_until_gate_A`, Reddit=``, products_0.25=`0.5809842433049522`.
13. Biggest bottleneck: products/reddit full streaming training resource; arxiv fullgraph teacher gate remains below 0.715.
14. Next dataset after arxiv/products/Reddit: ogbn-papers100M train-target-only dry-run to server execution.

## Stage Changes

- Added T24 scale-bucket full-node ratio policy and safety validation.
- Added arxiv filter-bank v4 / LabelReuse v3 wrappers and SFT v4 model aliases.
- Added products SFT signature cache and memmap recovery script with proxy promotion blocked.
- Added Reddit processed-cache loader and T24 Reddit SFT/condense entrypoints.
- Added unified T24 tables, ultra dry-run, tests, and configuration.

- Stage CSV: `experiments\tables\t24_stage_summary_seed42.csv`
