# Rank Diagnostics Summary

## Scope

- Seed: 42 only.
- Small ratios: 0.5%, 2.5%, 9.6%.
- Medium ratios: 0.5%, 6.0%, 12.0%.
- Diagnostics are computed from train-target relation demand/residual matrices only.

## Relation Diagnostics

| Dataset | Ratio | Relation | Stable rank | Entropy rank | Recon err | Acc | Macro-F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| acm | 0.5000% | paper--cite_ref-->paper | 1.1076 | 1.3756 | 0.2269 | 0.5925 | 0.5005 |
| acm | 0.5000% | paper--cited_by-->paper | 1.3332 | 1.7557 | 0.1755 | 0.5925 | 0.5005 |
| acm | 0.5000% | author--writes-->paper | 2.3730 | 4.9432 | 0.7883 | 0.5925 | 0.5005 |
| acm | 0.5000% | subject--subject_of-->paper | 1.0357 | 1.2042 | 0.1366 | 0.5925 | 0.5005 |
| acm | 0.5000% | term--term_in-->paper | 1.0086 | 1.0545 | 0.0574 | 0.5925 | 0.5005 |
| acm | 2.5000% | paper--cite_ref-->paper | 1.3369 | 1.7593 | 0.2647 | 0.6331 | 0.5338 |
| acm | 2.5000% | paper--cited_by-->paper | 1.0001 | 1.0010 | 0.6344 | 0.6331 | 0.5338 |
| acm | 2.5000% | author--writes-->paper | 2.8609 | 9.2415 | 0.7781 | 0.6331 | 0.5338 |
| acm | 2.5000% | subject--subject_of-->paper | 1.0933 | 1.5669 | 0.1621 | 0.6331 | 0.5338 |
| acm | 2.5000% | term--term_in-->paper | 1.0109 | 1.0673 | 0.0800 | 0.6331 | 0.5338 |
| acm | 9.6000% | paper--cite_ref-->paper | 1.2780 | 1.6898 | 0.1170 | 0.6596 | 0.6533 |
| acm | 9.6000% | paper--cited_by-->paper | 1.0002 | 1.0019 | 0.4288 | 0.6596 | 0.6533 |
| acm | 9.6000% | author--writes-->paper | 5.7765 | 41.4851 | 0.9093 | 0.6596 | 0.6533 |
| acm | 9.6000% | subject--subject_of-->paper | 1.1160 | 1.8142 | 0.1953 | 0.6596 | 0.6533 |
| acm | 9.6000% | term--term_in-->paper | 1.0313 | 1.1646 | 0.0694 | 0.6596 | 0.6533 |
| dblp | 0.5000% | paper--written_by-->author | 7.3620 | 13.0759 | 0.0005 | 0.8313 | 0.8237 |
| dblp | 2.5000% | paper--written_by-->author | 3.5481 | 12.5984 | 0.0192 | 0.8289 | 0.8213 |
| dblp | 9.6000% | paper--written_by-->author | 6.8047 | 40.2261 | 0.0361 | 0.8243 | 0.8165 |
| imdb | 0.5000% | director--directs-->movie | 3.0801 | 10.2886 | 0.9378 | 0.2926 | 0.2849 |
| imdb | 0.5000% | actor--acts_in-->movie | 1.6369 | 5.3310 | 0.8805 | 0.2926 | 0.2849 |
| imdb | 0.5000% | keyword--keyword_in-->movie | 4.2669 | 11.4990 | 0.6709 | 0.2926 | 0.2849 |
| imdb | 2.5000% | director--directs-->movie | 13.2402 | 24.6113 | 0.9895 | 0.3507 | 0.2956 |
| imdb | 2.5000% | actor--acts_in-->movie | 8.5517 | 23.6453 | 0.9916 | 0.3507 | 0.2956 |
| imdb | 2.5000% | keyword--keyword_in-->movie | 6.4408 | 20.7602 | 0.9217 | 0.3507 | 0.2956 |
| imdb | 9.6000% | director--directs-->movie | 16.4441 | 85.0120 | 0.9600 | 0.3004 | 0.2787 |
| imdb | 9.6000% | actor--acts_in-->movie | 23.3346 | 83.6023 | 0.9949 | 0.3004 | 0.2787 |
| imdb | 9.6000% | keyword--keyword_in-->movie | 15.0155 | 83.1663 | 0.9718 | 0.3004 | 0.2787 |
| ogbn-arxiv | 0.5000% | paper--cite_ref-->paper | 3.5765 | 13.6339 | 0.4057 | 0.4343 | 0.3015 |
| ogbn-arxiv | 0.5000% | paper--cited_by-->paper | 4.4236 | 19.2412 | 0.4897 | 0.4343 | 0.3015 |
| ogbn-arxiv | 6.0000% | paper--cite_ref-->paper | 7.7447 | 42.1552 | 0.5721 | 0.3921 | 0.2564 |
| ogbn-arxiv | 6.0000% | paper--cited_by-->paper | 11.3237 | 50.9930 | 0.5978 | 0.3921 | 0.2564 |
| ogbn-arxiv | 12.0000% | paper--cite_ref-->paper | 9.0804 | 47.9599 | 0.5493 | 0.4664 | 0.3060 |
| ogbn-arxiv | 12.0000% | paper--cited_by-->paper | 12.9420 | 55.6760 | 0.5746 | 0.4664 | 0.3060 |
| ogbn-products | 0.5000% | product--co_purchase-->product | 6.3579 | 32.2697 | 0.4125 | 0.4335 | 0.1954 |
| ogbn-products | 0.5000% | product--co_purchased_by-->product | 6.3579 | 32.2697 | 0.4133 | 0.4335 | 0.1954 |
| ogbn-products | 6.0000% | product--co_purchase-->product | 6.4364 | 34.7164 | 0.5298 | 0.5501 | 0.2451 |
| ogbn-products | 6.0000% | product--co_purchased_by-->product | 6.4361 | 34.7157 | 0.5230 | 0.5501 | 0.2451 |
| ogbn-products | 12.0000% | product--co_purchase-->product | 7.0816 | 35.8702 | 0.3401 | 0.5891 | 0.2643 |
| ogbn-products | 12.0000% | product--co_purchased_by-->product | 7.0816 | 35.8950 | 0.3716 | 0.5891 | 0.2643 |

## Hypothesis Checks

- DBLP flatness is supported when effective ranks and reconstruction errors remain low across ratios.
- IMDB failure is supported when non-target relations show high effective rank and high reconstruction error.
- ACM ratio sensitivity is supported when reconstruction quality and rank leave room for target-budget gains.
- Medium gaps are supported when one-hop reconstruction is moderate but accuracy remains below full-graph sanity levels, motivating diffusion features.

## Files

- CSV: `experiments\tables\rank_diagnostics_small_medium_seed42.csv`
- Report: `experiments\reports\rank_diagnostics_summary.md`
