# T23-SFT-Arxiv+Condense Stage Summary

## Stage Outputs

| dataset | best_row | accuracy | macro_f1 | selection_score | status |
|---|---|---|---|---|---|
| ogbn-arxiv | A3_true_sagn_lite_v3 | 0.7016645063061951 | 0.5048992808650066 | 0.7460688483932719 | completed_replay |
| ogbn-products | shadow_b1 | 0.6083323955535889 | 0.3340439060108459 |  | completed_proxy |
| dblp | current_reference | 0.924647867679596 | 0.920296385884285 |  | completed_replay |
| acm | ACM_H512_D0p3_CE | 0.9159584641456604 | 0.9164112210273743 | 0.9861617396275203 | completed_replay |
| ultra-dryrun | train_target_only_policy |  |  |  | completed |

## Required Answers

1. Arxiv best row is `A3_true_sagn_lite_v3` with acc `0.7016645063061951` and selection score `0.7460688483932719`.
2. Arxiv reached 0.715/0.725/0.740 gates: `False` / `False` / `False`.
3. Arxiv v3 head aliases and label-dropout diagnostics are implemented; replay metrics come from local T22 full-edge memmap runs.
4. Products fullgraph teacher reference is `0.7555780580193042`.
5. Products best condensed proxy row is `shadow_b1` at ratio `0.05%` with acc `0.6083323955535889`.
6. Products recovery uses no logits/KD/dense-P2/E-by-d flags; full streaming SFT recovery sweep is represented by proxy rows in default mode.
7. DBLP requested ratio grid rows: `30`.
8. DBLP best replayed SFT-condense row is `current_reference` with acc `0.924647867679596`.
9. ACM best tune row is `ACM_H512_D0p3_CE` with acc `0.9159584641456604`.
10. ACM condensed sweep gate status: `skipped_by_gate`.
11. Ultra dry-run rows written: `8`.
12. papers100M/MAG all-target cache is marked forbidden by T23 ultra policy; train-target-only is the allowed path.
13. Any forbidden promoted/input flags found: `False`.
14. Method note and config are included for T23 opt-in behavior.
15. Default Shadow-HGC-R-1 path remains unchanged; T23 is opt-in.

## Code Changes

- Added T23 filter bank v3 and LabelReuse v2 wrappers with train-label-only policy, Y0/Y4/Yres1 naming, fp16 memmap compatibility, and no E-by-d materialization.
- Added SAGN/GAMLP lite v3 aliases, label dropout diagnostics, and T23 selection score helpers.
- Added SFT signature, centroid/medoid/herding condensation helpers, b=2 nonnegative assignment wrapper, and recovery gap utilities.
- Added T23 arxiv/products/DBLP/ACM/ultra scripts, config, method note, and tests.

- Stage CSV: `experiments\tables\t23_stage_summary_seed42.csv`
