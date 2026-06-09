# Fullgraph Parity + Condensation Recovery Summary

## 0. Stage Scope And Code Changes

- Added fullgraph parity audit output with required hashes, split counts, schema counts, gate decisions, and resource fields.
- Added schema alignment audit with `current_processed` versus `full_schema` loader modes; default condensation loading remains unchanged.
- Added full-schema small-data loader entry point for alignment-only experiments while preserving the incoming-to-target default path.
- Added identity/prototype/shadow gap decomposition with explicit compatibility flags so mismatched diagnostic rows are retained but not promoted.
- Added explicit compiled block-stat APIs: `fit_block_stats`, `freeze_block_stats`, and `apply_block_stats`; stats are fit on original train target demand rows.
- Added KD v2 gate/log helpers and tests; KD rows are skipped unless teacher quality logs pass the gate.
- Added no-diffusion promoted-row validation so diffusion, products P2/two-hop LAD, CoverageMedoid, source anchors, and invalid KD rows are excluded from best summaries.
- Added the stage runner and five dataset-specific candidate tables required by the prompt.

## 1. Fullgraph Parity Status By Dataset

| dataset | best_variant | accuracy | gate | gate_passed | blocked |
|---|---|---|---|---|---|
| acm | fullgraph_sehgnn_lite_tuned | 0.9027384519577026 | 0.9 | True | False |
| dblp | fullgraph_dblp_full_schema_sehgnn_lite | 0.8066901564598083 | 0.9 | False | True |
| imdb | fullgraph_imdb_sehgnn_lite_MAM_MDM_MKM | 0.4244222342967987 | 0.55 | False | True |
| ogbn-arxiv | fullgraph_lad_table_teacher | 0.6615641117095947 | 0.68 | False | True |
| ogbn-products | fullgraph_lad_table_teacher | 0.6884398460388184 | 0.7 | False | True |

## 2. Schema Completeness Status By Dataset

| dataset | loader_name | target_type | metapath_available | metapath_missing | freehgc_or_hgb_alignment_status | notes |
|---|---|---|---|---|---|---|
| acm | current_processed | paper | ["PAP", "PSP", "PTP"] | [] | aligned | full_schema audit only; default condensation path remains incoming-to-target |
| acm | full_schema | paper | ["PAP", "PSP", "PTP"] | [] | aligned | full_schema audit only; default condensation path remains incoming-to-target |
| dblp | current_processed | author | ["APA"] | ["APVPA", "APTPA", "APCPA"] | partial | full_schema audit only; default condensation path remains incoming-to-target |
| dblp | full_schema | author | ["APA", "APVPA", "APTPA", "APCPA"] | [] | aligned | full_schema audit only; default condensation path remains incoming-to-target |
| imdb | current_processed | movie | ["MAM", "MDM", "MKM"] | [] | aligned | full_schema audit only; default condensation path remains incoming-to-target |
| imdb | full_schema | movie | ["MAM", "MDM", "MKM"] | [] | aligned | full_schema audit only; default condensation path remains incoming-to-target |
| ogbn-arxiv | ogb_homogeneous | paper | [] | [] | not_applicable | homogeneous OGB schema; full feature hashing is skipped by resource guard |
| ogbn-products | ogb_homogeneous | product | [] | [] | not_applicable | homogeneous OGB schema; full feature hashing is skipped by resource guard |

## 3. Identity Condensation Sanity Status

| dataset | ratio | fullgraph_acc | identity_condensed_acc | prototype_oracle_acc | shadow_hgc_acc | schema_compatible | bottleneck_label |
|---|---|---|---|---|---|---|---|
| acm | 0.12 | 0.9027384519577026 | 0.9027384519577026 |  | 0.8937677145004272 | False | training_head_bottleneck |
| dblp | 0.096 | 0.8066901564598083 | 0.8066901564598083 |  | 0.7845070362091064 | False | blocked_by_fullgraph_backbone |
| imdb | 0.05 | 0.4244222342967987 | 0.4244222342967987 |  | 0.42410993576049805 | False | blocked_by_fullgraph_backbone |
| ogbn-arxiv | 0.12 | 0.6615641117095947 | 0.6615641117095947 | 0.6143036484718323 | 0.5967738628387451 | True | blocked_by_fullgraph_backbone |
| ogbn-products | 0.12 | 0.6884398460388184 | 0.6884398460388184 | 0.6576123833656311 | 0.6586742401123047 | True | blocked_by_fullgraph_backbone |

## 4. Gap Decomposition Table

| dataset | full_to_identity_gap | identity_to_oracle_gap | oracle_to_shadow_gap | full_to_shadow_gap | compatibility_reason |
|---|---|---|---|---|---|
| acm | 0.0 |  |  | 0.008971 | missing identity/oracle/shadow row |
| dblp | 0.0 |  |  | 0.022183 | missing identity/oracle/shadow row |
| imdb | 0.0 |  |  | 0.000312 | missing identity/oracle/shadow row |
| ogbn-arxiv | 0.0 | 0.04726 | 0.01753 | 0.06479 | compatible_existing_rows |
| ogbn-products | 0.0 | 0.030827 | -0.001062 | 0.029766 | compatible_existing_rows |

## 5. Promoted Rows

| dataset | variant | requested_ratio | accuracy | macro_f1 | status | invalid_reasons |
|---|---|---|---|---|---|---|
| acm | S1_clean_metapath_sehgnn_tuned_h256_d0p3_lr0p003_class_balanced | 0.15 | 0.9027384519577026 | 0.9034235080083212 | completed | [] |

## 6. Blocked Rows And Reasons

| dataset | variant | requested_ratio | status | reason | invalid_reasons |
|---|---|---|---|---|---|
| dblp | S0_current_best | 0.005 | blocked_by_fullgraph_backbone | DBLP fullgraph/schema gate did not pass; rows are diagnostics and are not promoted | [] |
| dblp | S1_clean_APA_sehgnn | 0.005 | blocked_by_fullgraph_backbone | DBLP fullgraph/schema gate did not pass; rows are diagnostics and are not promoted | [] |
| dblp | S0_current_best | 0.065 | blocked_by_fullgraph_backbone | DBLP fullgraph/schema gate did not pass; rows are diagnostics and are not promoted | [] |
| dblp | S1_clean_APA_sehgnn | 0.065 | blocked_by_fullgraph_backbone | DBLP fullgraph/schema gate did not pass; rows are diagnostics and are not promoted | [] |
| dblp | S0_current_best | 0.096 | blocked_by_fullgraph_backbone | DBLP fullgraph/schema gate did not pass; rows are diagnostics and are not promoted | [] |
| dblp | S1_clean_APA_sehgnn | 0.096 | blocked_by_fullgraph_backbone | DBLP fullgraph/schema gate did not pass; rows are diagnostics and are not promoted | [] |
| imdb | S1_clean_MAM_MDM_MKM | 0.005 | blocked_by_fullgraph_backbone | IMDB fullgraph gate did not pass; clean rows are diagnostics and Path-LAD/source-anchor paths are not promoted | [] |
| imdb | S1_clean_MAM_MDM_MKM | 0.025 | blocked_by_fullgraph_backbone | IMDB fullgraph gate did not pass; clean rows are diagnostics and Path-LAD/source-anchor paths are not promoted | [] |
| imdb | S1_clean_MAM_MDM_MKM | 0.05 | blocked_by_fullgraph_backbone | IMDB fullgraph gate did not pass; clean rows are diagnostics and Path-LAD/source-anchor paths are not promoted | [] |
| ogbn-arxiv | LAD_reference | 0.06 | diagnostic_existing | existing no-diffusion LAD_reference diagnostic retained; no P2/diffusion path run | [] |
| ogbn-arxiv | LAD_reference_with_fixed_block_stats | 0.06 | skipped_blocked_by_fullgraph_backbone | ogbn-arxiv fullgraph teacher gate did not pass; no-diffusion LAD_reference retained as diagnostic | [] |
| ogbn-arxiv | stronger_table_head | 0.06 | skipped_blocked_by_fullgraph_backbone | ogbn-arxiv fullgraph teacher gate did not pass; no-diffusion LAD_reference retained as diagnostic | [] |
| ogbn-arxiv | LAD_reference | 0.12 | diagnostic_existing | existing no-diffusion LAD_reference diagnostic retained; no P2/diffusion path run | [] |
| ogbn-arxiv | LAD_reference_with_fixed_block_stats | 0.12 | skipped_blocked_by_fullgraph_backbone | ogbn-arxiv fullgraph teacher gate did not pass; no-diffusion LAD_reference retained as diagnostic | [] |
| ogbn-arxiv | stronger_table_head | 0.12 | skipped_blocked_by_fullgraph_backbone | ogbn-arxiv fullgraph teacher gate did not pass; no-diffusion LAD_reference retained as diagnostic | [] |
| ogbn-products | LAD_reference | 0.06 | diagnostic_existing | existing no-diffusion LAD_reference diagnostic retained; no P2/diffusion path run | [] |
| ogbn-products | LAD_reference_balanced_softmax | 0.06 | skipped_blocked_by_fullgraph_backbone | ogbn-products fullgraph teacher gate did not pass; products P2/diffusion paths are not run | [] |
| ogbn-products | LAD_reference_logit_adjusted | 0.06 | skipped_blocked_by_fullgraph_backbone | ogbn-products fullgraph teacher gate did not pass; products P2/diffusion paths are not run | [] |
| ogbn-products | LAD_reference_label_smoothing | 0.06 | skipped_blocked_by_fullgraph_backbone | ogbn-products fullgraph teacher gate did not pass; products P2/diffusion paths are not run | [] |
| ogbn-products | stronger_table_head | 0.06 | skipped_blocked_by_fullgraph_backbone | ogbn-products fullgraph teacher gate did not pass; products P2/diffusion paths are not run | [] |
| ogbn-products | LAD_reference | 0.12 | diagnostic_existing | existing no-diffusion LAD_reference diagnostic retained; no P2/diffusion path run | [] |
| ogbn-products | LAD_reference_balanced_softmax | 0.12 | skipped_blocked_by_fullgraph_backbone | ogbn-products fullgraph teacher gate did not pass; products P2/diffusion paths are not run | [] |
| ogbn-products | LAD_reference_logit_adjusted | 0.12 | skipped_blocked_by_fullgraph_backbone | ogbn-products fullgraph teacher gate did not pass; products P2/diffusion paths are not run | [] |
| ogbn-products | LAD_reference_label_smoothing | 0.12 | skipped_blocked_by_fullgraph_backbone | ogbn-products fullgraph teacher gate did not pass; products P2/diffusion paths are not run | [] |
| ogbn-products | stronger_table_head | 0.12 | skipped_blocked_by_fullgraph_backbone | ogbn-products fullgraph teacher gate did not pass; products P2/diffusion paths are not run | [] |

## 7. Dropped Components

- High-dimensional diffusion remains diagnostic-only and is not promoted.
- Products P2 / two-hop LAD and products diffusion were not run in this stage.
- CoverageMedoid and source anchors were not promoted.
- Old KD rows are not promoted; KD v2 is skipped unless teacher quality logs pass the gate.
- DBLP and IMDB condensation SOTA chasing is blocked while fullgraph/schema alignment remains below gate.

## 8. Next-Stage Recommendation

- ACM is the only dataset eligible for clean S1 tuning in this sprint if the fullgraph acceptable gate passes; use the best valid row from `acm_s1_clean_tuned_seed42.csv`.
- DBLP needs schema/backbone alignment before condensation claims; full-schema loading is now auditable but the fullgraph gate remains the decision point.
- IMDB needs a stronger aligned fullgraph backbone before Path-LAD/source-anchor or KD experiments are meaningful.
- arxiv/products should stay no-diffusion; recover the fullgraph teacher ceiling before spending more runs on compressed variants.

## 9. Acceptance Checklist

- Fullgraph parity table exists: `True` (`experiments\tables\fullgraph_parity_seed42.csv`)
- Schema alignment table exists: `True` (`experiments\tables\schema_alignment_audit_seed42.csv`)
- Identity audit table exists: `True` (`experiments\tables\identity_condensation_audit_seed42.csv`)
- ACM candidate table exists: `True` (`experiments\tables\acm_s1_clean_tuned_seed42.csv`)
- DBLP candidate table exists: `True` (`experiments\tables\dblp_schema_fixed_candidate_seed42.csv`)
- IMDB candidate table exists: `True` (`experiments\tables\imdb_fullgraph_first_candidate_seed42.csv`)
- arxiv candidate table exists: `True` (`experiments\tables\arxiv_no_diffusion_recovery_seed42.csv`)
- products candidate table exists: `True` (`experiments\tables\products_no_diffusion_recovery_seed42.csv`)
- Invalid rows are retained in artifacts but excluded from promoted best-row summaries.
- KD v2 is skipped unless teacher gate passes.

## 10. paper100M Local Trial

The local paper100M directory was tested with guarded memmap access:

- Dataset root: `D:\Shadow-HGC\dataset\paper100M`
- Memmap root: `D:\Shadow-HGC\dataset\paper100M\processed\papers100m_memmap`
- Nodes: `111059956`
- Edges: `1615685872`
- Train/valid/test nodes: `1207179` / `125265` / `214338`
- Feature dim: `128`
- Local smoke status: `completed_smoke`
- Smoke setting: `sample_train=20000`, `sample_valid=5000`, `epochs=10`, `no_diffusion=true`
- Smoke train/valid accuracy: `0.38175` / `0.33800`
- Observed classes in smoke: `172`
- Full-scale local status: `blocked_resource_guard`
- Conservative full-scale peak RAM estimate: `115.37 GB`
- Available RAM at trial time: `23.43 GB`
- Expected full edge scans: `4`
- Edge-slice cache estimate: `0.329 GB`
- Disk spill used by estimate: `false`

No hard OOM was triggered because the full run was stopped by the resource guard before allocating unsafe tensors. Use these commands on a larger server:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts/run_paper100m_local_trial.py --dataset-root D:/Shadow-HGC/dataset/paper100M --output-dir experiments/logs/paper100m_local_trial_seed42 --seed 42 --sample-train 200000 --sample-valid 50000 --epochs 50 --full-scale --no-diffusion
```

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts/dry_run_ultra.py --dataset ogbn-papers100M --ratios 0.001 0.0025 0.005 --output experiments/logs/paper100m_local_trial_seed42/paper100m_ultra_dry_run_server.json
```

Linux server template:

```bash
python scripts/run_paper100m_local_trial.py --dataset-root /path/to/paper100M --output-dir experiments/logs/paper100m_server_seed42 --seed 42 --sample-train 200000 --sample-valid 50000 --epochs 50 --full-scale --no-diffusion
```

Artifacts:

- Local trial JSON: `experiments\logs\paper100m_local_trial_seed42\paper100m_local_trial_seed42.json`
- Local trial CSV: `experiments\tables\paper100m_local_trial_seed42.csv`
- Local trial report: `experiments\reports\paper100m_local_trial_seed42.md`
- Ultra dry-run JSON: `experiments\logs\paper100m_local_trial_seed42\paper100m_ultra_dry_run_seed42.json`

## Files

- Schema audit: `experiments\tables\schema_alignment_audit_seed42.csv`
- Fullgraph parity: `experiments\tables\fullgraph_parity_seed42.csv`
- Identity audit: `experiments\tables\identity_condensation_audit_seed42.csv`
- ACM candidates: `experiments\tables\acm_s1_clean_tuned_seed42.csv`
- DBLP candidates: `experiments\tables\dblp_schema_fixed_candidate_seed42.csv`
- IMDB candidates: `experiments\tables\imdb_fullgraph_first_candidate_seed42.csv`
- arxiv candidates: `experiments\tables\arxiv_no_diffusion_recovery_seed42.csv`
- products candidates: `experiments\tables\products_no_diffusion_recovery_seed42.csv`
