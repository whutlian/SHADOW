# T23 SFT Arxiv + Condense Plan

## Scope

Implement T23 as an opt-in stage on top of the existing T21/T22 SFT/preprop stack. The default Shadow-HGC-R-1 path remains unchanged.

## Implementation Tasks

- Add T23 filter bank and LabelReuse v2 wrappers with the requested block names, fp16 memmap path, train-row stats, explicit citation-direction names, and no E-by-d materialization.
- Add SAGN/GAMLP lite v3 compatibility aliases, label dropout, diagnostics, and the `valid_acc + 0.05 * valid_macro_f1` selection score.
- Add SFT condensation helpers for signatures, class-wise budgets, centroid/medoid/herding selection, and b=2 nonnegative assignment diagnostics.
- Add T23 scripts for arxiv SFT boost, products recovery, DBLP ratio sweep, ACM tune/sweep, ultra dry-run, and an orchestrating stage runner.
- Add T23 config, method note, stage summary, and prompt-completion checklist.

## Verification

- Add and run T23 tests named in the prompt.
- Run the stage script with the local conda `pytorch` Python.
- Run the full test suite, or report any timeout/blocker with the targeted T23 test results.

## Delivery

- Write `experiments/tables/t23_stage_summary_seed42.csv`.
- Write `experiments/reports/t23_stage_summary.md`.
- Push current project state to GitHub after verification.
