# T23 Arxiv SFT Boost

Rows use the T23 v3 naming and selection score. The metrics are replayed from the existing local full-edge memmap SFT runs so the stage can compare against the known seed42 results without re-running all large OGB jobs.

| variant | model_type | accuracy | macro_f1 | valid_acc | valid_macro_f1 | selection_score | gate_0715 | gate_0725 | gate_0740 |
|---|---|---|---|---|---|---|---|---|---|
| A0_current_best_replay | gamlp_lite | 0.6530461082649219 | 0.4173269426774673 | 0.6687472733984362 | 0.4468387494868408 | 0.6910892108727782 | False | False | False |
| A1_add_X3_Xres2 | gamlp_lite_v3 | 0.6729008497417854 | 0.42834986600271846 | 0.6870364777341521 | 0.45326891608573605 | 0.7096999235384389 | False | False | False |
| A2_add_LabelReuse_Y1Y2Y3 | gamlp_lite_v3 | 0.6947513527971524 | 0.48676255326594575 | 0.7089499647639183 | 0.5106144301076361 | 0.7344806862693001 | False | False | False |
| A3_true_sagn_lite_v3 | sagn_lite_v3 | 0.7016645063061951 | 0.5048992808650066 | 0.7196550219806034 | 0.5282765282533703 | 0.7460688483932719 | False | False | False |
| A4_gamlp_recursive_v3 | gamlp_recursive_v2 | 0.6894636133572002 | 0.48908222402081003 | 0.7104265243800127 | 0.5182331667655019 | 0.7363381827182879 | False | False | False |
| A5_two_stage_sqrt_to_ce | gamlp_lite_v3 | 0.6684155299055614 | 0.4394560702948403 | 0.6877747575421994 | 0.4622497304725009 | 0.7108872440658245 | False | False | False |
| A6_A4_plus_A5 | gamlp_recursive_v2 | 0.6822624117852808 | 0.4636449392454682 | 0.700493305144468 | 0.4920471183500915 | 0.7250956610619725 | False | False | False |
| A7_A4_plus_LabelReuse_plus_two_stage | gamlp_recursive_v2 | 0.6892578647408596 | 0.48908966374168283 | 0.7101245008221753 | 0.5134796903776709 | 0.7357984853410588 | False | False | False |

- Best by selection score: `A3_true_sagn_lite_v3` with score `0.7460688483932719`.
- CSV: `experiments\tables\t23_arxiv_sft_boost_seed42.csv`
