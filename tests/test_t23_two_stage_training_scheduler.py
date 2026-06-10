from shadow_hgc.training.two_stage import select_best_t23_row, t23_selection_score


def test_t23_two_stage_selection_score_uses_requested_formula():
    assert t23_selection_score(0.7, 0.5) == 0.725
    rows = [
        {"variant": "a", "valid_acc": 0.70, "valid_macro_f1": 0.20, "accuracy": 0.80},
        {"variant": "b", "valid_acc": 0.69, "valid_macro_f1": 0.80, "accuracy": 0.70},
    ]
    assert select_best_t23_row(rows)["variant"] == "b"
