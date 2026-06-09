from shadow_hgc.logits.correct_smooth import select_best_validation_row


def test_t1_validation_selection_ignores_better_test_row():
    rows = [
        {"name": "valid_best", "valid_acc": 0.8, "valid_macro_f1": 0.4, "test_acc": 0.1},
        {"name": "test_best", "valid_acc": 0.7, "valid_macro_f1": 0.9, "test_acc": 0.99},
    ]

    selected = select_best_validation_row(rows)

    assert selected["name"] == "valid_best"
