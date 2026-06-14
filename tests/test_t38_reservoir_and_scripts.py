from __future__ import annotations

import re
from pathlib import Path

from shadow_hgc.sft.unified_reservoir import teacher_free_selection_signature


def test_t38_teacher_free_signature_ignores_valid_and_test_labels() -> None:
    base = teacher_free_selection_signature(
        candidate_ids=[1, 2, 3],
        train_labels={1: 0, 2: 1, 3: 1},
        valid_labels={10: 0},
        test_labels={20: 1},
    )
    changed = teacher_free_selection_signature(
        candidate_ids=[1, 2, 3],
        train_labels={1: 0, 2: 1, 3: 1},
        valid_labels={10: 99, 11: 98},
        test_labels={20: 97, 21: 96},
    )

    assert base == changed


def test_t38_unified_runner_does_not_assign_old_method_ids_to_main_rows() -> None:
    source = Path("scripts/run_t38_unified_stage.py").read_text(encoding="utf-8")
    forbidden_assignment = re.compile(
        r"method\s*=\s*[\"'](?:products_uca_|reddit_ttcpp_|reddit_stt_|scr_|stt_randcore_)",
        re.MULTILINE,
    )

    assert forbidden_assignment.search(source) is None
