from scripts.t23_common import no_forbidden_flags


def test_t23_no_forbidden_flags_helper_rejects_promoted_bad_rows():
    clean = {"uses_logits_as_input": False, "uses_kd": False, "uses_dense_p2": False}
    bad = dict(clean, uses_logits_as_input=True)
    assert no_forbidden_flags(clean) is True
    assert no_forbidden_flags(bad) is False
