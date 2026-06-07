from shadow_hgc.eval.status import exception_status


def test_exception_status_marks_memory_failures_as_oom():
    assert exception_status(MemoryError("allocation failed")) == "oom"
    assert exception_status(RuntimeError("CUDA out of memory while allocating")) == "oom"
    assert exception_status(RuntimeError("DefaultCPUAllocator: not enough memory")) == "oom"


def test_exception_status_marks_timeout_failures_as_oot():
    assert exception_status(TimeoutError("command timed out")) == "oot"


def test_exception_status_marks_other_failures_as_experiment_failed():
    assert exception_status(RuntimeError("shape mismatch")) == "experiment_failed"
