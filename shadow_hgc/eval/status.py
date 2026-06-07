from __future__ import annotations


def exception_status(exc: BaseException) -> str:
    if isinstance(exc, MemoryError):
        return "oom"
    if isinstance(exc, TimeoutError):
        return "oot"
    message = str(exc).lower()
    if (
        "out of memory" in message
        or "cannot allocate memory" in message
        or "not enough memory" in message
    ):
        return "oom"
    if "timed out" in message or "timeout" in message:
        return "oot"
    return "experiment_failed"
