from __future__ import annotations


STT_STUDENT_ALIASES = {
    "reddit_stt_gamlp_ratio_v2": "gamlp_lite",
    "reddit_stt_sagn_ratio_v2": "sagn_lite_v4",
    "products_stt_official": "sagn_lite_v4",
    "products_stt_balanced": "sagn_lite_v4",
    "arxiv_semantic_stt": "sagn_lite_v4",
}


def resolve_stt_student(method: str, default: str = "sagn_lite_v4") -> str:
    return STT_STUDENT_ALIASES.get(str(method), default)
