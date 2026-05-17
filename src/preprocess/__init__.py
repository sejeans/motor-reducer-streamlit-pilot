"""preprocess 서브패키지: chunker / shape_guard / nan_handler / panel_formatter 재export.

Rev. 2026-05-16-02:
- chunker.py 신규 — CSV 1개를 PNG 단위(2,000행)로 환원
- shape_guard.py — chunker 뒤단 길이 sanity check (정상 시 no-op)
- nan_handler.py — 선형 보간 + 결측 비율 반환
- panel_formatter.py — sktime panel `(N, 1, 2000)` float32 변환 + select_column 헬퍼
"""
from src.preprocess.chunker import split_into_chunks
from src.preprocess.nan_handler import clean_signal
from src.preprocess.panel_formatter import select_column, to_sktime_panel
from src.preprocess.shape_guard import ShapeGuardStats, enforce_length, get_stats

__all__ = [
    "split_into_chunks",
    "enforce_length",
    "get_stats",
    "ShapeGuardStats",
    "clean_signal",
    "select_column",
    "to_sktime_panel",
]
