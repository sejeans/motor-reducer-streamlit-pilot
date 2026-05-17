"""모터-감속기 5클래스 라벨 인코딩 매핑 (Phase 1 1-5)."""
from __future__ import annotations

from typing import Dict

# Phase 3 split·Phase 7 평가·Phase 6 추론 응답까지 동일 매핑을 공유해야 하므로 고정.
LABEL_MAP: Dict[str, int] = {
    "NORMAL": 0,
    "ECC10": 1,
    "ECC20": 2,
    "DEMAG": 3,
    "REDUC": 4,
}

INV_LABEL_MAP: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

__all__ = ["LABEL_MAP", "INV_LABEL_MAP"]
