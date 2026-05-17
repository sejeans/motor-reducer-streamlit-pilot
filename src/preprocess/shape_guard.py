"""Shape guard (Phase 2-1, Rev. 2026-05-16-02).

길이 보장 sanity check 모듈.

위치 부여 (Rev. 2026-05-16-02)
-----------------------------
chunker.py 가 정상 동작하면 모든 입력은 이미 길이 `target` 을 만족한다.
shape_guard 는 그 **뒤단** 에서 동작하는 방어적 sanity check 로 격하되었다.
- 정상 입력(이미 target 길이) → no-op, 원본 1D 배열을 그대로 반환
- 짧은 입력 → 0-pad 처리 후 pad 카운터 증가
- 긴 입력 → 절단 후 truncate 카운터 증가

설계 원칙
---------
1. **No Laziness**: pad/truncate 발생 시 WARN 로그 + 카운터로 가시화 (silent fix 금지).
2. **Simplicity**: 1D 배열에 한정. 다채널은 호출 측에서 컬럼별로 분리해 호출.
3. **Minimal Impact**: 출력 dtype 은 입력 dtype 유지 (float32 → float32).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.common.logger import get_logger

_LOGGER = get_logger("preprocess.shape_guard")


@dataclass
class ShapeGuardStats:
    """shape_guard 호출 통계 누적 컨테이너.

    test 에서 카운터 증감을 검증하고, 운영에서는 주기적으로 dump 해 health-check 한다.
    """

    pad_count: int = 0
    truncate_count: int = 0
    noop_count: int = 0
    history: list[tuple[int, int]] = field(default_factory=list)
    # history: List[(input_len, target_len)] — 비정상 호출만 기록

    def reset(self) -> None:
        self.pad_count = 0
        self.truncate_count = 0
        self.noop_count = 0
        self.history.clear()


# 모듈 전역 카운터 (싱글톤). 테스트에서 `_STATS.reset()` 으로 초기화.
_STATS = ShapeGuardStats()


def get_stats() -> ShapeGuardStats:
    """현재 누적된 shape_guard 통계 반환."""
    return _STATS


def enforce_length(arr: np.ndarray, target: int = 2_000) -> np.ndarray:
    """1D 배열의 길이를 정확히 `target` 으로 맞춘다.

    Parameters
    ----------
    arr : np.ndarray
        1D 신호 배열 (예: `signal_df["rms"].to_numpy()`).
    target : int, default 2000
        보장할 출력 길이 (Rev. 2026-05-16-02 기준 PNG 1장 = 2,000행).

    Returns
    -------
    np.ndarray
        shape `(target,)` 1D 배열. dtype 은 입력 보존.

    Raises
    ------
    ValueError
        `target <= 0` 이거나 `arr` 가 1D 가 아닌 경우.
    """
    if target <= 0:
        raise ValueError(f"target must be positive, got {target}")
    if arr.ndim != 1:
        raise ValueError(
            f"enforce_length expects 1D array, got shape={arr.shape}"
        )

    n = arr.shape[0]
    if n == target:
        _STATS.noop_count += 1
        return arr

    _STATS.history.append((n, target))
    if n < target:
        _STATS.pad_count += 1
        pad_width = target - n
        out = np.concatenate(
            [arr, np.zeros(pad_width, dtype=arr.dtype)], axis=0
        )
        _LOGGER.warning(
            "pad: input length %d < target %d → 0-pad %d samples",
            n, target, pad_width,
        )
        return out

    # n > target
    _STATS.truncate_count += 1
    _LOGGER.warning(
        "truncate: input length %d > target %d → drop %d samples",
        n, target, n - target,
    )
    return arr[:target]


__all__ = ["enforce_length", "get_stats", "ShapeGuardStats"]
