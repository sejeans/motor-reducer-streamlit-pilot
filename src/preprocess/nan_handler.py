"""NaN handler (Phase 2-2).

1D 신호의 NaN 을 선형 보간으로 메우고, 보간 실패 영역은 0 으로 대체한다.
결측 비율을 함께 반환해 데이터 품질 모니터링 가능.

설계 원칙
---------
1. **No Laziness**: 결측 비율을 반환값으로 가시화 (호출 측이 NFR-04 결함 Recall 미달
   원인 분석 시 활용).
2. **Simplicity**: numpy 만으로 선형 보간 구현 — pandas/scipy 의존 회피해
   pure-numpy 추론 경로 보전.
3. **Edge cases**:
   - 전 구간 NaN → 모두 0 으로 대체 후 ratio=1.0 반환 (silent 0-fill 대신 ratio 로 노출)
   - 가장자리 NaN → 인접 유효값으로 nearest-neighbor 채움 (np.interp 표준 동작)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from src.common.logger import get_logger

_LOGGER = get_logger("preprocess.nan_handler")


def clean_signal(arr: np.ndarray) -> Tuple[np.ndarray, float]:
    """1D 배열의 NaN 을 선형 보간 + 0 fallback 으로 제거한다.

    Parameters
    ----------
    arr : np.ndarray
        1D 신호 배열. dtype 은 float 계열 권장 (NaN 표현 가능).

    Returns
    -------
    cleaned : np.ndarray
        NaN 이 모두 제거된 1D 배열. dtype 은 입력 보존 (정수형이면 float 로 승격).
    nan_ratio : float
        입력 배열에서의 NaN 비율 (0.0 ~ 1.0).

    Raises
    ------
    ValueError
        `arr` 가 1D 가 아닌 경우.

    Notes
    -----
    - 보간 알고리즘: `np.interp(xp=valid_idx, fp=valid_val, x=all_idx)`
      → 가장자리는 자동으로 양 끝의 valid 값으로 확장 (nearest-neighbor)
    - 보간 실패(예: 전 구간 NaN) → 0 으로 fallback + WARN 로그
    """
    if arr.ndim != 1:
        raise ValueError(
            f"clean_signal expects 1D array, got shape={arr.shape}"
        )

    n = arr.shape[0]
    if n == 0:
        return arr.copy(), 0.0

    # dtype: 입력이 float 면 그대로, 정수면 float64 로 승격해 NaN 표현 가능하게.
    if np.issubdtype(arr.dtype, np.floating):
        out_dtype = arr.dtype
        work = arr.astype(out_dtype, copy=True)
    else:
        # 정수형: NaN 이 존재할 수 없으므로 ratio=0 으로 즉시 반환
        return arr.astype(arr.dtype, copy=True), 0.0

    nan_mask = np.isnan(work)
    nan_count = int(nan_mask.sum())
    nan_ratio = nan_count / n

    if nan_count == 0:
        return work, 0.0

    if nan_count == n:
        # 전 구간 NaN → 보간 불가, 0 fallback
        _LOGGER.warning(
            "all-NaN input (n=%d) → fallback to zeros, ratio=1.0", n,
        )
        return np.zeros(n, dtype=out_dtype), 1.0

    valid_idx = np.where(~nan_mask)[0]
    valid_val = work[valid_idx]
    all_idx = np.arange(n)
    # np.interp 는 가장자리에서 fp[0]/fp[-1] 로 자동 확장 (nearest-neighbor)
    interpolated = np.interp(all_idx, valid_idx, valid_val)
    work = interpolated.astype(out_dtype, copy=False)

    if np.isnan(work).any():
        # 이론상 위 분기로 모두 처리되지만, 방어 코드로 0 fallback
        _LOGGER.warning(
            "interpolation residual NaN remains (n=%d, ratio=%.4f) → "
            "fallback to 0 for residual",
            n, nan_ratio,
        )
        work = np.nan_to_num(work, nan=0.0).astype(out_dtype, copy=False)

    _LOGGER.info(
        "NaN interpolated: n=%d, nan_count=%d, ratio=%.4f", n, nan_count, nan_ratio,
    )
    return work, float(nan_ratio)


__all__ = ["clean_signal"]
