"""Panel formatter (Phase 2-3, 2-4).

CSV 의 신호 DataFrame 리스트를 sktime MiniRocket 이 요구하는 panel ndarray
`(N, 1, target=2000)` float32 로 변환한다.

설계 원칙
---------
1. **Simplicity**: univariate 만 지원 (best_column 정책상 채널 1개). 다채널 확장은
   별도 함수에서 처리해 본 함수 인터페이스를 단순 유지.
2. **No Laziness**: 컬럼 미존재/길이 불일치 등 데이터 오염을 KeyError/ValueError 로
   즉시 노출 (silent fix 금지).
3. **Minimal Impact**: dtype 은 float32 로 통일 (NFR-02 메모리 절감).
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from src.preprocess.shape_guard import enforce_length

_DEFAULT_TARGET = 2_000
_VALID_COLUMNS: frozenset[str] = frozenset(
    {"peak_freq_bin", "band_start", "band_end", "rms"}
)


def select_column(signal_df: pd.DataFrame, col_name: str) -> np.ndarray:
    """`signal_df` 에서 단일 컬럼을 1D float32 ndarray 로 추출한다 (Phase 2-4).

    Parameters
    ----------
    signal_df : pd.DataFrame
        CSVLoader.load() 또는 chunker.split_into_chunks() 의 출력.
    col_name : str
        선택할 컬럼명. 'peak_freq_bin' | 'band_start' | 'band_end' | 'rms' 중 하나.

    Returns
    -------
    np.ndarray
        1D float32 배열, shape `(len(signal_df),)`.

    Raises
    ------
    KeyError
        `col_name` 이 `signal_df.columns` 에 없을 때.
    ValueError
        `col_name` 이 정의된 4개 컬럼이 아닐 때 (오타 방지).
    """
    if col_name not in _VALID_COLUMNS:
        raise ValueError(
            f"col_name must be one of {sorted(_VALID_COLUMNS)}, got '{col_name}'"
        )
    if col_name not in signal_df.columns:
        raise KeyError(
            f"column '{col_name}' not in signal_df.columns={list(signal_df.columns)}"
        )
    return signal_df[col_name].to_numpy(dtype=np.float32, copy=False)


def to_sktime_panel(
    signals: Iterable[np.ndarray | pd.DataFrame],
    *,
    target: int = _DEFAULT_TARGET,
    col_name: str | None = None,
) -> np.ndarray:
    """sktime MiniRocket 입력 panel `(N, 1, target)` float32 를 생성한다 (Phase 2-3).

    `signals` 의 각 원소는:
      - 1D `np.ndarray` (이미 단일 채널 추출 완료), 또는
      - `pd.DataFrame` (4컬럼 중 1개 선택 필요, `col_name` 지정 필수)
    가 될 수 있다. 길이는 `shape_guard.enforce_length` 로 보정된다 (정상 입력은 no-op).

    Parameters
    ----------
    signals : Iterable[np.ndarray | pd.DataFrame]
        chunker 출력 chunk 들의 리스트, 혹은 이미 select_column 으로 추출한 1D 배열 리스트.
    target : int, default 2000
        panel 의 시간축 길이. PNG 1장 = 2,000행 (Rev. 2026-05-16-02).
    col_name : str, optional
        `signals` 원소가 DataFrame 일 때 선택할 컬럼명. 1D ndarray 입력 시 무시.

    Returns
    -------
    np.ndarray
        shape `(N, 1, target)`, dtype `float32`.

    Raises
    ------
    ValueError
        `signals` 가 비었거나, DataFrame 원소가 섞였는데 `col_name` 미지정인 경우.
    """
    signal_list: List[np.ndarray] = []
    for i, s in enumerate(signals):
        if isinstance(s, pd.DataFrame):
            if col_name is None:
                raise ValueError(
                    f"signals[{i}] is DataFrame but col_name=None. "
                    "Specify col_name to select a single channel."
                )
            arr = select_column(s, col_name)
        elif isinstance(s, np.ndarray):
            arr = s.astype(np.float32, copy=False)
            if arr.ndim != 1:
                raise ValueError(
                    f"signals[{i}] ndarray must be 1D, got shape={arr.shape}"
                )
        else:
            raise TypeError(
                f"signals[{i}] must be np.ndarray or pd.DataFrame, "
                f"got {type(s).__name__}"
            )

        arr = enforce_length(arr, target=target)
        signal_list.append(arr)

    if not signal_list:
        raise ValueError("signals is empty — cannot build panel")

    # stack → (N, target), reshape → (N, 1, target)
    panel = np.stack(signal_list, axis=0).astype(np.float32, copy=False)
    panel = panel.reshape(panel.shape[0], 1, panel.shape[1])
    return panel


__all__ = ["select_column", "to_sktime_panel"]
