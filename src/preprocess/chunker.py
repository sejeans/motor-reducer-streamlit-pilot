"""CSV 신호 chunker (Phase 2-X, Rev. 2026-05-16-02).

가변 길이 CSV(2000/4000/6000/8000/10000행) 를 PNG 1장 단위인 2,000행 chunk 로 분할한다.

설계 원칙
---------
1. **누수 차단(Option A)**: 같은 부모 CSV 에서 잘린 chunk 들은 동일 `group_key` 를
   상속하여 Phase 3 `StratifiedGroupKFold` 가 시간 인접·동일 모터 지문의 누수를 차단.
2. **명확한 실패**: chunk_size 배수가 아닌 입력은 `AssertionError` 로 즉시 노출
   (Silent truncation 금지 — root cause 를 가린다).
3. **추적성**: 각 chunk 에 0-indexed `chunk_id` 컬럼을 메타로 부착.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd


def split_into_chunks(
    signal_df: pd.DataFrame,
    chunk_size: int = 2_000,
    *,
    group_key: Optional[str] = None,
) -> List[pd.DataFrame]:
    """`signal_df` 를 `chunk_size` 행 단위로 분할한 DataFrame 리스트를 반환한다.

    Parameters
    ----------
    signal_df : pd.DataFrame
        CSVLoader.load() 의 두 번째 반환값. 길이는 `chunk_size` 의 배수여야 한다.
    chunk_size : int, default 2000
        PNG 1장에 대응하는 행 수 (Rev. 2026-05-16-02 기준 2000).
    group_key : str, optional
        부모 CSV 의 group_key. 지정 시 각 chunk DataFrame 의 attrs 와
        `chunk_meta` 컬럼에 기록된다.

    Returns
    -------
    List[pd.DataFrame]
        각 길이 정확히 `chunk_size`. 원본의 컬럼·dtype 보존.
        각 chunk 는 `.attrs["group_key"]` 와 `.attrs["chunk_id"]` (0-indexed) 를 갖는다.

    Raises
    ------
    AssertionError
        `len(signal_df) % chunk_size != 0` 일 때.
    ValueError
        `chunk_size <= 0` 일 때.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    n_rows = len(signal_df)
    assert n_rows % chunk_size == 0, (
        f"signal_df length {n_rows} is not a multiple of chunk_size {chunk_size}. "
        f"Expected one of {{2000, 4000, 6000, 8000, 10000}} for motor-reducer CSV."
    )

    n_chunks = n_rows // chunk_size
    chunks: List[pd.DataFrame] = []
    for i in range(n_chunks):
        # iloc 슬라이스는 view 일 수 있어 .copy() 로 안전한 독립 객체 보장
        chunk = signal_df.iloc[i * chunk_size : (i + 1) * chunk_size].copy()
        # attrs 는 pandas 가 보장하는 메타 슬롯 (직렬화 시 누락될 수 있으므로 메타용 한정)
        chunk.attrs["chunk_id"] = i
        if group_key is not None:
            chunk.attrs["group_key"] = group_key
        chunks.append(chunk)

    return chunks


__all__ = ["split_into_chunks"]
