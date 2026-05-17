"""단일 CSV 로더 (Phase 1 1-3): 메타 1행 + 신호 (rows, 4) 분리.

NFR-02 (엣지 메모리 200MB) 를 위해 신호 dtype 은 float32 로 다운캐스트한다.

Rev. 2026-05-16-02:
- 가변 길이 CSV 허용 (`{2000, 4000, 6000, 8000, 10000}`)
- 2000행 chunk 분할 책임은 Phase 2 `chunker.py` 로 이관
- 로더는 더 이상 길이 강제 검증을 하지 않고, 허용 집합 이탈 시에만 WARN
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.data.label_map import LABEL_MAP

_LOGGER = get_logger("data.csv_loader")

# CSV 가 가지고 있는 7 개의 메타 컬럼 (path_indexer 의 group_key 와는 별도 — CSV 내부값)
_META_COLS: tuple[str, ...] = (
    "group_key",
    "vehicle",
    "fault_class",
    "date",
    "sensor",
    "timestamp",
    "zsplit",
)
_SIGNAL_COLS: tuple[str, ...] = ("peak_freq_bin", "band_start", "band_end", "rms")
_INDEX_COL: str = "time_idx"

# Rev. 2026-05-16-02: PNG 1장당 2000행, CSV 1개 = 1~5개 PNG 연접 → 5가지 길이 허용
_ALLOWED_ROW_COUNTS: frozenset[int] = frozenset({2_000, 4_000, 6_000, 8_000, 10_000})


class CSVLoader:
    """모터-감속기 시계열 CSV 로더.

    `load(path)` 가 반환하는 `meta_df` 는 1행, `signal_df` 는 `(rows, 4)` 형태.
    rows 는 `{2000, 4000, 6000, 8000, 10000}` 중 하나여야 한다.
    2000행 chunk 분할은 `src/preprocess/chunker.py` 책임 (관심사 분리).
    """

    def load(self, path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        path = Path(path)
        # pandas 2.1 호환: dtype 지정 시 NaN → float32 자연 변환
        df = pd.read_csv(
            path,
            dtype={col: "float32" for col in _SIGNAL_COLS},
        )

        # 1) signal 분리: time_idx 가 임의 시작값일 수 있으므로 컬럼만 추출하고 인덱싱은 별도
        signal_df = df[list(_SIGNAL_COLS)].copy()
        if _INDEX_COL in df.columns:
            signal_df.index = pd.Index(df[_INDEX_COL].to_numpy(), name=_INDEX_COL)

        n_rows = len(signal_df)
        if n_rows not in _ALLOWED_ROW_COUNTS:
            _LOGGER.warning(
                "row count out of allowed set: %s has %d rows (allowed=%s)",
                path.name,
                n_rows,
                sorted(_ALLOWED_ROW_COUNTS),
            )

        # 2) meta 분리: 첫 행에서 메타 7컬럼 추출 → 단일 행 DataFrame
        missing = [c for c in _META_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"missing meta columns in {path.name}: {missing}"
            )
        meta_row = df.iloc[0][list(_META_COLS)].to_dict()
        # label_id 부착 (LABEL_MAP 미등록 클래스면 KeyError → 데이터 오염 즉시 노출)
        meta_row["label_id"] = LABEL_MAP[str(meta_row["fault_class"]).upper()]
        meta_df = pd.DataFrame([meta_row])

        return meta_df, signal_df


__all__ = ["CSVLoader"]
