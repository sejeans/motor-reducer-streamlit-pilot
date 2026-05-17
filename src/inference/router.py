"""추론 라우터 (PRD v1.3 §5, FR-11).

한 청크 입력에 대해:
  1. 전처리(`rms` 선택 + NaN 정리 + panel 변환)
  2. 센서별 모델 dispatcher 호출 → predict_proba
  3. AS = 1 - P(NORMAL) 계산
  4. latency 측정
을 수행하여 dict로 반환한다.

PRD v1.3 §2-6 Anomaly Score
---------------------------
AS = 1 - P(NORMAL), [0, 1] 범위. 청크 단위, 센서별 독립.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.data.label_map import INV_LABEL_MAP, LABEL_MAP
from src.inference.registry import ModelRegistry
from src.preprocess.nan_handler import clean_signal
from src.preprocess.panel_formatter import to_sktime_panel

_LOGGER = get_logger("inference.router")

_SIGNAL_COL = "rms"
_TARGET_LEN = 2_000
_NORMAL_ID = LABEL_MAP["NORMAL"]


@dataclass
class InferenceResult:
    """단일 청크 추론 결과."""

    sensor: str
    predicted_label: str
    predicted_label_id: int
    top_class_prob: float
    anomaly_score: float  # AS = 1 - P(NORMAL)
    class_probs: Dict[str, float]
    latency_ms: float
    model_key: str

    def as_dict(self) -> dict:
        return {
            "sensor": self.sensor,
            "predicted_label": self.predicted_label,
            "predicted_label_id": self.predicted_label_id,
            "top_class_prob": self.top_class_prob,
            "anomaly_score": self.anomaly_score,
            "class_probs": self.class_probs,
            "latency_ms": self.latency_ms,
            "model_key": self.model_key,
        }


class InferenceRouter:
    """청크 입력 → 추론 결과 dict 변환기."""

    def __init__(self, registry: ModelRegistry):
        self._registry = registry

    def predict_chunk(
        self,
        sensor: str,
        chunk_df: pd.DataFrame,
    ) -> InferenceResult:
        """단일 청크 DataFrame에 대해 추론 1회 수행."""
        t0 = time.perf_counter()

        rms_raw = chunk_df[_SIGNAL_COL].to_numpy(dtype=np.float64, copy=False)
        rms_clean, _ratio = clean_signal(rms_raw)
        rms_clean = rms_clean.astype(np.float32, copy=False)

        panel = to_sktime_panel([rms_clean], target=_TARGET_LEN)

        proba_batch = self._registry.predict_proba(sensor, panel)  # (1, 5)
        proba = proba_batch[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_id = int(np.argmax(proba))
        pred_label = INV_LABEL_MAP[pred_id]
        top_prob = float(proba[pred_id])
        anomaly_score = float(1.0 - proba[_NORMAL_ID])

        class_probs = {
            INV_LABEL_MAP[i]: float(proba[i]) for i in range(len(proba))
        }

        return InferenceResult(
            sensor=sensor,
            predicted_label=pred_label,
            predicted_label_id=pred_id,
            top_class_prob=top_prob,
            anomaly_score=anomaly_score,
            class_probs=class_probs,
            latency_ms=latency_ms,
            model_key=f"pilot_{sensor.lower()}",
        )


__all__ = ["InferenceRouter", "InferenceResult"]
