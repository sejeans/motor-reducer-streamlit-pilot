"""센서별 모델 레지스트리 (PRD v1.3 §5).

3개 joblib 번들을 메모리에 적재하고 sensor 키 기반 dispatcher를 제공한다.

본 데이터 단계 확장 경로
------------------------
- 키 차원: `(sensor)` 1차원 → `(vehicle, sensor)` 2차원
- 코드 변경: `self._models` 의 dict 키를 tuple로 확장
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np

from src.common.logger import get_logger
from src.data.label_map import INV_LABEL_MAP

_LOGGER = get_logger("inference.registry")

SENSORS: tuple[str, ...] = ("Current_U", "Vib_Motor", "Vib_TM")


class ModelRegistry:
    """센서별 모델 번들 3개를 보관·dispatch.

    각 번들은 `train_pilot.py`의 출력 스키마를 따른다:
        {sensor, minirocket, classifier, label_map, meta}
    """

    def __init__(self, models_dir: str | Path, sensors: Iterable[str] = SENSORS):
        self._models_dir = Path(models_dir)
        self._models: Dict[str, dict] = {}
        for sensor in sensors:
            self._load_one(sensor)

    def _load_one(self, sensor: str) -> None:
        path = self._models_dir / f"pilot_{sensor.lower()}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"missing model bundle: {path}")
        bundle = joblib.load(path)
        if bundle.get("sensor") != sensor:
            raise ValueError(
                f"sensor mismatch in {path}: expected={sensor}, "
                f"got={bundle.get('sensor')}"
            )
        self._models[sensor] = bundle
        _LOGGER.info(
            "loaded sensor=%s val_acc=%.4f kernels=%d",
            sensor,
            bundle["meta"].get("val_accuracy", float("nan")),
            bundle["meta"].get("num_kernels", -1),
        )

    @property
    def sensors(self) -> tuple[str, ...]:
        return tuple(self._models.keys())

    def get_bundle(self, sensor: str) -> dict:
        if sensor not in self._models:
            raise KeyError(f"sensor not registered: {sensor}")
        return self._models[sensor]

    def predict_proba(self, sensor: str, panel: np.ndarray) -> np.ndarray:
        """단일/배치 청크 panel `(N, 1, 2000)` → 확률 `(N, 5)` 반환.

        MiniRocket transform 후 LogisticRegression.predict_proba 호출.
        """
        bundle = self.get_bundle(sensor)
        minirocket = bundle["minirocket"]
        clf = bundle["classifier"]

        feats = minirocket.transform(panel)
        if hasattr(feats, "to_numpy"):
            feats = feats.to_numpy()
        proba = clf.predict_proba(feats)
        return np.asarray(proba, dtype=np.float64)

    @staticmethod
    def label_names() -> list[str]:
        """클래스 ID 순서대로 라벨 이름 리스트 (NORMAL, ECC10, ECC20, DEMAG, REDUC)."""
        return [INV_LABEL_MAP[i] for i in range(len(INV_LABEL_MAP))]


__all__ = ["ModelRegistry", "SENSORS"]
