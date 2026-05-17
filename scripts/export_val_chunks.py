"""val 청크 직렬화 스크립트 (one-shot).

`artifacts/sample_index.parquet` + `artifacts/splits_{sensor}.json`을 기반으로
val 셋 CSV들을 로드·청크 분할·rms 추출하여 `artifacts/val_chunks.npz`로 저장한다.

산출물:
    artifacts/val_chunks.npz
        signals:      (N, 2000) float32
        sensors:      (N,) str  ('Current_U' / 'Vib_Motor' / 'Vib_TM')
        fault_classes: (N,) str
        group_keys:   (N,) str
        vehicles:     (N,) str
        chunk_ids:    (N,) int32

이 파일 1개만 있으면 데모 실행 가능 (CSV 원본 불필요).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.logger import get_logger  # noqa: E402
from src.data.csv_loader import CSVLoader  # noqa: E402
from src.preprocess.chunker import split_into_chunks  # noqa: E402
from src.preprocess.nan_handler import clean_signal  # noqa: E402

_LOGGER = get_logger("scripts.export_val_chunks")

SENSORS = ("Current_U", "Vib_Motor", "Vib_TM")
_CHUNK_SIZE = 2_000
_SIGNAL_COL = "rms"


def export_val_chunks(
    sample_index_path: Path,
    splits_dir: Path,
    out_path: Path,
) -> None:
    df_index = pd.read_parquet(sample_index_path)
    loader = CSVLoader()

    signals: List[np.ndarray] = []
    sensors: List[str] = []
    fault_classes: List[str] = []
    group_keys: List[str] = []
    vehicles: List[str] = []
    chunk_ids: List[int] = []

    for sensor in SENSORS:
        splits_path = splits_dir / f"splits_{sensor.lower()}.json"
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        val_groups = set(splits["val_groups"])

        df_sensor = df_index[
            (df_index["sensor"] == sensor)
            & (df_index["group_key"].isin(val_groups))
        ].sort_values(["group_key"])

        _LOGGER.info(
            "sensor=%s val_groups=%d csv_files=%d",
            sensor, len(val_groups), len(df_sensor),
        )

        for row in df_sensor.itertuples(index=False):
            try:
                _meta, signal_df = loader.load(row.csv_path)
                chunks = split_into_chunks(
                    signal_df, chunk_size=_CHUNK_SIZE, group_key=row.group_key,
                )
            except Exception as exc:
                _LOGGER.warning("skip %s: %s", row.csv_path, exc)
                continue

            for c_idx, chunk_df in enumerate(chunks):
                rms_raw = chunk_df[_SIGNAL_COL].to_numpy(dtype=np.float64, copy=False)
                rms_clean, _ratio = clean_signal(rms_raw)
                signals.append(rms_clean.astype(np.float32, copy=False))
                sensors.append(sensor)
                fault_classes.append(row.fault_class)
                group_keys.append(row.group_key)
                vehicles.append(row.vehicle)
                chunk_ids.append(c_idx)

    if not signals:
        raise RuntimeError("no val chunks collected")

    signals_arr = np.stack(signals, axis=0)  # (N, 2000)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        signals=signals_arr,
        sensors=np.asarray(sensors),
        fault_classes=np.asarray(fault_classes),
        group_keys=np.asarray(group_keys),
        vehicles=np.asarray(vehicles),
        chunk_ids=np.asarray(chunk_ids, dtype=np.int32),
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    _LOGGER.info(
        "saved: %s (N=%d, shape=%s, size=%.2f MB)",
        out_path, signals_arr.shape[0], signals_arr.shape, size_mb,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="val 청크 npz 직렬화")
    parser.add_argument(
        "--sample-index",
        default=str(_PROJECT_ROOT / "artifacts" / "sample_index.parquet"),
    )
    parser.add_argument(
        "--splits-dir",
        default=str(_PROJECT_ROOT / "artifacts"),
    )
    parser.add_argument(
        "--out",
        default=str(_PROJECT_ROOT / "artifacts" / "val_chunks.npz"),
    )
    args = parser.parse_args()

    export_val_chunks(
        sample_index_path=Path(args.sample_index),
        splits_dir=Path(args.splits_dir),
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
