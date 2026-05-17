"""파일럿 학습 스크립트 (PRD v1.3 §2-4, §2-5).

센서별로 MiniRocket(num_kernels=10000) + LogisticRegression(multinomial)을 학습하여
`models/pilot_{sensor}.joblib` 3개를 저장한다.

PRD v1.3 핵심 결정
------------------
- 분류기: `LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0,
  class_weight='balanced', max_iter=1000, random_state=42)` (RidgeClassifierCV 교체)
- 특징 추출: `MiniRocket(num_kernels=10000, random_state=42)` (sktime)
- 입력: `(N, 1, 2000)` float32 panel (univariate, rms 컬럼)
- 청크: chunk_size=2000, 같은 부모 CSV의 chunk는 동일 group_key 상속
- split: `GroupShuffleSplit(n_splits=1, test_size=0.2)` 결과 splits_{sensor}.json 사용
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sktime.transformations.panel.rocket import MiniRocket

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.logger import get_logger  # noqa: E402
from src.data.csv_loader import CSVLoader  # noqa: E402
from src.data.label_map import LABEL_MAP  # noqa: E402
from src.preprocess.chunker import split_into_chunks  # noqa: E402
from src.preprocess.nan_handler import clean_signal  # noqa: E402
from src.preprocess.panel_formatter import to_sktime_panel  # noqa: E402

_LOGGER = get_logger("train.train_pilot")

SENSORS: tuple[str, ...] = ("Current_U", "Vib_Motor", "Vib_TM")
_CHUNK_SIZE = 2_000
_TARGET_LEN = 2_000
_SIGNAL_COL = "rms"
_NUM_KERNELS = 10_000
_SEED = 42


@dataclass
class ChunkSample:
    """한 청크의 학습용 단위 표본."""

    signal_1d: np.ndarray  # shape (2000,) float32
    label_id: int
    fault_class: str
    group_key: str
    vehicle: str
    chunk_id: int
    csv_path: str


def _load_chunks_for_groups(
    df_sensor: pd.DataFrame,
    target_groups: List[str],
) -> List[ChunkSample]:
    """주어진 group_key 리스트에 속한 CSV들을 로드·청크 분할·rms 추출."""
    loader = CSVLoader()
    samples: List[ChunkSample] = []
    target_set = set(target_groups)

    df_target = df_sensor[df_sensor["group_key"].isin(target_set)].copy()
    _LOGGER.info(
        "loading %d CSVs for %d target groups", len(df_target), len(target_set),
    )

    for row in df_target.itertuples(index=False):
        csv_path = row.csv_path
        group_key = row.group_key
        fault_class = row.fault_class
        vehicle = row.vehicle

        try:
            _meta_df, signal_df = loader.load(csv_path)
        except Exception as exc:
            _LOGGER.warning("skipping CSV (load error): %s | %s", csv_path, exc)
            continue

        # chunk_size 배수가 아니면 chunker가 AssertionError → 스킵하고 로그
        try:
            chunks = split_into_chunks(
                signal_df, chunk_size=_CHUNK_SIZE, group_key=group_key,
            )
        except AssertionError as exc:
            _LOGGER.warning("skipping CSV (chunker assert): %s | %s", csv_path, exc)
            continue

        label_id = LABEL_MAP[fault_class.upper()]
        for chunk_df in chunks:
            rms_raw = chunk_df[_SIGNAL_COL].to_numpy(dtype=np.float32, copy=False)
            rms_clean, _ratio = clean_signal(rms_raw.astype(np.float64))
            rms_clean = rms_clean.astype(np.float32, copy=False)
            samples.append(
                ChunkSample(
                    signal_1d=rms_clean,
                    label_id=label_id,
                    fault_class=fault_class,
                    group_key=group_key,
                    vehicle=vehicle,
                    chunk_id=int(chunk_df.attrs.get("chunk_id", -1)),
                    csv_path=str(csv_path),
                )
            )

    return samples


def _stack_panel(samples: List[ChunkSample]) -> Tuple[np.ndarray, np.ndarray]:
    """ChunkSample 리스트 → (X panel, y label_ids)."""
    if not samples:
        raise RuntimeError("empty samples list — nothing to stack")
    signals = [s.signal_1d for s in samples]
    X = to_sktime_panel(signals, target=_TARGET_LEN)
    y = np.array([s.label_id for s in samples], dtype=np.int64)
    return X, y


def train_one_sensor(
    df: pd.DataFrame,
    sensor: str,
    splits_path: Path,
    out_path: Path,
) -> dict:
    """한 센서의 모델을 학습·검증·저장하고 메타 dict를 반환."""
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    train_groups: List[str] = splits["train_groups"]
    val_groups: List[str] = splits["val_groups"]

    df_sensor = df[df["sensor"] == sensor].copy()
    if df_sensor.empty:
        raise RuntimeError(f"sensor={sensor}: no rows in sample_index")

    _LOGGER.info(
        "=== training sensor=%s: train_groups=%d val_groups=%d ===",
        sensor, len(train_groups), len(val_groups),
    )

    t0 = time.perf_counter()
    train_samples = _load_chunks_for_groups(df_sensor, train_groups)
    val_samples = _load_chunks_for_groups(df_sensor, val_groups)
    t_load = time.perf_counter() - t0
    _LOGGER.info(
        "sensor=%s: chunks loaded train=%d val=%d in %.2fs",
        sensor, len(train_samples), len(val_samples), t_load,
    )

    X_train, y_train = _stack_panel(train_samples)
    X_val, y_val = _stack_panel(val_samples)
    _LOGGER.info(
        "sensor=%s: X_train=%s y_train=%s X_val=%s",
        sensor, X_train.shape, y_train.shape, X_val.shape,
    )

    # MiniRocket fit + transform
    t0 = time.perf_counter()
    minirocket = MiniRocket(num_kernels=_NUM_KERNELS, random_state=_SEED)
    minirocket.fit(X_train)
    X_train_feat = minirocket.transform(X_train)
    X_val_feat = minirocket.transform(X_val)
    # sktime MiniRocket은 pandas DataFrame을 반환할 수 있음 → ndarray 변환
    if hasattr(X_train_feat, "to_numpy"):
        X_train_feat = X_train_feat.to_numpy()
    if hasattr(X_val_feat, "to_numpy"):
        X_val_feat = X_val_feat.to_numpy()
    t_feat = time.perf_counter() - t0
    _LOGGER.info(
        "sensor=%s: MiniRocket feat train=%s val=%s in %.2fs",
        sensor, X_train_feat.shape, X_val_feat.shape, t_feat,
    )

    # LogisticRegression(multinomial)
    t0 = time.perf_counter()
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=_SEED,
    )
    clf.fit(X_train_feat, y_train)
    t_fit = time.perf_counter() - t0

    # 검증
    y_pred = clf.predict(X_val_feat)
    val_acc = accuracy_score(y_val, y_pred)
    report = classification_report(
        y_val, y_pred,
        labels=list(LABEL_MAP.values()),
        target_names=list(LABEL_MAP.keys()),
        zero_division=0,
        output_dict=True,
    )
    _LOGGER.info(
        "sensor=%s: val_acc=%.4f (fit %.2fs)", sensor, val_acc, t_fit,
    )

    # 직렬화 (joblib zlib)
    bundle = {
        "sensor": sensor,
        "minirocket": minirocket,
        "classifier": clf,
        "label_map": LABEL_MAP,
        "meta": {
            "num_kernels": _NUM_KERNELS,
            "chunk_size": _CHUNK_SIZE,
            "signal_column": _SIGNAL_COL,
            "seed": _SEED,
            "val_accuracy": float(val_acc),
            "n_train_chunks": len(train_samples),
            "n_val_chunks": len(val_samples),
            "n_train_groups": len(train_groups),
            "n_val_groups": len(val_groups),
            "load_seconds": float(t_load),
            "feat_seconds": float(t_feat),
            "fit_seconds": float(t_fit),
            "splits_path": str(splits_path),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path, compress=("zlib", 3))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    _LOGGER.info("saved: %s (%.2f MB)", out_path, size_mb)

    return {
        "sensor": sensor,
        "val_accuracy": float(val_acc),
        "n_train_chunks": len(train_samples),
        "n_val_chunks": len(val_samples),
        "size_mb": float(size_mb),
        "out_path": str(out_path),
        "classification_report": report,
    }


def train_all_sensors(
    sample_index_path: str | Path,
    splits_dir: str | Path,
    models_dir: str | Path,
    sensors: tuple[str, ...] = SENSORS,
) -> List[dict]:
    sample_index_path = Path(sample_index_path)
    splits_dir = Path(splits_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(sample_index_path)
    _LOGGER.info("loaded sample_index: %s rows=%d", sample_index_path, len(df))

    results: List[dict] = []
    for sensor in sensors:
        splits_path = splits_dir / f"splits_{sensor.lower()}.json"
        if not splits_path.exists():
            raise FileNotFoundError(f"missing splits: {splits_path}")
        out_path = models_dir / f"pilot_{sensor.lower()}.joblib"
        result = train_one_sensor(
            df=df, sensor=sensor, splits_path=splits_path, out_path=out_path,
        )
        results.append(result)

    # 학습 요약 jsonl로 함께 저장
    summary_path = models_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {"sensors": results, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _LOGGER.info("training summary: %s", summary_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="파일럿 학습 (3개 센서)")
    parser.add_argument("--sample-index", default="artifacts/sample_index.parquet")
    parser.add_argument("--splits-dir", default="artifacts")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    results = train_all_sensors(
        sample_index_path=args.sample_index,
        splits_dir=args.splits_dir,
        models_dir=args.models_dir,
    )
    print("\n=== 학습 결과 ===")
    for r in results:
        print(
            f"  {r['sensor']:10s} val_acc={r['val_accuracy']:.4f} "
            f"chunks={r['n_train_chunks']}/{r['n_val_chunks']} "
            f"size={r['size_mb']:.2f}MB"
        )


if __name__ == "__main__":
    main()
