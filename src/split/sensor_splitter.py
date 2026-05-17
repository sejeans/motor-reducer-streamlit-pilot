"""센서별 GroupShuffleSplit (파일럿 단계, PRD v1.3 §2-3).

센서별로 부모 CSV(`group_key`) 단위 80/20 hold-out split을 생성하고,
`splits_{sensor}.json` 3개를 저장한다.

설계 원칙
---------
1. **누수 차단**: GroupShuffleSplit으로 group_key 누수 차단 후
   `assert_no_group_leakage()`로 사후 검증 (PSC-05).
2. **시드 회전**: train/val 둘 다 5클래스 ≥ 1 샘플 충족 시까지 시드 회전
   (최대 10회). 미충족 시 RuntimeError로 즉시 노출.
3. **본 데이터 단계 호환**: splits.json 스키마는 본 데이터 단계
   `StratifiedGroupKFold` 출력과 동일 키 사용.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.logger import get_logger  # noqa: E402

_LOGGER = get_logger("split.sensor_splitter")

SENSORS: tuple[str, ...] = ("Current_U", "Vib_Motor", "Vib_TM")
FAULT_CLASSES: tuple[str, ...] = ("NORMAL", "ECC10", "ECC20", "DEMAG", "REDUC")

_DEFAULT_TEST_SIZE = 0.2
_DEFAULT_SEED = 42
_MAX_RETRY = 10


def _split_one_sensor(
    df_sensor: pd.DataFrame,
    sensor: str,
    test_size: float,
    base_seed: int,
    max_retry: int,
) -> Dict:
    """단일 센서의 group_key를 train/val로 80/20 분할한다.

    train/val 모두 5클래스 ≥ 1 샘플 충족 시까지 시드 회전.
    """
    groups_df = (
        df_sensor.drop_duplicates("group_key")[["group_key", "fault_class"]]
        .reset_index(drop=True)
    )
    n_groups = len(groups_df)
    if n_groups < 5:
        raise RuntimeError(
            f"sensor={sensor}: only {n_groups} unique group_keys, cannot split"
        )

    for trial in range(max_retry):
        trial_seed = base_seed + trial
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=trial_seed)
        train_idx, val_idx = next(gss.split(groups_df, groups=groups_df["group_key"]))

        train_classes = set(groups_df.loc[train_idx, "fault_class"].unique())
        val_classes = set(groups_df.loc[val_idx, "fault_class"].unique())
        target_classes = set(FAULT_CLASSES)

        train_ok = train_classes == target_classes
        val_ok = len(val_classes) >= 1

        if train_ok and val_ok:
            train_groups = sorted(groups_df.loc[train_idx, "group_key"].tolist())
            val_groups = sorted(groups_df.loc[val_idx, "group_key"].tolist())
            assert_no_group_leakage(train_groups, val_groups, sensor=sensor)

            _LOGGER.info(
                "sensor=%s split ok: trial=%d, train_groups=%d, val_groups=%d, "
                "train_classes=%d, val_classes=%d",
                sensor, trial, len(train_groups), len(val_groups),
                len(train_classes), len(val_classes),
            )
            return {
                "sensor": sensor,
                "split_method": "GroupShuffleSplit",
                "n_splits": 1,
                "test_size": test_size,
                "used_seed": trial_seed,
                "trial_count": trial + 1,
                "n_train_groups": len(train_groups),
                "n_val_groups": len(val_groups),
                "train_classes": sorted(train_classes),
                "val_classes": sorted(val_classes),
                "train_groups": train_groups,
                "val_groups": val_groups,
            }

        _LOGGER.warning(
            "sensor=%s trial=%d (seed=%d) rejected: train_5cls=%s val_>=1cls=%s "
            "(train=%s, val=%s)",
            sensor, trial, trial_seed, train_ok, val_ok,
            sorted(train_classes), sorted(val_classes),
        )

    raise RuntimeError(
        f"sensor={sensor}: max_retry={max_retry} exhausted, "
        f"cannot find a split with train 5 classes and val >= 1 class"
    )


def assert_no_group_leakage(
    train_groups: List[str],
    val_groups: List[str],
    *,
    sensor: str = "",
) -> None:
    """train/val group_key 교집합 = ∅ 검증 (PSC-05)."""
    overlap = set(train_groups) & set(val_groups)
    if overlap:
        raise AssertionError(
            f"sensor={sensor}: group_key leakage detected ({len(overlap)} groups): "
            f"{sorted(overlap)[:5]}..."
        )


def build_splits(
    sample_index_path: str | Path,
    out_dir: str | Path,
    *,
    sensors: tuple[str, ...] = SENSORS,
    test_size: float = _DEFAULT_TEST_SIZE,
    base_seed: int = _DEFAULT_SEED,
    max_retry: int = _MAX_RETRY,
) -> Dict[str, Path]:
    """센서별로 splits_{sensor}.json 3개를 생성한다.

    Returns
    -------
    Dict[str, Path]
        센서명 → 저장된 json 경로 매핑.
    """
    sample_index_path = Path(sample_index_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(sample_index_path)
    _LOGGER.info("loaded sample_index: %s rows=%d", sample_index_path, len(df))

    out_paths: Dict[str, Path] = {}
    for sensor in sensors:
        df_sensor = df[df["sensor"] == sensor].copy()
        if df_sensor.empty:
            raise RuntimeError(f"sensor={sensor}: no rows in sample_index")

        splits = _split_one_sensor(
            df_sensor=df_sensor,
            sensor=sensor,
            test_size=test_size,
            base_seed=base_seed,
            max_retry=max_retry,
        )

        out_path = out_dir / f"splits_{sensor.lower()}.json"
        json_bytes = json.dumps(splits, indent=2, ensure_ascii=False).encode("utf-8")
        out_path.write_bytes(json_bytes)
        sha256 = hashlib.sha256(json_bytes).hexdigest()
        _LOGGER.info("saved: %s sha256=%s", out_path, sha256[:16])
        out_paths[sensor] = out_path

    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="센서별 GroupShuffleSplit 생성")
    parser.add_argument(
        "--sample-index",
        default="artifacts/sample_index.parquet",
        help="sample_index.parquet 경로",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts",
        help="splits_{sensor}.json 저장 디렉터리",
    )
    parser.add_argument("--test-size", type=float, default=_DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    args = parser.parse_args()

    out_paths = build_splits(
        sample_index_path=args.sample_index,
        out_dir=args.out_dir,
        test_size=args.test_size,
        base_seed=args.seed,
    )
    for sensor, path in out_paths.items():
        print(f"  {sensor}: {path}")


if __name__ == "__main__":
    main()
