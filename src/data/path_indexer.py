"""모터-감속기 CSV 트리 스캔 + 샘플 인덱스 DataFrame 생성 (Phase 1 1-1).

데이터 트리:
    {root}/{vehicle}/{fault_class}/{date}/{sensor}/*.csv

핵심 설계:
- `group_key` 는 파일명에서 `_{sensor}_timeseries.csv` 접미사를 제거한 prefix.
  → 동일 (timestamp, zsplit) 에서 나온 3센서 CSV 가 같은 group_key 를 공유하므로
    Phase 3 의 StratifiedGroupKFold 가 센서 누수까지 동시에 차단할 수 있다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# `python -m src.data.path_indexer` 와 `python src/data/path_indexer.py` 양쪽 호출 모두
# 지원하기 위한 sys.path 보정 (직접 실행 시 PROJECT_ROOT 가 sys.path 에 없을 수 있음).
if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.logger import get_logger  # noqa: E402
from src.data.label_map import LABEL_MAP  # noqa: E402

_LOGGER = get_logger("data.path_indexer")

VEHICLES: tuple[str, ...] = ("IONIQ", "KONA", "NIRO")
SENSORS: tuple[str, ...] = ("Current_U", "Vib_Motor", "Vib_TM")
FAULT_CLASSES: tuple[str, ...] = tuple(LABEL_MAP.keys())

_CSV_SUFFIX = "_timeseries.csv"


def _strip_group_key(file_name: str, sensor: str) -> Optional[str]:
    """파일명에서 `_{sensor}_timeseries.csv` 를 떼낸 prefix 를 반환.

    예) 22_06_09_12_26_53_01_zsplit001_Current_U_timeseries.csv
        → 22_06_09_12_26_53_01_zsplit001
    규약에 맞지 않으면 None.
    """
    suffix = f"_{sensor}{_CSV_SUFFIX}"
    if not file_name.endswith(suffix):
        return None
    return file_name[: -len(suffix)]


def build_sample_index(
    root: str | Path,
    vehicles: Iterable[str] = VEHICLES,
    sensors: Iterable[str] = SENSORS,
    fault_classes: Iterable[str] = FAULT_CLASSES,
) -> pd.DataFrame:
    """모터-감속기 CSV 샘플 인덱스 DataFrame 을 만든다.

    Returns
    -------
    pd.DataFrame
        columns = [csv_path, vehicle, fault_class, date, sensor, group_key]
        샘플 0건의 leaf 디렉터리는 WARN 로그 후 인덱스에서 제외된다.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"data root not found: {root_path}")

    rows: List[dict] = []
    leaves_found = 0
    leaves_empty: List[str] = []

    for vehicle in vehicles:
        for fault in fault_classes:
            fault_dir = root_path / vehicle / fault
            if not fault_dir.exists():
                # 해당 (차종, 고장) 자체가 데이터에 없는 경우는 상위 단위로 WARN
                _LOGGER.warning(
                    "missing fault_dir: %s/%s (skipped)", vehicle, fault
                )
                continue

            # 날짜 폴더는 1개 이상 존재할 수 있다 (예: 0609/0728/0922)
            date_dirs = [d for d in fault_dir.iterdir() if d.is_dir()]
            if not date_dirs:
                _LOGGER.warning(
                    "no date subfolders under %s/%s", vehicle, fault
                )
                continue

            for date_dir in date_dirs:
                for sensor in sensors:
                    sensor_dir = date_dir / sensor
                    if not sensor_dir.exists():
                        _LOGGER.warning(
                            "missing sensor leaf: %s (skipped)", sensor_dir
                        )
                        continue

                    leaves_found += 1
                    csv_files = sorted(sensor_dir.glob(f"*_{sensor}{_CSV_SUFFIX}"))
                    if not csv_files:
                        leaves_empty.append(str(sensor_dir))
                        _LOGGER.warning(
                            "empty leaf (0 csv): %s (excluded from index)",
                            sensor_dir,
                        )
                        continue

                    for csv_file in csv_files:
                        group_key = _strip_group_key(csv_file.name, sensor)
                        if group_key is None:
                            _LOGGER.warning(
                                "filename does not match suffix rule: %s (skipped)",
                                csv_file,
                            )
                            continue
                        rows.append(
                            {
                                "csv_path": str(csv_file.resolve()),
                                "vehicle": vehicle,
                                "fault_class": fault,
                                "date": date_dir.name,
                                "sensor": sensor,
                                "group_key": group_key,
                            }
                        )

    df = pd.DataFrame(
        rows,
        columns=["csv_path", "vehicle", "fault_class", "date", "sensor", "group_key"],
    )

    n_unique_leaf = df.groupby(
        ["vehicle", "sensor", "fault_class"], observed=True
    ).ngroups
    _LOGGER.info(
        "index built: total=%d rows, leaves_scanned=%d, leaves_empty=%d, "
        "unique(vehicle,sensor,fault)=%d",
        len(df),
        leaves_found,
        len(leaves_empty),
        n_unique_leaf,
    )
    return df


def _print_distribution(df: pd.DataFrame) -> None:
    """CLI 사용자를 위해 분포를 표준 출력으로 보여준다."""
    print(f"\n총 샘플 수: {len(df):,}")
    print(f"unique group_key: {df['group_key'].nunique():,}")
    print(f"unique (vehicle, sensor, fault_class): "
          f"{df.groupby(['vehicle','sensor','fault_class'], observed=True).ngroups}")

    print("\n[vehicle x fault_class] 샘플 수")
    print(df.pivot_table(
        index="vehicle", columns="fault_class", values="csv_path",
        aggfunc="count", fill_value=0,
    ).to_string())

    print("\n[sensor x fault_class] 샘플 수")
    print(df.pivot_table(
        index="sensor", columns="fault_class", values="csv_path",
        aggfunc="count", fill_value=0,
    ).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="motor-reducer 샘플 인덱스 빌더")
    parser.add_argument(
        "--root",
        required=True,
        help=r"데이터 루트, 예: 샘플데이터\03.합성데이터\1.모터_감속기_시계열",
    )
    parser.add_argument(
        "--out",
        default="artifacts/sample_index.parquet",
        help="출력 parquet 경로",
    )
    args = parser.parse_args()

    df = build_sample_index(args.root)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    _LOGGER.info("saved sample index: %s", out_path)

    _print_distribution(df)


if __name__ == "__main__":
    main()
