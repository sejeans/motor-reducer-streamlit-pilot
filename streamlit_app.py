"""모터-감속기 이상 탐지 데모 (Streamlit, PRD v1.3).

실행:
    streamlit run dashboard/streamlit_app.py

기능 (PRD v1.3 §5, FR-13~20)
-----------------------------
- "데모 시작" 버튼 → val 셋 청크가 3개 센서 라운드-로빈으로 순차 스트리밍
- 청크 도착 시마다 센서별 모델 추론 → AS / 5클래스 확률 / 로그 갱신
- 위젯 4종:
  1) AS 게이지 (센서 3개, 색상 임계 0.3/0.7)
  2) AS 시계열 라인 차트 (센서 3개 누적)
  3) 5클래스 확률 막대 차트 (센서 3개, 최신 청크)
  4) 이상 탐지 로그 테이블 (누적)
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent  # dashboard_streamlit/ (self-contained)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.inference.registry import ModelRegistry, SENSORS  # noqa: E402
from src.inference.router import InferenceRouter  # noqa: E402

# ============================================================
# 설정
# ============================================================
VAL_CHUNKS_PATH = _PROJECT_ROOT / "artifacts" / "val_chunks.npz"
MODELS_DIR = _PROJECT_ROOT / "models"
# 하위 호환용 (사이드바 메타 정보 표시에서만 사용, 없어도 무방)
SAMPLE_INDEX_PATH = _PROJECT_ROOT / "artifacts" / "sample_index.parquet"
SPLITS_DIR = _PROJECT_ROOT / "artifacts"

LABEL_NAMES = ["NORMAL", "ECC10", "ECC20", "DEMAG", "REDUC"]
LABEL_COLORS = {
    "NORMAL": "#22c55e",
    "ECC10":  "#facc15",
    "ECC20":  "#f97316",
    "DEMAG":  "#ef4444",
    "REDUC":  "#a855f7",
}
SENSOR_COLORS = {
    "Current_U": "#3b82f6",
    "Vib_Motor": "#10b981",
    "Vib_TM":    "#f59e0b",
}

CHUNK_INTERVAL_SEC = 0.5  # 청크 간격 (PRD v1.3 §3 리스크 — 0.5초 이상 권장)

st.set_page_config(
    page_title="모터-감속기 이상 탐지 데모",
    page_icon="🔧",
    layout="wide",
)


# ============================================================
# 자원 로드 (캐시)
# ============================================================
@st.cache_resource(show_spinner="모델 3개 로드 중...")
def get_registry() -> ModelRegistry:
    return ModelRegistry(MODELS_DIR)


@st.cache_resource
def get_router() -> InferenceRouter:
    return InferenceRouter(get_registry())


@st.cache_data(show_spinner="val 청크 큐 빌드 중...")
def build_chunk_queue() -> List[dict]:
    """`artifacts/val_chunks.npz`에서 청크를 읽어 3개 센서 라운드-로빈으로 인터리브.

    CSV 원본 의존 없음 — 사전 직렬화된 npz만 있으면 데모 재현 가능.
    """
    import numpy as np

    if not VAL_CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"val_chunks.npz not found: {VAL_CHUNKS_PATH}. "
            "Run `python scripts/export_val_chunks.py` first."
        )

    data = np.load(VAL_CHUNKS_PATH, allow_pickle=False)
    signals = data["signals"]          # (N, 2000) float32
    sensors_arr = data["sensors"]
    fault_classes = data["fault_classes"]
    group_keys = data["group_keys"]
    vehicles = data["vehicles"]
    chunk_ids = data["chunk_ids"]
    n = signals.shape[0]

    per_sensor: Dict[str, List[dict]] = {s: [] for s in SENSORS}
    for i in range(n):
        sensor = str(sensors_arr[i])
        if sensor not in per_sensor:
            continue
        chunk_df = pd.DataFrame({"rms": signals[i]})
        per_sensor[sensor].append({
            "sensor": sensor,
            "vehicle": str(vehicles[i]),
            "fault_class": str(fault_classes[i]),
            "group_key": str(group_keys[i]),
            "chunk_id": int(chunk_ids[i]),
            "chunk_df": chunk_df,
        })

    # 그룹/청크 순서 정렬 후 라운드-로빈 인터리브
    for sensor in SENSORS:
        per_sensor[sensor].sort(key=lambda d: (d["group_key"], d["chunk_id"]))

    queue: List[dict] = []
    max_len = max(len(per_sensor[s]) for s in SENSORS)
    for i in range(max_len):
        for sensor in SENSORS:
            if i < len(per_sensor[sensor]):
                queue.append(per_sensor[sensor][i])
    return queue


# ============================================================
# 위젯 렌더
# ============================================================
def _as_color(as_value: float) -> str:
    if as_value < 0.3:
        return "#22c55e"  # 녹색
    if as_value <= 0.7:
        return "#facc15"  # 황색
    return "#ef4444"      # 적색


def _build_gauge_fig(value: float, sensor: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"valueformat": ".3f", "font": {"size": 36}},
        title={"text": f"<b>{sensor}</b> Anomaly Score",
               "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar": {"color": _as_color(value)},
            "steps": [
                {"range": [0.0, 0.3], "color": "#dcfce7"},
                {"range": [0.3, 0.7], "color": "#fef9c3"},
                {"range": [0.7, 1.0], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def _build_prob_fig(probs: Dict[str, float], sensor: str) -> go.Figure:
    values = [probs[lbl] for lbl in LABEL_NAMES]
    colors = [LABEL_COLORS[lbl] for lbl in LABEL_NAMES]
    fig = go.Figure(go.Bar(
        x=LABEL_NAMES,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=f"<b>{sensor}</b>", font=dict(size=14)),
        height=240,
        yaxis=dict(range=[0, 1], title="확률"),
        margin=dict(l=30, r=10, t=40, b=30),
    )
    return fig


def _build_timeseries_fig(as_history: Dict[str, List[dict]]) -> go.Figure:
    fig = go.Figure()
    for sensor in SENSORS:
        rows = as_history.get(sensor, [])
        if not rows:
            continue
        xs = [r["step"] for r in rows]
        ys = [r["AS"] for r in rows]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=sensor,
            line=dict(color=SENSOR_COLORS[sensor], width=2),
            marker=dict(size=6),
        ))
    fig.add_hline(y=0.3, line_dash="dot", line_color="#facc15",
                  annotation_text="0.3", annotation_position="right")
    fig.add_hline(y=0.7, line_dash="dot", line_color="#ef4444",
                  annotation_text="0.7", annotation_position="right")
    fig.update_layout(
        height=320,
        xaxis_title="청크 인덱스 (스트리밍 순서)",
        yaxis_title="Anomaly Score",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def _step_counter() -> int:
    """단조 증가 카운터 — 각 렌더 호출이 고유 key 접미사를 갖도록."""
    st.session_state.render_step = st.session_state.get("render_step", 0) + 1
    return st.session_state.render_step


def update_gauges(
    gauge_phs: Dict[str, "st.delta_generator.DeltaGenerator"],
    latest_as: Dict[str, float],
    step: int,
) -> None:
    """센서별 AS 게이지를 placeholder에 갱신. step으로 key 유일성 확보."""
    for sensor in SENSORS:
        value = latest_as.get(sensor)
        ph = gauge_phs[sensor]
        if value is None:
            ph.markdown(f"**{sensor}**  \n_데이터 대기 중_")
        else:
            ph.plotly_chart(
                _build_gauge_fig(value, sensor),
                use_container_width=True,
                key=f"gauge_{sensor}_{step}",
            )


def update_prob_charts(
    prob_phs: Dict[str, "st.delta_generator.DeltaGenerator"],
    last_probs: Dict[str, Dict[str, float]],
    step: int,
) -> None:
    """센서별 5클래스 확률 막대를 placeholder에 갱신."""
    for sensor in SENSORS:
        probs = last_probs.get(sensor)
        ph = prob_phs[sensor]
        if probs is None:
            ph.markdown(f"**{sensor}** _대기 중_")
        else:
            ph.plotly_chart(
                _build_prob_fig(probs, sensor),
                use_container_width=True,
                key=f"prob_{sensor}_{step}",
            )


def update_timeseries(
    ts_ph: "st.delta_generator.DeltaGenerator",
    as_history: Dict[str, List[dict]],
    step: int,
) -> None:
    if not any(as_history.get(s) for s in SENSORS):
        ts_ph.markdown("_시계열 대기 중_")
        return
    ts_ph.plotly_chart(
        _build_timeseries_fig(as_history),
        use_container_width=True,
        key=f"ts_chart_{step}",
    )


def update_log(
    log_ph: "st.delta_generator.DeltaGenerator",
    log_rows: List[dict],
) -> None:
    if not log_rows:
        log_ph.markdown("_로그 대기 중_")
        return
    df_log = pd.DataFrame(log_rows[::-1])
    log_ph.dataframe(
        df_log, use_container_width=True, height=360, hide_index=True,
    )


# ============================================================
# 메인
# ============================================================
def main() -> None:
    st.title("🔧 모터-감속기 이상 탐지 데모")
    st.caption(
        "파일럿 단계 (PRD v1.3) · MiniROCKET + LogisticRegression(multinomial) · "
        "센서별 모델 3개 · Anomaly Score = 1 − P(NORMAL)"
    )

    # 사이드바: 메타 정보
    with st.sidebar:
        st.header("ℹ️ 데모 정보")
        registry = get_registry()
        for sensor in SENSORS:
            bundle = registry.get_bundle(sensor)
            meta = bundle["meta"]
            st.markdown(
                f"**{sensor}**\n"
                f"- val_acc: `{meta['val_accuracy']:.3f}`\n"
                f"- chunks (train/val): `{meta['n_train_chunks']}/{meta['n_val_chunks']}`\n"
                f"- kernels: `{meta['num_kernels']:,}`"
            )
        st.divider()
        st.caption(f"청크 간격: {CHUNK_INTERVAL_SEC}s")

    # 컨트롤
    col1, col2, col3 = st.columns([1, 1, 4])
    start_btn = col1.button("▶ 데모 시작", type="primary", use_container_width=True)
    reset_btn = col2.button("⟲ 초기화", use_container_width=True)

    if reset_btn:
        for key in [
            "as_history", "last_probs", "log_rows", "latest_as", "completed",
            "render_step",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    # 세션 상태 초기화
    if "as_history" not in st.session_state:
        st.session_state.as_history = {s: [] for s in SENSORS}
        st.session_state.last_probs = {s: None for s in SENSORS}
        st.session_state.log_rows = []
        st.session_state.latest_as = {s: None for s in SENSORS}
        st.session_state.completed = False

    router = get_router()
    queue = build_chunk_queue()
    total = len(queue)

    st.markdown("---")

    # 진행률
    progress_ph = st.empty()

    # 위젯 영역 — 각 센서별 placeholder 사전 생성 (slot 교체 패턴)
    st.subheader("센서별 Anomaly Score")
    gauge_cols = st.columns(3)
    gauge_phs: Dict[str, "st.delta_generator.DeltaGenerator"] = {
        sensor: gauge_cols[i].empty() for i, sensor in enumerate(SENSORS)
    }

    st.subheader("Anomaly Score 시계열")
    ts_ph = st.empty()

    st.subheader("5클래스 확률 (최신 청크)")
    prob_cols = st.columns(3)
    prob_phs: Dict[str, "st.delta_generator.DeltaGenerator"] = {
        sensor: prob_cols[i].empty() for i, sensor in enumerate(SENSORS)
    }

    st.subheader("이상 탐지 로그")
    log_ph = st.empty()

    def render_all() -> None:
        step = _step_counter()
        update_gauges(gauge_phs, st.session_state.latest_as, step)
        update_timeseries(ts_ph, st.session_state.as_history, step)
        update_prob_charts(prob_phs, st.session_state.last_probs, step)
        update_log(log_ph, st.session_state.log_rows)

    render_all()
    progress_ph.progress(
        len(st.session_state.log_rows) / total if total else 0.0,
        text=f"청크 {len(st.session_state.log_rows)} / {total}",
    )

    # 데모 실행
    if start_btn:
        st.session_state.completed = False
        start_idx = len(st.session_state.log_rows)
        for i in range(start_idx, total):
            item = queue[i]
            sensor = item["sensor"]
            result = router.predict_chunk(sensor, item["chunk_df"])

            st.session_state.as_history[sensor].append({
                "step": i,
                "AS": result.anomaly_score,
            })
            st.session_state.last_probs[sensor] = result.class_probs
            st.session_state.latest_as[sensor] = result.anomaly_score
            st.session_state.log_rows.append({
                "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "step": i,
                "sensor": sensor,
                "true_label": item["fault_class"],
                "predicted_label": result.predicted_label,
                "top_prob": round(result.top_class_prob, 3),
                "AS": round(result.anomaly_score, 3),
                "latency_ms": round(result.latency_ms, 1),
                "vehicle": item["vehicle"],
                "group_key": item["group_key"],
                "chunk_id": item["chunk_id"],
            })

            render_all()
            progress_ph.progress(
                (i + 1) / total,
                text=f"청크 {i + 1} / {total} · 현재: {sensor}",
            )

            time.sleep(CHUNK_INTERVAL_SEC)

        st.session_state.completed = True
        st.success(f"데모 완료. 총 {total}개 청크 처리.")


if __name__ == "__main__":
    main()
