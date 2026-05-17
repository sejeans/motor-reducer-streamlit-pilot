# 모터-감속기 이상 탐지 데모 (Streamlit)

EV 모터-감속기 시계열 신호 기반 5클래스(NORMAL / ECC10 / ECC20 / DEMAG / REDUC) 이상 탐지 파이프라인의 **파일럿 대시보드**.

> 시스템분석/설계 과제 · 홍익대 정보산업공학과 7조 · 박세진(C321027) · 박솔
> PRD: 본 데이터(600GB) 진입 전 feasibility 검증용 파일럿 단계.

## 데모 화면

- 3개 센서(`Current_U` / `Vib_Motor` / `Vib_TM`) 청크가 라운드-로빈으로 순차 스트리밍
- 위젯 4종:
  - **Anomaly Score (AS) 게이지** (센서별 3개, 색상 임계 0.3 / 0.7)
  - **AS 시계열 라인 차트** (3센서 누적)
  - **5클래스 확률 막대 차트** (센서별 최신 청크)
  - **이상 탐지 로그 테이블** (누적)

`AS = 1 − P(NORMAL)`. 0=정상, 1=결함 확정.

## 빠른 시작

```bash
# 1. 의존성 설치
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt

# 2. 데모 실행
streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501` 열리면 **"▶ 데모 시작"** 클릭.

> 모델 3개(`models/pilot_*.joblib`)와 val 청크(`artifacts/val_chunks.npz`)가 repo에 포함돼 clone 후 즉시 실행 가능. CSV 원본 불필요.

## 폴더 구조

```
dashboard_streamlit/
├── streamlit_app.py          # 데모 앱
├── requirements.txt
├── artifacts/
│   ├── val_chunks.npz        # 직렬화된 val 청크 51개 (0.32MB)
│   ├── sample_index.parquet  # 원본 CSV 인덱스 (참고용)
│   └── splits_*.json (3개)   # 센서별 train/val split
├── models/
│   ├── pilot_current_u.joblib
│   ├── pilot_vib_motor.joblib
│   └── pilot_vib_tm.joblib   # 각 0.4MB
├── scripts/
│   └── export_val_chunks.py  # val 청크 재생성 (CSV 원본 있을 때)
└── src/
    ├── common/               # 로거
    ├── data/                 # CSVLoader, label_map
    ├── preprocess/           # chunker, nan_handler, panel_formatter
    ├── split/                # 센서별 GroupShuffleSplit
    ├── train/                # MiniRocket + LogisticRegression 학습
    └── inference/            # ModelRegistry, InferenceRouter
```

## 모델 사양

| 항목 | 값 |
|---|---|
| 특징 추출 | MiniRocket (sktime, `num_kernels=10000`) |
| 분류기 | LogisticRegression (multinomial, lbfgs, `class_weight='balanced'`) |
| 입력 | `(N, 1, 2000)` float32 — `rms` 컬럼 단일 채널 |
| 청크 크기 | 2,000 행 (chunker) |
| 모델 분리 | 센서별 3개 (차종 합산) |
| Split | `GroupShuffleSplit(n_splits=1, test_size=0.2, seed=42)` — 부모 CSV 누수 차단 |

> 파일럿 단계는 **end-to-end 동작 검증**이 목표. 정량 성능 목표 없음. 본 데이터(600GB) 단계에서 (차종 × 센서) 9-모델 분리 + StratifiedGroupKFold + 정량 NFR(≤100ms 추론, ≤200MB 디스크) 평가 활성화.

## 재학습 (선택)

CSV 원본 있는 환경에서:

```bash
python -m src.split.sensor_splitter
python -m src.train.train_pilot
python scripts/export_val_chunks.py
```

## 의존성

- Python 3.10+
- sktime 0.26.x · scikit-learn 1.4.x · pandas 2.1.x · numpy 1.26.x
- streamlit ≥1.30 · plotly ≥5.18

## 라이선스

학내 과제 산출물. 별도 명시 없음.
