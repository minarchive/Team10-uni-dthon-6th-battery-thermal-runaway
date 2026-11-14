# val0.9195 — 시계열 전용 배터리 열폭주 6단계 분류 모델

이미지 없이 CSV 센서 시계열만으로 학습한 LightGBM 모델과, 단계가 내려가지 않게 하는 **확신 cummax 후처리**를 함께 담았습니다.

| 항목 | 값 |
|---|---|
| Validation Macro F1 (후처리 적용) | **0.9195** |
| Validation Macro F1 (후처리 없음) | 0.9104 |
| 학습 데이터 | Train만 (33,522행, 78개 실험). Validation은 학습에 사용하지 않음 |
| 평가 조건 | Validation의 500행 초과 실험을 Test처럼 500행으로 균등 다운샘플 (`--cap500`) |
| 사전학습 모델 | 없음 |

## 파일

| 파일 | 내용 |
|---|---|
| `model.joblib` | 학습된 가중치 (LightGBM + 피처 목록 735개 + τ=0.7) |
| `features.py` | 피처 생성, 모델 설정, 확신 cummax 후처리, 500행 다운샘플 |
| `predict.py` | 추론 → 후처리 → 제출 CSV 저장. 라벨이 있으면 Macro F1과 bat_class별 표 출력 |
| `train.py` | `model.joblib` 재학습 (같은 데이터면 같은 모델) |
| `requirements.txt` | 검증한 라이브러리 버전 (Python 3.10) |

## 실행

```bash
pip install -r requirements.txt
DATA=/path/to/battery_data_light      # train/ validation/ test/ sample_submission.csv 가 있는 폴더

# 1) Test 제출 파일 만들기
python predict.py --input $DATA/test/test.csv --output submission.csv \
                  --sample_submission $DATA/sample_submission.csv

# 2) Validation 점수 재현 (0.9195가 나와야 함)
python predict.py --input $DATA/validation/validation.csv --output val_pred.csv --cap500

#    후처리 없는 점수 (0.9104)
python predict.py --input $DATA/validation/validation.csv --output val_raw.csv --cap500 --no_postprocess

# 3) (선택) 재학습
python train.py --data_dir $DATA --out model.joblib                       # 제공된 가중치와 동일
python train.py --data_dir $DATA --out model_trval.joblib --train_val     # Train+Val로 학습 (검증 점수 없음)
```

이미지 폴더는 필요 없습니다. CSV만 있으면 됩니다. 피처 계산과 추론은 CPU로 몇 초면 끝납니다.

## 모델 구성

**피처 735개.** 모두 같은 실험 안에서 현재 시점 이하의 관측만 사용합니다.
- **사용 신호 20개**
  - 전압, 온도 t1·t2, 압력, 가스 10종, `atmp_t`, `thermal_t`, `indu_t`
  - 파생 신호: `max(t1,t2)`, `t1−t2`, 가스 합계
  - 사용하지 않은 컬럼: `meas_a`(전부 0), `meas_t1_plus_t2`(t1+t2와 중복), `indu_t.1`(결측 79%)
- **누적 피처:** 처음부터 현재까지의 최댓값·최솟값, 시작값 대비 변화, 최고점 대비 하락폭, 최고점 이후 경과 시간
- **과거 윈도우 피처:** 30/60/120/300초 구간의 평균·표준편차·최대·최소·편차·기울기·초당 변화량
  - 윈도우는 **행 수가 아니라 초 기준**입니다. Test의 긴 실험은 500행으로 줄어 있어 간격이 10초·15초로 섞여 있기 때문입니다.
- **임계 도달 이후 경과 시간**
  - t1·t2·t_max가 45/50/60/80/100/150/200/300°C를 처음 넘은 뒤 경과 시간
  - 전압이 시작값의 90/50/10% 아래로 처음 떨어진 뒤 경과 시간
- **배터리 조건 컬럼:** `bat_class, bat_cc, soc_status, tr_method`, 그리고 `relative_time_s`

**분류기:** LightGBM, `class_weight='balanced'`, 500 trees, lr 0.03, num_leaves 31, colsample 0.3, seed 42

**후처리 (확신 cummax, τ=0.7):** `features.conf_cummax`
- 실험별로 시간순으로 처리합니다.
- 예측 확률이 0.7 이상인 단계만 "확정 단계"로 누적합니다.
- 출력은 `max(확정 단계, 현재 예측)`입니다. 그래서 단계가 내려가지 않고, 확신이 낮은 일시적 오예측은 이후 행으로 이어지지 않습니다.
- 근거: 학습 데이터에서 같은 실험 안의 라벨은 한 번도 내려가지 않습니다(97개 실험 모두).
- 과거 예측만 사용하므로 미래 정보 누수가 없습니다.

## 검증 결과 (Validation, 500행 다운샘플, 후처리 적용)

F1 % (Macro는 해당 bat_class에 있는 라벨만 평균)

| bat_class | L1 | L2 | L3 | L4 | L5 | L6 | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 원통형 | 100.0 | 77.3 | 92.1 | 85.3 | 96.5 | 97.2 | 91.4 |
| 2 파우치형 | 100.0 | 85.0 | 94.2 | 83.3 | 77.8 | 90.7 | 88.5 |
| 3 각형 | 100.0 | 84.1 | 94.0 | 93.6 | 90.0 | 85.3 | 91.2 |
| 합계 | 100.0 | 83.2 | 93.7 | 91.5 | 93.0 | 90.3 | **91.9** |

Train 실험 단위 5-fold 교차검증(500행 다운샘플, 후처리 적용) Macro F1은 0.904입니다.

## 알려진 한계
- **L2(2단계)가 가장 약합니다.** 오류는 대부분 2→3 전환 시점 ±2분 안에서 생깁니다. 1→2는 t1이 45°C에 도달하는 시점으로 정확히 정해지지만, 2→3은 센서 값만으로 정해지는 규칙이 없습니다.
- **파우치형 L4·L5는 학습 샘플이 매우 적습니다**(84행, 55행).
- Validation은 19개 실험뿐이라 점수가 ±0.02~0.03 정도 흔들릴 수 있습니다.

## 재현 확인 (2026-11-14)
- 다른 디렉터리에서 `predict.py`를 실행해 Validation 0.9195와 0.9104를 재현했습니다.
- Test 제출 파일이 원본 실험 코드의 결과와 행 단위로 동일합니다.
- `train.py`로 재학습한 모델의 예측 확률이 `model.joblib`과 동일합니다.
