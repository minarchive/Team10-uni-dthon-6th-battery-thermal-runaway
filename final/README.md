# final — 최종 제출 모델 (Public 0.9156)

배터리 열폭주 6단계 분류(Macro-F1). 이 폴더는 Public 0.9156을 낸 제출 파일(`v3w_0.csv`)을 그대로 재현하는 코드·가중치입니다.
기반은 팀원이 만든 파이프라인(`sota_0.8812/` → `v3_sota_et0.8920/`)이고, 최종본은 여기에 **ExtraTrees 하나를 더(leaf 크기만 다르게) 추가**한 것입니다.

## 1. 한눈에 보기
```
센서 CSV ─► 시계열 피처 166개 (features.py, mode='rich', 같은 실험의 현재·과거 관측만, 시간 창은 초 단위)
              ├─ 일반 실험 (tr_method 1) ──► 분기 A: 피처 166개
              └─ 과충전 실험 (tr_method 2) ─► 분기 C: 압력 절대값·v_peak_ratio 제외 153개
                   각 분기를 3개 모델이 예측해 확률을 가중 평균 (`weights/config.json`의 `ens_weights` = LightGBM 0.2083, ET20 0.2083, ET40 0.5833)
                     ① LightGBM (600트리, lr 0.03, colsample 0.5, 클래스 가중 없음)
                     ② ExtraTrees (500트리, min_samples_leaf=20, max_features=0.3)
                     ③ ExtraTrees (500트리, min_samples_leaf=40, max_features=0.3)   ← 최종본에서 추가한 것
                                                        │ P_시계열
열화상 ─► [현재−첫 프레임, 현재−15초 전, 현재−60초 전] 차이 3채널 ─► EfficientNet-B0 × 시드 2개 평균 ─► P_이미지
                                                        │
            P = 0.94·P_시계열 + 0.06·P_이미지 ─► 인과적 단조 디코딩(prefix Viterbi) ─► 라벨 1~6
```

## 2. 구현 상세
| 부분 | 내용 | 파일 |
|---|---|---|
| 피처 | 센서 통계 166개(시차 차이, 창 평균·표준편차·최대·최소, 누적 극값, 임계 도달 후 경과시간 등). 가스·주변 온도 제외, 시간 창은 `relative_time_s` 기준이라 5/10/15초 간격 모두 같은 의미 | `features.py` |
| 과충전 분기 C | 과충전 실험은 압력 **절대 수준**(Test 각형 과충전 2개의 시작 압력이 학습 범위의 3~7배)과 과충전 배율을 뺀 피처로 예측 | `model.py: cols_without_abs_pressure` |
| LightGBM | 설정은 `model.py: LGB_PARAMS` | `train.py` |
| ExtraTrees | A/C 각각 학습. leaf 20은 `weights/et.joblib`, leaf 40은 `weights/et40.joblib` | `train_et.py --leaf 20/40` |
| 이미지 | EfficientNet-B0(ImageNet 사전학습), 입력은 실험 기준 차이 3채널, 학습 때 5/10/15초 격자 증강, 시드 42·7 평균. 혼합 비중 α=0.06 | `model.py`, `train.py`, `weights/vision_seed*.pt` |
| 디코딩 | 각 시점까지의 확률로 라벨이 내려가지 않는 최적 경로(prefix Viterbi)의 마지막 상태. 하이퍼파라미터 없음, 미래 행 미사용 | `model.py: viterbi` |
| 추론 | 입력 CSV의 행만 사용. 세 모델 확률 평균 → 이미지 혼합 → Viterbi | `test.py` |

학습 데이터는 Train(78) + Validation(19) 전체 실험입니다. 모든 입력은 같은 실험의 현재·과거 관측만 씁니다(외부 데이터 없음, 이미지 백본의 ImageNet 사전학습 가중치만 사용).

## 환경 / 학습·검증 구성 / 사전학습 모델
- **환경**: Ubuntu 22.04, Python 3.10.12, GPU NVIDIA RTX 5080(이미지 추론은 CPU로도 가능), `requirements.txt` 참고(torch는 버전 미지정).
- **학습 구성**: Train(78개 실험) + Validation(19개 실험) 전체로 최종 학습. 클래스 가중치 없음, 시드 42 고정.
- **검증 구성**: 실험 단위 5-fold(같은 실험의 프레임이 학습·검증에 섞이지 않음)의 OOF, 제공 Validation, 그리고 Test 입력과 가까운 학습 실험 30개로 한 '닮은 실험' 평가. 자세한 수치는 5절.
- **시계열 피처**: 센서(T1, T2, thermal, 전압, 압력, 보조 센서) 시차 차이·창 평균·표준편차·최대·최소·기울기·누적 극값·임계(45/50/60℃ 등, 전압 하락) 도달 후 경과시간 등 166개. 전부 같은 실험의 현재·과거 행으로만 계산하고 창은 `relative_time_s`(초) 기준.
- **사전학습 모델**: 이미지 모델 EfficientNet-B0의 ImageNet 사전학습 가중치(torchvision `IMAGENET1K_V1`)만 사용했고 학습 때 미세조정했습니다. 외부 데이터와 다른 사전학습 모델은 쓰지 않았습니다.

## 3. 재현
```bash
pip install -r requirements.txt
DATA=/path/to/battery_data_light          # train/ validation/ test/ sample_submission.csv

# (선택) 재학습 → weights/
python train.py    --data_dir $DATA --vision_seeds 42,7 --alpha_trm1 0.06 --alpha_trm2 0.06   # lgbm.joblib, vision_seed*.pt
python train_et.py --data_dir $DATA --leaf 20 --out weights/et.joblib
python train_et.py --data_dir $DATA --leaf 40 --out weights/et40.joblib

# 추론 (weights/ 그대로 사용)
python test.py --input $DATA/test/test.csv --image_root $DATA/test --output submission.csv
```
확인: 위 `test.py`를 다시 돌린 결과가 DAKER에 제출한 `v3w_0.csv`와 **다른 행 0개**(3,451행 전부 일치)입니다. 로컬 점검은 `python check_local.py --data_dir $DATA` (같은 입력 재실행, 행 순서 섞기, 실험 나눠 넣기, 실험마다 앞 60%만 넣기).

## 4. 점수 이력 (Public)
| 제출 | Public |
|---|---|
| 팀 v1: LightGBM A/C + 이미지 | 0.8812 |
| v3: v1 + ExtraTrees(leaf 20) 확률 평균 | 0.89 |
| v3 + ExtraTrees(leaf 40) 추가, 세 모델 1/3씩 (`v3x_et3`) | 0.9070 |
| 같은 세 모델, 가중치 0.25 / 0.25 / 0.5 (`v3z_g05`) | 0.9091 |
| **최종: 가중치 LightGBM 0.2083 / ET20 0.2083 / ET40 0.5833 (이 폴더, `v3w_0`)** | **0.9156** |
| (참고) 서버에서 만든 다른 후보들 | 우리 단독 LightGBM 0.868, v3 + 우리 모델 0.3 블렌드 0.8732, 경계별 전문 모델 0.8469 |

## 5. 왜 ExtraTrees를 하나 더 넣었나 (그리고 아는 것/모르는 것)
- ET는 LightGBM보다 확률이 부드럽습니다(L2/L3 프레임의 평균 최대확률: LightGBM 0.97, ET 0.88). 확률을 평균하면 Viterbi가 L2↔L3 경계에서 한 프레임의 강한 확신에 덜 휘둘립니다.
- **우리 5-fold CV로는 이 효과가 보이지 않습니다**(ET를 더하면 OOF가 −0.002 정도 내려가고, 단독 ET는 0.87). Public에서만 올랐습니다. CV는 이 문제의 Test 점수를 잘 예측하지 못했고(우리가 시도한 변경 4개 중 2개의 방향이 반대), 그래서 아래 두 가지를 같이 봤습니다.
  1. **Test 닮은 실험 평가**: 각 Test 실험과 입력 프로필(그룹, 시작 온도·압력·전압 등)이 가장 가까운 같은 그룹 학습 실험 3개(총 30개)의 OOF만으로 비교. Test 입력만 쓰고 라벨은 평가에만 사용. 이미 Public을 아는 변경 4개에서 방향이 모두 맞았습니다.
  2. v3와 달라지는 Test 행 수 확인(행 하나가 약 0.0004~0.0005를 움직이므로 변경 폭을 먼저 본다).
- 이 평가에서 `LightGBM + ET + ET40` 평균이 Test 닮은 실험 +0.0008(개선 12 / 악화 5)이었고 Public에서 +0.017 올랐습니다. 다만 Public을 보며 고른 변경이라 **Private에서는 이만큼 오르지 않을 수 있습니다**(승자의 저주).
- 우리 쪽에서 효과가 없었던 것(참고): 우리 LightGBM(피처 754개) 블렌드, 과충전 규칙 교체, 이미지를 LightGBM 입력으로 쓰기, 증강(시간축 배속·500행 압축·건너뛰기), 경계별 전문 모델, XGBoost/MLP 멤버, 피처를 더 넣는 전진 선택, LSTM/AE.

## 6. 팀원에게
- 이 폴더만으로 재현됩니다. 팀원 코드(`features.py`, `model.py`, `train.py`)는 수정하지 않았고, `train_et.py`에 `--leaf/--out` 옵션을, `test.py`에 ET 두 개를 읽어 평균하는 부분을 추가했습니다.
- 추가 실험을 하신다면 한 번에 하나만, v3 대비 바뀌는 Test 행 수를 먼저 확인하고 올리는 것을 권합니다.
- 학습된 가중치는 `weights/`에 있습니다(파일당 21MB 이하, 합계 약 87MB). 대회 데이터와 제출 CSV는 레포에 올리지 않았습니다.
