# 이미지 + 시계열 결합 학습 계획

작성: 2026-11-14. 기준 모델은 `val0.9195`(시계열 735피처 LightGBM, CV 0.904 / Val 0.9195).
전제: **최종 제출은 이미지와 시계열을 반드시 함께 사용**한다. 따라서 "이미지가 도움이 되는가"가 아니라 **"이미지를 어떤 방식으로 넣는 게 가장 좋은가"**를 고르는 실험이다. 기대 지점은 2→3 전환(센서 규칙이 없는 유일한 경계).

## 0. 환경 · 자원

| 항목 | 내용 |
|---|---|
| Python | `/home/nxcolelxu/unid/venv` (python3 -m venv, torch cu124, lightgbm, scikit-learn, transformers, timm) |
| 데이터 | `/home/nxcolelxu/unid/data/battery_data_light` (setup.sh의 DATA_URL을 gdown으로 받음) |
| 사전학습 가중치 | HF `facebook/dinov2-small`, `facebook/dinov2-base` (Apache-2.0), timm `convnext_tiny`, `efficientnet_b0` (ImageNet). 캐시 `hf_cache/` |

## 1. 공통 평가 규칙 (README 5-3과 동일)

- **선택은 Train CV로만**: Train 78개 실험 GroupKFold(5), 시드 3개(0,1,2) 평균
- 검증 fold는 Test 조건대로 **500행 캡** 후 평가, 후처리는 **확신 cummax τ=0.7**
- 지표: Macro F1 (라벨 1~6)
- **Val 19개 실험은 기준선(v3)과 CV 1위 설정만** 마지막에 확인. Val 보고 설정 안 바꿈
- 모든 이미지 피처는 같은 실험 안의 **현재 시점 이하 프레임만** 사용 (첫 프레임, 30초 전 프레임, 과거 윈도우)
- PCA·스케일러는 fold의 학습 실험으로만 fit. Val/Test는 transform만
- 파인튜닝 모델 출력을 LightGBM 피처로 쓸 때는 반드시 **OOF**(해당 실험을 안 본 fold 모델의 출력)

## 2. 단계별 계획

### 1단계 — 임베딩 추출 (`embed.py`)

고정 DINOv2 ViT-S/14, ViT-B/14로 Train·Val·Test 전 프레임(~47,000장) 임베딩 추출. 224×224, bf16 추론. CLS 토큰 + 패치 평균(ViT-S 768차원, ViT-B 1536차원), float16 저장.
- 학습 없음. 프레임 한 장씩 독립 처리라 누수 없음.
- 예상 시간: ViT-S 2~3분, ViT-B 5~8분 (GPU). 이미지 로딩이 병목이라 worker 6개.
- "이미지가 정보가 되는지" 확인하는 이진 분류 단계는 **건너뜀** (이미지는 어차피 반드시 사용).

### 2단계 — 임베딩을 시계열 피처로 바꿔 LightGBM에 추가 (`exp_img.py --stage 2`) — 방식 A

이미지 피처(실험별 시간순, 과거만 사용):

| 피처 | 설명 | 개수 (k=32) |
|---|---|---:|
| `img_z{j}` | 현재 프레임 임베딩 PCA 성분 | 32 |
| `img_z{j}_mean{w}` | 과거 w초(30/60/120) 윈도우 평균 | 96 |
| `img_z{j}_delta{w}` | w초 전 프레임 대비 변화량 | 96 |
| `img_zdrift{w}` | w초 전 대비 PCA 공간 이동 거리(L2) | 3 |
| `img_cos_first`, `_cummax` | **t=0 첫 프레임과의 코사인 거리**와 그 누적최대 (연기·화염 출현 시 급증) | 2 |
| `img_cos_first_mean{w}` | 위 값의 윈도우 평균 | 3 |
| `img_cos_prev`, `img_cos_lag{w}` | 직전 프레임 / w초 전 프레임과의 코사인 거리 | 4 |

비교 설정 (6클래스, CV×3시드, 캡, 확신 cummax):

| 이름 | 피처 |
|---|---|
| `v3` | 시계열 735 (참고용 기준선, 제출 후보 아님) |
| `v3+cos` | + 코사인 거리 피처만 (9개, PCA 없음) |
| `v3+img_k32` | + PCA 32 전체 이미지 피처 (+236) |
| `v3+img_k64` | + PCA 64 (+460) |

- DINOv2-S와 DINOv2-B 두 임베딩으로 각각 실행 (`res_dinov2s.json`, `res_dinov2b.json`)
- CV 1위 이미지 설정과 v3만 Val 확인 (Train 전체 학습, Val 500행 캡)
- 특히 볼 것: **2단계 F1**(현재 CV 0.838 / Val 0.832)과 2↔3 경계 오류 수

### 3단계 — 소형 백본 파인튜닝 (`finetune.py`) — 방식 B (2단계와 무관하게 실행)

| 항목 | 설정 |
|---|---|
| 백본 | timm `convnext_tiny`(28M), `efficientnet_b0`(5M). ImageNet 사전학습, 6채널 입력(첫 conv 가중치 timm이 자동 조정) |
| 입력 | 224×224. **현재 프레임 RGB 3ch + (현재 − 30초 전 프레임) 3ch**. 30초 전이 없으면 첫 프레임. 모두 과거 |
| 캐시 | 전 프레임을 224 uint8로 RAM 캐시(~7GB) → 디스크 병목 제거 |
| 폴드 | GroupKFold(5, seed 0) 실험 단위. 검증 fold는 500행 캡 후 평가 |
| 샘플링 | **에폭마다 실험별 최대 150프레임 무작위 추출** (78실험 → 에폭당 ~11,000장). 거의 같은 연속 프레임 반복 학습 방지 |
| 손실 | 클래스 가중치 CE(√balanced로 완화) + label smoothing 0.1 |
| 증강 | random crop(0.85~1.0)·이동·좌우 뒤집기만. 색상 변형 없음(열화상 색 = 온도) |
| 최적화 | AdamW lr 1e-4(효B0 3e-4), wd 0.05, OneCycle cos, bf16 AMP, grad clip 2.0, drop_path 0.1 |
| 에폭 | 6 (짧게. 실험 78개뿐이라 길면 과적합) |
| 출력 | `ft_*.npz`: Train은 **OOF 확률**, Val·Test는 5개 fold 모델 평균 확률. `<out>_models/fold_*.pt`: fold별 CNN 체크포인트 보존 |
| 예상 시간 | ConvNeXt-T 에폭당 ~1분 → fold당 ~8분, 5 fold ~40분. EffB0 그 절반 |

이미지 단독 CV/Val 점수도 같이 기록(시계열 없이 얼마나 되는지).

### 4단계 — 스태킹 (`exp_stack_img.py`)

OOF 확률을 LightGBM 피처로 추가:

| 피처 | 설명 |
|---|---|
| `cnn_p1..p6` | 현재 프레임 클래스 확률 |
| `cnn_stage` | 기대 단계 Σ k·p_k |
| `cnn_p*_cummax`, `cnn_stage_cummax` | 누적최대 |
| `cnn_p*_mean30/60`, `cnn_stage_mean30/60` | 과거 윈도우 평균 |
| `cnn_stage_delta30/60` | w초 전 대비 변화 |

비교: `v3`, `cnn_only`, `v3+cnn`, `v3+cnn+img_k32`, 백본 2개 평균(`ft_convnext_tiny + ft_efficientnet_b0`).

주의: LightGBM CV의 학습 fold 행 OOF 확률을 만든 CNN 모델이 평가 fold 실험을 봤을 수 있어(2차 누수) **CV는 조금 낙관적**일 수 있다. CNN은 Val을 전혀 안 봤으므로 **스태킹의 최종 판단은 Val 비중을 높여서** 본다(두 수치 모두 보고).

## 3. 실행 순서 (SLURM)

```bash
cd /home/nxcolelxu/unid/img_exp
sbatch run_embed.sh   # 방식 A: DINOv2-S/B 임베딩 → exp_img.py --stage 2   (GPU 10분 + CPU 1~2시간)
sbatch run_ft.sh      # 방식 B: convnext_tiny, efficientnet_b0 파인튜닝 → 스태킹 (GPU ~1시간 + CPU 1시간)
squeue -u $USER; tail -f logs/*.out
```

두 잡은 독립이라 다른 GPU에서 동시에 돈다. 결과는 `res_*.json`, `logs/*.log`.

## 4. 결정 규칙

이미지는 반드시 들어가므로 후보는 **이미지를 포함한 설정끼리** 비교한다. `v3`는 "이미지가 얼마나 보태는지" 보는 참고값.

1. 방식 A 후보(`v3+cos`, `v3+img_k32`, `v3+img_k64` × DINOv2-S/B)와 방식 B 후보(`v3+cnn`, `v3+cnn+img_k32`, 백본 평균) 중 **CV post F1 1위**를 고른다. 차이가 시드 표준편차(~0.002) 안이면 더 단순한 쪽(피처 적고 백본 작은 쪽).
2. 방식 B는 2차 누수로 CV가 낙관적일 수 있다. 방식 B가 CV 1위여도 **Val이 방식 A 1위보다 낮으면 방식 A**를 택한다.
3. 대회 서버(16GB)에서 재현 가능해야 하므로 백본은 **DINOv2-S 또는 ConvNeXt-T 이하**. DINOv2-B가 더 좋아도 차이가 작으면 S.
4. 어느 설정도 v3를 못 넘으면: 이미지 피처 수를 최소(코사인 거리 9개)로 줄여 손해를 가장 적게 하는 설정을 제출한다.
5. 채택되면 README에 모델명 / 출처 / 버전 / 라이선스 기재, `val0.9195`와 같은 구조의 패키지(`features.py`·`predict.py`)로 정리.

## 5. 예상 결과 기록 위치

| 파일 | 내용 |
|---|---|
| `emb_dinov2s.npz`, `emb_dinov2b.npz` | 임베딩 |
| `res_dinov2s.json`, `res_dinov2b.json` | 2단계(방식 A) CV/Val 점수 |
| `ft_convnext_tiny.npz`, `ft_efficientnet_b0.npz` | OOF/Val/Test 확률 |
| `ft_*_models/fold_*.pt` | 재사용 가능한 파인튜닝 CNN fold 체크포인트 |
| `sub_ts_*.model.joblib` | Train(+Val)로 학습한 스태킹 분류기와 피처 열 목록. `make_submission.py --model_in`으로 재학습 없이 추론 |
| `res_stack_*.json` | 4단계 점수 |
| `logs/` | 전체 로그 |

## 6. 인과 시퀀스 모델 후속 실험 (PCA 제외)

목표: 행 단위 분류와 확신 cummax에 의존하던 구조를 실험별 시간 시퀀스로 확장한다. Train 78개 실험에서 먼저 GroupKFold seed 0으로 선별하고, 개선 후보만 seed 1·2로 확인한다. Val은 최종 확인으로만 사용한다.

1. **인과 HSMM 필터**: `v3+cos`의 fold별 OOF 확률을 관측 likelihood로 사용한다. 단계 전이는 단조 증가로 제한하고, 체류 시간 분포는 각 fold 학습 실험에서만 추정한다. 체류 시간 기반 모델이므로 일반 HMM의 기하분포 대신 HSMM/명시적 체류 시간 필터를 사용한다. Test/Val은 `relative_time_s`로 순방향 필터링한다.
2. **Frozen DINO-S GRU**: 기존 `emb_dinov2s.npz`의 CLS+patch mean(768d)을 재사용한다. PCA 없이 학습형 768→64 projection + 단방향 GRU(128 hidden) + 6-class head를 학습한다. 입력은 fold 안에서 표준화한 v3 시계열 피처와 9개 cosine 피처이며, hidden state는 실험 경계마다 초기화한다.
3. **앙상블 확인**: fold가 같은 `v3+cos`, HSMM, GRU의 OOF 확률을 우선 50:50으로 결합해 비교한다. 가중치 탐색은 별도 nested CV 없이 하지 않는다.
4. **부분 DINO 미세조정**: Frozen GRU가 `v3+cos` 대비 CV post F1을 0.002 이상 개선할 때만 진행을 검토한다. DINOv2-S의 마지막 Transformer block과 LayerNorm만 풀고, encoder의 앞부분은 고정한다. 이 단계는 임베딩 캐시를 재사용하지 않고 이미지에서 encoder를 다시 계산한다.

모든 단계에서 CV fold는 실험 단위다. Train/Val/Test 평가와 test 추론은 과거 프레임만 사용한다. fold별 전처리 통계, OOF 확률, 분류기 가중치 및 학습 설정을 저장한다. 진행 상황은 `status_server.py` 페이지에 `seq_progress.json`, fold 로그 및 결과 파일을 연결해 표시한다.

초기 seed-0 결과 (`res_sequence.json`; 동일 XGBoost CUDA backend 기준): `v3+cos` CV post 0.9027 / Val 0.9209, 인과 HSMM 0.8978 / 0.9146, frozen DINO-S GRU 0.7930 / 0.8327, `v3+cos`+GRU 확률 평균 0.9038 / 0.9186. GRU는 기준선 개선 조건을 충족하지 않아 마지막 DINO block 미세조정은 일단 보류한다. HSMM 필터도 cummax 기준선보다 낮아 현재 설정은 채택하지 않는다. 이 실험의 LightGBM 라이브러리는 클러스터에서 NCCL 누락으로 로드할 수 없어 XGBoost CUDA backend를 사용했으므로, 기존 방식 A LightGBM 점수와 직접 비교하지 말고 같은 run 안의 차이로 해석한다.
