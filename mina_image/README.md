# mina_image — 열폭주 단계 예측: 이미지 인코더 비교 연구

Uni-DTHON 배터리 열폭주 6단계 분류(Macro F1)에서 **열화상 이미지만으로** 어디까지 가능한지 보고, 여러 vision encoder와 그 중간 레이어를 체계적으로 비교하는 작업 공간. 최종 목표는 시계열 센서 + 이미지 융합이다.

## 현재 계획 (2026-11-14 기준, 이 절이 최신 결정)

### 이미지 EDA로 확인된 사실 (학습 300장 표본 + 라벨별 60장 시각 확인, 전체 스캔은 `scripts/` 에서 재실행)
- 컬러바·눈금·텍스트·회색/검정/흰 테두리 없음 → **crop 불필요**.
- RGB 의사색, 이미지당 고유색 17~256개(중앙 약 210) → 256색 이하 팔레트, 컬러맵 역변환(픽셀→상대온도 인덱스) 가능성 높음.
- 색 순서: 자홍(저온) → 파랑 → 청록 → 초록 → 노랑 → 빨강 → 흰색(고온). 5~6단계는 빨강·흰색 영역과 셀 격자 윤곽이 커짐. 1단계 일부는 센서 노이즈.
- 크기 36~336(W) × 76~308(H). bat_class와 연동(class1 약 57×87, class2 약 321~327×255~308, class3 151×151 또는 336×297) → **크기가 지름길**. 라벨별 평균 크기도 다름(1단계 293×257 → 6단계 124×137).

### 전처리 정책 (인코더 투입 전)
1. crop 없음, 원본 전체 사용.
2. 긴 변 224로 맞추는 종횡비 유지 리사이즈(bicubic) + 정사각 패딩(데이터 테두리 최빈색). reflect 패딩 금지.
3. 정규화는 각 모델 공식 mean/std만. 실험별 z-score·실험 내 최대/최소 정규화 금지(미래 정보 규칙).
4. 증강 기본 없음. hue/밝기 변환 금지. 파인튜닝에서만 소폭 crop/translate.
5. W,H 메타는 인코더 입력에서 제외, B0 하한선 전용.
6. 팔레트 LUT로 온도 인덱스 1채널을 만들어(원해상도 변환 후 리사이즈, 프레임 단독) 3채널 복제 입력도 캐시.
7. ablation: 패딩 vs 왜곡 리사이즈, 입력 112/224/336 (인코더 1~2개만).

### 1차 인코더 패널 (RTX 5080 16GB, frozen 추출 + 레이어 스윕)
| 순위 | 인코더 | 근거 |
| --- | --- | --- |
| 1 | DINOv2 ViT-S/14, ViT-B/14 | frozen feature·밀집 표현 최강 |
| 2 | ConvNeXt-S (IN-22k→1k) | 현대 CNN, stage 중간층 풀링 |
| 3 | ResNet50 | 기준 모델 |
| 4 | DINO ResNet50 (ViT-S/16은 제외) | 자기지도 CNN 비교 |
| 5 | SigLIP 또는 OpenCLIP ViT-B/16 | 언어정렬, 중간층 유리 가설 |
| 6 | MAE ViT-B/16 | 저·중간층 색·질감 보존 |

2차: ViT-B/16 augreg, EfficientNet-B3, Swin-T, DINOv2-L, 무작위 초기화. 열화상 특화 가중치는 가용성 조사 후(1차 제외).

### 다음 실행 순서
1) 서버 `git pull` + CUDA 12.8+ PyTorch 환경 → 2) `data/folds.parquet` → 3) B0·B1 → 4) 팔레트 LUT·B2 → 5) DINOv2-S·ResNet50로 파이프라인 검증 후 나머지 → 6) 레이어 히트맵 → 상위 3~4개 파인튜닝.
설계 문서(Claude Docs): https://claude.ai/artifact/DzTecpPUpLfy5m3LpujzPA

### 라벨(단계)별 이미지 분석 (`data_label_split/level_1..6/`, `outputs/figures/collage_level_*.png`, 표본 기준)
- Train+Val 라벨 프레임 수: L1 18,560 · L2 6,110 · L3 12,535 · L4 2,203 · L5 3,059 · L6 1,255. 라벨은 97개 실험 전부에서 단조 비감소.
- 단계 시작 시각 중앙값(초): L2 335 · L3 540 · L4 985 · L5 1,145 · L6 1,390, 실험 간 편차 큼(L6 시작 445~7,610초) → 시간만으로 단계 추정 불가.
- 단계 지속 프레임(평균): L1 191 · L2 63 · L3 129 · L4 23 · L5 34 · L6 13 → 후반 단계는 짧아 실험당 샘플이 적음.
- **라벨×bat_class 쏠림**: class2는 L1에 11,330프레임(61%)이고 L4~L6에는 459프레임뿐. L1은 class2, L4~L6은 class1·3 위주 → 이미지 크기만으로 단계를 어느 정도 맞힐 수 있는 지름길(B0 필수).
- L1: 전체가 자홍/보라, 셀 격자가 희미(class1은 센서 노이즈). L2: 중앙부 청록~초록 시작, 일부 셀·단자가 따뜻. L3: 셀 격자 전체가 초록~노랑, 중앙이 주황. L4: 빨강·흰색 코어 등장, 일부 프레임은 흰색 포화. L5: 흰색 포화 영역이 넓고 셀 경계가 번짐. L6: L5와 비슷하나 셀이 깨진 듯 뜯긴 질감·주변 확산이 더 거칠고, 일부는 확대된 프레임.
- 구분이 어려운 쌍: L4↔L5, L5↔L6(둘 다 빨강/흰색 고온), L1↔L2(저온 단계).
- 같은 단계 안에서도 배터리 형태·카메라 거리·확대·팔레트 포화 정도가 달라 실험 간 변동이 단계 간 변동만큼 큼.

### 과적합 대책 (모든 실험 공통)
- 분할은 항상 실험 단위(`folds.parquet`), fit이 필요한 모든 것(표준화·wd·early stopping epoch)은 train fold 안쪽 실험 단위 inner split에서만 선택.
- 결과표에 `gap_train_minus_val`(inner train F1 − fold val F1)을 항상 기록, 큰 레이어/인코더는 정규화 강화.
- 프레임이 강하게 상관(같은 실험 5초 간격)이라 프레임 수가 아닌 실험 97개가 실질 표본 크기. 고차원 레이어일수록 wd 격자를 넓게.
- 지름길 점검: B0(크기)·B0c(조건) 점수를 기준선으로 두고, 인코더가 이를 못 넘으면 지름길만 학습한 것으로 판단. 크기 정책(pad↔distort) ablation으로 확인.
- 파인튜닝 단계: 낮은 lr, layer-wise lr decay, 소폭 crop/translate, label smoothing, 3시드.
- (나중에 검토) 파인튜닝용 헤드에 LeakyReLU + Dropout 사용 여부 — frozen probe 단계 결과를 본 뒤 결정.

## 실행(서버, `mina_image/` 에서)
`scripts/make_folds.py` → `scripts/baseline_b0.py` → `scripts/extract_features.py --encoder <이름>` → `scripts/probe_cv.py --encoder <이름>` → `scripts/summarize.py`. 라벨별 폴더 복사는 `scripts/split_by_label.py`, 콜라주·통계는 `scripts/eda_label_collage.py`. 인코더 이름: dinov2_s, dinov2_b, convnext_s, resnet50, siglip_b, mae_b, dino_r50. 스크립트는 로컬에서 문법만 확인했고 GPU 실행은 서버 첫 실행이 검증이다.

## 디렉터리
```
mina_image/
├── README.md        # 이 문서 (유일한 .md)
├── configs/         # 인코더·실험별 yaml
├── src/{data,encoders,probe,analysis}/
├── scripts/         # 추출·학습·평가 진입점
├── features/        # 캐시된 인코더 feature (<encoder>/<layer>.npy)
├── outputs/{logs,preds,figures}/
├── notebooks/       # EDA
└── data_link -> /home/ubuntu/data/battery_data_light
```

## 작업 원칙
1. frozen feature 추출 → 캐시 → probe를 기본 파이프라인으로 둔다.
2. 모든 비교는 동일한 분할(실험 단위 GroupKFold)과 동일한 지표(Macro F1)로 한다.
3. OOF/test 확률을 항상 저장해 시계열 모델과 스태킹·융합에 재사용한다.
4. 미래 프레임 사용 금지 규칙(§5)을 지킨다.

## 0. 과제 및 데이터 이해

### 1. 과제 한 줄 요약
각 관측 시점(행)의 열화상 이미지(+센서)로 배터리 열폭주 진행 단계(1~6)를 분류. 평가 지표 **Macro F1**. (평가 세부는 DAKER 대회 페이지 기준)

### 2. 데이터 요약
| Split | 행 | 실험 수 | 라벨 |
|---|---|---|---|
| Train | 33,522 | 78 | O |
| Validation | 10,200 | 19 | O |
| Test | 3,451 | 11 | X |

- 이미지: `data/battery_data_light/{split}/images/<sample_id>.png`, RGB 의사색(pseudo-color), **크기 가변**(가로 36~336, 세로 76~308px, 표본 기준). 색→온도 변환 기준은 제공되지 않음.
- 시간 간격: Train/Val 5초 고정, **Test는 5·10·15초 혼재** → 프레임 수가 아니라 `relative_time_s` 기준으로 다룰 것.
- 실험 조건(`bat_class`, `bat_cc`, `soc_status`, `tr_method`)은 실험 내 상수.
- 라벨 분포(Train/Val): 1·3단계 70%+, 6단계 약 2.5~3%. Val은 5단계 비중(11.1%)이 Train(5.7%)의 약 2배 → **분포 차이 존재**.

### 3. 이미지 단독 실험 관점의 핵심 관찰/가설
| # | 관찰 | 의미 / 확인 필요 사항 |
|---|---|---|
| 1 | 라벨은 시간 경과에 따른 **진행 단계**(순서형, 대체로 단조) | 단일 프레임으로는 1 vs 3, 4 vs 5 등 인접 단계 구분이 어려울 수 있음. 오류가 인접 단계에 몰리는지 확인 |
| 2 | 이미지 크기가 가변 | 크기 자체가 실험/배터리 형태(`bat_class`)의 **지름길(shortcut) 신호**일 수 있음. 리사이즈 정책(왜곡 vs 패딩)에 따른 성능 변화와, 크기 메타 feature만으로 낼 수 있는 성능을 baseline으로 측정 |
| 3 | 의사색 이미지 | 컬러맵/컬러바/텍스트 오버레이 여부를 EDA로 확인. 픽셀색→상대온도 근사(컬러맵 역변환)가 가능하면 강한 feature가 될 수 있음 |
| 4 | 이미지 통계가 베이스라인 feature | 베이스라인(`uni_d_baseline_v7`)은 이미지 통계 + 센서 + 과거 방향 시계열 → 트리 모델. 우리 이미지 only 결과의 비교 기준점으로 사용 |
| 5 | `thermal_t`는 이미지와 별개 수치 | 이미지 only 실험에서는 사용하지 않음(센서 취급). 이미지 feature와의 상관은 분석용으로만 확인 |
| 6 | 클래스 불균형 | class weight / balanced sampling / 확률 임계 보정(로짓 bias 튜닝)으로 Macro F1 직접 최적화 |
| 7 | 실험 수가 적음(97개) | 같은 실험의 프레임은 강하게 상관 → 프레임 단위 분할은 **누수**. 반드시 실험 단위 분할 |

### 4. 열려 있는 질문 (EDA로 우선 해소)
- [ ] 이미지 크기 분포 vs `bat_class`/`bat_cc`/`tr_method`/`label` 관계
- [ ] 컬러바·눈금·텍스트가 이미지에 포함되는가? 포함 시 crop 필요 여부
- [ ] 라벨별 대표 이미지 시각 확인, 단계별 시각적 차이가 있는가
- [ ] 단계 전이 시점이 실험 간 일관적인가(시간 vs 라벨 곡선)
- [ ] Train/Val/Test의 이미지 크기·색 분포 shift 여부
- [ ] 이미지가 동일/유사한 중복 프레임이 많은가(정지 구간)

### 5. 최종 목표와의 연결
이미지 단독 → (a) 최고 인코더/레이어 선정, (b) 프레임별 임베딩·OOF 확률을 저장 → (c) 센서 시계열 모델에 과거 방향 시계열 feature로 결합. 따라서 **임베딩은 반드시 프레임 단위로 캐시**한다.

## 1. 실행 계획: Phase 2 Frozen 인코더 우선

**목표:** 16GB GPU 한 장에서 이미지 단독 성능과 인코더·레이어 차이를 측정한다. Phase 0의 분할/입력 점검 후 Phase 2만 수행해도 된다. Phase 1의 이미지 통계 baseline, Phase 3~5 및 파인튜닝은 선택 사항이며 아래 시간 예산에 포함하지 않는다. 현재 코드는 구현 전이므로 소요 시간은 측정값이 아닌 계획치다.

### 1차 모델 패널과 순서

| 순서/ID | 모델 및 가중치 | 비교 목적 | 입력 | 시작 batch | 전체 추출 예산* |
|---|---|---|---|---:|---:|
| E01 | torchvision `ResNet50_Weights.IMAGENET1K_V2` | 지도학습 CNN 기준 | 224 | 32 | 10~30분 |
| E02 | DINOv2 ViT-S/14 `facebook/dinov2-small` | 경량 자기지도 기준 | 224 | 16 | 15~40분 |
| E03 | DINOv2 ViT-B/14 `facebook/dinov2-base` | AnyThermal과 동일 구조의 RGB 사전학습 대조군 | 224 | 8 | 25~60분 |
| E04 | [AnyThermal](https://huggingface.co/theairlabcmu/AnyThermal) ViT-B/14 | 열화상 사전학습 효과 | 224 | 8 | 25~60분 |
| E05 | torchvision `ViT_B_16_Weights.IMAGENET1K_V1` | 지도학습 ViT 비교 | 224 | 8 | 25~60분 |

*Train+Val+Test 47,173장, 한 모델을 한 번만 추출하는 대략적인 벽시계 시간. 다운로드, 구현, probe 학습은 제외. 배치·디코딩·스토리지에 따라 크게 달라질 수 있다. 각 체크포인트의 실제 로드, 전처리, 접근 조건과 대회 허용 여부는 실행 전 확인한다. 시간 부족 시 E01~E04만 수행하고 E04 로드가 막히면 E01~E03으로 완료한다. ConvNeXt, CLIP, MAE 등은 1차 결과 이후 선택한다.

### 시간 예산과 실행 순서

| 단계 | 작업 | 예상 벽시계 시간 |
|---|---|---:|
| 준비 | 기존 venv 확인, 추가 패키지·가중치 설치, 데이터 경로와 라벨 점검, group fold 고정 | 1~2시간 |
| 구현·스모크 테스트 | 100~500장으로 모델 로드, 레이어 hook, 캐시 정렬, GPU 메모리와 처리량 확인 | 2~4시간 |
| 특징 추출 | E01~E05 순차 실행, 전체 47,173장/모델 | 1.7~4.2시간 + 다운로드 |
| 평가·정리 | 고정 probe, 5-fold, OOF·단계별 F1 및 최종/중간층 비교 | 2~5시간 |
| 합계 | 첫 구현 포함, 실패·다운로드 제외 | **약 7~15시간** |

추출 전 각 모델에서 워밍업 후 대표 이미지 500~1,000장의 초당 처리량과 최대 VRAM을 측정한다. `47,173 / 초당 이미지 수`에 I/O·저장 여유 20~30%를 더해 모델별 예산을 갱신한다. probe 및 모델 선택 비용은 CPU와 수렴 속도에 의존하므로 별도 기록한다. 시간이 부족하면 E05와 선택 분석을 미루고, 레이어 수를 줄이며 fold 수나 그룹 분할을 임의로 줄이지 않는다.

### 16GB 운영 규칙

- 한 번에 모델 하나만 GPU에 올리고 `eval()` + `torch.inference_mode()`로 추출한다. 우선 fp16 autocast를 쓰고, 비정상 값이 있으면 fp32로 재검증한다. 시작 batch는 보장치가 아니며 OOM이면 절반씩 줄인다.
- 224 고정, 종횡비 유지 후 패딩을 기본으로 하고 모델별 공식 정규화·보간을 기록한다. 비교 시 같은 공간 정책을 유지하되 모델 고유 정규화를 적용한다. DINOv2/AnyThermal은 patch 크기 14의 배수를 사용한다.
- CNN은 stage 1~4 GAP, ViT는 block 3/6/9/12의 CLS와 patch mean을 같은 forward에서 풀링해 CPU로 옮긴다. 모델별 최종 정규화 적용 여부를 기록한다. 원본 token 전체와 GPU activation을 누적 저장하지 않는다.
- fp16 pooled feature와 `sample_id`, split, 레이어, 체크포인트 revision, 전처리 및 dtype을 저장한다. 파일 정렬을 검증하고 완료된 모델은 재추출하지 않는다. 필요한 경우 상위 모델만 전 블록으로 확장한다.

### AnyThermal 논문 반영

[AnyThermal 논문](https://arxiv.org/abs/2602.06203)은 DINOv2-B/14의 RGB teacher에서 열화상 student로 지식을 증류하고, 흑백 열화상을 3채널로 변환해 사용한다. 평가 과제는 주로 장소 인식·분할·깊이이므로 배터리 단계 분류의 성능 우위를 보장하지 않는다. 공개 [모델 카드](https://huggingface.co/theairlabcmu/AnyThermal)와 [구현](https://github.com/castacks/AnyThermal)을 기준으로 체크포인트·전처리를 검증한다.

1차 주 비교는 DINOv2-B와 AnyThermal 모두 **원본 RGB 의사색**을 사용한다. 추가 예산이 있으면 두 모델 모두 동일한 grayscale 변환 후 3채널 복제 실험을 실시한다. 일반 grayscale은 의사색의 온도 순서를 보장하지 않으므로 이를 온도 복원으로 해석하지 않는다. 논문의 paired RGB–thermal 증류를 현재 데이터로 재학습하는 것은 이번 범위가 아니다.

### 완료 기준

- [ ] 고정된 실험 단위 fold와 클래스 분포 확인
- [ ] E01~E04의 최종층 및 지정 중간층 feature 추출, E05는 시간 허용 시 실행
- [ ] 같은 probe로 fold 평균±표준편차, pooled OOF Macro F1, 클래스별 F1과 혼동행렬 기록
- [ ] 모델별 처리량·최대 VRAM·추출/학습 시간, 체크포인트 revision 및 오류 기록
- [ ] 최종 선택 모델의 OOF와 Test 확률·행 ID 저장 (Test는 모델 선택에 사용하지 않음)

## 2. 인코더 후보 카탈로그

> 상태 표기: ☐ 미조사 / ◐ 로드·추출 확인 / ✔ 평가 완료. 가중치 출처·라이선스는 사용 전 확인하고 `README`(제출 템플릿의 "사용한 사전학습 모델" 항목)에 기록한다. 아래 모델명 일부는 조사 시점에 실제 존재·가중치 접근 가능 여부를 반드시 재확인할 것(특히 열화상 특화).

### A. 지도학습 CNN (ImageNet)
| 모델 | 레이어 단위 | 비고 | 상태 |
|---|---|---|---|
| ResNet-18/50/101 (torchvision/timm) | stage1~4 (+stem) | 기준 모델. 작은 입력에도 강건 | ☐ |
| ResNeXt / Wide-ResNet | stage | ResNet 계열 용량 효과 | ☐ |
| EfficientNet-B0~B3 | block 그룹 | 소형·효율 | ☐ |
| ConvNeXt-T/S/B | stage1~4 | 현대 CNN. 중간층 풀링 실험에 적합 | ☐ |
| RegNet | stage | 선택 | ☐ |

### B. 지도학습 ViT 계열
| 모델 | 레이어 단위 | 비고 | 상태 |
|---|---|---|---|
| ViT-S/B/16 (augreg, IN-21k→1k) | block 1~12 | 지도학습 ViT 기준 | ☐ |
| DeiT | block | 선택 | ☐ |
| Swin-T/B | stage/block | 계층형, 가변 해상도에 유리 | ☐ |

### C. 자기지도 / 대규모 표현 (Frozen feature에 강점)
| 모델 | 레이어 단위 | 비고 | 상태 |
|---|---|---|---|
| DINO (v1) ViT-S/B, ResNet50 | block | 요청 항목. CNN/ViT 모두 존재 | ☐ |
| DINOv2 ViT-S/B/L (+ registers) | block 1~N, patch token | 밀집 feature 강함. patch=14 → 해상도 14 배수 | ☐ |
| DINOv3 (접근성 확인) | block | 조사 대상 | ☐ |
| MAE ViT-B | block | 복원 학습 → 저·중간층 정보 보존, 파인튜닝에서 강함 | ☐ |
| CLIP / OpenCLIP / SigLIP (image tower) | block | 의미 중심. 열화상엔 도메인 갭 큼 → 중간층 유리 가설 | ☐ |

### D. 열화상/적외선(Thermal/IR) 특화 — **조사 필요**
열화상 이미지로 사전학습된 공개 가중치가 있는지 아래 순서로 조사:
1. HuggingFace Hub / timm / torchvision / Papers with Code에서 `thermal`, `infrared`, `FLIR`, `LLVIP`, `RGB-T`, `TIR` 키워드 검색
2. 후보 유형: (a) FLIR/LLVIP 등 열화상 검출·분할 backbone, (b) RGB-T 멀티모달 백본, (c) IR 객체 추적(SiamRPN류 등) 백본, (d) 열화상 대상 자기지도 사전학습 모델
3. 평가 항목: 가중치 공개 여부, 라이선스, 입력 규격, 의사색/흑백 중 어떤 형식으로 학습됐는지(**본 데이터는 RGB 의사색**이므로 흑백 IR 학습 모델은 도메인 갭 가능)
- 가용 모델이 없거나 부적합하면 대안: **(i) ImageNet 모델을 이 데이터로 도메인 적응(자기지도 계속 학습: DINO/MAE류), (ii) 컬러맵 역변환으로 온도 근사 1채널 이미지를 만들어 입력**.
- 결과는 이 표에 모델별 한 줄로 추가: `이름 | 출처 | 라이선스 | 학습 데이터 | 입력형식 | 상태`.

### E. 비교 대조군
| 대조군 | 목적 |
|---|---|
| 무작위 초기화 동일 아키텍처(Frozen / FT) | 사전학습의 순수 기여도 |
| 이미지 통계 GBM (Phase 1 B1) | 딥 feature의 필요성 |
| 소형 CNN from scratch | 작은 입력·소규모 데이터 적합성 |

### 공통 인터페이스 (구현 시)
```python
class Encoder:
    name: str
    def preprocess(img) -> Tensor            # 모델 고유 정규화 + 리사이즈 정책
    def layer_names() -> list[str]           # hook 지점
    def extract(batch) -> dict[str, Tensor]  # layer_name -> (B, D) pooled feature (+ 선택: patch grid)
```
- 풀링 기본값: CNN은 GAP, ViT는 CLS와 patch-mean 둘 다 저장. 공간 보존이 필요하면 2×2/4×4 adaptive pool도 저장.
- 입력 크기: 데이터가 작음(36~336px) → 업샘플링 비율이 인코더 성능에 영향. 해상도 ablation 포함.

### 패널 구성

1차 실행 대상과 우선순위는 §1 표를 따른다. 이 카탈로그의 나머지는 후속 후보이다.

## 3. 중간 레이어(Intermediate Layer) 활용 전략

### 동기
- 열화상 의사색 이미지는 자연 이미지와 도메인 갭이 크다. 마지막 레이어는 ImageNet/언어 의미에 특화돼 있어, **저·중간 레이어의 색·질감·형태(온도 분포 패턴) 정보가 더 유용**할 수 있다는 가설.
- 열폭주 단계는 "고온 영역의 크기·강도·모양" 같은 저·중수준 시각 단서에 가까울 가능성이 높다.

### 분석 (무엇이 어디에 있는가)
| 분석 | 방법 | 산출 |
|---|---|---|
| 레이어별 linear probe | 각 레이어 pooled feature → LR, 실험 단위 CV Macro F1 | 인코더×레이어 히트맵 |
| 레이어별 클래스 분리도 | 단계별 centroid 거리/실루엣, LDA 분리 비 | 레이어별 분리도 곡선 |
| 표현 유사도 | linear CKA: 레이어×레이어(모델 내), 모델×모델(같은 상대 깊이) | CKA 행렬 |
| 단계별 오류 | 레이어별 단계별 F1, 인접 단계 혼동 | 어떤 단계가 어느 깊이에서 풀리는지 |
| 정보 중복성 | 레이어 간 probe 확률 상관, 앙상블 이득 | 보완적 레이어 쌍 |
| 시각화 | CNN Grad-CAM, ViT attention rollout/patch feature PCA | 모델이 보는 영역이 고온 영역인지 확인 |

### 활용 방안 (어떻게 쓸 것인가) — 단순한 것부터
| ID | 방식 | 설명 | 비고 |
|---|---|---|---|
| L1 | 최종 레이어만 | 기준 | |
| L2 | 단일 최적 레이어 | 히트맵 argmax | 선택 편향 방지 위해 fold 내부에서 선택 |
| L3 | 다층 concat | 선택 레이어들을 풀링 후 concat → 표준화 → LR | 차원 큼 → PCA/L2 강정 |
| L4 | 학습형 가중합 | 레이어별 feature를 투영(공통 차원) 후 softmax 가중합 (ELMo/probing 스타일) | 학습된 가중치가 곧 레이어 중요도 |
| L5 | 레이어 앙상블 | 레이어별 probe 확률의 평균/스태킹 | 구현 단순, 강력한 기준 |
| L6 | 풀링 변형 | GAP / GeM / max / 어텐션 풀링, ViT는 CLS vs patch mean | 열 영역은 "국소 최대"가 중요 → max·GeM 가설 |
| L7 | 공간 보존 | 마지막 feature map을 2×2, 4×4로 풀링 후 flatten | 뜨거운 부위의 위치/분포 반영 |
| L8 | FPN/경량 융합 헤드 | 여러 stage feature map을 정렬·합산하는 소형 헤드 학습 | 파인튜닝 단계에서만 |
| L9 | 하이브리드 | 레이어 가중합 + 센서 feature (융합 단계에서) | Phase 5 이후 |

### 운영 규칙
- 레이어 선택·가중치 학습은 **반드시 fold 내부(train fold)** 에서만 하고, 검증 fold로 점수를 보고(선택 누수 방지).
- 1차는 모든 레이어에서 C=1로 고정. 후속 튜닝은 모든 레이어에 같은 탐색 격자를 쓰고 outer train 내부에서 수행.
- 선택한 레이어 feature는 fp16 캐시. ViT-B 12블록×768 ≈ 프레임당 9.2K float → 47K 프레임 ≈ 약 0.9GB(fp16). 용량 문제 없음.
- patch-grid까지 저장하는 건 상위 후보 1~2개만(용량·시간 절약).

### 기대 가설 (검증 대상, 결론 아님)
1. 지도학습 CNN은 중간 stage(2~3)가 최종 stage보다 좋을 수 있다.
2. DINOv2는 최종 레이어도 강하고 patch 기반 풀링에서 이득이 클 수 있다.
3. CLIP류는 중간 레이어가 최종보다 확실히 낫고 전체적으로는 열세일 수 있다.
4. 다층 앙상블(L5)이 단일 최적 레이어(L2)보다 안정적이다.
5. 파인튜닝 후엔 레이어 간 차이가 줄어든다.

## 4. 평가 프로토콜

### 분할
- **단위: `experiment_id`.** 같은 실험의 프레임은 절대 train/valid에 나뉘지 않는다.
- 사용 데이터: Train(78) + Validation(19) = **97 실험**을 합쳐 행별 라벨 기준 `StratifiedGroupKFold(k=5, shuffle=True, random_state=42)`. 클래스별 실험 수와 fold별 라벨·조건 분포를 확인하고, 희귀 클래스가 빠진 fold가 있으면 분할을 재검토한 뒤 `data/folds.parquet`에 저장, 모든 실험이 공유.
- 보조 보고: 필요 시 별도 모델을 Train만으로 학습해 제공된 **Train→Validation 단일 분할** 점수를 기록(대회 베이스라인과 직접 비교용). Val은 5단계 비중이 크므로 CV와 차이가 날 수 있음을 감안.
- Test(11 실험)는 최종 예측에만 사용. 분포·크기 분석(레이블 없이)은 허용하되 모델 선택에 쓰지 않는다.

### 지표
- **주 지표: Macro F1** (6 클래스 단순 평균).
- 보조: 단계별 F1/Recall, 혼동행렬, 인접 단계 오류 비율(|예측−정답|≤1), 순서 오차(MAE of stage), 실험별 점수 분포(약한 실험 식별).
- 보고 형식: fold 평균 ± 표준편차, 그리고 pooled OOF Macro F1 둘 다.

### 불균형 대응 (공정 비교를 위해 고정)
- Probe 기본: StandardScaler + Logistic Regression(C=1, L2, `class_weight='balanced'`), scaler는 각 train fold에서만 fit. 레이어별 점수는 탐색적으로 보고하며 최적 레이어의 성능 추정은 outer train 내부 group CV로 선택 후 outer fold에서 평가한다.
- 사후 보정: OOF 로짓에 클래스별 bias를 학습해 Macro F1 최대화 — fold 내부에서만 튜닝, 사용 여부를 표에 명시.
- 인코더 간 비교는 "보정 없음" 기준 표를 주표로, "보정 후" 표를 보조로 둔다.

### 공정성 규칙
| 항목 | 규칙 |
|---|---|
| 전처리 | 정책 ablation을 하는 실험 외에는 전 인코더 동일 공간 정책(입력 해상도·리사이즈 방식 명시), 모델별 공식 정규화 |
| probe | 동일 알고리즘·동일 하이퍼 탐색 격자 |
| 시드 | 최소 3시드(파인튜닝), probe는 고정 분할 + 결정적 |
| 튜닝 | 1차는 고정 C=1로 튜닝 없이 비교. 추후 하이퍼·레이어 선택 시 outer train 내부 group CV로 선택 |
| 모델 선택 | OOF 점수 기준. Validation 단일 분할 점수로 선택하지 않음 |
| 기록 | 모든 실행을 이 README의 실험 로그에 설정 파일 해시·점수와 함께 기록 |

### 주의 (누수/착시)
- 이미지 크기·파일 속성이 실험 식별 정보를 담고 있을 수 있음 → B0 baseline으로 정량화, 크기를 통일했을 때 성능이 떨어지는지 확인.
- 정규화 통계(StandardScaler, PCA)는 train fold에서만 fit.
- 재현 확인: 제출 시 운영진이 재실행할 수 있으므로 추론 파이프라인은 입력 CSV 행만으로 동작해야 함(05).

## 5. 규칙 및 제약

> 출처: Uni-DTHON 데이터 자료집. **자료집과 DAKER 대회 페이지가 다르면 대회 페이지가 우선.** 제출 직전 대회 페이지의 평가·참가 규정을 다시 확인할 것.

### 미래 시점 정보 사용 금지 (이미지 파이프라인에 직접 적용)
각 시점의 예측은 **같은 실험의 현재 및 과거 관측만** 사용.
| 구분 | 내용 |
|---|---|
| 허용 | 현재 프레임, 같은 실험의 과거 프레임(embedding EMA, 지난 k초 변화량 등), 실험 조건값, Train/Val에서 계산해 **고정한** 통계(정규화 평균·분산, PCA, 학습된 가중치) |
| 금지 | 같은 실험의 미래 프레임 사용, 예측 대상 실험 전체 통계로 정규화(예: 실험별 z-score), 이후 관측을 이용한 이동 창·보간·결측 대체, **실험 종료 시각·전체 행 수 활용** |
| 주의 | 이미지 정규화를 "실험 내 최대/최소 온도"로 하면 금지 위반. 컬러맵 역변환 시에도 프레임 단독 규칙만 사용 |
| 주의 | Test 간격이 5/10/15초 → 과거 창은 행 수가 아니라 `relative_time_s` 차이로 정의 |
| 금지 | `label`을 입력으로 사용 |

### 구현 시 체크 (코드 리뷰 항목)
- [ ] 시계열/집계 연산이 `groupby(experiment_id)` 내에서 `relative_time_s` 오름차순 + **causal**(shift/rolling closed to past)인가
- [ ] fit이 필요한 모든 변환이 train 데이터에서만 학습되고 파일로 고정되는가
- [ ] 추론 코드가 test CSV의 한 행(및 그 이전 행)만으로 동작하는가 — 미래 행을 지워도 같은 예측이 나오는지 **자동 테스트**(truncation test) 작성
- [ ] 사용한 사전학습 모델의 출처·버전·라이선스를 README에 기록했는가

### 데이터 사실 (코드 가정)
- `meas_a`는 상수, `meas_t1_plus_t2`는 파생. 결측은 `indu_t`, `indu_t.1`뿐. (센서 쪽 사실이며 이미지 only 단계에서는 사용하지 않음)
- `image_path`는 split 폴더 기준 상대경로(`images/<sample_id>.png`).
- 제출: `sample_submission.csv`와 동일한 행 수(3,451)·순서, `label` 1~6 정수.

### 환경 제약
| 항목 | 상태 |
|---|---|
| GPU | NVIDIA RTX 5080, 16GB (Blackwell, sm_120 → PyTorch CUDA 12.8+ 빌드 필요) |
| 디스크 | 여유 약 860GB |
| 파이썬 패키지 | `/home/ubuntu/venv`: torch 2.11.0+cu128, torchvision 0.26.0+cu128 설치 확인. 추가 패키지는 필요 시 설치하고 버전 기록 |
| 데이터 위치 | `/home/ubuntu/data/battery_data_light` (심볼릭 링크: `data_link`) |
| 베이스라인 참고 | `/home/ubuntu/data/baseline/uni_d_baseline_v7 (1)/uni_d_baseline_v7/` (model.py, train.py, test.py) |
| 외부 가중치 | 다운로드 필요(인터넷 가능 여부 확인). 대회에서 사전학습 모델 사용이 허용되는지 대회 페이지에서 확인 후 진행 |

## 6. 실험 로그

규칙: 한 번 실행 = 한 줄. 설정 파일(`configs/*.yaml`)과 `outputs/preds/` 파일명에 `ID`를 사용.
CV = 실험 단위 5-fold Macro F1 (평균±표준편차), OOF = pooled OOF Macro F1, Val = Train→Val 단일 분할.

### Phase 1 하한선
| ID | 설명 | 입력 | CV | OOF | Val | 비고 |
|---|---|---|---|---|---|---|
| B0 | 크기 메타 + GBM | W,H,ratio | | | | shortcut |
| B1 | 이미지 통계 + GBM | | | | | |
| B2 | 컬러맵 역변환 온도 통계 | | | | | |
| B3 | 소형 CNN scratch | | | | | |

### Phase 2 Frozen 인코더 (최종 레이어 / 최적 레이어 / 다층)
| ID | 인코더 | 입력 해상도·정책 | 최종층 F1 | 최적층(층, F1) | 다층 L3/L5 | 비고 |
|---|---|---|---|---|---|---|
| E01 | ResNet50 (sup) | 224/패딩 | | | | |
| E02 | DINOv2-S | | | | | |
| E03 | DINOv2-B | | | | | |
| E04 | AnyThermal | | | | | |
| E05 | ViT-B/16 (sup) | | | | | |
| E06 | MAE ViT-B | | | | | |
| E07 | CLIP/SigLIP | | | | | |
| E08 | 열화상 특화 #1 | | | | | |

### Phase 3 중간 레이어 기법 (최상위 인코더 기준)
| ID | 기법 | 인코더 | CV | 최종층 대비 Δ | 비고 |
|---|---|---|---|---|---|

### Phase 4 파인튜닝
| ID | 인코더 | 방식(LP/부분/LoRA/FT) | 증강 | 손실 | CV (3 seed) | 비고 |
|---|---|---|---|---|---|---|

### 결정 기록
| 날짜 | 결정 | 근거 |
|---|---|---|
| 2026-11-14 | 실험 단위 StratifiedGroupKFold(5), Train+Val 통합 | 실험 수 적고 프레임 상관 큼, Val 분포가 Train과 다름 |
| 2026-11-14 | frozen feature 캐시 + probe 우선, 파인튜닝은 상위 후보만 | 인코더·레이어 비교 비용 최소화 |
