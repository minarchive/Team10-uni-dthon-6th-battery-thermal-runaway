# experiment1 — EDA + LightGBM(센서 인과 피처) + 이미지 게이팅

서버 실행 (데이터 기본 경로 `/home/ubuntu/data/battery_data_light`, 다르면 `DATA_DIR` 지정):

```bash
cd experiment1
pip install -r requirements.txt            # venv 가 .venv 에 있으면 env.sh 가 자동 활성화
bash run_features.sh                       # 이미지 통계 캐시 (cache/, 1회)
bash run_train.sh                          # ablation -> 설정 선택 -> outputs/submission.csv
bash run_gated.sh                          # 센서 주도 + 이미지 게이팅 -> outputs/submission_gated.csv
bash run_notebook.sh                       # EDA 노트북 experiment1.ipynb 생성/실행
# 전부: nohup bash run_all.sh > logs/run_all.out 2>&1 &
tail -f logs/latest.log                    # 실시간 로그 (logs/<단계>_<시각>.log)
```

- 스모크 테스트: `bash run_train.sh --smoke`, `bash run_gated.sh --smoke`
- 모든 시계열 피처는 같은 실험의 현재·과거 관측만 사용 (미래 정보 금지 규칙 준수)
- 대회 규정상 `cache/`, `outputs/`, `logs/*.log`, 실행된 노트북은 커밋하지 않음 (.gitignore)

## 이후 추가된 실험 (모두 CPU, 실행 스크립트는 `run_*.sh`, 로그는 `logs/`)
| 스크립트 | 내용 |
|---|---|
| `PROTOCOL.md`, `run_protocol.sh` | 규칙 후보를 실험 단위 검정으로 채택/기각하는 고정 프로토콜(Test 모사 평가). 라운드 1 결과 전부 기각, 라운드 2 는 중단 |
| `run_eda_v2.sh`, `src/make_eda_v2.py` | 사건 정렬 EDA: 경계 사전, 모달리티 정보 검정(이미지는 센서에 정보 없음), 경계 변위·천장 |
| `run_eda_v3.sh`, `src/make_eda_v3.py` | 분할 구성 순열 검정, 이상치, 오류 분해 확장, 단계 지속시간, 저용량 모델 |
| `run_evidence.sh` | 스케일링·이미지 가공 방식의 직접 측정 |
| `run_events.sh`, `src/events.py`, `src/rules.py` | 사건·규칙 피처(인과성 검사 포함) |
| `run_team.sh`, `src/team_pipeline.py`, `src/team_suite.py` | `mina_image/scripts/final_lgb.py` 로컬 재현(0.9147 / 0.9184 일치) + XGBoost·ExtraTrees·배깅·앙상블·증강 스위트 |
| `run_team.sh team_physics`, `src/team_physics.py` | 리뷰 논문 기반 물리 피처(Arrhenius ln(dT/dt) vs 1/T, 히터 대비 자체 발열) 시험. 인과성 검사 포함, B0 대비 실험 단위 CI. 결과: 채택 근거 없음 |

주의: 각형·과충전 Test 실험 2개의 시작 압력이 학습 최대의 3~7배다(학습 실험 7개뿐). CV 점수는 이 분포 이동을 보지 못하므로 CV 만으로 채택하지 말 것.
