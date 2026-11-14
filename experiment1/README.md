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
