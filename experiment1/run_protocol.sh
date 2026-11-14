#!/usr/bin/env bash
# PROTOCOL.md 자동 실행 (후보 E1~E7 그리디 평가 -> 채택 규칙으로 최종 submission_protocol.csv).  --smoke 지원
# 실시간 로그: tail -f logs/latest.log   결과: outputs/protocol/{results.csv,decisions.json,final.json}
cd "$(dirname "$0")"; source ./env.sh
run_logged protocol python -u src/protocol_runner.py "$@"
