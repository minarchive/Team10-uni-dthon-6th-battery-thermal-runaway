#!/usr/bin/env bash
# 팀 LightGBM 로컬 재현/확장 실험 (CPU).  사용: bash run_team.sh team_b0 | team_suite ...
cd "$(dirname "$0")"; source ./env.sh
run_logged "$1" python -u "src/$1.py" "${@:2}"
