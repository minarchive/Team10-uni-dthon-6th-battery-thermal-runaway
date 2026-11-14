#!/usr/bin/env bash
# 공통 환경. 서버에 맞게 DATA_DIR / VENV 를 export 해서 덮어쓸 수 있음.
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_DIR="${DATA_DIR:-/home/ubuntu/data/battery_data_light}"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$EXP_DIR/src:${PYTHONPATH:-}"
export N_JOBS="${N_JOBS:-$(nproc 2>/dev/null || echo 8)}"
# 가상환경이 있으면 활성화 (VENV 로 경로 지정, 기본: 저장소 루트 .venv 또는 experiment1/.venv)
for v in "${VENV:-}" "$EXP_DIR/.venv" "$EXP_DIR/../.venv" "$EXP_DIR/../mina_image/.venv"; do
  [ -n "$v" ] && [ -f "$v/bin/activate" ] && { source "$v/bin/activate"; break; }
done
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/outputs" "$EXP_DIR/cache"

# run_logged <이름> <명령...> : 화면 + logs/<이름>_<시각>.log 동시 기록, logs/latest_<이름>.log 심볼릭 링크 갱신
run_logged() {
  local name=$1; shift
  local f="$EXP_DIR/logs/${name}_$(date +%Y%m%d_%H%M%S).log"
  : > "$f"
  ln -sf "$f" "$EXP_DIR/logs/latest_${name}.log" 2>/dev/null || true
  ln -sf "$f" "$EXP_DIR/logs/latest.log" 2>/dev/null || true
  echo "[$(date +%T)] START $name -> $f" | tee -a "$f"
  "$@" 2>&1 | tee -a "$f"; local rc=${PIPESTATUS[0]}
  echo "[$(date +%T)] END   $name (exit=$rc)" | tee -a "$f"
  return $rc
}
