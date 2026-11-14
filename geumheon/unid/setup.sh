#!/usr/bin/env bash
# Uni-DTHON 팀 서버 초기 세팅 스크립트
# 팀에서 한 명만 한 번 실행하세요:  bash setup.sh
# 이미 끝난 단계는 자동으로 건너뛰므로, 중간에 실패하면 다시 실행해도 됩니다.
set -e

# ▼▼▼ 아래 두 줄의 따옴표 안에 구글 드라이브 파일 링크를 붙여 넣으세요 (따옴표 " " 는 지우지 마세요) ▼▼▼
# 이 setup.sh를 받은 드라이브 폴더 안의 파일을 우클릭 → 공유 → 링크 복사 (구글 계정 로그인 상태에서)
# DATA_URL     ← battery_data_for_participants.tar 파일의 링크
# BASELINE_URL ← uni_d_baseline_v7.zip 파일의 링크
DATA_URL="https://drive.google.com/file/d/1qXoN_aAhX6lY-B3fZ3kWA3a_Nn8SkG1n/view?usp=share_link"
BASELINE_URL="https://drive.google.com/file/d/1WTYj-GhVl8LugZVI6TsLq-lp-QAXOA7x/view?usp=share_link"


for u in "$DATA_URL" "$BASELINE_URL"; do
  case "$u" in
    https://drive.google.com/file/*) ;;
    https://drive.google.com/drive/folders/*)
      echo "❌ 폴더 링크가 들어 있습니다. 폴더 안의 '파일'을 우클릭 → 공유 → 링크 복사로 다시 넣으세요."; exit 1 ;;
    *)
      echo "❌ setup.sh 맨 위 DATA_URL / BASELINE_URL에 구글 드라이브 파일 링크를 넣지 않았습니다."
      echo "   링크는 https://drive.google.com/file/d/... 로 시작해야 합니다."; exit 1 ;;
  esac
done

VENV="${VENV:-$HOME/venv}"
DATA_DIR="${DATA_DIR:-$HOME/data}"
START=$(date +%s)
step() { echo; echo "===== $1 ====="; }
# 구글 드라이브 공유 링크 다운로드 (gdown 버전에 따라 --fuzzy 필요 여부가 달라 둘 다 시도)
gdl() { gdown "$1" -O "$2" || gdown --fuzzy "$1" -O "$2"; }
# tar / zip 자동 판별해서 풀기
extract() {
  mkdir -p "$2"
  if tar -tf "$1" >/dev/null 2>&1; then tar -xf "$1" -C "$2"; else python -m zipfile -e "$1" "$2"; fi
}

step "1/5 Python 가상환경($VENV)"
if [ ! -x "$VENV/bin/python" ]; then
  if ! python3 -m venv "$VENV" 2>/dev/null; then
    echo "python3-venv가 없어 설치합니다 (sudo 비밀번호를 물으면 서버 비밀번호 입력)"
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv
    python3 -m venv "$VENV"
  fi
else
  echo "이미 있음 → 건너뜀"
fi
source "$VENV/bin/activate"
pip install -q --upgrade pip

step "2/5 torch(GPU) 설치"
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "이미 설치됨 → 건너뜀"
else
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

step "3/5 Jupyter, gdown 설치"
pip install -q jupyterlab ipykernel gdown

step "4/5 데이터·베이스라인 내려받기 ($DATA_DIR)"
mkdir -p "$DATA_DIR"
if [ ! -d "$DATA_DIR/battery_data_light" ]; then
  rm -rf "$DATA_DIR/data.archive"
  gdl "$DATA_URL" "$DATA_DIR/data.archive"
  if [ -d "$DATA_DIR/data.archive" ]; then
    echo "⚠ DATA_URL이 폴더 링크입니다. 데이터는 tar(또는 zip) 파일 하나의 링크여야 합니다."
    exit 1
  fi
  extract "$DATA_DIR/data.archive" "$DATA_DIR"
  rm -f "$DATA_DIR/data.archive"
  if [ ! -d "$DATA_DIR/battery_data_light" ]; then
    echo "⚠ 압축은 풀렸지만 battery_data_light 폴더가 없습니다. 아래 구조를 운영진에게 보여 주세요."
    ls "$DATA_DIR"
    exit 1
  fi
else
  echo "데이터 이미 있음 → 건너뜀"
fi
if [ -z "$(ls -A "$DATA_DIR/baseline" 2>/dev/null)" ]; then
  rm -rf "$DATA_DIR/baseline.archive" "$DATA_DIR/baseline"
  gdl "$BASELINE_URL" "$DATA_DIR/baseline.archive"
  if [ -d "$DATA_DIR/baseline.archive" ]; then
    # 폴더 링크였던 경우: 받은 파일들을 그대로 사용
    mkdir -p "$DATA_DIR/baseline"
    mv "$DATA_DIR/baseline.archive" "$DATA_DIR/baseline/uni_d_baseline_v7"
  else
    extract "$DATA_DIR/baseline.archive" "$DATA_DIR/baseline"
    rm -f "$DATA_DIR/baseline.archive"
  fi
else
  echo "베이스라인 이미 있음 → 건너뜀"
fi

step "5/5 VS Code 기본 Python을 $VENV로 지정"
MS=~/.vscode-server/data/Machine
mkdir -p "$MS"
python - "$MS/settings.json" "$VENV/bin/python" <<'PY'
import json, os, sys, shutil
path, py = sys.argv[1], sys.argv[2]
data = {}
if os.path.exists(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        data = json.loads(text) if text else {}
    except Exception:
        print("settings.json을 읽을 수 없어 건드리지 않음 (VS Code에서 직접 python.defaultInterpreterPath 추가)")
        sys.exit(0)
    shutil.copy(path, path + ".bak")
data["python.defaultInterpreterPath"] = py
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print("설정 완료:", py, "(기존 파일은 settings.json.bak으로 백업)" if os.path.exists(path + ".bak") else "")
PY

echo
echo "===== 확인 ====="
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
ls "$DATA_DIR" "$DATA_DIR/battery_data_light"
echo "걸린 시간: $(( ($(date +%s) - START) / 60 ))분"
echo
echo "완료! 팀원은 각자 아래처럼 베이스라인을 본인 폴더로 복사해서 쓰세요:"
echo "  mkdir -p ~/<내이름> && cp -r $DATA_DIR/baseline/* ~/<내이름>/"