"""GPU 장기 큐 러너: outputs/queue/jobs.txt 의 'name|args' 줄을 위에서부터 mm_fuse.py 로 실행(기본 인자 BASE 뒤에 붙임). 한 줄 끝날 때마다 qsummary.py 로 REPORT 갱신.
큐가 비면 auto_next(기준 대비 +0.003 이상 이득을 낸 변형들을 합친 조합 설정)를 추가하고, 그것도 없으면 대기하며 jobs.txt 에 줄이 추가되길 기다린다. outputs/queue/STOP 파일이 있으면 종료.
실패한 작업은 failed 로 기록하고 다음으로 넘어간다. 사용: nohup python scripts/qrunner.py &"""
import os, re, subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parents[1]; Q = R / "outputs/queue"; JOBS, DONE, STOP = Q / "jobs.txt", Q / "done.txt", Q / "STOP"
BASE = "--gas sum --rules 1 --heads tr --tab 100"; REF = "all_hdtr"; ENV = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=str(R / "scripts"))
log = lambda *m: print(f"[{time.strftime('%H:%M:%S')}]", *m, flush=True)
def jobs(): return [tuple(l.strip().split("|", 1)) for l in JOBS.read_text().splitlines() if "|" in l and not l.startswith("#")]
def done(): return {l.split("|")[0] for l in DONE.read_text().splitlines()} if DONE.exists() else set()
def score(run):
    import numpy as np
    f = R / f"outputs/preds/ft_mm_all_hdtr_{run}_oof_gated.npy"; return f.exists()
def gated_table():
    """run -> gated OOF (REPORT.md 표에서 읽음)"""
    t = {}
    for l in (R / "outputs/REPORT.md").read_text().splitlines() if (R / "outputs/REPORT.md").exists() else []:
        m = re.match(r"\| (\S+) \| ([0-9.]+) \|", l)
        if m: t[m.group(1)] = float(m.group(2))
    return t
def auto_next():
    t = gated_table(); base = t.get(REF)
    if base is None: return False
    win = [(n, a) for n, a in jobs() if t.get("all_hdtr_" + n, 0) >= base + 0.003 and not n.startswith("combo")]
    win.sort(key=lambda x: -t["all_hdtr_" + x[0]]); existing = {n for n, _ in jobs()}
    if len(win) >= 2:
        for k in (2, 3, len(win)):
            if k > len(win): continue
            nm = "combo" + "_".join(n for n, _ in win[:k])
            if nm not in existing:
                with open(JOBS, "a") as f: f.write(f"{nm}|{' '.join(a for _, a in win[:k])}\n")
                log("auto_next: 조합 추가", nm); return True
    return False
def summarize():
    subprocess.run([sys.executable, "scripts/qsummary.py"], cwd=R, env=ENV, capture_output=True)
while not STOP.exists() and not (Path("/tmp/mm14.out").exists() and "MM14_DONE" in Path("/tmp/mm14.out").read_text()): time.sleep(5)
summarize()
while not STOP.exists():
    pend = [(n, a) for n, a in jobs() if n not in done()]
    if not pend:
        if not auto_next(): time.sleep(120)
        continue
    n, a = pend[0]
    if n.startswith("FT_"): name = n[3:]; cmd = [sys.executable, "scripts/finetune.py", "--name", name] + a.split(); logn = f"outputs/logs/ft_{name}.log"      # 이미지 단독 파인튜닝(OOF/test 확률 -> LightGBM 이미지 피처 소스)
    else: name = f"all_hdtr_{n}"; cmd = [sys.executable, "scripts/mm_fuse.py", "--name", name] + (BASE + " " + a).split(); logn = f"outputs/logs/mm_{name}.log"
    log("START", name, BASE, a); t0 = time.time()
    with open(R / logn, "w") as lf: rc = subprocess.run(cmd, cwd=R, env=ENV, stdout=lf, stderr=subprocess.STDOUT).returncode
    with open(DONE, "a") as f: f.write(f"{n}|{'ok' if rc == 0 else 'FAIL'}|{time.time() - t0:.0f}s|{time.strftime('%H:%M:%S')}\n")
    log("DONE" if rc == 0 else "FAIL", name, f"{time.time() - t0:.0f}s"); summarize()
log("STOP")
