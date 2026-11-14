"""CPU 장기 큐: outputs/queue/cpu_jobs.txt 의 'name|args' 를 final_lgb.py(--name final_lgb_<name>)로 순서대로 실행하고, 끝날 때마다 hard_exp.py 로 final_lgb 대비 실험 단위 부트스트랩 비교를 outputs/CPU_REPORT.md 에 갱신.
큐가 비면 120초마다 jobs 파일을 다시 읽는다. outputs/queue/CPU_STOP 파일이 있으면 종료."""
import os, subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parents[1]; Q = R / "outputs/queue"; J, D, STOP = Q / "cpu_jobs.txt", Q / "cpu_done.txt", Q / "CPU_STOP"; ENV = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=str(R / "scripts"))
def jobs(): return [tuple(l.strip().split("|", 1)) for l in J.read_text().splitlines() if "|" in l and not l.startswith("#")]
def done(): return {l.split("|")[0] for l in D.read_text().splitlines()} if D.exists() else set()
def report():
    names = [f"final_lgb_{n}_oof" for n, _ in jobs() if n in done() and (R / f"outputs/preds/final_lgb_{n}_oof.npy").exists()]
    if not names: return
    out = subprocess.run([sys.executable, "scripts/hard_exp.py", "final_lgb_oof"] + names, cwd=R, env=ENV, capture_output=True, text=True).stdout
    keep = [l for l in out.splitlines() if l.startswith("모델별") or "부트스트랩" in l or l.startswith("  final_lgb") or "final_lgb_oof" in l and "fold" in l]
    (R / "outputs/CPU_REPORT.md").write_text("# CPU 큐 결과 (final_lgb 대비)\n\n```\n" + "\n".join(keep) + "\n```\n\n실행 이력: outputs/queue/cpu_done.txt\n")
while not STOP.exists():
    pend = [(n, a) for n, a in jobs() if n not in done()]
    if not pend: time.sleep(120); continue
    n, a = pend[0]; t0 = time.time()
    with open(R / f"outputs/logs/final_lgb_{n}.log", "w") as lf: rc = subprocess.run([sys.executable, "scripts/final_lgb.py", "--name", f"final_lgb_{n}", "--n_jobs", "8"] + a.split(), cwd=R, env=ENV, stdout=lf, stderr=subprocess.STDOUT).returncode
    with open(D, "a") as f: f.write(f"{n}|{'ok' if rc == 0 else 'FAIL'}|{time.time() - t0:.0f}s|{time.strftime('%H:%M:%S')}\n")
    report()
