"""이미지 단독 파인튜닝(FT_*)이 끝나 OOF/test 가 생기면, 그 확률을 LightGBM 이미지 피처로 쓰는 CPU 작업(img_<name>_td)을 cpu_jobs.txt 에 자동 추가."""
import time
from pathlib import Path
R = Path(__file__).resolve().parents[1]; Q = R / "outputs/queue"
while not (Q / "CPU_STOP").exists():
    done = (Q / "done.txt").read_text().splitlines() if (Q / "done.txt").exists() else []; cj = (Q / "cpu_jobs.txt").read_text()
    for l in done:
        n = l.split("|")[0]
        if n.startswith("FT_") and "|ok|" in l:
            nm = n[3:]; job = f"img_{nm}_td"
            if job not in cj and (R / f"outputs/preds/ft_{nm}_oof.npy").exists() and (R / f"outputs/preds/ft_{nm}_test.npy").exists():
                (Q / "cpu_jobs.txt").open("a").write(f"{job}|--img ft_{nm} --test_dist 1\n"); cj += job
    time.sleep(60)
