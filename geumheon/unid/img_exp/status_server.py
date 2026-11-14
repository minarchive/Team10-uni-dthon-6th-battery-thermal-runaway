"""실험 진행 상황 실시간 페이지.  python status_server.py [port]  → http://localhost:8765

/ : HTML 페이지(5초마다 갱신)   /status.json : 현재 상태
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path('/home/nxcolelxu/unid')
EXP = ROOT / 'img_exp'
DATA_TOTAL = 2.99e9
USER = os.environ.get('USER', 'nxcolelxu')
_prev = {}   # 속도 추정용 (키 → (time, value))


def rate(key, value):
    now = time.time()
    t0, v0 = _prev.get(key, (now, value))
    if key not in _prev or now - t0 > 60:
        _prev[key] = (now, value)
    return (value - v0) / (now - t0) if now - t0 > 1 else 0.0


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ''


def tail(path, n=6):
    try:
        lines = Path(path).read_text(errors='replace').replace('\r', '\n').splitlines()
        return [l for l in lines if l.strip()][-n:]
    except Exception:
        return []


def fmt_eta(sec):
    if sec is None or sec != sec or sec < 0:
        return '—'
    sec = int(sec)
    return f'{sec//3600}시간 {sec%3600//60}분' if sec >= 3600 else (f'{sec//60}분 {sec%60}초' if sec >= 60 else f'{sec}초')


def stage(name, state, pct=None, detail='', eta=None, lines=None):
    return dict(name=name, state=state, pct=pct, detail=detail, eta=eta, lines=lines or [])


def download():
    d = ROOT / 'data' / 'battery_data_light'
    if (d / 'train' / 'train.csv').exists():
        n = sum(len(list((d / s / 'images').glob('*.png'))) for s in ['train', 'validation', 'test'] if (d / s / 'images').exists())
        return stage('데이터 다운로드 · 압축 해제', 'done', 100, f'{d} (이미지 {n:,}장)')
    parts = glob.glob(str(ROOT / 'data' / '*.part'))
    if parts:
        sz = os.path.getsize(parts[0])
        r = rate('dl', sz)
        eta = (DATA_TOTAL - sz) / r if r > 1e4 else None
        return stage('데이터 다운로드', 'running', sz / DATA_TOTAL * 100,
                     f'{sz/1e9:.2f} / {DATA_TOTAL/1e9:.2f} GB  ({r/1e6:.1f} MB/s)', fmt_eta(eta))
    if (ROOT / 'data' / 'data.archive').exists():
        return stage('압축 해제', 'running', None, 'tar 풀는 중')
    return stage('데이터 다운로드', 'pending')


def venv():
    sp = glob.glob(str(ROOT / 'venv' / 'lib' / 'python3*' / 'site-packages'))
    pkgs = ['torch', 'torchvision', 'lightgbm', 'sklearn', 'pandas', 'transformers', 'timm']
    have = [p for p in pkgs if sp and os.path.isdir(os.path.join(sp[0], p))]
    pip_running = bool(sh(f'pgrep -u {USER} -f "venv/bin/pip" ').strip())
    done = len(have) == len(pkgs)
    st = 'done' if done else ('running' if pip_running else 'failed')
    detail = f'설치됨: {", ".join(have) or "없음"}' + ('' if done else f'  | 남음: {", ".join(p for p in pkgs if p not in have)}')
    return stage('가상환경 설치 (venv)', st, len(have) / len(pkgs) * 100, detail)


def squeue():
    out = sh(f'squeue -u {USER} -h -o "%i|%j|%T|%M|%N|%r"')
    jobs = {}
    for l in out.splitlines():
        f = l.split('|')
        if len(f) >= 6:
            jobs.setdefault(f[1], []).append(dict(id=f[0], state=f[2], time=f[3], node=f[4], reason=f[5]))
    return jobs


def latest_log(pattern):
    fs = sorted(glob.glob(str(EXP / 'logs' / pattern)), key=os.path.getmtime)
    return fs[-1] if fs else None


def embed_stages(jobs):
    log = latest_log('embed_*.out')
    job = (jobs.get('embed') or [None])[0]
    txt = Path(log).read_text(errors='replace') if log else ''
    out = []
    for key, name in [('dinov2s', 'DINOv2-S 임베딩 추출'), ('dinov2b', 'DINOv2-B 임베딩 추출')]:
        f = EXP / f'emb_{key}.npz'
        if f.exists():
            out.append(stage(name, 'done', 100, f'{f.name} ({f.stat().st_size/1e6:.0f} MB)'))
            continue
        seg = txt.split(f'--out emb_{key}' if False else 'frames')  # noqa
        m = re.findall(r'(\d+)/(\d+) (\d+)s', txt)
        if not job:
            out.append(stage(name, 'pending' if not log else 'failed', detail='' if not log else '잡 종료, 결과 없음 — 로그 확인'))
        elif job['state'] != 'RUNNING':
            out.append(stage(name, 'queued', detail=f'SLURM {job["state"]} ({job["reason"]})'))
        elif m and not (EXP / f'emb_{key}.npz').exists() and (key == 'dinov2s' or (EXP / 'emb_dinov2s.npz').exists()):
            i, n, s = map(int, m[-1])
            r = rate(f'emb_{key}', i)
            out.append(stage(name, 'running', i / n * 100, f'{i:,} / {n:,} 프레임, {s}s 경과', fmt_eta((n - i) / r if r > 0 else None), tail(log, 3)))
        else:
            out.append(stage(name, 'running' if key == 'dinov2s' or (EXP / 'emb_dinov2s.npz').exists() else 'pending', None,
                             '시작 중 (모델 로드 / 이미지 읽는 중)', lines=tail(log, 3)))
    # exp_img 결과
    for key, name in [('dinov2s', '방식 A: LightGBM 실험 (DINOv2-S)'), ('dinov2b', '방식 A: LightGBM 실험 (DINOv2-B)')]:
        res = EXP / f'res_{key}.json'
        lg = EXP / 'logs' / f'exp_{key}.log'
        cfg_total = 4
        if res.exists():
            r = json.loads(res.read_text())
            done = [k for k in r if k.startswith('cv6/')]
            val = [k for k in r if k.startswith('val/')]
            st = 'done' if val else ('running' if (job and job['state'] == 'RUNNING') else 'failed')
            out.append(stage(name, st, (len(done) + len(val)) / (cfg_total + 2) * 100,
                             f'CV 완료 {len(done)}/{cfg_total}' + (', Val 확인 완료' if val else ''), lines=tail(lg, 6)))
        elif lg.exists():
            out.append(stage(name, 'running' if job and job['state'] == 'RUNNING' else 'failed', 0, '', lines=tail(lg, 4)))
        else:
            out.append(stage(name, 'pending'))
    return out


def ft_stages(jobs):
    out = []
    EPOCHS = 6
    for key, name, jn in [('convnext_tiny', '방식 B: ConvNeXt-Tiny 파인튜닝', 'ft_convnext'), ('efficientnet_b0', '방식 B: EfficientNet-B0 파인튜닝', 'ft_effb0')]:
        job = (jobs.get(jn) or [None])[0]
        lg = EXP / 'logs' / f'ft_{key}.log'
        npz = EXP / f'ft_{key}.npz'
        if npz.exists():
            lines = [l for l in tail(lg, 40) if l.startswith('CV') or l.startswith('VAL')]
            out.append(stage(name, 'done', 100, '', lines=lines))
            continue
        if not lg.exists():
            out.append(stage(name, 'queued' if job and job['state'] != 'RUNNING' else 'pending',
                             detail=f'SLURM {job["state"]} ({job["reason"]})' if job and job['state'] != 'RUNNING' else ''))
            continue
        txt = lg.read_text(errors='replace')
        folds = len(re.findall(r'=== fold', txt))
        eps = len(re.findall(r'^\s+ep\d+', txt, re.M))
        total = 5 * EPOCHS
        r = rate(f'ft_{key}', eps)
        st = 'running' if job and job['state'] == 'RUNNING' else 'failed'
        detail = f'fold {folds}/5, 에폭 {eps}/{total}' if folds else '이미지 캐시 만드는 중 (~47,000장 → RAM)'
        out.append(stage(name, st, eps / total * 100, detail, fmt_eta((total - eps) / r if r > 0 else None), tail(lg, 4)))
    for key, name in [('convnext', '스태킹 (ConvNeXt)'), ('effb0', '스태킹 (EffB0)'), ('both', '스태킹 (둘 평균)')]:
        # Stack jobs are submitted separately as stack_effb0 and stack_both.
        # Keep compatibility with the former single job named "stack".
        job = (jobs.get(f'stack_{key}') or jobs.get('stack') or [None])[0]
        active = ('running' if job['state'] == 'RUNNING' else
                  'queued' if job['state'] in ('PENDING', 'CONFIGURING') else 'failed') if job else 'failed'
        res = EXP / f'res_stack_{key}.json'
        lg = EXP / 'logs' / f'stack_{key}.log'
        if res.exists():
            r = json.loads(res.read_text())
            st = 'done' if any(k.startswith('val/') for k in r) else active
            out.append(stage(name, st, 100 if st == 'done' else 50, '', lines=tail(lg, 8)))
        elif lg.exists():
            out.append(stage(name, active, 0, '', lines=tail(lg, 4)))
        else:
            out.append(stage(name, 'queued' if job else 'pending',
                             detail=f'SLURM {job["state"]} ({job["reason"]})' if job else ''))
    return out


def sequence_stages(jobs):
    job = (jobs.get('seq') or [None])[0]
    prog_file = EXP / 'seq_progress.json'
    prog = {}
    if prog_file.exists():
        try:
            prog = json.loads(prog_file.read_text())
        except Exception:
            pass
    active = ('running' if job and job['state'] == 'RUNNING' else
              'queued' if job and job['state'] in ('PENDING', 'CONFIGURING') else 'failed')
    lines = tail(EXP / 'logs' / 'seq.log', 6)

    hsmm = EXP / 'res_hsmm.json'
    seq = EXP / 'res_sequence.json'
    if hsmm.exists():
        out_h = stage('인과 HSMM 필터 (v3+cos 확률)', 'done', 100, 'GroupKFold OOF 및 Val 완료', lines=lines)
    else:
        pct = prog.get('pct', 0) if prog.get('phase', '').startswith('HSMM') else 0
        out_h = stage('인과 HSMM 필터 (v3+cos 확률)', active, pct,
                      prog.get('detail', '') if active == 'running' else '', lines=lines)

    if seq.exists():
        out_g = stage('PCA 없는 frozen DINO-S GRU', 'done', 100, 'GroupKFold OOF 및 Val 완료', lines=lines)
        out_e = stage('GRU 확률 앙상블 비교', 'done', 100, 'OOF/Val 확률 평균 평가 완료', lines=lines)
        try:
            r = json.loads(seq.read_text())
            baseline = r['cv/v3+cos'].get('cummax', r['cv/v3+cos'].get('post', 0))
            gru = r['cv/gru_frozen'].get('cummax', r['cv/gru_frozen'].get('post', 0))
            delta = gru - baseline
            detail = f'frozen GRU CV Δ {delta:+.4f}; 마지막 DINO block 미세조정은 Δ≥0.002일 때 진행 검토'
            out_p = stage('DINO-S 마지막 block 미세조정 (성능 기준 통과 시)', 'pending', None, detail)
        except Exception:
            out_p = stage('DINO-S 마지막 block 미세조정 (성능 기준 통과 시)', 'pending')
    else:
        running_gru = prog.get('phase') == 'Frozen DINO-S GRU'
        out_g = stage('PCA 없는 frozen DINO-S GRU', active,
                      prog.get('pct', 0) if running_gru else None,
                      prog.get('detail', '') if running_gru and active == 'running' else '', lines=lines)
        out_e = stage('GRU 확률 앙상블 비교', 'queued' if active == 'running' else active,
                      None, 'HSMM와 GRU OOF 확률 산출 후 비교')
        out_p = stage('DINO-S 마지막 block 미세조정 (성능 기준 통과 시)', 'pending', None,
                      'frozen GRU 결과를 보고 조건부 진행')
    return [out_h, out_g, out_e, out_p]


def results():
    rows = []
    for f in sorted(EXP.glob('res_*.json')):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        for k, v in r.items():
            if k.startswith('cv6/') or k.startswith('cv/'):
                name = k.split('/', 1)[1]
                raw = v.get('raw', 0)
                post = v.get('post', v.get('cummax', v.get('filtered', 0)))
                val = r.get(f'val/{name}', {})
                val_post = val.get('post', val.get('cummax', val.get('filtered')))
                rows.append(dict(src=f.stem, cfg=name,
                                 cv_raw=raw[0] if isinstance(raw, (list, tuple)) else raw,
                                 cv_post=post[0] if isinstance(post, (list, tuple)) else post,
                                 cv_std=post[1] if isinstance(post, (list, tuple)) and len(post) > 1 else 0,
                                 val_raw=val.get('raw'), val_post=val_post,
                                 per_class=v.get('per_class_post', v.get('per_class'))))
    return rows


def status():
    jobs = squeue()
    gpu = sh(f'squeue -u {USER} -h -o "%i %j %T %M %N %b"')
    stages = [download(), venv()] + embed_stages(jobs) + ft_stages(jobs) + sequence_stages(jobs)
    return dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), stages=stages, slurm=gpu.strip().splitlines(),
                results=results(), baseline=dict(cv_post=0.9040, val_post=0.9195, cv_raw=0.9039, val_raw=0.9104))


HTML = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Uni-D 이미지 실험 진행</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--mut:#6b7280;--line:#e5e7eb;--ok:#16a34a;--run:#2563eb;--q:#d97706;--bad:#dc2626;--pend:#9ca3af}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0f1115;--card:#171a21;--fg:#e6e8eb;--mut:#9aa3af;--line:#2a2f3a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,"Segoe UI",Roboto,"Noto Sans KR",sans-serif;padding:16px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);font-size:13px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px}
.row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.name{font-weight:600;flex:1;min-width:200px}
.badge{font-size:12px;padding:2px 8px;border-radius:999px;color:#fff;background:var(--pend)}
.done{background:var(--ok)}.running{background:var(--run)}.queued{background:var(--q)}.failed{background:var(--bad)}
.bar{height:8px;background:var(--line);border-radius:4px;margin:8px 0 4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--run);transition:width .5s}
.done .bar i{background:var(--ok)}.detail{color:var(--mut);font-size:13px}.eta{font-variant-numeric:tabular-nums}
pre{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px;font-size:12px;overflow-x:auto;margin:8px 0 0;white-space:pre-wrap}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}.best{font-weight:700;color:var(--ok)}
h2{font-size:15px;margin:18px 0 8px}
</style></head><body>
<h1>이미지 + 시계열 실험 진행 상황</h1>
<div class="sub">5초마다 갱신 · 마지막 갱신 <span id="t">—</span> · 기준선 v3: CV 0.9040 / Val 0.9195 (확신 cummax)</div>
<div id="stages"></div>
<h2>SLURM 큐</h2><div class="card"><pre id="slurm">—</pre></div>
<h2>결과 (Macro F1, 500행 캡, 확신 cummax τ=0.7)</h2><div class="card"><div id="res">아직 없음</div></div>
<script>
const L={done:'완료',running:'진행 중',queued:'대기(큐)',pending:'대기',failed:'중단/오류'};
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function tick(){try{const r=await fetch('/status.json',{cache:'no-store'});const d=await r.json();
document.getElementById('t').textContent=d.time;
document.getElementById('stages').innerHTML=d.stages.map(s=>`<div class="card ${s.state}"><div class="row"><span class="name">${esc(s.name)}</span>
${s.eta&&s.state==='running'?`<span class="eta detail">남은 시간 약 ${esc(s.eta)}</span>`:''}<span class="badge ${s.state}">${L[s.state]||s.state}</span></div>
${s.pct!=null?`<div class="bar"><i style="width:${Math.min(100,s.pct).toFixed(1)}%"></i></div>`:''}
<div class="detail">${s.pct!=null?s.pct.toFixed(1)+'% · ':''}${esc(s.detail||'')}</div>
${s.lines&&s.lines.length?`<pre>${esc(s.lines.join('\n'))}</pre>`:''}</div>`).join('');
document.getElementById('slurm').textContent=d.slurm.length?d.slurm.join('\n'):'실행 중인 잡 없음';
if(d.results.length){const bestBySrc=Object.fromEntries([...new Set(d.results.map(x=>x.src))].map(s=>[s,Math.max(...d.results.filter(x=>x.src===s).map(x=>x.cv_post))]));
document.getElementById('res').innerHTML=`<table><tr><th>출처</th><th>설정</th><th>CV raw</th><th>CV post</th><th>±</th><th>Val raw</th><th>Val post</th><th>F1/클래스 (post)</th></tr>`+
`<tr><td>기준선</td><td>v3 (시계열만)</td><td>${d.baseline.cv_raw.toFixed(4)}</td><td>${d.baseline.cv_post.toFixed(4)}</td><td></td><td>${d.baseline.val_raw.toFixed(4)}</td><td>${d.baseline.val_post.toFixed(4)}</td><td></td></tr>`+
d.results.map(x=>`<tr><td>${esc(x.src)}</td><td>${esc(x.cfg)}</td><td>${x.cv_raw.toFixed(4)}</td><td class="${x.cv_post===bestBySrc[x.src]?'best':''}">${x.cv_post.toFixed(4)}</td><td>${x.cv_std.toFixed(4)}</td><td>${x.val_raw!=null?x.val_raw.toFixed(4):'—'}</td><td>${x.val_post!=null?x.val_post.toFixed(4):'—'}</td><td>${x.per_class?x.per_class.map(v=>v.toFixed(3)).join(' '):''}</td></tr>`).join('')+'</table>'}
}catch(e){document.getElementById('t').textContent='연결 실패 ('+e+')'}}
tick();setInterval(tick,5000);
</script></body></html>'''


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith('/status.json'):
            body, ct = json.dumps(status(), ensure_ascii=False).encode(), 'application/json; charset=utf-8'
        else:
            body, ct = HTML.encode(), 'text/html; charset=utf-8'
        self.send_response(200)
        self.send_header('Content-Type', ct)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f'http://localhost:{port}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', port), H).serve_forever()
