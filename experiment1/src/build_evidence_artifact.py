"""근거 아티팩트 HTML 생성: 템플릿의 {{FIG_x}} 를 outputs/evidence/fig_x.png 의 data URI 로 치환 (집계 그림만, 원본 프레임 없음)."""
import base64
from pathlib import Path
R = Path(__file__).resolve().parents[1]
t = (R / "src" / "evidence_artifact_template.html").read_text(encoding="utf-8")
for name in ["modality", "displacement", "boundary_auc", "jitter", "team_compare", "real_experiment_1", "real_experiment_2", "real_experiment_3", "real_stage_collage", "real_framing", "real_processing", "real_relative", "image_variants_left"]:
    fn = name if name.startswith("real_") else f"fig_{name}"
    b = base64.b64encode((R / "outputs" / "evidence" / f"{fn}.png").read_bytes()).decode()
    t = t.replace("{{FIG_%s}}" % name, "data:image/png;base64," + b)
assert "{{FIG_" not in t
out = R / "outputs" / "evidence" / "evidence_artifact_real.html"
out.write_text(t, encoding="utf-8"); print(out, round(out.stat().st_size / 1e6, 2), "MB")
