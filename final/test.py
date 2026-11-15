"""추론: python test.py --input <test.csv> --image_root <이미지 폴더> --output <submission.csv>
입력 CSV에 있는 행만 사용한다. 피처·이미지 입력은 같은 실험의 현재·과거 행으로만 계산하므로, 입력 일부만 넣어도 각 행의 예측은 같다.
최종 구조: P_시계열 = (LightGBM + ExtraTrees(leaf 20) + ExtraTrees(leaf 40)) / 3  (각각 일반 실험용 A, 과충전 실험용 C 분기)
          P = 0.94 * P_시계열 + 0.06 * P_이미지(EfficientNet-B0 시드 2개 평균)  ->  인과적 단조 디코딩(prefix Viterbi)"""
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
import features as FT
from model import VisionNet, first_index, load_images, past_index, seed_everything, viterbi, vision_input


def tree_proba(W, name, X, t2):
    E = joblib.load(W / name); P = E['A'].predict_proba(X[E['cols']].values.astype(np.float32))
    if t2.any(): P[t2] = E['C'].predict_proba(X.loc[X.index[t2], E['cols_c']].values.astype(np.float32))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True); ap.add_argument('--image_root', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--weights', default='weights')
    a = ap.parse_args(); seed_everything(); W = Path(a.weights); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv(a.input); cfg = json.load(open(W / 'config.json')); L = joblib.load(W / 'lgbm.joblib')
    X = FT.build(df, 'rich'); t2 = df.tr_method.values == 2                          # 과충전 실험: 압력 절대값 제외 모델(C)
    P = L['A'].predict_proba(X[L['cols']])
    if t2.any(): P[t2] = L['C'].predict_proba(X.loc[X.index[t2], L['cols_c']])
    members = [P] + [tree_proba(W, n, X, t2) for n in ('et.joblib', 'et40.joblib') if (W / n).exists()]
    P = np.mean(members, 0)                                                         # 멤버 확률 평균 (LightGBM, ET20, ET40 각 1/3)
    images = load_images(df, a.image_root); f0, p15, p60 = first_index(df), past_index(df, 15), past_index(df, 60); Pv = 0
    for s in cfg['vision_seeds']:
        net = VisionNet(pretrained=False).to(dev); net.load_state_dict(torch.load(W / f'vision_seed{s}.pt', map_location=dev)); net.eval(); out = []
        with torch.no_grad():
            for b in range(0, len(df), 256):
                r = np.arange(b, min(b + 256, len(df)))
                out.append(torch.softmax(net(vision_input(images, r, f0[r], p15[r], p60[r], dev)).float(), 1).cpu().numpy())
        Pv = Pv + np.concatenate(out) / len(cfg['vision_seeds'])
    al = np.where(t2, cfg['alpha_trm2'], cfg['alpha_trm1'])[:, None]
    P = (1 - al) * P + al * Pv
    pred = viterbi(df, P)
    sub = pd.DataFrame({'sample_id': df.sample_id.values, 'label': pred.astype(int)})
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); sub.to_csv(a.output, index=False)
    print(f'saved {a.output}  ({len(sub)}행)  분포 {sub.label.value_counts().sort_index().to_dict()}')


if __name__ == '__main__':
    main()
