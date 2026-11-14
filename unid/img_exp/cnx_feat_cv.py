"""ConvNeXt(Train fold 파인튜닝) OOF 확률 → 코사인 9 + 확률 윈도우 피처 → 이미지 XGB. 0.9·센서 + 0.1·이미지 CV."""
import numpy as np, pandas as pd, os, sys
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
import final_oc as O
from final_oc import F, load, gbm, img_features, D

class Ident:
    def transform(self, E): return E

def cnx_feats(fr, emb):
    return img_features(fr, emb, Ident(), 6, use_pca=True, use_cos=True).loc[fr.sample_id.values]

def main():
    tr = load(D, 'train').reset_index(drop=True); tr['sample_id'] = tr.sample_id.astype(str)
    tc = F.cap_rows(tr).reset_index(drop=True)
    c = np.load('ft_convnext_tiny.npz', allow_pickle=True)
    emb = pd.DataFrame(c['probs'], index=c['sample_id'].astype(str)) + 1e-6
    Xf, Xc = cnx_feats(tr, emb), cnx_feats(tc, emb)
    y = tr.label.astype(int).to_numpy()
    P = np.zeros((len(tc), 6))
    for s in (0, 1, 2):
        for a, b in GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id):
            m = ~tc.experiment_id.isin(set(tr.experiment_id.iloc[a])).to_numpy()
            P[m] += gbm(s).fit(Xf.iloc[a], y[a]).predict_proba(Xc.loc[tc.sample_id[m].values]) / 3
        print('seed', s, 'done', flush=True)
    oof = np.load('final_oc_gpu/oof.npz')
    np.save('final_oc_gpu/oof_I_cnxfeat.npy', P)
    yc = tc.label.to_numpy(int); oc = (tc.tr_method == 2).to_numpy()
    def sc(Q):
        l = F.postprocess(tc, Q, 0.7)
        return (f1_score(yc, l, labels=F.LABELS, average='macro'), f1_score(yc[oc], l[oc], labels=F.LABELS, average='macro'),
                f1_score(yc, l, labels=F.LABELS, average=None).round(3).tolist())
    print('image-only CNX-feat XGB: all %.4f OC %.4f' % sc(P)[:2], f'({Xf.shape[1]} feats)')
    for nm in ('S_v3A', 'S_oc', 'S_oc_aug'):
        print(f'{nm:9s} 0.9 + 0.1*CNX-feat XGB: all %.4f OC %.4f %s' % sc(0.9 * oof[nm] + 0.1 * P))

main()
