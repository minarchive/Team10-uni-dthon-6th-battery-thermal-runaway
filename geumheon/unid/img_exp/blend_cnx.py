"""센서 모델(final_oc) + ConvNeXt-Tiny 파인튜닝 이미지 확률 0.1 결합: CV/Val 확인 + Test 제출.

- CV : final_oc_gpu/oof.npz(센서 OOF, Train 캡 행) + ft_convnext_tiny.npz(같은 GroupKFold(5, seed 0) OOF)
- Val: 센서 모델을 Train만으로 다시 학습(3시드) + ft_convnext_tiny.npz Val 확률(Train fold 모델 5개 평균)
- Test: final_oc_gpu/test_probs_<S>.npy(Train+Val 5시드) + ft_convnext_tiny_trval.npz(Train+Val 파인튜닝)
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import final_oc as O
from final_oc import F, load, gbm, sensor_feats, make_aug, D

W = 0.1
OUT = O.M.HERE / 'final_oc_gpu'
LAB = F.LABELS


def sc(fr, P):
    lab = F.postprocess(fr, P, 0.7)
    y = fr.label.astype(int).to_numpy(); oc = (fr.tr_method == 2).to_numpy()
    f = lambda m: float(f1_score(y[m], lab[m], labels=LAB, average='macro'))
    return dict(all=f(np.ones(len(y), bool)), overcharge=f(oc),
                per_class=f1_score(y, lab, labels=LAB, average=None).round(4).tolist())


def cnn_probs(path, ids):
    c = np.load(path, allow_pickle=True)
    return pd.DataFrame(c['probs'], index=c['sample_id'].astype(str)).loc[ids].to_numpy()


def main():
    tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
    for f in (tr, va, te):
        f['sample_id'] = f.sample_id.astype(str)
    tr_cap, va_cap = F.cap_rows(tr).reset_index(drop=True), F.cap_rows(va).reset_index(drop=True)
    res = {}
    # CV
    oof = np.load(OUT / 'oof.npz')
    c_tr = cnn_probs('ft_convnext_tiny.npz', tr_cap.sample_id.values)
    for nm in ('S_v3A', 'S_oc', 'S_oc_aug'):
        res[nm] = dict(cv=sc(tr_cap, (1 - W) * oof[nm] + W * c_tr))
    # Val
    aug_tr = make_aug(tr, 0)
    Str, Sva, Sag = sensor_feats(tr), sensor_feats(va_cap), sensor_feats(aug_tr)
    ytr = tr.label.astype(int).to_numpy()
    no_oc = [c for c in Str.columns if not c.startswith('oc_')]
    c_va = cnn_probs('ft_convnext_tiny.npz', va_cap.sample_id.values)
    for nm, cols, aug in (('S_v3A', no_oc, False), ('S_oc', list(Str.columns), False), ('S_oc_aug', list(Str.columns), True)):
        X, y = (pd.concat([Str, Sag]), np.r_[ytr, aug_tr.label.astype(int).to_numpy()]) if aug else (Str, ytr)
        pS = sum(gbm(s).fit(X[cols], y).predict_proba(Sva.loc[va_cap.sample_id.values, cols]) for s in (0, 1, 2)) / 3
        res[nm]['val'] = sc(va_cap, (1 - W) * pS + W * c_va)
        O.log(f'{nm:9s} +ConvNeXt {W}: CV {res[nm]["cv"]["all"]:.4f} (OC {res[nm]["cv"]["overcharge"]:.4f}) | '
              f'Val {res[nm]["val"]["all"]:.4f} (OC {res[nm]["val"]["overcharge"]:.4f}) {res[nm]["val"]["per_class"]}')
    (OUT / 'results_convnext.json').write_text(json.dumps(res, indent=1))
    # Test
    c_te = cnn_probs('ft_convnext_tiny_trval.npz', te.sample_id.values)
    ss = pd.read_csv(D / 'sample_submission.csv', dtype={'sample_id': str})
    for nm in ('S_v3A', 'S_oc_aug'):
        pS = np.load(OUT / f'test_probs_{nm}.npy')
        lab = F.postprocess(te, (1 - W) * pS + W * c_te, 0.7)
        sub = ss[['sample_id']].merge(pd.DataFrame({'sample_id': te.sample_id.values, 'label': lab}), on='sample_id', how='left')
        assert sub.label.notna().all() and len(sub) == len(ss)
        p = OUT / f'submission_{nm}_convnext{W}.csv'
        sub.to_csv(p, index=False)
        O.log(f'wrote {p.name} dist {sub.label.value_counts().sort_index().to_dict()}')
    O.log('ALL DONE')


if __name__ == '__main__':
    main()
