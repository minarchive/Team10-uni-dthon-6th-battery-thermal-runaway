import sys, time; sys.path.insert(0, '/home/nxcolelxu/unid/img_exp')
import numpy as np
import exp_cluster_moe as C
C.BK = C.XGBBackend()
tr = C.load(C.D, 'train').reset_index(drop=True)
ex = sorted(tr.experiment_id.unique())[:30]; tr = tr[tr.experiment_id.isin(ex)].reset_index(drop=True)
X = C.F.build_features(tr).to_numpy(np.float32); y = tr.label.to_numpy(int) - 1; e = tr.experiment_id.astype(str).to_numpy()
a = np.isin(e, ex[:24]); b = ~a
t = time.time(); ctx = C.FoldCtx(X[a], y[a], e[a], 0); print('ctx', round(time.time() - t), flush=True)
p0 = C.BK.predict(ctx.base, X[b][:50]); nt = ctx.base.num_boosted_rounds()
g, K, h = C.em_groups(ctx, C.profiles(tr), 2); print('K', K, h, round(time.time() - t), flush=True)
exs, gate = C.fit_moe(ctx, g, K)
print('base unchanged', ctx.base.num_boosted_rounds() == nt, np.allclose(p0, C.BK.predict(ctx.base, X[b][:50])), 'expert rounds', exs[0].num_boosted_rounds())
Pm, Pk, G = C.predict_moe(exs, gate, X[b]); print(Pm.shape, Pm.sum(1)[:3].round(4), G[:2].round(3), round(time.time() - t))
