"""fusion_cv.py / make_submission.py 공용: 이미지 신호 컬럼 만들기.

변형(variant) 정의 — 모두 시계열 팀 features.py 의 SIGNALS 에 '추가 신호'로 들어가 같은 인과적 feature(누적/시간 창/임계)가 계산된다(신호당 약 35개).
  ts     : 추가 신호 없음(시계열만)
  light  : img_exp(이미지 기대 단계), img_p456(P(단계>=4))                 -> 이미지를 '살짝만'
  probs6 : img_p1..img_p6 + img_exp                                        -> 이미지 확률 전체(후기 결합)
  emb    : 프리즈 임베딩(features/<enc>/<layer>.npy)의 PCA 성분 emb_0..emb_{k-1} -> early fusion. test 행도 캐시에 있어 파인튜닝 test 예측이 필요 없다
  blend  : (feature 가 아니라) 확률 가중 평균 P=(1-w)P_ts + w P_img  -> fusion_cv 의 가중치 스윕, make_submission --blend_w
PCA 는 라벨 있는 행(Train+Val)에서만 적합한다(Test 는 변환만). CV 에서는 검증 fold 의 임베딩도 PCA 적합에 들어간다(라벨은 쓰지 않는 비지도 변환이라 누수는 작지만 0은 아님).
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import ROOT, table

L = list(range(1, 7))
IMG_VARIANTS = {"light": ["img_exp", "img_p456"], "probs6": [f"img_p{k}" for k in L] + ["img_exp"]}


def add_image_cols(df, probs_name, models):
    """df(sample_id 있음)에 img_p1..6, img_exp, img_p456 을 붙인다. 라벨 행=OOF, test 행=fold 평균(없으면 NaN)."""
    img = pd.read_parquet(ROOT / "outputs" / "preds" / f"{probs_name}.parquet")
    img["sample_id"] = img.sample_id.astype(str)
    for k in L:
        img[f"img_p{k}"] = np.mean([img[f"{m}_p{k}"].values for m in models], axis=0)
    P = img[[f"img_p{k}" for k in L]].values
    img["img_exp"], img["img_p456"] = P @ np.array(L), P[:, 3:].sum(1)
    cols = ["sample_id"] + [f"img_p{k}" for k in L] + ["img_exp", "img_p456"]
    return df.merge(img[cols], on="sample_id", how="left")


def emb_columns(sample_ids, fit_mask, spec, k, seed=0):
    """spec='enc:layer'. 표준화·PCA 는 fit_mask 행으로만 적합. 반환: DataFrame(emb_0..emb_{k-1}), 입력 순서."""
    enc, layer = spec.split(":")
    t = table()
    E = np.load(ROOT / "features" / enc / f"{layer}.npy").astype(np.float32)
    pos = pd.Series(np.arange(len(t)), index=t.sample_id.astype(str).values)
    X = E[pos.loc[np.asarray(sample_ids, dtype=str)].values]
    fit = np.where(fit_mask)[0]
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
    rng = np.random.default_rng(seed)
    sub = rng.choice(fit, min(20000, len(fit)), replace=False)
    pca = PCA(k, svd_solver="randomized", random_state=seed).fit((X[sub] - mu) / sd)
    Z = pca.transform((X - mu) / sd).astype(np.float32)
    print(f"임베딩 {spec}: {X.shape[1]}차원 -> PCA {k}개, 설명 분산 {pca.explained_variance_ratio_.sum():.3f}", flush=True)
    return pd.DataFrame(Z, columns=[f"emb_{i}" for i in range(k)])
