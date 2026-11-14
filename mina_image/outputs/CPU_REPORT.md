# CPU 큐 결과 (final_lgb 대비)

```
모델별 합산 F1: {'final_lgb_oof': 0.9184, 'final_lgb_soc_w15_oo': 0.9145, 'final_lgb_soc_w2_oof': 0.9134, 'final_lgb_cap500oc_s': 0.9171, 'final_lgb_cap500oc_k': 0.9191, 'final_lgb_cap300oc_o': 0.9136, 'final_lgb_cap400oc_o': 0.9149, 'final_lgb_cap250oc_o': 0.9151, 'final_lgb_cap500oc_f': 0.9147, 'final_lgb_xgb_td_oof': 0.9203, 'final_lgb_xgb_oof': 0.9154, 'final_lgb_td_p05_oof': 0.9168, 'final_lgb_td_p15_oof': 0.9214, 'final_lgb_td_p2_oof': 0.9185, 'final_lgb_td_wide_oo': 0.9206, 'final_lgb_td_p15_wid': 0.9214, 'final_lgb_img_dino_o': 0.9092, 'final_lgb_img_dino_t': 0.9073, 'final_lgb_img_ord_td': 0.9105, 'final_lgb_oc_logit_o': 0.9202, 'final_lgb_oc_speed_o': 0.9183, 'final_lgb_oc_logit_t': 0.9223, 'final_lgb_oc_speed_t': 0.9204, 'final_lgb_cmp_eval_o': 0.9206, 'final_lgb_cmp_train_': 0.9229, 'final_lgb_dropp_oof': 0.9109, 'final_lgb_dropp_td_o': 0.9095, 'final_lgb_bym_oof': 0.9006, 'final_lgb_bym_dropp_': 0.8941, 'final_lgb_img_eff_ti': 0.905, 'final_lgb_noovr_oof': 0.9147} | 500행 상한 가중: {'final_lgb_oof': 0.9204, 'final_lgb_soc_w15_oo': 0.9182, 'final_lgb_soc_w2_oof': 0.9172, 'final_lgb_cap500oc_s': 0.9194, 'final_lgb_cap500oc_k': 0.923, 'final_lgb_cap300oc_o': 0.9172, 'final_lgb_cap400oc_o': 0.9183, 'final_lgb_cap250oc_o': 0.9184, 'final_lgb_cap500oc_f': 0.9186, 'final_lgb_xgb_td_oof': 0.9224, 'final_lgb_xgb_oof': 0.9192, 'final_lgb_td_p05_oof': 0.9193, 'final_lgb_td_p15_oof': 0.9247, 'final_lgb_td_p2_oof': 0.9221, 'final_lgb_td_wide_oo': 0.9225, 'final_lgb_td_p15_wid': 0.9247, 'final_lgb_img_dino_o': 0.9157, 'final_lgb_img_dino_t': 0.9144, 'final_lgb_img_ord_td': 0.9165, 'final_lgb_oc_logit_o': 0.9219, 'final_lgb_oc_speed_o': 0.9192, 'final_lgb_oc_logit_t': 0.9239, 'final_lgb_oc_speed_t': 0.9213, 'final_lgb_cmp_eval_o': 0.9225, 'final_lgb_cmp_train_': 0.9247, 'final_lgb_dropp_oof': 0.9149, 'final_lgb_dropp_td_o': 0.9142, 'final_lgb_bym_oof': 0.9084, 'final_lgb_bym_dropp_': 0.9053, 'final_lgb_img_eff_ti': 0.9121, 'final_lgb_noovr_oof': 0.9173}
                                     bc  tr  soc     cc    n  fold  acc_final_lgb_oof
실험 단위 부트스트랩(500행 가중 F1): 평균 ± 표준편차(5~95%)
  final_lgb_oof              0.9193 ± 0.0084 (0.905~0.932)
  final_lgb_soc_w15_oof      0.9172 ± 0.0083 (0.903~0.930)
  final_lgb_soc_w2_oof       0.9162 ± 0.0087 (0.901~0.930)
  final_lgb_cap500oc_soc_w15 0.9184 ± 0.0084 (0.905~0.931)
  final_lgb_cap500oc_k1_oof  0.9158 ± 0.0092 (0.900~0.931)
  final_lgb_cap500oc_k4_oof  0.9223 ± 0.0077 (0.909~0.935)
  final_lgb_cap300oc_oof     0.9161 ± 0.0086 (0.902~0.930)
  final_lgb_cap400oc_oof     0.9172 ± 0.0086 (0.903~0.931)
  final_lgb_cap250oc_oof     0.9174 ± 0.0088 (0.903~0.931)
  final_lgb_cap500oc_full_oo 0.9177 ± 0.0086 (0.903~0.931)
  final_lgb_xgb_td_oof       0.9211 ± 0.0080 (0.907~0.933)
  final_lgb_xgb_oof          0.9181 ± 0.0084 (0.904~0.931)
  final_lgb_td_p05_oof       0.9182 ± 0.0085 (0.904~0.931)
  final_lgb_td_p15_oof       0.9237 ± 0.0072 (0.911~0.935)
  final_lgb_td_p2_oof        0.9209 ± 0.0079 (0.907~0.933)
  final_lgb_td_wide_oof      0.9213 ± 0.0080 (0.908~0.934)
  final_lgb_td_p15_wide_oof  0.9237 ± 0.0072 (0.911~0.935)
  final_lgb_img_dino_oof     0.9149 ± 0.0088 (0.900~0.929)
  final_lgb_img_dino_td_oof  0.9147 ± 0.0085 (0.900~0.928)
  final_lgb_img_ord_td_oof   0.9156 ± 0.0095 (0.899~0.930)
  final_lgb_oc_logit_oof     0.9207 ± 0.0085 (0.906~0.934)
  final_lgb_oc_speed_oof     0.9183 ± 0.0088 (0.903~0.933)
  final_lgb_oc_logit_td_oof  0.9228 ± 0.0081 (0.909~0.936)
  final_lgb_oc_speed_td_oof  0.9203 ± 0.0084 (0.907~0.933)
  final_lgb_cmp_eval_only_oo 0.9193 ± 0.0084 (0.905~0.932)
  final_lgb_cmp_train_k2_oof 0.9235 ± 0.0077 (0.910~0.936)
  final_lgb_cmp_train_k4_oof 0.9223 ± 0.0077 (0.909~0.935)
  final_lgb_cmp_eval_only_td 0.9213 ± 0.0080 (0.908~0.934)
  final_lgb_cmp_train_k2_td_ 0.9237 ± 0.0076 (0.910~0.936)
  final_lgb_img_dino_tidx_td 0.9131 ± 0.0088 (0.898~0.927)
  final_lgb_dropp_oof        0.9139 ± 0.0096 (0.898~0.929)
  final_lgb_dropp_td_oof     0.9131 ± 0.0093 (0.898~0.927)
  final_lgb_bym_oof          0.9084 ± 0.0138 (0.884~0.929)
  final_lgb_bym_dropp_oof    0.9014 ± 0.0144 (0.877~0.924)
  final_lgb_bym_dropp_td_oof 0.9054 ± 0.0142 (0.880~0.927)
  final_lgb_img_eff_tidx_td_ 0.9110 ± 0.0103 (0.893~0.926)
  final_lgb_noovr_oof        0.9161 ± 0.0090 (0.901~0.930)
  final_lgb_soc_w15_oof      - final_lgb_oo: -0.0021  [-0.0056, +0.0009] 개선 비율 0.11
  final_lgb_soc_w2_oof       - final_lgb_oo: -0.0030  [-0.0067, -0.0001] 개선 비율 0.03
  final_lgb_cap500oc_soc_w15 - final_lgb_oo: -0.0009  [-0.0031, +0.0013] 개선 비율 0.24
  final_lgb_cap500oc_k1_oof  - final_lgb_oo: -0.0035  [-0.0095, +0.0006] 개선 비율 0.09
  final_lgb_cap500oc_k4_oof  - final_lgb_oo: +0.0030  [-0.0032, +0.0126] 개선 비율 0.74
  final_lgb_cap300oc_oof     - final_lgb_oo: -0.0032  [-0.0068, -0.0005] 개선 비율 0.01
  final_lgb_cap400oc_oof     - final_lgb_oo: -0.0021  [-0.0055, +0.0009] 개선 비율 0.09
  final_lgb_cap250oc_oof     - final_lgb_oo: -0.0019  [-0.0058, +0.0017] 개선 비율 0.13
  final_lgb_cap500oc_full_oo - final_lgb_oo: -0.0016  [-0.0056, +0.0015] 개선 비율 0.20
  final_lgb_xgb_td_oof       - final_lgb_oo: +0.0019  [-0.0022, +0.0064] 개선 비율 0.79
  final_lgb_xgb_oof          - final_lgb_oo: -0.0012  [-0.0047, +0.0020] 개선 비율 0.28
  final_lgb_td_p05_oof       - final_lgb_oo: -0.0011  [-0.0043, +0.0017] 개선 비율 0.22
  final_lgb_td_p15_oof       - final_lgb_oo: +0.0044  [-0.0031, +0.0142] 개선 비율 0.83
  final_lgb_td_p2_oof        - final_lgb_oo: +0.0016  [-0.0039, +0.0066] 개선 비율 0.76
  final_lgb_td_wide_oof      - final_lgb_oo: +0.0020  [-0.0019, +0.0066] 개선 비율 0.84
  final_lgb_td_p15_wide_oof  - final_lgb_oo: +0.0044  [-0.0031, +0.0142] 개선 비율 0.83
  final_lgb_img_dino_oof     - final_lgb_oo: -0.0044  [-0.0133, +0.0028] 개선 비율 0.15
  final_lgb_img_dino_td_oof  - final_lgb_oo: -0.0046  [-0.0136, +0.0030] 개선 비율 0.14
  final_lgb_img_ord_td_oof   - final_lgb_oo: -0.0037  [-0.0121, +0.0033] 개선 비율 0.19
  final_lgb_oc_logit_oof     - final_lgb_oo: +0.0015  [-0.0016, +0.0042] 개선 비율 0.82
  final_lgb_oc_speed_oof     - final_lgb_oo: -0.0010  [-0.0074, +0.0041] 개선 비율 0.37
  final_lgb_oc_logit_td_oof  - final_lgb_oo: +0.0035  [-0.0011, +0.0080] 개선 비율 0.94
  final_lgb_oc_speed_td_oof  - final_lgb_oo: +0.0011  [-0.0059, +0.0078] 개선 비율 0.64
  final_lgb_cmp_eval_only_oo - final_lgb_oo: +0.0000  [+0.0000, +0.0000] 개선 비율 0.00
  final_lgb_cmp_train_k2_oof - final_lgb_oo: +0.0042  [-0.0014, +0.0135] 개선 비율 0.89
  final_lgb_cmp_train_k4_oof - final_lgb_oo: +0.0030  [-0.0032, +0.0126] 개선 비율 0.74
  final_lgb_cmp_eval_only_td - final_lgb_oo: +0.0020  [-0.0019, +0.0066] 개선 비율 0.84
  final_lgb_cmp_train_k2_td_ - final_lgb_oo: +0.0044  [-0.0021, +0.0139] 개선 비율 0.86
  final_lgb_img_dino_tidx_td - final_lgb_oo: -0.0061  [-0.0150, +0.0030] 개선 비율 0.10
  final_lgb_dropp_oof        - final_lgb_oo: -0.0054  [-0.0117, -0.0004] 개선 비율 0.01
  final_lgb_dropp_td_oof     - final_lgb_oo: -0.0062  [-0.0131, -0.0003] 개선 비율 0.02
  final_lgb_bym_oof          - final_lgb_oo: -0.0109  [-0.0365, +0.0089] 개선 비율 0.18
  final_lgb_bym_dropp_oof    - final_lgb_oo: -0.0179  [-0.0426, -0.0006] 개선 비율 0.01
  final_lgb_bym_dropp_td_oof - final_lgb_oo: -0.0138  [-0.0404, +0.0056] 개선 비율 0.10
  final_lgb_img_eff_tidx_td_ - final_lgb_oo: -0.0082  [-0.0196, +0.0013] 개선 비율 0.04
  final_lgb_noovr_oof        - final_lgb_oo: -0.0032  [-0.0076, +0.0016] 개선 비율 0.08
final_lgb_oof: 실제 fold 5개 F1 표준편차 0.0157 | 무작위 19개 실험 부분집합 표준편차 0.0196 | 무작위 11개(500행 상한) 표준편차 0.0270
```

실행 이력: outputs/queue/cpu_done.txt
