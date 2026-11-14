"""joint_moe2 (센서 트랜스포머 + 이미지 + 게이트 MoE)에서 이미지 백본만 DINOv2 → timm CNN(EfficientNet)으로 교체.

바뀐 것은 이미지 인코더뿐이다. 데이터 로딩(비율 유지 이미지, 30초 전 프레임), 센서 인과 트랜스포머, 전문가 3개,
경계별 게이트, 손실, GroupKFold(5, seed 0), 500행 캡 + 확신 cummax 평가, 출력 형식은 joint_moe2.py와 같다.

이미지 인코더
  현재 프레임, 30초 전 프레임 → 같은 CNN (가중치 공유) → 마지막 특징맵 (C×h×w)
  → [현재 평균풀링, 현재 최대풀링, 현재 평균 − 과거 평균] (3C) → LayerNorm → Dropout → Linear → GELU → z_img (256)
  DINO 버전의 [CLS, 패치 평균, CLS 변화]에 대응한다.
학습 범위: --n_unfreeze = 학습할 마지막 stage 수 (efficientnet_b0는 7 stage, 7이면 stem까지 전부 학습).
  고정된 stage의 BatchNorm은 eval 모드로 둔다(통계 갱신 안 함).
학습률: 백본 head(conv_head, bn2) --lr, stage는 뒤에서부터 --llrd 배씩 감소. 센서·헤드·게이트는 --head_lr.

    python joint_moe2_cnn.py --data_dir <data> --out joint_moe2_eb0 --model efficientnet_b0 --n_unfreeze 7 --lr 3e-4
"""
import timm
import torch
import torch.nn as nn

import joint_moe as J1
import joint_moe2 as M2
from finetune_dino import K


class JointMoE2CNN(nn.Module):
    def __init__(self, name, n_unfreeze, sensor_dim, L, hid=256, drop_path=0.1):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=True, num_classes=0, global_pool='', drop_path_rate=drop_path)
        bb = self.backbone
        self.n_stages = len(bb.blocks)
        self.n_unfreeze = min(n_unfreeze, self.n_stages)
        for p in bb.parameters():
            p.requires_grad = False
        self.trainable_mods = []
        if self.n_unfreeze > 0:
            self.trainable_mods += [bb.conv_head, bb.bn2]
            self.trainable_mods += list(bb.blocks[self.n_stages - self.n_unfreeze:])
            if self.n_unfreeze == self.n_stages:
                self.trainable_mods += [bb.conv_stem, bb.bn1]
        for m in self.trainable_mods:
            for p in m.parameters():
                p.requires_grad = True
        d = bb.num_features
        self.img_proj = nn.Sequential(nn.LayerNorm(3 * d), nn.Dropout(0.1), nn.Linear(3 * d, hid), nn.GELU())
        self.sen_enc = M2.SensorDecoder(sensor_dim, L, out=hid)
        self.head_s = nn.Linear(hid, K)
        self.head_i = nn.Linear(hid, K)
        self.head_j = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, K))
        self.gate = nn.Sequential(nn.Linear(2 * hid + 3, 64), nn.GELU(), nn.Linear(64, 3 * K))

    def train(self, mode=True):
        super().train(mode)
        if mode:                                   # 고정된 부분의 BN 통계는 사전학습 값 유지
            self.backbone.eval()
            for m in self.trainable_mods:
                m.train()
        return self

    def forward(self, x6, ctx, cur):
        b = x6.shape[0]
        x = torch.cat([x6[:, :3], x6[:, 3:]], 0).contiguous(memory_format=torch.channels_last)
        f = self.backbone.forward_features(x)
        avg, mx = f.mean((2, 3)), f.amax((2, 3))
        zi = self.img_proj(torch.cat([avg[:b], mx[:b], avg[:b] - avg[b:]], 1))
        zs = self.sen_enc(ctx['Xp'], ctx['T'], ctx['H'][cur], cur)
        zz = torch.cat([zs, zi], 1)
        S, I, J = self.head_s(zs), self.head_i(zi), self.head_j(zz)
        w = self.gate(torch.cat([zz, ctx['SZ'][cur]], 1)).view(b, 3, K).softmax(1)
        return (w * torch.stack([S, I, J], 1)).sum(1), S, I, w


def param_groups(model, args):
    bb = model.backbone
    fast = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    groups = [{'params': fast, 'lr': args.head_lr}]
    if model.n_unfreeze == 0:
        return groups
    groups.append({'params': [*bb.conv_head.parameters(), *bb.bn2.parameters()], 'lr': args.lr})
    S = model.n_stages
    for i in range(S - model.n_unfreeze, S):
        groups.append({'params': list(bb.blocks[i].parameters()), 'lr': args.lr * args.llrd ** (S - i)})
    if model.n_unfreeze == S:
        groups.append({'params': [*bb.conv_stem.parameters(), *bb.bn1.parameters()], 'lr': args.lr * args.llrd ** (S + 1)})
    return groups


# joint_moe2.train_fold 가 전역 이름으로 참조하는 모델·파라미터 그룹만 교체하고 나머지는 그대로 실행
M2.JointMoE2 = JointMoE2CNN
J1.param_groups = param_groups

if __name__ == '__main__':
    M2.main()
