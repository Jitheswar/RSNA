"""Multi-series 2.5D knee MRI model: shared timm encoder -> slice tokens -> study transformer -> per-target queries."""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

N_TYPES = 12


class GeM(nn.Module):
    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))

    def forward(self, x):  # [N,C,H,W]
        p = self.p.clamp(1.0, 6.0)
        return x.float().clamp(min=1e-6).pow(p).mean((-2, -1)).pow(1.0 / p)


class KneeNet(nn.Module):
    def __init__(self, backbone: str, n_targets=12, d=384, n_layers=2, n_heads=8, dropout=0.1,
                 pretrained=True, drop_path=0.1, n_aux=0, grad_ckpt=False, head="tf", img=None):
        super().__init__()
        self.head_type = head
        kw = {"drop_path_rate": drop_path} if drop_path else {}
        self.is_vit = any(k in backbone for k in ("vit_", "eva02", "deit"))
        if self.is_vit and img:
            kw["img_size"] = img  # pretrained position embeddings are resampled to this grid at load time
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="", **kw)
        if grad_ckpt:
            self.enc.set_grad_checkpointing(True)
        C = self.enc.num_features * (2 if self.is_vit else 1)  # ViT readout: [CLS, mean patch token]
        self.pool = GeM()
        self.proj = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, d))
        self.type_emb = nn.Embedding(N_TYPES, d)
        self.pos = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.tf = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.queries = nn.Parameter(torch.randn(n_targets, d) * 0.02)
        self.xattn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.qnorm = nn.LayerNorm(d)
        if head == "pool":  # simple: per-plane masked mean+max of encoder features -> MLP (no attention)
            Cf = C
            self.pool_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(6 * Cf, 512), nn.GELU(), nn.Dropout(0.2),
                                           nn.Linear(512, n_targets))
        if head == "mil":  # per-finding gated attention-MIL over window tokens (no transformer)
            self.mil_att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, n_targets))
        self.head_w = nn.Parameter(torch.randn(n_targets, d) * 0.02)
        self.head_b = nn.Parameter(torch.zeros(n_targets))
        self.n_aux = n_aux
        if n_aux:
            self.aux_head = nn.Linear(d, n_aux)
        cfg = self.enc.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1) * 255.0)

    def encode(self, imgs):  # imgs float [N,3,H,W] in 0..255
        f = self.enc.forward_features((imgs - self.mean) / self.std)
        if f.ndim == 3:  # ViT-style tokens [N,L,C]
            if self.is_vit:
                return torch.cat([f[:, 0], f[:, self.enc.num_prefix_tokens:].mean(1)], -1).float()
            return f.mean(1)
        return self.pool(f)

    def forward(self, x, types, pos, present, micro=0, smask=None):
        """x: uint8/float [B,S,K,3,H,W]; types [B,S]; pos [B,S,K,2]; present [B,S]; smask [B,S,K] (slice valid)."""
        B, S, K = x.shape[:3]
        valid = present > 0.5  # [B,S]
        vimg = valid.unsqueeze(2).expand(B, S, K)
        if smask is not None:
            vimg = vimg & (smask > 0.5)
        dt = self.mean.dtype  # float32 in training (bf16 autocast); float16 after net.half() for pure-fp16 inference
        imgs = x[vimg].to(dt)  # [N,3,H,W]
        n_img = imgs.shape[0]
        if micro:
            # fixed-size chunks (last one zero-padded) keep conv input shapes constant -> cuDNN autotunes once
            pad_n = (-n_img) % micro
            if pad_n:
                imgs = torch.cat([imgs, imgs.new_zeros((pad_n,) + imgs.shape[1:])])
            feats = torch.cat([self.encode(imgs[i:i + micro]) for i in range(0, imgs.shape[0], micro)])[:n_img]
        else:
            feats = self.encode(imgs)
        feats = feats.to(dt)
        if self.head_type == "pool":
            Cf = feats.shape[-1]
            full = feats.new_zeros(B, S, K, Cf)
            full[vimg] = feats
            plane = torch.div(types, 4, rounding_mode="floor")  # 0 sag, 1 cor, 2 ax
            parts = []
            for pl in range(3):
                msk = (vimg & (plane == pl).unsqueeze(2)).float().unsqueeze(-1)  # [B,S,K,1]
                cnt = msk.sum((1, 2)).clamp(min=1.0)
                mean = (full * msk).sum((1, 2)) / cnt
                mx = (full * msk + (msk - 1) * 1e4).amax((1, 2))
                mx = torch.where(msk.sum((1, 2)) > 0, mx, torch.zeros_like(mx))
                parts += [mean, mx]
            return self.pool_head(torch.cat(parts, -1)), None
        tok_v = self.proj(feats)  # [N,d]
        d = tok_v.shape[-1]
        tok = tok_v.new_zeros(B, S, K, d)
        tok[vimg] = tok_v
        tok = tok + self.type_emb(types).unsqueeze(2) + self.pos(pos)
        tok = tok.view(B, S * K, d)
        pad = (~vimg).reshape(B, S * K)
        allpad = pad.all(1)
        if allpad.any():  # never let a study attend to nothing
            pad = pad.clone()
            pad[allpad, 0] = False
        if self.head_type == "mil":
            h = self.norm(tok)
            a = self.mil_att(h).float().masked_fill(pad.unsqueeze(-1), float("-inf")).softmax(1)  # [B,T,12]
            ctx = torch.einsum("btn,btd->bnd", a, h.float())
            logits = (ctx * self.head_w).sum(-1) + self.head_b
        else:
            h = self.norm(self.tf(tok, src_key_padding_mask=pad))
            q = self.queries.unsqueeze(0).expand(B, -1, -1)
            ctx, _ = self.xattn(q, h, h, key_padding_mask=pad)
            ctx = self.qnorm(ctx + q)
            logits = (ctx * self.head_w).sum(-1) + self.head_b
        if self.n_aux:
            m = (~pad).to(h.dtype).unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            return logits, self.aux_head(pooled)
        return logits, None


@torch.no_grad()
def gpu_augment(x, present, gen=None, rot=10.0, scale=(0.9, 1.1), shift=0.06, crop=None, zoom=1.0):
    """Per-series consistent random affine + intensity jitter on GPU. x uint8 [B,S,K,3,H,W] -> float."""
    B, S, K, C, H, W = x.shape
    dev = x.device
    xf = x.float()
    n = B * S
    ang = (torch.rand(n, device=dev, generator=gen) * 2 - 1) * math.radians(rot)
    sc = torch.empty(n, device=dev).uniform_(scale[0], scale[1], generator=gen)
    tx = (torch.rand(n, device=dev, generator=gen) * 2 - 1) * shift
    ty = (torch.rand(n, device=dev, generator=gen) * 2 - 1) * shift
    sc = sc * zoom  # zoom > 1 samples a central sub-window (higher effective resolution)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)  # [n,2,3]
    theta = theta.repeat_interleave(K, 0)
    imgs = xf.view(n * K, C, H, W)
    out_hw = (crop, crop) if crop else (H, W)
    grid = F.affine_grid(theta, (n * K, C) + out_hw, align_corners=False)
    imgs = F.grid_sample(imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # intensity: per-series gamma, contrast, brightness
    g = torch.exp(torch.empty(n, device=dev).uniform_(-0.25, 0.25, generator=gen)).repeat_interleave(K).view(-1, 1, 1, 1)
    c = torch.empty(n, device=dev).uniform_(0.85, 1.15, generator=gen).repeat_interleave(K).view(-1, 1, 1, 1)
    b = torch.empty(n, device=dev).uniform_(-12, 12, generator=gen).repeat_interleave(K).view(-1, 1, 1, 1)
    imgs = (255.0 * (imgs / 255.0).clamp(0, 1) ** g) * c + b
    imgs = imgs.clamp(0, 255)
    return imgs.view(B, S, K, C, *out_hw)


def center_zoom(x, zoom, out):
    """Deterministic central zoom + resize for inference/validation. x float/uint8 [B,S,K,C,H,W] -> float."""
    B, S, K, C, H, W = x.shape
    th = torch.zeros(B * S * K, 2, 3, device=x.device)
    th[:, 0, 0] = 1.0 / zoom
    th[:, 1, 1] = 1.0 / zoom
    grid = F.affine_grid(th, (B * S * K, C, out, out), align_corners=False)
    y = F.grid_sample(x.reshape(B * S * K, C, H, W).float(), grid, mode="bilinear", padding_mode="zeros",
                      align_corners=False)
    return y.view(B, S, K, C, out, out)
