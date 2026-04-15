"""
Adaptive Flow Matching for climate super-resolution.
Fotiadis et al., ICML 2025 (arXiv:2410.19814).

Encoder (DualEncoderUNet) produces deterministic μ.
Flow UNet learns velocity v_θ(x_t, t, condition) along OT paths.
Adaptive σ_z via EMA of encoder RMSE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import DualEncoderUNet


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        t = t.view(-1).float()
        half = self.dim // 2
        emb = math.log(10000.0) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device).float() * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimeCondDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)

    def forward(self, x, t_emb):
        h = F.relu(self.bn1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        return F.relu(self.bn2(self.conv2(h)))


class TimeCondUNet(nn.Module):
    """UNet with time conditioning for flow velocity prediction."""

    def __init__(self, in_ch=12, out_ch=1, base=64, time_dim=256):
        super().__init__()
        f = base
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim),
            nn.GELU(), nn.Linear(time_dim, time_dim))

        self.enc1 = TimeCondDoubleConv(in_ch, f, time_dim)
        self.enc2 = TimeCondDoubleConv(f, f*2, time_dim)
        self.enc3 = TimeCondDoubleConv(f*2, f*4, time_dim)
        self.enc4 = TimeCondDoubleConv(f*4, f*8, time_dim)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = TimeCondDoubleConv(f*8, f*16, time_dim)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec4 = TimeCondDoubleConv(f*16 + f*8, f*8, time_dim)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = TimeCondDoubleConv(f*8 + f*4, f*4, time_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = TimeCondDoubleConv(f*4 + f*2, f*2, time_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = TimeCondDoubleConv(f*2 + f, f, time_dim)
        self.final = nn.Conv2d(f, out_ch, 1)

    def _match(self, x, ref):
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, ref.shape[2:], mode="bilinear", align_corners=True)
        return x

    def forward(self, x_t, t, x_dyn, x_stat):
        x = torch.cat([x_t, x_dyn, x_stat], dim=1)
        te = self.time_mlp(t)
        e1 = self.enc1(x, te)
        e2 = self.enc2(self.pool(e1), te)
        e3 = self.enc3(self.pool(e2), te)
        e4 = self.enc4(self.pool(e3), te)
        bn = self.bottleneck(self.pool(e4), te)
        d4 = self.dec4(torch.cat([self._match(self.up4(bn), e4), e4], 1), te)
        d3 = self.dec3(torch.cat([self._match(self.up3(d4), e3), e3], 1), te)
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], 1), te)
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], 1), te)
        return self.final(d1)


class AFMModel(nn.Module):
    def __init__(self, dynamic_ch=9, static_ch=2, out_ch=1, base=64,
                 time_dim=256, sigma_z_init=1.0, sigma_z_ema=0.999,
                 encoder_loss_weight=0.1):
        super().__init__()
        self.encoder = DualEncoderUNet(dynamic_ch, static_ch, out_ch, base)
        self.flow_net = TimeCondUNet(out_ch + dynamic_ch + static_ch, out_ch, base, time_dim)
        self.register_buffer("sigma_z", torch.tensor(sigma_z_init))
        self.sigma_z_ema = sigma_z_ema
        self.encoder_loss_weight = encoder_loss_weight

    @torch.no_grad()
    def _update_sigma(self, mu, target):
        new = (mu - target).pow(2).mean().sqrt().clamp(min=1e-4)
        self.sigma_z.mul_(self.sigma_z_ema).add_(new * (1 - self.sigma_z_ema))

    def forward(self, x_dyn, x_stat, x_target=None, extract_features=False):
        if x_target is None:
            return self.encoder(x_dyn, x_stat, extract_features=extract_features)

        if extract_features:
            mu, features = self.encoder(x_dyn, x_stat, extract_features=True)
        else:
            mu = self.encoder(x_dyn, x_stat)

        self._update_sigma(mu, x_target)
        B = mu.shape[0]
        z = mu + self.sigma_z * torch.randn_like(mu)
        t = torch.rand(B, device=mu.device)
        te = t[:, None, None, None]
        x_t = (1 - te) * z + te * x_target
        u = x_target - z

        v = self.flow_net(x_t, t, x_dyn, x_stat)
        flow_loss = F.mse_loss(v, u)
        enc_loss = F.mse_loss(mu, x_target)
        total = flow_loss + self.encoder_loss_weight * enc_loss

        result = {"flow_loss": flow_loss, "encoder_loss": enc_loss,
                  "total_loss": total, "sigma_z": self.sigma_z.item()}
        if extract_features:
            result["features"] = features
        return result

    @torch.no_grad()
    def sample(self, x_dyn, x_stat, n_samples=1, steps=20, method="midpoint"):
        self.eval()
        mu = self.encoder(x_dyn, x_stat)
        all_s = []
        for _ in range(n_samples):
            x = mu + self.sigma_z * torch.randn_like(mu)
            dt = 1.0 / steps
            for i in range(steps):
                tv = i * dt
                t = torch.full((mu.shape[0],), tv, device=mu.device)
                if method == "euler":
                    x = x + self.flow_net(x, t, x_dyn, x_stat) * dt
                else:  # midpoint
                    v1 = self.flow_net(x, t, x_dyn, x_stat)
                    tm = torch.full_like(t, tv + dt / 2)
                    v2 = self.flow_net(x + v1 * dt / 2, tm, x_dyn, x_stat)
                    x = x + v2 * dt
            all_s.append(x)
        return torch.stack(all_s, dim=1)

    @torch.no_grad()
    def deterministic_predict(self, x_dyn, x_stat):
        self.eval()
        return self.encoder(x_dyn, x_stat)
