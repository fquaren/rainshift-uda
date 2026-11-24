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
    """mode="afm" (default, every run to date): Adaptive Flow Matching. The
    encoder's output is mu, the flow starts from z = mu + sigma_z * eps, and
    the loss is flow + encoder_loss_weight * MSE(mu, y).

    mode="cfm" (PLAN_COSMO.md C6, the third rung of the model ladder): plain
    conditional flow matching with the SAME encoder and flow network. The
    encoder's output is a cond_ch-channel conditioning field, concatenated to
    the flow net's inputs; the flow starts from N(0, I); the loss is the flow
    loss alone, so the encoder learns only what the flow needs. There is no
    mu, no encoder MSE and no sigma_z (the buffer stays at 1 and is unused).
    The encoder's bottleneck is the same place in both modes, so every
    alignment loss acts on the same features on all three rungs (UNet, AFM,
    CFM)."""

    def __init__(self, dynamic_ch=9, static_ch=2, out_ch=1, base=64,
                 time_dim=256, sigma_z_init=1.0, sigma_z_ema=0.999,
                 encoder_loss_weight=0.1, mode="afm", cond_ch=8,
                 det_members=4, det_steps=10):
        super().__init__()
        if mode not in ("afm", "cfm"):
            raise ValueError(f"mode {mode!r} not in ('afm', 'cfm')")
        self.mode = mode
        self.out_ch = out_ch
        self.cond_ch = cond_ch if mode == "cfm" else 0
        enc_out = out_ch if mode == "afm" else cond_ch
        self.encoder = DualEncoderUNet(dynamic_ch, static_ch, enc_out, base)
        self.flow_net = TimeCondUNet(out_ch + dynamic_ch + static_ch + self.cond_ch, out_ch, base, time_dim)
        self.register_buffer("sigma_z", torch.tensor(sigma_z_init))
        self.sigma_z_ema = sigma_z_ema
        self.encoder_loss_weight = encoder_loss_weight
        # CFM point forecast: the mean of det_members samples of det_steps
        # midpoint steps, with fixed noise (see deterministic_predict).
        self.det_members = det_members
        self.det_steps = det_steps

    @torch.no_grad()
    def _update_sigma(self, mu, target):
        new = (mu - target).pow(2).mean().sqrt().clamp(min=1e-4)
        self.sigma_z.mul_(self.sigma_z_ema).add_(new * (1 - self.sigma_z_ema))

    def _flow_dyn(self, x_dyn, enc_out):
        """Dynamic inputs as the flow net sees them: raw for AFM, raw plus the
        encoder's conditioning field for CFM."""
        return x_dyn if self.mode == "afm" else torch.cat([x_dyn, enc_out.to(x_dyn.dtype)], dim=1)

    def forward(self, x_dyn, x_stat, x_target=None, extract_features=False):
        # x_dyn arrives at native input resolution (80x80) when the dataset
        # yields raw inputs (upsample_on_gpu=True). Bring it to the target grid
        # (200x200) here, on GPU, before the encoder. Statics are already 200x200.
        if x_dyn.shape[-1] != x_stat.shape[-1]:
            x_dyn = F.interpolate(
                x_dyn, size=x_stat.shape[-2:],
                mode="bicubic", align_corners=False,
            )

        # Encoder only (UDA target passes): AFM returns mu, CFM the
        # conditioning field; with extract_features, also the feature dict.
        if x_target is None:
            return self.encoder(x_dyn, x_stat, extract_features=extract_features)

        if extract_features:
            enc, features = self.encoder(x_dyn, x_stat, extract_features=True)
        else:
            enc = self.encoder(x_dyn, x_stat)

        B = x_target.shape[0]
        if self.mode == "afm":
            mu = enc
            self._update_sigma(mu, x_target)
            z = mu + self.sigma_z * torch.randn_like(mu)
        else:
            z = torch.randn_like(x_target)
        t = torch.rand(B, device=x_target.device)
        te = t[:, None, None, None]
        x_t = (1 - te) * z + te * x_target
        u = x_target - z

        v = self.flow_net(x_t, t, self._flow_dyn(x_dyn, enc), x_stat)
        flow_loss = F.mse_loss(v, u)
        if self.mode == "afm":
            enc_loss = F.mse_loss(mu, x_target)
            total = flow_loss + self.encoder_loss_weight * enc_loss
        else:
            enc_loss = torch.zeros((), device=x_target.device)
            total = flow_loss

        result = {"flow_loss": flow_loss, "encoder_loss": enc_loss,
                  "total_loss": total, "sigma_z": self.sigma_z.item()}
        if self.mode == "afm":
            result["mu"] = mu  # exposed so output-level losses reuse it
        if extract_features:
            result["features"] = features
        return result

    def _match_dyn(self, x_dyn, x_stat):
        """Bring x_dyn to the static/target grid. The encoder does this
        internally, but flow_net concatenates x_dyn with x_t and x_stat, so it
        must be upsampled here too or torch.cat fails on the spatial dims when
        the dataset yields native-resolution inputs (upsample_on_gpu=True)."""
        if x_dyn.shape[-1] != x_stat.shape[-1]:
            x_dyn = F.interpolate(
                x_dyn, size=x_stat.shape[-2:], mode="bicubic", align_corners=False
            )
        return x_dyn

    def start_state(self, x_dyn, x_stat, generator=None):
        """Encoder pass and the sampler's starting point: (flow-net dynamic
        input, z0). AFM: z0 = mu + sigma_z * eps. CFM: z0 = eps. Used by
        sample() and by train_afm._adabn_afm, so both start from the same
        distribution. x_dyn must already be on the target grid."""
        enc = self.encoder(x_dyn, x_stat)
        shape = (x_stat.shape[0], self.out_ch, *x_stat.shape[-2:])
        noise = torch.randn(shape, generator=generator, device=x_stat.device, dtype=torch.float32)
        z0 = enc.float() + self.sigma_z * noise if self.mode == "afm" else noise
        return self._flow_dyn(x_dyn, enc), z0

    @torch.no_grad()
    def sample(self, x_dyn, x_stat, n_samples=1, steps=20, method="midpoint", generator=None):
        """generator: an optional torch.Generator on the model's device, so a
        validation score can draw the same noise every epoch and every run."""
        was_training = self.training
        self.eval()
        x_dyn = self._match_dyn(x_dyn, x_stat)
        enc = self.encoder(x_dyn, x_stat)
        f_dyn = self._flow_dyn(x_dyn, enc)
        B = x_stat.shape[0]
        shape = (B, self.out_ch, *x_stat.shape[-2:])
        all_s = []
        for _ in range(n_samples):
            # The ODE state stays float32. Under evaluate.py's bf16 autocast
            # the network outputs are bf16, and integrating in bf16 would add
            # a rounding error at every step.
            noise = torch.randn(shape, generator=generator, device=x_stat.device, dtype=torch.float32)
            x = enc.float() + self.sigma_z * noise if self.mode == "afm" else noise
            dt = 1.0 / steps
            for i in range(steps):
                tv = i * dt
                t = torch.full((B,), tv, device=x_stat.device)
                if method == "euler":
                    x = x + self.flow_net(x, t, f_dyn, x_stat).float() * dt
                else:  # midpoint
                    v1 = self.flow_net(x, t, f_dyn, x_stat).float()
                    tm = torch.full_like(t, tv + dt / 2)
                    v2 = self.flow_net(x + v1 * dt / 2, tm, f_dyn, x_stat).float()
                    x = x + v2 * dt
            all_s.append(x)
        if was_training:
            self.train()
        return torch.stack(all_s, dim=1)

    @torch.no_grad()
    def deterministic_predict(self, x_dyn, x_stat):
        """AFM: the encoder mean mu. CFM has no mean: the point forecast is
        the mean of det_members samples (det_steps midpoint steps) with noise
        from a generator seeded 0 on every call, so it is reproducible and the
        same for every model. Costs det_members * 2 * det_steps flow-net calls;
        train_afm.py does not call it per epoch for CFM (it selects on CRPS)."""
        # Do NOT flip module mode here: this used to leave the model
        # permanently in eval() after any call, silently freezing BatchNorm
        # for the rest of training. Callers control mode. (sample() switches
        # to eval and restores the previous mode itself.)
        if self.mode == "afm":
            return self.encoder(x_dyn, x_stat)
        g = torch.Generator(device=x_stat.device)
        g.manual_seed(0)
        return self.sample(x_dyn, x_stat, n_samples=self.det_members, steps=self.det_steps,
                           generator=g).mean(dim=1)
