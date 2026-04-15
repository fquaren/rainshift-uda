import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DualEncoderUNet(nn.Module):
    """
    Dual-encoder UNet for climate super-resolution.

    Dynamic atmospheric variables pass through a 4-level encoder + bottleneck.
    Static topographic variables pass through a 2-level encoder and are
    injected at the two finest decoder levels via skip connections.

    BatchNorm is used throughout to enable AdaBN as a UDA baseline.

    Architecture (200x200 input, base_features=64):

        Dynamic encoder                Static encoder
        ─────────────────               ──────────────
        L1: 9  → 64   (200)            S1: 2  → 32  (200)
        L2: 64 → 128  (100)            S2: 32 → 64  (100)
        L3: 128→ 256  (50)
        L4: 256→ 512  (25)
        BN: 512→1024  (12)

        Decoder
        ──────
        U4: 1024     → cat(skip4=512)              → 512  (25)
        U3: 512      → cat(skip3=256)              → 256  (50)
        U2: 256      → cat(skip2=128, stat2=64)    → 128  (100)
        U1: 128      → cat(skip1=64,  stat1=32)    → 64   (200)

        Output: 64 → 1 (200)
    """

    def __init__(
        self,
        dynamic_channels: int = 9,
        static_channels: int = 2,
        out_channels: int = 1,
        base_features: int = 64,
    ):
        super().__init__()
        f = base_features  # shorthand

        # --- Dynamic encoder (4 levels) ---
        self.dyn_enc1 = DoubleConv(dynamic_channels, f)       # -> f
        self.dyn_enc2 = DoubleConv(f, f * 2)                  # -> 2f
        self.dyn_enc3 = DoubleConv(f * 2, f * 4)              # -> 4f
        self.dyn_enc4 = DoubleConv(f * 4, f * 8)              # -> 8f
        self.pool = nn.MaxPool2d(2, 2)

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(f * 8, f * 16)           # -> 16f

        # --- Static encoder (2 levels) ---
        self.stat_enc1 = DoubleConv(static_channels, f // 2)  # -> f/2
        self.stat_enc2 = DoubleConv(f // 2, f)                # -> f

        # --- Decoder ---
        # Each level: upsample -> cat(skip [+ static]) -> DoubleConv
        # U4: 16f up -> cat 8f = 24f -> 8f
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec4 = DoubleConv(f * 16 + f * 8, f * 8)

        # U3: 8f up -> cat 4f = 12f -> 4f
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = DoubleConv(f * 8 + f * 4, f * 4)

        # U2: 4f up -> cat 2f + f (static) = 7f -> 2f
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = DoubleConv(f * 4 + f * 2 + f, f * 2)

        # U1: 2f up -> cat f + f//2 (static) = 3.5f -> f
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = DoubleConv(f * 2 + f + f // 2, f)

        # --- Output head ---
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def _match_size(self, x, target):
        """Interpolate x to match target's spatial dims (handles pooling rounding)."""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=True)
        return x

    def forward(self, x_dyn, x_stat, extract_features=False):
        """
        Args:
            x_dyn:  (B, dynamic_channels, H, W) - atmospheric variables
            x_stat: (B, static_channels, H, W)  - topographic variables
            extract_features: if True, also return dict of intermediate features
                              for domain alignment losses

        Returns:
            out: (B, out_channels, H, W)
            features (optional): dict with keys 'bottleneck', 'enc4', 'enc3',
                                 'enc2' for multi-scale UDA losses
        """

        # --- Dynamic encoder ---
        d1 = self.dyn_enc1(x_dyn)           # (B, f, H, W)
        d2 = self.dyn_enc2(self.pool(d1))   # (B, 2f, H/2, W/2)
        d3 = self.dyn_enc3(self.pool(d2))   # (B, 4f, H/4, W/4)
        d4 = self.dyn_enc4(self.pool(d3))   # (B, 8f, H/8, W/8)
        bn = self.bottleneck(self.pool(d4)) # (B, 16f, H/16, W/16)

        # --- Static encoder ---
        s1 = self.stat_enc1(x_stat)         # (B, f/2, H, W)
        s2 = self.stat_enc2(self.pool(s1))  # (B, f, H/2, W/2)

        # --- Decoder ---
        u4 = self.up4(bn)
        u4 = self._match_size(u4, d4)
        u4 = self.dec4(torch.cat([u4, d4], dim=1))

        u3 = self.up3(u4)
        u3 = self._match_size(u3, d3)
        u3 = self.dec3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(u3)
        u2 = self._match_size(u2, d2)
        s2 = self._match_size(s2, d2)
        u2 = self.dec2(torch.cat([u2, d2, s2], dim=1))

        u1 = self.up1(u2)
        u1 = self._match_size(u1, d1)
        s1 = self._match_size(s1, d1)
        u1 = self.dec1(torch.cat([u1, d1, s1], dim=1))

        # No activation: output is in standardized log space where
        # negative values represent below-mean log-precipitation.
        # Physical non-negativity is enforced at inverse_transform().
        out = self.final_conv(u1)

        if extract_features:
            features = {
                "bottleneck": bn,  # deepest, most semantic
                "enc4": d4,        # 1/8 resolution
                "enc3": d3,        # 1/4 resolution
                "enc2": d2,        # 1/2 resolution
            }
            return out, features

        return out


if __name__ == "__main__":
    model = DualEncoderUNet(dynamic_channels=9, static_channels=2, out_channels=1)
    x_dyn = torch.randn(2, 9, 200, 200)
    x_stat = torch.randn(2, 2, 200, 200)

    # Standard forward
    out = model(x_dyn, x_stat)
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}] (standardized log space)")

    # With feature extraction for UDA
    out, feats = model(x_dyn, x_stat, extract_features=True)
    for k, v in feats.items():
        print(f"  {k}: {v.shape}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params / 1e6:.2f}M")