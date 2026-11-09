"""CPU checks for the GAP-ablation additions to uda.py. Run inside a job.

1. mmd_loss with its defaults (GAP, biased) equals the implementation every
   seeded run used, so those runs stay reproducible.
2. pool_features shapes: gap (B, C), grid4 (B, 32 C).
3. Biased minus unbiased is the kernel-diagonal term, which is there even
   for identical distributions; the unbiased estimate averages ~0 there.
4. A purely spatial shift (same channel means, activation moved from the
   left to the right half) is invisible to GAP and visible to grid4.
5. Gradients reach the features through grid4; mmd_diagnostics returns its
   four keys; mmd_multiscale_loss runs with grid4 at the four UNet levels.
"""

import torch
import torch.nn.functional as F

from uda import _median_heuristic_bandwidth, mmd_diagnostics, mmd_loss, mmd_multiscale_loss, pool_features

torch.manual_seed(0)


def old_mmd(src_feat, tgt_feat):
    src = F.adaptive_avg_pool2d(src_feat, 1).flatten(1)
    tgt = F.adaptive_avg_pool2d(tgt_feat, 1).flatten(1)
    bw = _median_heuristic_bandwidth(src_feat, tgt_feat)
    loss = 0.0
    for b in [bw * (2.0**k) for k in (-2, -1, 0, 1, 2)]:
        g = 1.0 / (2.0 * b * b)
        loss = loss + torch.exp(-g * torch.cdist(src, src).pow(2)).mean() \
            + torch.exp(-g * torch.cdist(tgt, tgt).pow(2)).mean() - 2 * torch.exp(-g * torch.cdist(src, tgt).pow(2)).mean()
    return loss / 5


B, C = 32, 64
s, t = torch.randn(B, C, 12, 12), torch.randn(B, C, 12, 12) + 0.3
assert torch.allclose(mmd_loss(s, t), old_mmd(s, t), atol=1e-6), "default mmd_loss changed"

assert pool_features(s, "gap").shape == (B, C)
assert pool_features(s, "grid4").shape == (B, 32 * C)

same = [(mmd_loss(a, b), mmd_loss(a, b, unbiased=True))
        for a, b in ((torch.randn(B, C, 12, 12), torch.randn(B, C, 12, 12)) for _ in range(40))]
bias = sum(x - y for x, y in same) / len(same)
unb = sum(y for _, y in same) / len(same)
assert bias > 0, bias
assert abs(unb) < 0.5 * bias, (unb, bias)

def blob(side):
    x = 0.1 * torch.randn(B, C, 12, 12)
    if side == "left":
        x[..., :6] += 1.0
    else:
        x[..., 6:] += 1.0
    return x

L, R = blob("left"), blob("right")
d = mmd_diagnostics(L, R)
assert set(d) == {"mmd_gap", "null_gap", "mmd_grid4", "null_grid4"}, d
assert d["mmd_grid4"] > 10 * max(d["null_grid4"], 1e-6), d
assert d["mmd_gap"] < 0.1 * d["mmd_grid4"], d

x = torch.randn(B, C, 12, 12, requires_grad=True)
mmd_loss(x, t, pool="grid4", unbiased=True).backward()
assert x.grad is not None and x.grad.abs().sum() > 0

shapes = {"enc2": (8, 100, 100), "enc3": (16, 50, 50), "enc4": (32, 25, 25), "bottleneck": (64, 12, 12)}
fs = {k: torch.randn(8, *v) for k, v in shapes.items()}
ft = {k: torch.randn(8, *v) + 0.2 for k, v in shapes.items()}
v = mmd_multiscale_loss(fs, ft, pool="grid4", unbiased=True)
assert torch.isfinite(v), v

print("mmd pooling OK", {k: round(val, 5) for k, val in d.items()})
