"""
Unsupervised Domain Adaptation methods for climate super-resolution.

All functions operate on raw tensors and are model-agnostic.
Feature-level methods expect (B, C, H, W) feature maps.
Output-level methods expect (B, 1, H, W) predictions.

Methods are grouped by where they act:
    Input-level:   fda_transfer
    Feature-level: coral_loss, mmd_loss, mmd_multiscale_loss, dann_loss + DomainDiscriminator
    Output-level:  spectral_density_loss
    Test-time:     apply_adabn
    Post-hoc:      fit_quantile_mapping, apply_quantile_mapping
"""

from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# Feature-level losses
# ================================================================


def coral_loss(src_feat: torch.Tensor, tgt_feat: torch.Tensor) -> torch.Tensor:
    """
    Deep CORAL: align second-order statistics of source and target features.

    Applies global average pooling, then computes the squared Frobenius norm
    of the difference between source and target covariance matrices.

    Sun & Saenko, ECCV-W 2016. arXiv:1607.01719
    """
    src = F.adaptive_avg_pool2d(src_feat, 1).flatten(1)  # (B_s, C)
    tgt = F.adaptive_avg_pool2d(tgt_feat, 1).flatten(1)  # (B_t, C)

    d = src.size(1)
    n_s, n_t = src.size(0), tgt.size(0)

    src_c = src - src.mean(0, keepdim=True)
    tgt_c = tgt - tgt.mean(0, keepdim=True)

    cov_s = (src_c.T @ src_c) / max(n_s - 1, 1)
    cov_t = (tgt_c.T @ tgt_c) / max(n_t - 1, 1)

    return (cov_s - cov_t).pow(2).sum() / (4 * d * d)


def mmd_loss(
    src_feat: torch.Tensor,
    tgt_feat: torch.Tensor,
    bandwidths: Optional[list] = None,
) -> torch.Tensor:
    """
    Multi-kernel Maximum Mean Discrepancy with Gaussian RBF kernels, on
    single-level GAP-pooled features.

    Gretton et al., JMLR 2012; Long et al., ICML 2015 (DAN).
    """
    if bandwidths is None:
        bandwidths = [0.01, 0.1, 1.0, 10.0, 100.0]

    src = F.adaptive_avg_pool2d(src_feat, 1).flatten(1)
    tgt = F.adaptive_avg_pool2d(tgt_feat, 1).flatten(1)

    def _pairwise_sq_dist(a, b):
        return torch.cdist(a, b, p=2.0).pow(2)

    d_ss = _pairwise_sq_dist(src, src)
    d_tt = _pairwise_sq_dist(tgt, tgt)
    d_st = _pairwise_sq_dist(src, tgt)

    loss = torch.tensor(0.0, device=src.device, dtype=src.dtype)
    for bw in bandwidths:
        gamma = 1.0 / (2.0 * bw * bw)
        k_ss = torch.exp(-gamma * d_ss)
        k_tt = torch.exp(-gamma * d_tt)
        k_st = torch.exp(-gamma * d_st)
        loss = loss + k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()

    return loss / len(bandwidths)


def mmd_multiscale_loss(
    src_feats: Dict[str, torch.Tensor],
    tgt_feats: Dict[str, torch.Tensor],
    levels: Iterable[str] = ("enc2", "enc3", "enc4", "bottleneck"),
    bandwidths: Optional[list] = None,
    median_heuristic: bool = True,
) -> torch.Tensor:
    """
    Multi-scale MMD: average single-scale MMD losses across several encoder
    levels. Aligning features at multiple depths probes both shallow and
    semantic shift, which is the original motivation of DAN/JAN-style
    alignment (Long et al., ICML 2015/2017).

    Args:
        src_feats, tgt_feats: dicts keyed by layer name, each value (B, C, H, W).
        levels: which keys to include. Silently drops keys absent from the dicts.
        bandwidths: RBF bandwidths passed through to mmd_loss when
            ``median_heuristic`` is False.
        median_heuristic: if True (default) the bandwidth at each level is
            derived from the median pairwise distance of the stacked source and
            target features at that level (Gretton et al., JMLR 2012).
            Multi-bandwidth kernels spanning 2^-2..2^2 around the median are
            used, which is the standard DAN configuration.
    """
    active_levels = [lvl for lvl in levels if lvl in src_feats and lvl in tgt_feats]
    if not active_levels:
        raise KeyError(
            f"No requested MMD levels are present in feature dicts. "
            f"Requested {list(levels)}, available {list(src_feats)}."
        )

    losses = []
    for lvl in active_levels:
        if median_heuristic:
            bw = _median_heuristic_bandwidth(src_feats[lvl], tgt_feats[lvl])
            bws = [bw * (2.0**k) for k in (-2, -1, 0, 1, 2)]
        else:
            bws = bandwidths
        losses.append(mmd_loss(src_feats[lvl], tgt_feats[lvl], bandwidths=bws))

    return torch.stack(losses).mean()


@torch.no_grad()
def _median_heuristic_bandwidth(
    src_feat: torch.Tensor,
    tgt_feat: torch.Tensor,
    max_samples: int = 512,
) -> float:
    """
    Median of pairwise Euclidean distances between GAP-pooled features from
    the concatenated source and target batch. Bandwidth is computed with
    torch.no_grad so it enters the RBF kernel as a scalar constant and does
    not receive gradient. Capped at max_samples per domain to avoid O(n^2)
    memory on large batches.
    """
    src = F.adaptive_avg_pool2d(src_feat, 1).flatten(1)
    tgt = F.adaptive_avg_pool2d(tgt_feat, 1).flatten(1)

    if src.size(0) > max_samples:
        src = src[:max_samples]
    if tgt.size(0) > max_samples:
        tgt = tgt[:max_samples]

    z = torch.cat([src, tgt], dim=0)
    d = torch.cdist(z, z, p=2.0)
    # take upper triangle excluding the zero diagonal
    iu = torch.triu_indices(d.size(0), d.size(1), offset=1)
    med = d[iu[0], iu[1]].median().clamp(min=1e-6).item()
    return med


# ================================================================
# Adversarial domain adaptation (DANN)
# ================================================================


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """
    Small MLP for domain classification in DANN.

    Input: GAP-pooled feature vector (B, C).
    Output: domain logit (B, 1).

    Ganin et al., JMLR 2016.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def dann_loss(
    discriminator: DomainDiscriminator,
    src_feat: torch.Tensor,
    tgt_feat: torch.Tensor,
    grl_alpha: float = 1.0,
) -> torch.Tensor:
    """
    DANN adversarial loss with gradient reversal.

    The gradient reversal layer ensures the encoder learns to confuse the
    discriminator while the discriminator learns to separate domains.
    Both are updated in one backward pass.
    """
    src_pooled = F.adaptive_avg_pool2d(src_feat, 1).flatten(1)
    tgt_pooled = F.adaptive_avg_pool2d(tgt_feat, 1).flatten(1)

    src_reversed = grad_reverse(src_pooled, grl_alpha)
    tgt_reversed = grad_reverse(tgt_pooled, grl_alpha)

    src_logits = discriminator(src_reversed)
    tgt_logits = discriminator(tgt_reversed)

    # Source = 0, Target = 1
    src_labels = torch.zeros_like(src_logits)
    tgt_labels = torch.ones_like(tgt_logits)

    loss = F.binary_cross_entropy_with_logits(src_logits, src_labels) + F.binary_cross_entropy_with_logits(
        tgt_logits, tgt_labels
    )

    return loss * 0.5


def dann_grl_schedule(epoch: int, n_epochs: int) -> float:
    """
    Progressive GRL alpha schedule from Ganin et al.
    Ramps from 0 to 1 over training via a sigmoid.
    """
    p = epoch / n_epochs
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0


def lambda_uda_schedule(
    epoch: int,
    n_epochs: int,
    lambda_max: float,
    kind: str = "sigmoid",
) -> float:
    """
    Schedule for the UDA loss weight. Matches the DANN GRL ramp when
    kind == 'sigmoid', a linear warm-up when 'linear', or a constant when
    'fixed'.

    Args:
        epoch: 1-based epoch index.
        n_epochs: total number of epochs.
        lambda_max: asymptotic (fixed) weight.
        kind: 'fixed', 'linear', or 'sigmoid'.
    """
    if kind == "fixed":
        return float(lambda_max)
    p = max(0.0, min(1.0, epoch / max(n_epochs, 1)))
    if kind == "linear":
        return float(lambda_max * p)
    if kind == "sigmoid":
        return float(lambda_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))
    raise ValueError(f"Unknown schedule kind: {kind!r}")


# ================================================================
# Output-level losses
# ================================================================


def _radial_psd(x: torch.Tensor) -> torch.Tensor:
    """
    Compute radially averaged power spectral density.

    Args:
        x: (B, C, H, W) spatial field

    Returns:
        psd: (n_bins,) radially averaged PSD
    """
    B, C, H, W = x.shape
    fft = torch.fft.rfft2(x.float(), dim=(-2, -1))
    power = (fft.abs().pow(2)).mean(dim=(0, 1))  # (H, W//2+1)

    # Wavenumber grid
    ky = torch.fft.fftfreq(H, device=x.device)
    kx = torch.fft.rfftfreq(W, device=x.device)
    kr = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)

    # Radial binning
    n_bins = min(H, W) // 2
    k_edges = torch.linspace(0, 0.5, n_bins + 1, device=x.device)
    psd = torch.zeros(n_bins, device=x.device, dtype=x.dtype)

    for i in range(n_bins):
        mask = (kr >= k_edges[i]) & (kr < k_edges[i + 1])
        if mask.any():
            psd[i] = power[mask].mean()

    return psd


def spectral_density_loss(
    pred_src: torch.Tensor,
    pred_tgt: torch.Tensor,
) -> torch.Tensor:
    """
    Match radially averaged PSD between source and target predictions.
    Comparison in log space for numerical stability across decades of power.
    """
    psd_src = _radial_psd(pred_src)
    psd_tgt = _radial_psd(pred_tgt)

    # Log-space MSE, masking zero bins
    valid = (psd_src > 0) & (psd_tgt > 0)
    if not valid.any():
        return torch.tensor(0.0, device=pred_src.device)

    return F.mse_loss(
        torch.log(psd_src[valid]),
        torch.log(psd_tgt[valid]),
    )


# ================================================================
# Input-level transform
# ================================================================


@torch.no_grad()
def fda_transfer(
    src_img: torch.Tensor,
    tgt_img: torch.Tensor,
    beta: float = 0.01,
) -> torch.Tensor:
    """
    Fourier Domain Adaptation: replace low-frequency amplitude of source
    with that of target.

    Yang & Soatto, CVPR 2020. arXiv:2004.05498

    Args:
        src_img: (B, C, H, W) source input
        tgt_img: (B, C, H, W) target input (randomly sampled)
        beta: fraction of spectrum to swap (0.0 = no change, 1.0 = full swap)

    Returns:
        adapted source image with target low-frequency style
    """
    src_f = torch.fft.fft2(src_img.float(), dim=(-2, -1))
    tgt_f = torch.fft.fft2(tgt_img.float(), dim=(-2, -1))

    src_amp = torch.fft.fftshift(src_f.abs(), dim=(-2, -1))
    tgt_amp = torch.fft.fftshift(tgt_f.abs(), dim=(-2, -1))
    src_phase = src_f.angle()

    B, C, H, W = src_img.shape
    cy, cx = H // 2, W // 2
    h = max(int(H * beta), 1)
    w = max(int(W * beta), 1)

    src_amp[:, :, cy - h : cy + h, cx - w : cx + w] = tgt_amp[:, :, cy - h : cy + h, cx - w : cx + w]

    src_amp = torch.fft.ifftshift(src_amp, dim=(-2, -1))
    result = src_amp * torch.exp(1j * src_phase)
    return torch.fft.ifft2(result, dim=(-2, -1)).real.to(src_img.dtype)


# ================================================================
# Test-time adaptation
# ================================================================


@torch.no_grad()
def apply_adabn(
    model: nn.Module,
    target_loader,
    device: torch.device,
) -> None:
    """
    Adaptive Batch Normalization: replace BN running statistics with
    target-domain statistics. Parameter-free, no training required.

    Li et al., Pattern Recognition 2018. arXiv:1603.04779

    Modifies model in-place.
    """
    # Reset running stats
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None  # use cumulative moving average

    model.train()
    for x_dyn, x_stat, _ in target_loader:
        x_dyn = x_dyn.to(device, non_blocking=True)
        x_stat = x_stat.to(device, non_blocking=True)
        model(x_dyn, x_stat)
    model.eval()


# ================================================================
# Post-hoc: Quantile mapping
# ================================================================


def fit_quantile_mapping(
    source_data: np.ndarray,
    target_data: np.ndarray,
    n_quantiles: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit empirical quantile mapping from source to target distribution.

    Both arrays should be in physical units (after inverse_transform).

    Returns:
        (src_quantiles, tgt_quantiles): arrays of shape (n_quantiles,)
    """
    quantiles = np.linspace(0, 1, n_quantiles)
    src_q = np.quantile(source_data.ravel(), quantiles)
    tgt_q = np.quantile(target_data.ravel(), quantiles)
    return src_q, tgt_q


def apply_quantile_mapping(
    predictions: np.ndarray,
    src_quantiles: np.ndarray,
    tgt_quantiles: np.ndarray,
) -> np.ndarray:
    """Apply fitted quantile mapping to model predictions."""
    shape = predictions.shape
    mapped = np.interp(predictions.ravel(), src_quantiles, tgt_quantiles)
    return mapped.reshape(shape)
