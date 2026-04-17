"""
Joint-distribution OT via Sinkhorn for input-level domain adaptation.

Generalises quantile mapping from per-channel 1D alignment to joint alignment
over all input channels, respecting inter-channel correlations (e.g. CAPE-tp,
wind-tp). Applied as a test-time transform on target inputs, mapping them into
the source input distribution so the source-trained model can process them
without distribution shift.

Pipeline:
  1. Subsample source and target inputs at the pixel level (each pixel is a
     C-dim vector where C = number of input channels).
  2. Fit Sinkhorn OT plan P between the two subsampled empirical distributions.
  3. Compute barycentric target -> source map: T(y_j) = Σ_i P_ij * x_i / Σ_i P_ij.
  4. At test time, for each target pixel y, transport it by k-NN barycentric
     projection onto the fitted anchors.

Mathematical note: in the UDA convention here, "source" = training domain,
"target" = test domain. We map target -> source at test time (not the other
way around) so that the trained model sees inputs it was optimised for.

References:
  Cuturi, "Sinkhorn Distances", NeurIPS 2013 (arXiv:1306.0895)
  Courty et al., "Optimal Transport for Domain Adaptation", PAMI 2016
  Flamary et al., POT library (https://pythonot.github.io)
"""

from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
#  Sinkhorn algorithm (log-domain, batched, GPU)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sinkhorn_log(cost, reg, n_iter=200, tol=1e-6):
    """
    Log-domain Sinkhorn for entropic OT between uniform marginals.

    Args:
        cost: (n, m) cost matrix
        reg: entropic regularisation strength (epsilon)
        n_iter: max iterations
        tol: convergence tolerance on marginal error

    Returns:
        P: (n, m) transport plan, rows and cols sum to 1/n and 1/m
    """
    n, m = cost.shape
    device = cost.device
    dtype = cost.dtype

    log_a = -torch.log(torch.tensor(float(n), device=device, dtype=dtype))
    log_b = -torch.log(torch.tensor(float(m), device=device, dtype=dtype))
    log_a = log_a.expand(n)
    log_b = log_b.expand(m)

    K = -cost / reg  # log-kernel
    u = torch.zeros(n, device=device, dtype=dtype)
    v = torch.zeros(m, device=device, dtype=dtype)

    for _ in range(n_iter):
        u_new = log_a - torch.logsumexp(K + v[None, :], dim=1)
        v_new = log_b - torch.logsumexp(K + u_new[:, None], dim=0)
        diff = (u_new - u).abs().max().item()
        u, v = u_new, v_new
        if diff < tol:
            break

    return torch.exp(K + u[:, None] + v[None, :])


# ---------------------------------------------------------------------------
#  Subsampling: pixels from a dataset
# ---------------------------------------------------------------------------

def _subsample_pixels(x_tensor, n_pixels, rng):
    """
    Draw n_pixels random pixel-vectors from a tensor of shape (N, C, H, W).

    Returns: (n_pixels, C) float32 numpy array.
    """
    N, C, H, W = x_tensor.shape
    total = N * H * W
    n_pixels = min(n_pixels, total)

    # flatten to (N*H*W, C) by first transposing
    flat = x_tensor.permute(0, 2, 3, 1).reshape(-1, C)
    idx = rng.choice(total, size=n_pixels, replace=False)
    return flat[idx].float().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
#  OT map fitting
# ---------------------------------------------------------------------------

class SinkhornOTMap:
    """
    Joint-distribution OT map from target inputs to source inputs.

    After fitting, use .transform() to map a tensor (N, C, H, W) from the
    target domain into the source input space.

    Args:
        n_anchors: number of pixel anchors per domain (default 5000).
        reg: Sinkhorn entropic regularisation (default 0.1, in cost units).
        n_iter: max Sinkhorn iterations.
        k: number of neighbours for barycentric projection at test time.
        device: torch device.
    """

    def __init__(self, n_anchors=5000, reg=0.1, n_iter=300, k=10,
                 device="cuda", seed=42):
        self.n_anchors = n_anchors
        self.reg = reg
        self.n_iter = n_iter
        self.k = k
        self.device = device
        self.rng = np.random.default_rng(seed)

        # populated by fit()
        self.anchors_tgt = None        # (n_anchors, C) — target pixels
        self.anchors_tgt_mapped = None # (n_anchors, C) — their barycentric images in source
        self.channel_mean = None
        self.channel_std = None

    def fit(self, src_x, tgt_x):
        """
        Fit OT between subsampled source and target pixel distributions.

        Args:
            src_x: (N_s, C, H, W) source inputs (torch.Tensor, CPU or GPU)
            tgt_x: (N_t, C, H, W) target inputs
        """
        src_pix = _subsample_pixels(src_x, self.n_anchors, self.rng)
        tgt_pix = _subsample_pixels(tgt_x, self.n_anchors, self.rng)

        # standardise per-channel for a well-scaled cost matrix
        stacked = np.concatenate([src_pix, tgt_pix], axis=0)
        self.channel_mean = stacked.mean(axis=0, keepdims=True).astype(np.float32)
        self.channel_std = (stacked.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)

        src_z = (src_pix - self.channel_mean) / self.channel_std
        tgt_z = (tgt_pix - self.channel_mean) / self.channel_std

        # cost = squared Euclidean in standardised space
        src_t = torch.from_numpy(src_z).to(self.device)
        tgt_t = torch.from_numpy(tgt_z).to(self.device)
        cost = torch.cdist(tgt_t, src_t, p=2.0).pow(2)

        # normalise cost so reg is interpretable across domains
        cost = cost / cost.median().clamp(min=1e-8)

        P = sinkhorn_log(cost, reg=self.reg, n_iter=self.n_iter)  # (n_tgt, n_src)

        # barycentric projection: each target anchor maps to weighted mean of source
        # T(y_j) = (P[j,:] @ src) / P[j,:].sum()
        row_sums = P.sum(dim=1, keepdim=True).clamp(min=1e-12)
        mapped_z = (P @ src_t) / row_sums  # (n_tgt, C) in standardised space

        # de-standardise back to original units
        mapped = mapped_z.cpu().numpy() * self.channel_std + self.channel_mean

        self.anchors_tgt = tgt_pix  # in original (untransformed) units
        self.anchors_tgt_mapped = mapped.astype(np.float32)

        return self

    @torch.no_grad()
    def transform(self, x):
        """
        Map a batch of target inputs to source space.

        Args:
            x: (N, C, H, W) torch.Tensor (any device)

        Returns:
            transformed tensor with same shape and device as x
        """
        assert self.anchors_tgt is not None, "Call fit() first"
        device = x.device
        orig_dtype = x.dtype
        N, C, H, W = x.shape

        # flatten to pixels, standardise with the fit-time stats
        flat = x.permute(0, 2, 3, 1).reshape(-1, C).float()  # (N*H*W, C)
        mean = torch.from_numpy(self.channel_mean).to(device)
        std = torch.from_numpy(self.channel_std).to(device)
        flat_z = (flat - mean) / std

        anchors = torch.from_numpy(self.anchors_tgt).to(device)
        anchors_z = (anchors - mean) / std
        mapped = torch.from_numpy(self.anchors_tgt_mapped).to(device)
        mapped_z = (mapped - mean) / std

        # batched k-NN with inverse-distance weights
        mapped_flat_z = _knn_barycentric(flat_z, anchors_z, mapped_z, k=self.k)

        # de-standardise
        mapped_flat = mapped_flat_z * std + mean

        # reshape back
        out = mapped_flat.reshape(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out.to(orig_dtype)

    def save(self, path):
        """Save fitted map to .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            anchors_tgt=self.anchors_tgt,
            anchors_tgt_mapped=self.anchors_tgt_mapped,
            channel_mean=self.channel_mean,
            channel_std=self.channel_std,
            config=np.array([self.n_anchors, self.reg, self.n_iter, self.k]),
        )

    @classmethod
    def load(cls, path, device="cuda"):
        d = np.load(path)
        cfg = d["config"]
        obj = cls(
            n_anchors=int(cfg[0]), reg=float(cfg[1]),
            n_iter=int(cfg[2]), k=int(cfg[3]), device=device,
        )
        obj.anchors_tgt = d["anchors_tgt"]
        obj.anchors_tgt_mapped = d["anchors_tgt_mapped"]
        obj.channel_mean = d["channel_mean"]
        obj.channel_std = d["channel_std"]
        return obj


# ---------------------------------------------------------------------------
#  k-NN barycentric projection (batched, memory-aware)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _knn_barycentric(query, anchors, mapped, k=10, chunk=8192):
    """
    Inverse-distance-weighted k-NN projection.

    Args:
        query:   (Q, C) points to map
        anchors: (A, C) anchor points in same space as query
        mapped:  (A, C) where each anchor was mapped to
        k: number of neighbours
        chunk: query batch size to cap peak memory

    Returns:
        (Q, C) mapped queries
    """
    Q = query.shape[0]
    out = torch.empty_like(query)
    eps = 1e-8

    for start in range(0, Q, chunk):
        end = min(start + chunk, Q)
        q = query[start:end]                          # (b, C)
        d = torch.cdist(q, anchors, p=2.0)            # (b, A)
        vals, idx = torch.topk(d, k=k, largest=False, dim=1)  # (b, k)

        # inverse-distance weights; exact-match anchors get all the weight
        w = 1.0 / (vals + eps)
        w = w / w.sum(dim=1, keepdim=True)            # (b, k)

        # gather mapped anchors: mapped[idx] has shape (b, k, C)
        gathered = mapped[idx]                         # (b, k, C)
        out[start:end] = (w.unsqueeze(-1) * gathered).sum(dim=1)

    return out


# ---------------------------------------------------------------------------
#  Convenience: fit from dataset paths
# ---------------------------------------------------------------------------

def fit_ot_from_npy(src_path, tgt_path, n_anchors=5000, reg=0.1,
                    n_iter=300, k=10, device="cuda", subset=None):
    """
    Load training-split inputs from two .npy dataset dirs and fit the OT map.

    Args:
        src_path, tgt_path: region directories produced by convert_zarr_to_npy.py
        subset: cap on number of samples loaded from each (None = all)
    """
    src_x = np.load(Path(src_path) / "train_x.npy", mmap_mode="r")
    tgt_x = np.load(Path(tgt_path) / "train_x.npy", mmap_mode="r")

    if subset is not None:
        src_x = np.asarray(src_x[:subset])
        tgt_x = np.asarray(tgt_x[:subset])
    else:
        src_x = np.asarray(src_x)
        tgt_x = np.asarray(tgt_x)

    src_t = torch.from_numpy(src_x)
    tgt_t = torch.from_numpy(tgt_x)

    mapper = SinkhornOTMap(n_anchors=n_anchors, reg=reg, n_iter=n_iter,
                           k=k, device=device)
    mapper.fit(src_t, tgt_t)
    return mapper