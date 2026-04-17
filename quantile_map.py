"""
Quantile mapping as a test-time input transform.

Per-channel 1D alignment: for each channel, fit empirical CDFs on source and
target training inputs separately, then map target values through
  x_tgt -> F_src^{-1}(F_tgt(x_tgt))
at test time. This is the baseline that multi-channel-joint methods like
Sinkhorn OT should beat if inter-channel correlations matter for downscaling.

Reference: Cannon et al., J. Climate 2015 (bias correction via quantile
mapping). Applied to UDA for downscaling by Harder et al., RainShift 2025
(arXiv:2507.04930), who apply it to the precipitation channel only.

Interface matches sinkhorn.SinkhornOTMap for drop-in comparison.
"""

from pathlib import Path

import numpy as np
import torch

# precipitation channel indices in the default input ordering
# (cape, cp, sp, tclw, tcw, tisr, tp, u, v)
_TP_INDEX = 6
_CP_INDEX = 1


class QuantileMap:
    """
    Per-channel quantile mapping from target to source input distribution.

    Args:
        n_quantiles: number of quantile anchors per channel (default 1000,
            matching the RainShift paper).
        channels: which channel indices to transform. Use None for all,
            or e.g. [6] for tp only (paper default), [1, 6] for cp+tp.
    """

    def __init__(self, n_quantiles=1000, channels=None):
        self.n_quantiles = n_quantiles
        self.channels = channels  # None = all; list = subset
        self.src_quantiles = None  # (C, n_quantiles)
        self.tgt_quantiles = None  # (C, n_quantiles)
        self.n_channels = None

    def fit(self, src_x, tgt_x):
        """
        Fit empirical CDFs on the two input distributions.

        Args:
            src_x, tgt_x: (N, C, H, W) torch.Tensor or np.ndarray
        """
        if isinstance(src_x, torch.Tensor):
            src_x = src_x.cpu().numpy()
        if isinstance(tgt_x, torch.Tensor):
            tgt_x = tgt_x.cpu().numpy()

        C = src_x.shape[1]
        self.n_channels = C

        qs = np.linspace(0.0, 1.0, self.n_quantiles).astype(np.float32)
        src_q = np.empty((C, self.n_quantiles), dtype=np.float32)
        tgt_q = np.empty((C, self.n_quantiles), dtype=np.float32)

        chans = range(C) if self.channels is None else self.channels

        for c in range(C):
            if c not in chans:
                # identity mapping — both quantile arrays equal
                src_q[c] = tgt_q[c] = np.linspace(-1.0, 1.0, self.n_quantiles,
                                                   dtype=np.float32)
                continue
            src_q[c] = np.quantile(src_x[:, c].ravel(), qs).astype(np.float32)
            tgt_q[c] = np.quantile(tgt_x[:, c].ravel(), qs).astype(np.float32)

        self.src_quantiles = src_q
        self.tgt_quantiles = tgt_q
        return self

    @torch.no_grad()
    def transform(self, x):
        """
        Map target-domain inputs to source-domain distribution per channel.

        Args:
            x: (N, C, H, W) torch.Tensor
        Returns:
            same shape and device, same dtype
        """
        assert self.src_quantiles is not None, "Call fit() first"
        device, orig_dtype = x.device, x.dtype
        N, C, H, W = x.shape

        # move to CPU once, apply per-channel np.interp, return
        x_np = x.detach().float().cpu().numpy()
        out = np.empty_like(x_np)
        chans = range(C) if self.channels is None else self.channels

        for c in range(C):
            if c not in chans:
                out[:, c] = x_np[:, c]
                continue
            flat = x_np[:, c].ravel()
            mapped = np.interp(flat, self.tgt_quantiles[c], self.src_quantiles[c])
            out[:, c] = mapped.reshape(N, H, W)

        return torch.from_numpy(out).to(device=device, dtype=orig_dtype)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            src_quantiles=self.src_quantiles,
            tgt_quantiles=self.tgt_quantiles,
            n_quantiles=np.array(self.n_quantiles),
            channels=np.array(self.channels if self.channels is not None else [-1]),
        )

    @classmethod
    def load(cls, path):
        d = np.load(path)
        ch = d["channels"].tolist()
        channels = None if ch == [-1] else ch
        obj = cls(n_quantiles=int(d["n_quantiles"]), channels=channels)
        obj.src_quantiles = d["src_quantiles"]
        obj.tgt_quantiles = d["tgt_quantiles"]
        obj.n_channels = obj.src_quantiles.shape[0]
        return obj


# ---------------------------------------------------------------------------
#  Convenience: fit from dataset paths
# ---------------------------------------------------------------------------

def fit_qm_from_npy(src_path, tgt_path, n_quantiles=1000, channels=None,
                    subset=None):
    """
    Load training-split inputs from two .npy dataset dirs and fit the QM.

    Args:
        src_path, tgt_path: region directories
        channels: None (all), "tp" (precipitation only, paper default),
                  "precip" (tp + cp), or list of indices
        subset: cap on samples to load per domain
    """
    # resolve shorthand channel specs
    if channels == "tp":
        channels = [_TP_INDEX]
    elif channels == "precip":
        channels = [_CP_INDEX, _TP_INDEX]

    src_x = np.load(Path(src_path) / "train_x.npy", mmap_mode="r")
    tgt_x = np.load(Path(tgt_path) / "train_x.npy", mmap_mode="r")

    if subset is not None:
        src_x = np.asarray(src_x[:subset])
        tgt_x = np.asarray(tgt_x[:subset])
    else:
        src_x = np.asarray(src_x)
        tgt_x = np.asarray(tgt_x)

    qm = QuantileMap(n_quantiles=n_quantiles, channels=channels)
    qm.fit(src_x, tgt_x)
    return qm