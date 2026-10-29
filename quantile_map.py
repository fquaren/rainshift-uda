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

Ties. Precipitation has a point mass at the dry floor, so many target
quantiles are equal. np.interp over tied anchors put every dry target pixel
at the TOP of its tied run, i.e. at the source value of probability F_tgt(dry),
which is wet whenever the source is less often dry. transform() now uses the
mid-rank of a tied run (PLAN.md P15).

Space. QM is invariant to any monotone transform applied identically to both
domains, so fitting and applying in the model's input space (log, then
z-scored with SOURCE stats, for both domains) equals mapping in mm, provided
the fit and the application use the same pipeline. fit_qm_from_zarr does.
"""

from pathlib import Path

import numpy as np
import torch

from data.dataset import DEFAULT_INPUT_VARS

# precipitation channel indices in the default input ordering
# (cape, cp, sp, tclw, tcw, tisr, tp, u, v)
_TP_INDEX = DEFAULT_INPUT_VARS.index("tp")
_CP_INDEX = DEFAULT_INPUT_VARS.index("cp")
_CHANNEL_SPECS = {"tp": [_TP_INDEX], "precip": [_CP_INDEX, _TP_INDEX], "all": None}


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
        Map target-domain inputs to source-domain distribution per channel,
        on x's device. A value is placed on the target quantile grid (linear
        between anchors; mid-rank inside a run of tied anchors; clamped
        outside the fitted range) and read off the source grid at the same
        position.

        Args:
            x: (N, C, H, W) torch.Tensor
        Returns:
            same shape and device, same dtype
        """
        assert self.src_quantiles is not None, "Call fit() first"
        C = x.shape[1]
        chans = range(C) if self.channels is None else self.channels
        tq_all = torch.as_tensor(self.tgt_quantiles, dtype=torch.float64, device=x.device)
        sq_all = torch.as_tensor(self.src_quantiles, dtype=torch.float64, device=x.device)
        n = tq_all.shape[1]
        out = x.clone()

        for c in chans:
            tq, sq = tq_all[c], sq_all[c]
            v = x[:, c].double().contiguous()
            lo = torch.searchsorted(tq, v, right=False)   # first anchor >= v
            hi = torch.searchsorted(tq, v, right=True)    # first anchor > v
            i1 = hi.clamp(1, n - 1)
            i0 = i1 - 1
            t0, t1 = tq[i0], tq[i1]
            frac = ((v - t0) / (t1 - t0).clamp_min(1e-12)).clamp(0.0, 1.0)
            pos = i0.double() + frac
            pos = torch.where(hi > lo, (lo + hi - 1).double() / 2, pos).clamp(0.0, n - 1)
            j0 = pos.floor().long().clamp(0, n - 2)
            w = pos - j0.double()
            out[:, c] = (sq[j0] * (1 - w) + sq[j0 + 1] * w).to(x.dtype)

        return out

    def restrict(self, channels):
        """Copy that maps only `channels` ("tp", "precip", "all" or indices).
        Valid only for a map fitted on all channels."""
        assert self.channels is None, "restrict() needs a map fitted on all channels"
        if isinstance(channels, str):
            channels = _CHANNEL_SPECS[channels]
        obj = QuantileMap(n_quantiles=self.n_quantiles, channels=channels)
        obj.src_quantiles, obj.tgt_quantiles = self.src_quantiles, self.tgt_quantiles
        obj.n_channels = self.n_channels
        return obj

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
    if isinstance(channels, str):
        channels = _CHANNEL_SPECS[channels]

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


def _zarr_fit_sample(path, stats, n_chunks, px_stride):
    """Training-split inputs of one domain, in the model's input space (log
    and z-score with `stats`, native 80x80), from n_chunks chunks spread
    evenly over the split, every px_stride-th pixel in each direction."""
    from data.dataset_zarr import ClimateSRDatasetZarr

    ds = ClimateSRDatasetZarr(str(path), "train", stats=stats)
    ids = ds.chunk_ids
    pick = np.linspace(0, len(ids) - 1, min(n_chunks, len(ids))).round().astype(int)
    blocks = [ds._read_chunk(ids[i])[0][:, :, ::px_stride, ::px_stride] for i in pick]
    return np.concatenate(blocks, axis=0)


def fit_qm_from_zarr(src_path, tgt_path, src_stats, n_quantiles=1000, channels=None,
                     n_chunks=40, px_stride=2):
    """
    Fit the QM on the TRAINING splits of two Zarr domains, both normalised
    with the SOURCE stats, as every evaluation does. 40 chunks spread over
    ~20 years cover all seasons (the first N samples would be one season).

    Args:
        channels: None (all), "tp" (RainShift's choice), "precip" (tp + cp),
                  "all", or a list of indices
    """
    if isinstance(channels, str):
        channels = _CHANNEL_SPECS[channels]
    src_x = _zarr_fit_sample(src_path, src_stats, n_chunks, px_stride)
    tgt_x = _zarr_fit_sample(tgt_path, src_stats, n_chunks, px_stride)
    qm = QuantileMap(n_quantiles=n_quantiles, channels=channels)
    qm.fit(src_x, tgt_x)
    return qm