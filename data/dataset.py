"""
Dataset classes for RainShift climate super-resolution.

ClimateSRDatasetNPY — memory-mapped .npy loader (recommended).

Memory-mapped loading means the OS pages data from disk on demand, so the
full dataset never needs to fit in RAM. The z-score normalisation is applied
per-sample in __getitem__ using precomputed channel-wise (mean, std) tensors.
Cost: ~10 µs per sample on GH200 NVLink — negligible vs. the model forward.

Normalization pipeline:
  1. Unit conversion (tp, cp: m -> mm)          [baked into .npy]
  2. Log transform (tp, cp, precipitation, z)   [baked into .npy]
  3. Z-score per domain                          [applied per sample in __getitem__]
  4. Land-sea mask unchanged                     [passthrough]
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import zarr
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
#  Constants — must match convert_zarr_to_npy.py exactly
# ---------------------------------------------------------------------------

_LOG_VARS = {"tp", "cp", "precipitation", "z"}
_PASSTHROUGH_VARS = {"lsm"}
_UNIT_SCALE = {"tp": 1000.0, "cp": 1000.0}
_LOG_EPS = 1e-5

DEFAULT_INPUT_VARS = ["cape", "cp", "sp", "tclw", "tcw", "tisr", "tp", "u", "v"]
DEFAULT_STATIC_VARS = ["lsm", "z"]


# ===================================================================
#  NPY dataset — memory-mapped, lazy normalisation
# ===================================================================


class ClimateSRDatasetNPY(Dataset):
    """
    Memory-mapped dataset loading pre-converted .npy files.

    Expected directory layout (created by convert_zarr_to_npy.py):
        {data_path}/train_x.npy    (N, C_in, H, W)
        {data_path}/train_y.npy    (N, 1, H, W)
        {data_path}/test_x.npy
        {data_path}/test_y.npy
        {data_path}/static.npy     (C_static, H, W)
        {data_path}/stats.json     {var: [mean, std]}

    Z-score is applied per-sample in __getitem__ via precomputed tensors.
    Static covariates are normalised once at __init__ (small: 2×200×200).

    Args:
        data_path: directory containing .npy files for one region
        split: 'train', 'validation', or 'test'
        stats: normalization stats dict; if None, loads from stats.json
        input_vars: variable name list (for stats lookup ordering)
        output_var: target variable name
        static_vars: static variable name list
        validation_split_pct: fraction held for validation
        seed: random seed for train/val split
        subset_size: cap on samples (for debugging)
        augment: apply random flips + 90° rotations (train only)
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        stats: dict = None,
        input_vars: list = None,
        output_var: str = "precipitation",
        static_vars: list = None,
        validation_split_pct: float = 0.2,
        seed: int = 42,
        subset_size: int = None,
        augment: bool = False,
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be 'train', 'validation', or 'test'")

        self.input_vars = input_vars or DEFAULT_INPUT_VARS
        self.output_var = output_var
        self.static_vars = static_vars or DEFAULT_STATIC_VARS
        self.augment = augment and split == "train"

        root = Path(data_path)

        # --- load stats ---
        if stats is not None:
            self.stats = stats
        else:
            with open(root / "stats.json") as f:
                self.stats = json.load(f)

        # --- memory-map arrays (no RAM allocation) ---
        file_prefix = "test" if split == "test" else "train"
        self._x_mmap = np.load(root / f"{file_prefix}_x.npy", mmap_mode="r")
        self._y_mmap = np.load(root / f"{file_prefix}_y.npy", mmap_mode="r")

        # --- compute sample indices for this split ---
        n_total = self._x_mmap.shape[0]

        if split in ("train", "validation"):
            indices = np.arange(n_total)
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)

            if subset_size is not None and subset_size < n_total:
                indices = indices[:subset_size]

            n = len(indices)
            split_point = int(n * (1 - validation_split_pct))
            if split == "train":
                self.indices = np.sort(indices[:split_point])
            else:
                self.indices = np.sort(indices[split_point:])
        else:
            if subset_size is not None and subset_size < n_total:
                self.indices = np.arange(subset_size)
            else:
                self.indices = np.arange(n_total)

        # --- precompute z-score tensors for fast per-sample normalisation ---
        # x: (C_in,) mean and std
        x_mean = np.zeros(len(self.input_vars), dtype=np.float32)
        x_std = np.ones(len(self.input_vars), dtype=np.float32)
        for c, var in enumerate(self.input_vars):
            if var in _PASSTHROUGH_VARS:
                continue
            m, s = self.stats[var]
            x_mean[c] = m
            x_std[c] = s + 1e-8

        # shapes for broadcasting: (C, 1, 1)
        self._x_mean = torch.from_numpy(x_mean).reshape(-1, 1, 1)
        self._x_std = torch.from_numpy(x_std).reshape(-1, 1, 1)

        # y: scalar mean/std
        ym, ys = self.stats[self.output_var]
        self._y_mean = float(ym)
        self._y_std = float(ys) + 1e-8

        # --- static: small, normalise once and keep in memory ---
        static = np.load(root / "static.npy").copy()  # (C_s, H, W)
        for c, var in enumerate(self.static_vars):
            if var in _PASSTHROUGH_VARS:
                continue
            m, s = self.stats[var]
            static[c] = (static[c] - m) / (s + 1e-8)
        self.static = torch.from_numpy(static).float()

        print(
            f"[NPY-mmap] {split}: {len(self)} samples | "
            f"x={list(self._x_mmap.shape)} "
            f"static={list(self.static.shape)}"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        # read from mmap (triggers OS page-in, ~µs on NVMe)
        x = torch.from_numpy(self._x_mmap[real_idx].copy()).float()
        y = torch.from_numpy(self._y_mmap[real_idx].copy()).float()

        # per-sample z-score
        x = (x - self._x_mean) / self._x_std
        y = (y - self._y_mean) / self._y_std

        s = self.static

        # augmentation: random flip + 90° rotation
        if self.augment:
            x, s, y = self._augment(x, s, y)

        return x, s, y

    @staticmethod
    def _augment(x, s, y):
        # random horizontal flip
        if torch.rand(1).item() > 0.5:
            x = x.flip(-1)
            s = s.flip(-1)
            y = y.flip(-1)
        # random vertical flip
        if torch.rand(1).item() > 0.5:
            x = x.flip(-2)
            s = s.flip(-2)
            y = y.flip(-2)
        # random 90° rotation (0, 1, 2, or 3 times)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            x = torch.rot90(x, k, [-2, -1])
            s = torch.rot90(s, k, [-2, -1])
            y = torch.rot90(y, k, [-2, -1])
        return x, s, y


# ===================================================================
#  Utilities
# ===================================================================


def load_domain_stats(data_path: str) -> dict:
    """Load pre-computed stats from .npy region directory."""
    with open(Path(data_path) / "stats.json") as f:
        return json.load(f)


def inverse_transform(prediction: np.ndarray, var_name: str, stats: dict) -> np.ndarray:
    """Map model output back to physical units (mm for precipitation)."""
    if var_name not in _PASSTHROUGH_VARS:
        mean, std = stats[var_name]
        if isinstance(mean, list):
            mean, std = mean[0], mean[1]
        prediction = prediction * (std + 1e-8) + mean

    if var_name in _LOG_VARS:
        prediction = np.exp(prediction) - _LOG_EPS
        prediction = np.clip(prediction, 0.0, None)

    return prediction


def compute_domain_stats(cluster_path: str, input_vars=None, output_var="precipitation", static_vars=None) -> dict:
    """Compute per-domain (mean, std) on zarr training data. Post log-transform."""
    input_vars = input_vars or DEFAULT_INPUT_VARS
    static_vars = static_vars or DEFAULT_STATIC_VARS

    z_in = zarr.open(f"{cluster_path}/train_data_in.zarr", mode="r")
    z_out = zarr.open(f"{cluster_path}/train_data_out.zarr", mode="r")
    ds_static = xr.open_dataset(f"{cluster_path}/static_variables.nc").load()

    stats = {}
    for var in input_vars:
        if var in _PASSTHROUGH_VARS:
            continue
        data = z_in[var][:].astype(np.float64)
        if var in _UNIT_SCALE:
            data = data * _UNIT_SCALE[var]
        if var in _LOG_VARS:
            data = np.log(np.clip(data, 0.0, None) + _LOG_EPS)
        stats[var] = (float(np.mean(data)), float(np.std(data)))

    data = z_out[output_var][:].astype(np.float64)
    if output_var in _UNIT_SCALE:
        data = data * _UNIT_SCALE[output_var]
    if output_var in _LOG_VARS:
        data = np.log(np.clip(data, 0.0, None) + _LOG_EPS)
    stats[output_var] = (float(np.mean(data)), float(np.std(data)))

    for var in static_vars:
        if var in _PASSTHROUGH_VARS:
            continue
        data = ds_static[var].values.astype(np.float64)
        if var in _UNIT_SCALE:
            data = data * _UNIT_SCALE[var]
        if var in _LOG_VARS:
            data = np.log(np.clip(data, 0.0, None) + _LOG_EPS)
        stats[var] = (float(np.mean(data)), float(np.std(data)))

    return stats
