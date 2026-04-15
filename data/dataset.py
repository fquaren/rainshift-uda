"""
Dataset classes for RainShift climate super-resolution.

Two loaders:
  ClimateSRDataset      — reads directly from zarr (original format)
  ClimateSRDatasetNPY   — reads pre-converted .npy files (fast, recommended)

Use convert_zarr_to_npz.py to create .npy files first, then use the NPY class.

Normalization pipeline (both classes):
  1. Unit conversion (tp, cp: m -> mm)          [baked into .npy]
  2. Log transform (tp, cp, precipitation, z)   [baked into .npy]
  3. Z-score per domain                          [applied at init]
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
#  Constants — must match convert_zarr_to_npz.py exactly
# ---------------------------------------------------------------------------

_LOG_VARS = {"tp", "cp", "precipitation", "z"}
_PASSTHROUGH_VARS = {"lsm"}
_UNIT_SCALE = {"tp": 1000.0, "cp": 1000.0}
_LOG_EPS = 1e-5

DEFAULT_INPUT_VARS = ["cape", "cp", "sp", "tclw", "tcw", "tisr", "tp", "u", "v"]
DEFAULT_STATIC_VARS = ["lsm", "z"]


# ===================================================================
#  NPY dataset (recommended — fast, everything pre-transformed)
# ===================================================================

class ClimateSRDatasetNPY(Dataset):
    """
    Fast dataset loading pre-converted .npy files.

    Expected directory layout (created by convert_zarr_to_npz.py):
        {data_path}/train_x.npy    (N, C_in, H, W)   log-transformed, upsampled
        {data_path}/train_y.npy    (N, 1, H, W)       log-transformed
        {data_path}/test_x.npy
        {data_path}/test_y.npy
        {data_path}/static.npy     (C_static, H, W)   log-transformed where applicable
        {data_path}/stats.json     {var: [mean, std]}  computed on train split

    Z-score is applied in bulk at __init__ (vectorised, ~1 second).
    __getitem__ is pure tensor indexing — zero overhead per sample.

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
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be 'train', 'validation', or 'test'")

        self.input_vars = input_vars or DEFAULT_INPUT_VARS
        self.output_var = output_var
        self.static_vars = static_vars or DEFAULT_STATIC_VARS

        root = Path(data_path)

        # --- load stats ---
        if stats is not None:
            self.stats = stats
        else:
            with open(root / "stats.json") as f:
                self.stats = json.load(f)

        # --- load arrays ---
        file_prefix = "test" if split == "test" else "train"
        x = np.load(root / f"{file_prefix}_x.npy")  # (N, C_in, H, W)
        y = np.load(root / f"{file_prefix}_y.npy")  # (N, 1, H, W)
        static = np.load(root / "static.npy")        # (C_s, H, W)

        # --- train / validation split ---
        n = x.shape[0]
        if split in ("train", "validation"):
            indices = np.arange(n)
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)

            if subset_size is not None and subset_size < n:
                indices = indices[:subset_size]
                n = subset_size

            split_point = int(n * (1 - validation_split_pct))
            if split == "train":
                indices = np.sort(indices[:split_point])
            else:
                indices = np.sort(indices[split_point:])

            x = x[indices]
            y = y[indices]

        elif subset_size is not None and subset_size < n:
            x = x[:subset_size]
            y = y[:subset_size]

        # --- apply z-score in bulk (vectorised) ---
        x = self._zscore_inputs(x)
        y = self._zscore_output(y)
        static = self._zscore_static(static)

        # --- store as contiguous tensors ---
        self.x = torch.from_numpy(np.ascontiguousarray(x))
        self.y = torch.from_numpy(np.ascontiguousarray(y))
        self.static = torch.from_numpy(np.ascontiguousarray(static))

        print(f"[NPY] {split}: {len(self)} samples | "
              f"x={list(self.x.shape)} y={list(self.y.shape)} "
              f"static={list(self.static.shape)}")

    def _zscore_inputs(self, x: np.ndarray) -> np.ndarray:
        """Z-score each input channel. x: (N, C, H, W)."""
        x = x.copy()
        for c, var in enumerate(self.input_vars):
            if var in _PASSTHROUGH_VARS:
                continue
            mean, std = self.stats[var]
            x[:, c] = (x[:, c] - mean) / (std + 1e-8)
        return x

    def _zscore_output(self, y: np.ndarray) -> np.ndarray:
        """Z-score the target. y: (N, 1, H, W)."""
        mean, std = self.stats[self.output_var]
        return ((y - mean) / (std + 1e-8)).astype(np.float32)

    def _zscore_static(self, s: np.ndarray) -> np.ndarray:
        """Z-score static channels. s: (C_s, H, W)."""
        s = s.copy()
        for c, var in enumerate(self.static_vars):
            if var in _PASSTHROUGH_VARS:
                continue
            mean, std = self.stats[var]
            s[c] = (s[c] - mean) / (std + 1e-8)
        return s

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.static, self.y[idx]


# ===================================================================
#  Zarr dataset (original, slower but no preprocessing step)
# ===================================================================

class ClimateSRDataset(Dataset):
    """Dataset reading directly from zarr. See module docstring."""

    def __init__(
        self,
        cluster_path: str,
        split: str = "train",
        normalization_stats: dict = None,
        input_vars: list = None,
        output_var: str = "precipitation",
        static_vars: list = None,
        validation_split_pct: float = 0.2,
        seed: int = 42,
        subset_size: int = None,
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be 'train', 'validation', or 'test'")
        if normalization_stats is None:
            raise ValueError("normalization_stats is required.")

        self.input_vars = input_vars or DEFAULT_INPUT_VARS
        self.output_var = output_var
        self.static_vars = static_vars or DEFAULT_STATIC_VARS
        self.stats = normalization_stats

        all_vars = set(self.input_vars) | {self.output_var} | set(self.static_vars)
        missing = (all_vars - _PASSTHROUGH_VARS) - set(self.stats.keys())
        if missing:
            raise ValueError(f"Missing normalization stats for: {missing}")

        file_prefix = "test_data" if split == "test" else "train_data"
        ds_in = xr.open_zarr(
            f"{cluster_path}/{file_prefix}_in.zarr", consolidated=True
        )
        ds_static = xr.open_dataset(
            f"{cluster_path}/static_variables.nc"
        ).load()

        self.num_samples_total = ds_in.sizes["time"]
        all_indices = np.arange(self.num_samples_total)

        if subset_size is not None and subset_size < self.num_samples_total:
            rng = np.random.RandomState(seed)
            rng.shuffle(all_indices)
            all_indices = all_indices[:subset_size]
            self.num_samples_total = subset_size
        elif split in ("train", "validation"):
            rng = np.random.RandomState(seed)
            rng.shuffle(all_indices)

        if split == "test":
            self.indices = all_indices
        else:
            split_point = int(self.num_samples_total * (1 - validation_split_pct))
            if split == "train":
                self.indices = all_indices[:split_point]
            else:
                self.indices = all_indices[split_point:]

        self.indices = np.sort(self.indices)

        static_data_list = []
        for var in self.static_vars:
            data = ds_static[var].values.astype(np.float32)
            static_data_list.append(self._normalize(data, var))
        self.static_tensor = torch.from_numpy(
            np.stack(static_data_list, axis=0)
        ).float()

        print(f"Loading {len(self.indices)} samples into RAM for {split} split...")
        z_in = zarr.open(f"{cluster_path}/{file_prefix}_in.zarr", mode="r")
        z_out = zarr.open(f"{cluster_path}/{file_prefix}_out.zarr", mode="r")

        x_list, y_list = [], []
        for idx in self.indices:
            var_data = []
            for var in self.input_vars:
                val = z_in[var][idx].astype(np.float32)
                var_data.append(self._normalize(val, var))
            x_list.append(np.stack(var_data, axis=0))

            y_val = z_out[self.output_var][idx].astype(np.float32)
            y_list.append(self._normalize(y_val, self.output_var))

        self.x_data = np.stack(x_list, axis=0)
        self.y_data = np.stack(y_list, axis=0)
        print(f"Data loaded. x: {self.x_data.shape}, y: {self.y_data.shape}")

    def __len__(self):
        return len(self.indices)

    def _normalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        if var_name in _UNIT_SCALE:
            data = data * _UNIT_SCALE[var_name]
        if var_name in _LOG_VARS:
            data = np.log(np.clip(data, 0.0, None) + _LOG_EPS)
        if var_name in _PASSTHROUGH_VARS:
            return data
        mean, std = self.stats[var_name]
        return (data - mean) / (std + 1e-8)

    def __getitem__(self, idx):
        x_dynamic = torch.from_numpy(self.x_data[idx]).float()
        y_target = torch.from_numpy(self.y_data[idx]).float().unsqueeze(0)

        target_h, target_w = y_target.shape[-2], y_target.shape[-1]
        x_dynamic = x_dynamic.unsqueeze(0)
        x_dynamic = F.interpolate(
            x_dynamic, size=(target_h, target_w),
            mode="bicubic", align_corners=False,
        )
        x_dynamic = x_dynamic.squeeze(0)

        return x_dynamic, self.static_tensor, y_target


# ===================================================================
#  Utilities
# ===================================================================

def compute_domain_stats(cluster_path: str, input_vars=None, output_var="precipitation",
                         static_vars=None) -> dict:
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


def load_domain_stats(data_path: str) -> dict:
    """Load pre-computed stats from .npy region directory."""
    with open(Path(data_path) / "stats.json") as f:
        return json.load(f)


def inverse_transform(prediction: np.ndarray, var_name: str,
                      stats: dict) -> np.ndarray:
    """Map model output back to physical units (mm for precipitation)."""
    if var_name not in _PASSTHROUGH_VARS:
        mean, std = stats[var_name]
        if isinstance(mean, list):
            mean, std = mean[0], mean[1]  # handle json list format
        prediction = prediction * (std + 1e-8) + mean

    if var_name in _LOG_VARS:
        prediction = np.exp(prediction) - _LOG_EPS
        prediction = np.clip(prediction, 0.0, None)

    return prediction