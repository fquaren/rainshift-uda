"""
Convert RainShift zarr data to .npy for fast in-memory loading.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import xarray as xr
import zarr
from numpy.lib.format import open_memmap
from tqdm import tqdm

_LOG_VARS = {"tp", "cp", "precipitation", "z"}
_PASSTHROUGH_VARS = {"lsm"}
_UNIT_SCALE = {"tp": 1000.0, "cp": 1000.0}
_LOG_EPS = 1e-5

INPUT_VARS = ["cape", "cp", "sp", "tclw", "tcw", "tisr", "tp", "u", "v"]
OUTPUT_VAR = "precipitation"
STATIC_VARS = ["lsm", "z"]


def transform_var(data: np.ndarray, var_name: str) -> np.ndarray:
    if var_name in _UNIT_SCALE:
        data = data * _UNIT_SCALE[var_name]
    if var_name in _LOG_VARS:
        data = np.log(np.clip(data, 0.0, None) + _LOG_EPS)
    return data.astype(np.float32)


def upsample_batch(x: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    chunk = 256
    out = np.empty((x.shape[0], x.shape[1], target_h, target_w), dtype=np.float32)
    for i in range(0, x.shape[0], chunk):
        t = torch.from_numpy(x[i : i + chunk])
        t = F.interpolate(t, size=(target_h, target_w), mode="bicubic", align_corners=False)
        out[i : i + chunk] = t.numpy()
    return out


def convert_region(zarr_root: Path, out_dir: Path, region: str):
    zarr_path = zarr_root / region
    out_path = out_dir / region
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Converting: {region}")

    # Static variables
    ds_static = xr.open_dataset(str(zarr_path / "static_variables.nc")).load()
    static_list = [transform_var(ds_static[var].values, var) for var in STATIC_VARS]
    static = np.stack(static_list, axis=0) 
    target_h, target_w = static.shape[1], static.shape[2]
    np.save(out_path / "static.npy", static)

    for split in ("train", "test"):
        prefix = "test_data" if split == "test" else "train_data"
        z_in = zarr.open(str(zarr_path / f"{prefix}_in.zarr"), mode="r")
        z_out = zarr.open(str(zarr_path / f"{prefix}_out.zarr"), mode="r")

        first_var = INPUT_VARS[0]
        n = z_in[first_var].shape[0]
        in_h, in_w = z_in[first_var].shape[1], z_in[first_var].shape[2]

        x_out_path = out_path / f"{split}_x.npy"
        y_out_path = out_path / f"{split}_y.npy"
        
        # Initialize memmap arrays directly on disk
        x_memmap = open_memmap(x_out_path, mode='w+', dtype=np.float32, shape=(n, len(INPUT_VARS), target_h, target_w))
        y_memmap = open_memmap(y_out_path, mode='w+', dtype=np.float32, shape=(n, 1, target_h, target_w))

        chunk_size = 512
        
        # Accumulators for out-of-core variance calculation
        if split == "train":
            sum_x = np.zeros(len(INPUT_VARS), dtype=np.float64)
            sum_sq_x = np.zeros(len(INPUT_VARS), dtype=np.float64)
            sum_y, sum_sq_y = 0.0, 0.0
            pixel_count = 0

        for i in tqdm(range(0, n, chunk_size), desc=f"Processing {split}"):
            end = min(i + chunk_size, n)
            
            # Process input chunk
            x_channels = []
            for var in INPUT_VARS:
                raw = z_in[var][i:end].astype(np.float32)
                x_channels.append(transform_var(raw, var))
            x_chunk = np.stack(x_channels, axis=1)

            if in_h != target_h or in_w != target_w:
                x_chunk = upsample_batch(x_chunk, target_h, target_w)
            
            x_memmap[i:end] = x_chunk

            # Process output chunk
            y_raw = z_out[OUTPUT_VAR][i:end].astype(np.float32)
            y_chunk = transform_var(y_raw, OUTPUT_VAR)[:, np.newaxis, :, :]
            y_memmap[i:end] = y_chunk

            # Accumulate statistics
            if split == "train":
                elements_in_chunk = (end - i) * target_h * target_w
                pixel_count += elements_in_chunk
                
                for c, var in enumerate(INPUT_VARS):
                    if var not in _PASSTHROUGH_VARS:
                        vals = x_chunk[:, c].astype(np.float64)
                        sum_x[c] += np.sum(vals)
                        sum_sq_x[c] += np.sum(vals ** 2)
                
                y_vals = y_chunk.astype(np.float64)
                sum_y += np.sum(y_vals)
                sum_sq_y += np.sum(y_vals ** 2)

        x_memmap.flush()
        y_memmap.flush()

        if split == "train":
            stats = {}
            for c, var in enumerate(INPUT_VARS):
                if var not in _PASSTHROUGH_VARS:
                    mean = sum_x[c] / pixel_count
                    var_val = (sum_sq_x[c] / pixel_count) - (mean ** 2)
                    stats[var] = [float(mean), float(np.sqrt(max(var_val, 0.0)))]
            
            mean_y = sum_y / pixel_count
            var_y = (sum_sq_y / pixel_count) - (mean_y ** 2)
            stats[OUTPUT_VAR] = [float(mean_y), float(np.sqrt(max(var_y, 0.0)))]

            for c, var in enumerate(STATIC_VARS):
                if var not in _PASSTHROUGH_VARS:
                    stats[var] = [float(np.mean(static[c])), float(np.std(static[c]))]

            with open(out_path / "stats.json", "w") as f:
                json.dump(stats, f, indent=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_root", required=True, type=Path)
    p.add_argument("--out_root", required=True, type=Path)
    p.add_argument("--regions", nargs="+", required=True)
    args = p.parse_args()

    for region in args.regions:
        convert_region(args.zarr_root, args.out_root, region)

if __name__ == "__main__":
    main()