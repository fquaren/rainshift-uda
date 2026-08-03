"""
Zarr-native streaming dataset for RainShift super-resolution.

Reads directly from the RainShift Zarr stores, avoiding the .npy conversion
step and, crucially, the per-sample random-access I/O that makes .npy over a
networked filesystem catastrophically slow. Data is read chunk-aligned along
the time axis (chunk size 200 in the RainShift stores), so one sequential read
serves 200 samples. A rolling shuffle buffer provides SGD mixing without ever
holding the full dataset in RAM.

Layout assumptions (verified for RainShift):
  {domain}/train_data_in.zarr    inputs, dims (time, y, x), 9 variables, 80x80
  {domain}/train_data_out.zarr   target 'precipitation', (time, 200, 200)
  {domain}/test_data_in.zarr     test inputs
  {domain}/test_data_out.zarr    test target
  {domain}/static_variables.nc   static fields at target resolution (200x200)
  {domain}/normalization_stats.json  per-variable [mean, std]

Split semantics (chunk-partitioned, no shuffle across the boundary):
  train       chunks [0, N_chunks - val_chunks) of train_data_*.zarr
  validation  chunks [N_chunks - val_chunks, N_chunks) of train_data_*.zarr
  test        all chunks of test_data_*.zarr

Transforms (must match the .npy pipeline they replace -- VERIFY against the
current dataset.py before committing):
  - units: tp, cp inputs are in metres; multiply by 1000 to mm.
  - log: precip channels (tp, cp, precipitation) get log1p after unit scaling.
  - z-score: per-variable (mean, std) from normalization_stats.json.
  - inputs upsampled 80 -> 200 (bicubic) to match the target grid.
  - static fields normalised and concatenated as a separate tensor.

This is an IterableDataset: the DataLoader must NOT pass shuffle= or rely on
len(); the dataset owns shuffling (train only) and worker sharding.
"""

import json
import math
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import IterableDataset, get_worker_info
import time as _time

def _dlog(msg):
    print(f"[{_time.strftime('%H:%M:%S')}] [zarr] {msg}", flush=True)


DEFAULT_INPUT_VARS = ["cape", "cp", "sp", "tclw", "tcw", "tisr", "tp", "u", "v"]
TARGET_VAR = "precipitation"
STATIC_VARS = ["lsm", "z"]

# --- transform constants, must match convert_zarr_to_npy.py / dataset.py ---
_LOG_VARS = {"tp", "cp", "precipitation", "z"}      # z (geopotential) IS log-transformed
_PASSTHROUGH_VARS = {"lsm"}                          # lsm: no units, no log, no z-score
_UNIT_SCALE = {"tp": 1000.0, "cp": 1000.0}          # m -> mm, applied BEFORE log
_LOG_EPS = 1e-5

_CHUNK = 200            # RainShift time-chunk size
_TARGET_HW = 200        # target spatial resolution


def _transform_channel(arr: np.ndarray, var: str, mean: float, std: float) -> np.ndarray:
    """
    Full transform for one variable block, replicating dataset.py /
    compute_domain_stats exactly:
      units (tp,cp x1000) -> log(clip(x,0)+eps) for _LOG_VARS -> z-score.
    Passthrough vars (lsm) are returned unchanged. Stats were computed in
    log space, so z-score is applied after the log.
    """
    if var in _PASSTHROUGH_VARS:
        return arr.astype(np.float32)
    # Compute in float64 to match compute_domain_stats and avoid precision loss
    # on large-magnitude channels (e.g. z ~ 5e4, where log then (x-mean)/tiny_std
    # loses float32 precision).
    a = arr.astype(np.float64)
    # Scrub non-finite values (IMERG gaps etc.) in physical space, BEFORE
    # the log, matching convert_zarr_to_npy and compute_stats.py. Without
    # this a single NaN/inf pixel makes the loss NaN for the whole batch,
    # which is the melanesia-as-source failure (task=nan from epoch 1).
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if var in _UNIT_SCALE:
        a = a * _UNIT_SCALE[var]
    if var in _LOG_VARS:
        a = np.log(np.clip(a, 0.0, None) + _LOG_EPS)
    return ((a - mean) / (std + 1e-8)).astype(np.float32)


def _compute_and_cache_stats(root: Path, input_vars=None) -> dict:
    """Compute per-variable (mean, std) from a domain's TRAINING Zarr, matching
    compute_domain_stats: units (tp,cp x1000) -> log(clip+eps) for _LOG_VARS ->
    global mean/std in float64. Passthrough vars (lsm) get no entry. Cached to
    normalization_stats.json so workers and later runs reuse it."""
    input_vars = input_vars or DEFAULT_INPUT_VARS
    di = xr.open_zarr(root / "train_data_in.zarr")
    do = xr.open_zarr(root / "train_data_out.zarr")
    ds_static = xr.open_dataset(root / "static_variables.nc")

    def _stat(a, var):
        a = a.astype(np.float64)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        if var in _UNIT_SCALE:
            a = a * _UNIT_SCALE[var]
        if var in _LOG_VARS:
            a = np.log(np.clip(a, 0.0, None) + _LOG_EPS)
        return float(a.mean()), float(a.std())

    stats = {}
    for v in input_vars:
        if v in _PASSTHROUGH_VARS:
            continue
        stats[v] = _stat(di[v].values, v)
    stats[TARGET_VAR] = _stat(do[TARGET_VAR].values, TARGET_VAR)
    for v in STATIC_VARS:
        if v in _PASSTHROUGH_VARS:
            continue
        stats[v] = _stat(ds_static[v].values, v)

    di.close(); do.close(); ds_static.close()
    (root / "normalization_stats.json").write_text(json.dumps(stats))
    return stats


class ClimateSRDatasetZarr(IterableDataset):
    def __init__(
        self,
        domain_path,
        split: str,
        stats: dict = None,                     # <-- ADDED: source stats, threaded from build_loaders
        input_vars=None,
        val_chunks: int = 88,   # ~10%, strided across time (was 175 contiguous)
        split_mode: str = "strided",   # 'strided' (spread across time) or 'contiguous'
        purge_gap: int = 1,             # train chunks within this many of a val chunk are dropped
        shuffle_buffer_chunks: int = 8,
        subset_chunks: int = None,
        upsample_on_gpu: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.root = Path(domain_path)
        self.split = split
        self.input_vars = input_vars or DEFAULT_INPUT_VARS
        self.val_chunks = val_chunks
        self.split_mode = split_mode
        self.purge_gap = purge_gap
        self.shuffle_buffer_chunks = shuffle_buffer_chunks
        self.subset_chunks = subset_chunks
        self.upsample_on_gpu = upsample_on_gpu
        self.seed = seed

        # --- stats: prefer externally supplied (source) stats; else load the
        #     cached file; else compute from THIS domain's training Zarr and
        #     cache. Auto-compute must run in the main process before workers
        #     fork — never lazily inside a worker.
        if stats is not None:
            self.stats = stats
        else:
            stats_path = self.root / "normalization_stats.json"
            if stats_path.exists():
                self.stats = json.loads(stats_path.read_text())
            else:
                self.stats = _compute_and_cache_stats(self.root, self.input_vars)

        # Static fields. lsm is passthrough (raw); z is log-transformed and
        # z-scored, exactly as in compute_domain_stats.
        static_ds = xr.open_dataset(self.root / "static_variables.nc")
        static_layers = []
        for v in STATIC_VARS:
            raw = static_ds[v].values.astype(np.float32)  # (200, 200)
            if v in _PASSTHROUGH_VARS:
                static_layers.append(raw)                 # lsm: unchanged
            else:
                m, s = self.stats[v]                      # <-- self.stats, not stats
                static_layers.append(_transform_channel(raw, v, m, s))  # z: log+zscore
        static_ds.close()
        self.static = torch.from_numpy(np.stack(static_layers, axis=0)).float()  # (S,200,200)

        # Resolve which stores and which chunk range this split reads.
        if split == "test":
            self.in_path = self.root / "test_data_in.zarr"
            self.out_path = self.root / "test_data_out.zarr"
        else:
            self.in_path = self.root / "train_data_in.zarr"
            self.out_path = self.root / "train_data_out.zarr"

        # Determine total sample count and chunk boundaries.
        di = xr.open_zarr(self.in_path)
        self.n_total = di.sizes["time"]
        self.n_chunks = math.ceil(self.n_total / _CHUNK)
        di.close()

        # --- train/validation split over chunks ------------------------------
        # 'strided' (default): validation chunks spread UNIFORMLY across the full
        # time axis (every k-th chunk), so val spans all seasons rather than a
        # single contiguous temporal tail -- which for seasonally structured
        # precipitation makes the tail one season and inflates apparent
        # overfitting. 'contiguous' keeps the old last-N-chunks split.
        #
        # purge_gap: train chunks within `purge_gap` of a validation chunk are
        # dropped (used nowhere) to prevent temporal-autocorrelation leakage
        # between adjacent train/val chunks (blocked/purged CV). 0 disables.
        if split == "test":
            self.chunk_ids = list(range(self.n_chunks))
        else:
            all_ids = list(range(self.n_chunks))
            n_val = min(self.val_chunks, self.n_chunks - 1)

            if self.split_mode == "contiguous":
                val_ids = set(all_ids[self.n_chunks - n_val:])
            elif self.split_mode == "strided":
                if n_val > 0:
                    stride = self.n_chunks / n_val
                    offset = self.seed % max(int(stride), 1)
                    val_ids = set(
                        min(int(offset + i * stride), self.n_chunks - 1)
                        for i in range(n_val)
                    )
                else:
                    val_ids = set()
            else:
                raise ValueError(f"Unknown split_mode '{self.split_mode}'")

            if split == "validation":
                self.chunk_ids = sorted(val_ids)
            elif split == "train":
                purged = set()
                for v in val_ids:
                    for d in range(1, self.purge_gap + 1):
                        purged.add(v - d)
                        purged.add(v + d)
                self.chunk_ids = [
                    c for c in all_ids if c not in val_ids and c not in purged
                ]
            else:
                raise ValueError(f"Unknown split '{split}'")

        if self.subset_chunks is not None:
            self.chunk_ids = self.chunk_ids[: self.subset_chunks]

        if not self.chunk_ids:
            raise ValueError(
                f"No chunks for split '{split}' (n_chunks={self.n_chunks}, "
                f"val_chunks={self.val_chunks}, mode={self.split_mode}). "
                f"Reduce val_chunks."
            )

    # -- helpers -----------------------------------------------------------

    def _stores(self):
        """Lazily open and cache the input/output stores per process. Opening
        once (rather than per chunk) avoids repeated metadata round-trips over
        a networked filesystem. Handles are process-local, so this is safe
        after a DataLoader worker fork."""
        if getattr(self, "_di", None) is None:
            self._di = xr.open_zarr(self.in_path)
            self._do = xr.open_zarr(self.out_path)
        return self._di, self._do

    _logged_first_read = False

    def _read_chunk(self, chunk_id: int):
        """Read one time-chunk of inputs+target, apply transforms, return
        (x, y) numpy blocks of shape (n, C, 80, 80) and (n, 1, 200, 200)."""
        t0 = chunk_id * _CHUNK
        t1 = min(t0 + _CHUNK, self.n_total)

        di, do = self._stores()
        # if not self._logged_first_read:
        #     _dlog(f"{self.split}: first chunk {chunk_id} store opened, reading arrays...")

        x_layers = []
        for v in self.input_vars:
            m, s = self.stats[v] if v not in _PASSTHROUGH_VARS else (0.0, 1.0)
            block = di[v].isel(time=slice(t0, t1)).values.astype(np.float32)  # (n,80,80)
            x_layers.append(_transform_channel(block, v, m, s))
        x = np.stack(x_layers, axis=1)  # (n, C, 80, 80)

        m, s = self.stats[TARGET_VAR]
        y_block = do[TARGET_VAR].isel(time=slice(t0, t1)).values.astype(np.float32)  # (n,200,200)
        y = _transform_channel(y_block, TARGET_VAR, m, s)[:, np.newaxis, :, :]  # (n,1,200,200)
        # if not self._logged_first_read:
        #     _dlog(f"{self.split}: first chunk read OK (x={x.shape}, y={y.shape})")
        #     self._logged_first_read = True

        return x, y

    def _upsample(self, x: np.ndarray) -> np.ndarray:
        """Bicubic 80 -> 200 for a chunk block (n, C, 80, 80) -> (n, C, 200, 200).
        Done per chunk (vectorised) to amortise cost. Skipped if upsample is
        deferred to GPU in the model forward."""
        xt = torch.from_numpy(x)
        xt = F.interpolate(xt, size=(_TARGET_HW, _TARGET_HW), mode="bicubic", align_corners=False)
        return xt.numpy()

    def _worker_chunk_ids(self):
        """Shard chunk ids across dataloader workers (strided, disjoint)."""
        info = get_worker_info()
        ids = list(self.chunk_ids)
        if info is None:
            return ids
        return ids[info.id :: info.num_workers]

    # -- iteration ---------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        ids = self._worker_chunk_ids()
        _wi = get_worker_info()
        _wid = _wi.id if _wi else 0
        # _dlog(f"{self.split} worker {_wid}: {len(ids)} chunks assigned; reading first...")

        # Per-epoch chunk-order shuffle (train only). Seed varies by worker and
        # by a per-iterator counter so successive epochs differ.
        rng = np.random.default_rng(self.seed + (get_worker_info().id if get_worker_info() else 0))
        if self.split == "train":
            rng.shuffle(ids)

        buffer_x, buffer_y = [], []
        buf_cap = self.shuffle_buffer_chunks * _CHUNK

        def _flush_shuffled():
            order = rng.permutation(len(buffer_x))
            for k in order:
                yield buffer_x[k], buffer_y[k]

        for cid in ids:
            x, y = self._read_chunk(cid)
            if not self.upsample_on_gpu:
                x = self._upsample(x)
            n = x.shape[0]

            if self.split == "train":
                for k in range(n):
                    buffer_x.append(x[k])
                    buffer_y.append(y[k])
                if len(buffer_x) >= buf_cap:
                    yield from self._emit(buffer_x, buffer_y, rng, shuffle=True)
                    buffer_x, buffer_y = [], []
            else:
                # val/test: yield in order, no buffering
                for k in range(n):
                    yield self._pack(x[k], y[k])

        if self.split == "train" and buffer_x:
            yield from self._emit(buffer_x, buffer_y, rng, shuffle=True)

    def _emit(self, bx, by, rng, shuffle):
        order = rng.permutation(len(bx)) if shuffle else range(len(bx))
        for k in order:
            yield self._pack(bx[k], by[k])

    def _pack(self, x_np, y_np):
        x = torch.from_numpy(np.ascontiguousarray(x_np)).float()
        y = torch.from_numpy(np.ascontiguousarray(y_np)).float()
        return x, self.static, y


class JointZarrDataset(IterableDataset):
    """
    Oracle joint-training stream: yields labelled samples from source THEN
    target (both train splits) into a single stream, for the --joint_training
    upper bound where source+target labels are used under one loss. This is the
    iterable equivalent of ConcatDataset (which does not work with
    IterableDataset). Distinct from the UDA per-step draw, which uses two
    independent loaders instead.

    Chunks from both domains are shuffled together each epoch (train semantics)
    so the two domains are interleaved rather than presented in two blocks.
    """

    def __init__(self, src_ds: "ClimateSRDatasetZarr", tgt_ds: "ClimateSRDatasetZarr"):
        super().__init__()
        assert src_ds.split == "train" and tgt_ds.split == "train"
        self.src = src_ds
        self.tgt = tgt_ds

    def __iter__(self):
        # Tag each chunk with its owning dataset, shuffle the combined list,
        # then stream. Worker sharding is applied to the combined chunk list.
        tagged = [(self.src, c) for c in self.src.chunk_ids] + [
            (self.tgt, c) for c in self.tgt.chunk_ids
        ]
        info = get_worker_info()
        if info is not None:
            tagged = tagged[info.id :: info.num_workers]
        rng = np.random.default_rng(self.src.seed + (info.id if info else 0))
        rng.shuffle(tagged)

        buf = []
        cap = self.src.shuffle_buffer_chunks * _CHUNK
        for ds, cid in tagged:
            x, y = ds._read_chunk(cid)
            if not ds.upsample_on_gpu:
                x = ds._upsample(x)
            for k in range(x.shape[0]):
                buf.append((ds, x[k], y[k]))
            if len(buf) >= cap:
                for j in rng.permutation(len(buf)):
                    ds_k, xk, yk = buf[j]
                    yield ds_k._pack(xk, yk)
                buf = []
        if buf:
            for j in rng.permutation(len(buf)):
                ds_k, xk, yk = buf[j]
                yield ds_k._pack(xk, yk)