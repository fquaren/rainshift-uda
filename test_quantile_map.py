"""CPU check of quantile_map.QuantileMap on synthetic data. Run inside a job
(scripts/field_mean_check.sh runs it before the smoke pass when QM is set):
    sbatch --partition cpu --time 00:10:00 --mem 4G --chdir /scratch/fquareng \
        --output outputs/%j --error job_errors/%j \
        --wrap 'source <CODE_ROOT>/scripts/python_env.sh; cd <CODE_ROOT>; run_python test_quantile_map.py'
"""

import numpy as np
import torch

from quantile_map import _CP_INDEX, _TP_INDEX, QuantileMap

rng = np.random.default_rng(0)
N, C, H, W = 400, 9, 8, 8
FLOOR = -2.0


def with_floor(z, dry):
    """Continuous values above FLOOR, with a fraction `dry` set to FLOOR."""
    x = FLOOR + np.abs(z) + 1e-3
    x[rng.random(x.shape) < dry] = FLOOR
    return x


src = rng.normal(0.0, 1.0, (N, C, H, W)).astype(np.float32)
tgt = rng.normal(1.0, 2.0, (N, C, H, W)).astype(np.float32)
# tp: a point mass at the dry floor, 50 % in the source, 60 % in the target.
src[:, _TP_INDEX] = with_floor(rng.normal(size=(N, H, W)), 0.5)
tgt[:, _TP_INDEX] = with_floor(rng.normal(size=(N, H, W)), 0.6)

qm = QuantileMap(n_quantiles=1000).fit(src, tgt)
out = qm.transform(torch.from_numpy(tgt)).numpy()
assert out.shape == tgt.shape and out.dtype == np.float32

# 1. A continuous channel is mapped to the source marginal.
c = 0
assert abs(out[:, c].mean() - src[:, c].mean()) < 0.02, out[:, c].mean()
assert abs(out[:, c].std() - src[:, c].std()) < 0.02, out[:, c].std()
qs = np.linspace(0.01, 0.99, 99)
assert np.abs(np.quantile(out[:, c], qs) - np.quantile(src[:, c], qs)).max() < 0.03

# 2. Ties: every dry target pixel (rank mid-point 0.30) lands on the source
#    floor (source dry fraction 0.50). np.interp over the tied anchors, as the
#    old transform did, would put them at the source value of rank 0.60, wet.
dry = tgt[:, _TP_INDEX] == FLOOR
assert np.allclose(out[:, _TP_INDEX][dry], FLOOR), np.unique(out[:, _TP_INDEX][dry])[:5]
old = np.interp(tgt[:, _TP_INDEX][dry], qm.tgt_quantiles[_TP_INDEX], qm.src_quantiles[_TP_INDEX])
print(f"old np.interp transform: {(old > FLOOR + 1e-3).mean():.2f} of dry target pixels mapped wet "
      f"(new: {(out[:, _TP_INDEX][dry] > FLOOR + 1e-3).mean():.2f})")
# The wet target pixels then fill the source distribution above its 0.60 rank.
wet_out = out[:, _TP_INDEX][~dry]
assert abs(np.median(wet_out) - np.quantile(src[:, _TP_INDEX], 0.8)) < 0.05

# 3. Outside the fitted range values clamp to the source extremes.
ext = torch.from_numpy(tgt[:2].copy())
ext[:, 0] = 1e3
ext[:, 3] = -1e3
o = qm.transform(ext).numpy()
assert np.allclose(o[:, 0], qm.src_quantiles[0, -1]) and np.allclose(o[:, 3], qm.src_quantiles[3, 0])

# 4. restrict(): only the named channels move, the rest are bit-identical.
for arm, chans in (("tp", [_TP_INDEX]), ("precip", [_CP_INDEX, _TP_INDEX])):
    o = qm.restrict(arm).transform(torch.from_numpy(tgt)).numpy()
    for k in range(C):
        if k in chans:
            assert np.allclose(o[:, k], out[:, k]), (arm, k)
        else:
            assert np.array_equal(o[:, k], tgt[:, k]), (arm, k)

# 5. Same result on the GPU, if there is one.
if torch.cuda.is_available():
    g = qm.transform(torch.from_numpy(tgt).cuda()).cpu().numpy()
    assert np.allclose(g, out, atol=1e-5)

# 6. save / load round trip.
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "qm.npz"
    qm.restrict("precip").save(p)
    q2 = QuantileMap.load(p)
    assert q2.channels == [_CP_INDEX, _TP_INDEX]
    assert np.allclose(q2.transform(torch.from_numpy(tgt)).numpy(),
                       qm.restrict("precip").transform(torch.from_numpy(tgt)).numpy())

print("test_quantile_map: all checks passed")
