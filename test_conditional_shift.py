"""CPU checks of the model-based conditional-shift test (conditional_shift.py,
PLAN.md P2) and of the split-half chunk selection. Run inside a job;
scripts/run_cond_test.sh and the SMOKE task of scripts/run_half_split.sh run it.

(a) Smooth per-model error fields (one field per model, shared by every
    sample): a single quadruple's block-bootstrap CI under-covers R = 1,
    while the spread over independent quadruples covers it. This is the
    problem the 3 x 2 replication solves.
(b) to_phys / to_mm against data.dataset_zarr._transform_channel on RAW
    values (tp in m -> log(1000 tp + 1e-5)), not only round trips.
(c) Cross-year pair selection, Latin quadruples, half_chunk_ids purge.
"""

import itertools

import numpy as np
import torch

from conditional_shift import (
    auc,
    from_phys,
    half_pairs,
    half_seed,
    latin_quadruples,
    oof_propensity,
    pairwise,
    replicate_summary,
    seed_pairs,
    summarise,
    to_mm,
    to_phys,
    vectors,
)
from data.dataset import inverse_transform
from data.dataset_zarr import _transform_channel, half_chunk_ids

rng = np.random.default_rng(0)

# --- (c) split halves: disjoint, never touching, parity right, raises when empty
n_total = 175_320  # RainShift training record
ids = list(range(-(-n_total // 200)))
h0, h1 = half_chunk_ids(ids, 0, n_total), half_chunk_ids(ids, 1, n_total)
assert not set(h0) & set(h1)
for half, hs in ((0, h0), (1, h1)):
    for c in hs:
        t0, t1 = c * 200, min((c + 1) * 200, n_total) - 1
        assert t0 // 8760 == t1 // 8760 and (t0 // 8760) % 2 == half
s0, s1 = set(h0), set(h1)
assert not any(c + 1 in s1 or c - 1 in s1 for c in s0), "halves touch"
# the four aligned boundaries (every fifth year, 219 chunks): both neighbours dropped
for k in (5, 10, 15, 20):
    c = k * 8760 // 200
    if c < len(ids):
        assert c not in s0 | s1 and c - 1 not in s0 | s1, k
straddle = [c for c in ids if (c * 200) // 8760 != (min((c + 1) * 200, n_total) - 1) // 8760]
assert not set(straddle) & (s0 | s1)
try:
    half_chunk_ids([0, 1], 1, n_total)
    raise AssertionError("empty half must raise")
except ValueError:
    pass
assert half_chunk_ids([], 0, n_total) == []
print(f"half_chunk_ids: {len(h0)} + {len(h1)} of {len(ids)} chunks, halves never touch")

# --- (c) pair selection: cross-year D, (h0, h1) N, Latin quadruples
keys = [f"{d}/h{h}r{r}" for d in ("A", "B") for h in (0, 1) for r in range(3)]
seed = {k: half_seed(int(k[3]), int(k[5])) for k in keys}
assert sorted(set(seed.values())) == list(range(42, 48))
g = half_pairs(keys, "A", "B")
assert len(g["D"]) == 18 and len(g["NA"]) == 9 and len(g["NB"]) == 9
for a, b in g["D"] + g["NA"] + g["NB"]:
    assert a[3] != b[3], ("cross-year only", a, b)
    assert seed[a] != seed[b], ("no shared seed", a, b)
quads = latin_quadruples(keys, "A", "B")
used = [m for q in quads for m in q["models"]]
assert len(used) == len(set(used)) == 12, "every model once"
for q in quads:
    assert len({seed[m] for m in q["models"]}) == 4, q
    assert all(a[3] != b[3] for a, b in q["D"]), q
sp = seed_pairs(["A/s42", "A/s43", "B/s42", "B/s43"], "A", "B")
assert sorted(sp["D"]) == [("A/s42", "B/s43"), ("A/s43", "B/s42")], sp
print("pairs: D cross-year only, no shared seed, Latin quadruples disjoint")

# --- (b) spaces against the dataset transform on raw values
stats = {"cape": [300.0, 80.0], "tp": [-5.0, 3.0], "z": [10.5, 0.2], "precipitation": [-6.0, 3.5]}
raw = {"cape": rng.gamma(1.0, 300.0, (4, 5, 5)), "lsm": rng.random((4, 5, 5)),
       "tp": np.where(rng.random((4, 5, 5)) < 0.5, 0.0, rng.gamma(0.5, 0.002, (4, 5, 5)))}  # m
names = list(raw)
z = np.stack([_transform_channel(raw[v], v, *(stats[v] if v in stats else (0.0, 1.0))) for v in names], 1)
ph = to_phys(torch.from_numpy(z).double(), stats, names).numpy()
assert np.allclose(ph[:, 0], raw["cape"], rtol=1e-5, atol=1e-2)
assert np.array_equal(ph[:, 1], raw["lsm"].astype(np.float32).astype(np.float64)), "lsm passes through"
assert np.allclose(ph[:, 2], np.log(1000.0 * raw["tp"] + 1e-5), atol=1e-5), "tp: log(1000 tp + 1e-5)"
p_mm = np.where(rng.random((3, 6, 6)) < 0.5, 0.0, rng.lognormal(-1, 1.5, (3, 6, 6)))
zp = _transform_channel(p_mm, "precipitation", *stats["precipitation"])
back = to_mm(torch.from_numpy(zp).double(), stats).numpy()
assert np.allclose(back, p_mm, rtol=1e-4, atol=1e-6), "to_mm inverts the target transform"
assert np.allclose(back, inverse_transform(zp.astype(np.float64), "precipitation", stats), rtol=1e-10, atol=1e-12)
sB = {"cape": [100.0, 50.0], "tp": [-3.0, 2.0], "z": [10.0, 0.3]}
a = torch.from_numpy(ph)
assert torch.allclose(to_phys(from_phys(a, sB, names), sB, names), a, atol=1e-10)
print("spaces: to_phys / to_mm match _transform_channel on raw values")

# --- AUC: perfect, chance, ties
assert auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
assert auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.5
assert abs(auc(rng.random(4000), rng.integers(0, 2, 4000)) - 0.5) < 0.03

# --- propensity: separable -> AUC ~ 1; same distribution -> AUC ~ 0.5; unequal n is balanced
n, d = 2000, 20
gr = np.repeat(np.arange(n // 200), 200)
_, _, a_sep = oof_propensity(rng.normal(0, 1, (n, d)), rng.normal(3, 1, (n, d)), gr, gr)
pa, pb, a_null = oof_propensity(rng.normal(0, 1, (n, d)), rng.normal(0, 1, (n, d)), gr, gr)
assert a_sep > 0.99 and abs(a_null - 0.5) < 0.05, (a_sep, a_null)
assert (pb >= 0.1).mean() > 0.95
_, pb_u, _ = oof_propensity(rng.normal(0, 1, (4 * n, d)), rng.normal(0, 1, (n, d)),
                            np.repeat(np.arange(4 * n // 200), 200), gr)
assert abs(np.median(pb_u) - 0.5) < 0.1, ("balanced prior", np.median(pb_u))
print(f"propensity: AUC separable {a_sep:.3f}, null {a_null:.3f}, unequal-n median {np.median(pb_u):.2f}")


# --- decomposition with i.i.d. noise: shared -> R ~ 1, b2 ~ 0; shifted -> b2 ~ bias^2
def simulate(bias, sd_a=0.3, sd_b=0.5, n=4000, pix=64):
    f_star = rng.normal(0, 1, (n, pix))
    ks = ["A/h0r0", "A/h1r0", "B/h0r0", "B/h1r0"]
    L = np.stack([f_star + bias + rng.normal(0, sd_a, (n, pix)),
                  f_star + bias + rng.normal(0, sd_a, (n, pix)),
                  f_star + rng.normal(0, sd_b, (n, pix)),
                  f_star + rng.normal(0, sd_b, (n, pix))])
    pairs = list(itertools.combinations(range(4), 2))
    dis = pairwise(torch.from_numpy(L)[..., None], pairs).numpy()
    D, NA, NB = vectors(ks, pairs, dis, half_pairs(ks, "A", "B"))
    block = np.repeat(np.arange(n // 200), 200) // 4
    return summarise(D, NA, NB, block, np.ones(n, bool), n_boot=200)


s0, s1 = simulate(0.0), simulate(0.4)
assert abs(s0["R"] - 1) < 0.03 and abs(s0["b2"]) < 0.01 and s0["R_ci"][0] < 1 < s0["R_ci"][1], s0
assert abs(s1["b2"] - 0.16) < 0.01 and s1["R_ci"][0] > 1, s1
print(f"decomposition: shared R {s0['R']:.3f} b2 {s0['b2']:.4f}; bias 0.4 R {s1['R']:.3f} b2 {s1['b2']:.4f}")

# --- (a) smooth per-model error fields: single-quadruple CI vs replicate spread
side, n_s, n_rep = 16, 2000, 200
yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
basis = np.stack([np.cos(np.pi * p * (xx + 0.5) / side) * np.cos(np.pi * q * (yy + 0.5) / side)
                  for p in range(5) for q in range(5)]).reshape(25, -1)  # 25 smooth modes
basis /= np.sqrt((basis ** 2).mean(1, keepdims=True))
block = np.arange(n_s) // 25 // 4  # 80 "chunks" of 25 samples, blocks of 4
pairs12 = list(itertools.combinations(range(len(keys)), 2))
cover_single, cover_rep, widths = 0, 0, []
for _ in range(n_rep):
    fields = rng.normal(0, 0.3 / 5, (len(keys), 25)) @ basis  # (12, pix), sd ~ 0.3 per pixel
    # disagreement per sample: fixed field difference plus i.i.d. per-sample noise
    noise = rng.normal(0, 0.1, (len(keys), n_s, 8))
    dis = np.stack([((fields[a] - fields[b]) ** 2).mean() + ((noise[a] - noise[b]) ** 2).mean(1)
                    for a, b in pairs12])
    qs = [summarise(*vectors(keys, pairs12, dis, q), block, np.ones(n_s, bool), n_boot=200)
          for q in latin_quadruples(keys, "A", "B")]
    cover_single += qs[0]["R_ci"][0] <= 1 <= qs[0]["R_ci"][1]
    rep = replicate_summary(qs)["R"]
    cover_rep += rep["t_ci"][0] <= 1 <= rep["t_ci"][1]
    widths.append(qs[0]["R_ci"][1] - qs[0]["R_ci"][0])
print(f"smooth error fields, {n_rep} draws: single-quadruple block CI covers R=1 in "
      f"{cover_single / n_rep:.0%} (median width {np.median(widths):.3f}); "
      f"t interval over 3 quadruples covers in {cover_rep / n_rep:.0%}")
assert cover_single / n_rep < 0.5, "the single-replicate CI should under-cover"
assert cover_rep / n_rep > 0.8, "the replicate interval should roughly cover"

# --- empty mask
assert summarise(np.ones(10), np.ones(10), np.ones(10), np.zeros(10), np.zeros(10, bool)) == {"n": 0}
print("test_conditional_shift: all passed")
