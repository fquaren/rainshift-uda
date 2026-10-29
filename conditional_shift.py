"""Model-based conditional-shift test (PLAN.md P2).

Question: on inputs both domains produce, do the conditionals P(Y|X) of two
RainShift domains differ? The binned test (conditional_shift_binned.py)
conditions on CAPE and TCW only. This one lets trained networks condition on
all inputs, and must control four things to be read at all.

1. Units. Every model sees its inputs in ITS OWN source-normalised space, as
   in evaluate.py, and every prediction is mapped back to mm before any
   comparison. Discrepancies are mean squared differences in log(P + floor)
   at the D1 floors (0.1 mm/h headline, 0.01 as a check).

2. Estimation noise. Two estimates of ONE conditional disagree too. For
   models f_A, f_B whose errors around a shared f* are independent and
   centred,
       E d(f_A, f_B) = (E d(f_A, f_A') + E d(f_B, f_B')) / 2,
   with f_A' a second estimate from A's data. So
       R  = D_AB / ((N_A + N_B) / 2)     is 1 under a shared conditional, and
       b2 = D_AB - (N_A + N_B) / 2       estimates the mean squared difference
                                         of the two conditional means.
   The identity needs every pair entering D and N to be independent. Hence:
     half  models trained and selected on alternate 365-day blocks of the
           training years (train_unet.py --train_half), 3 replicates per
           (domain, half): half 0 with seeds 42/43/44, half 1 with 45/46/47,
           trees unet_seed<S>_half<h>/. --seed also sets their shuffle, so
           each has its own init AND data order. N_A pairs (A/h0, A/h1); D_AB
           uses CROSS-YEAR pairs only, (A/h0, B/h1) and (A/h1, B/h0): no pair
           in D or N shares years or a seed. The primary null.
     seed  the seeded source-only models (42/43/44): initialisation only,
           since --seed never reaches their dataset. D from pairs of
           different seeds. Too small a null; kept to show by how much.
   Training variance. One quadruple of half models is one draw of four
   smooth error fields; the chunk bootstrap holds them fixed and its CI
   under-covers (test_conditional_shift.py simulates this). So R and b2 are
   reported (a) pooled over all cross-year / within-domain pairs, with the
   block bootstrap, and (b) per disjoint "Latin" quadruple q = (A/h0 r_q,
   A/h1 r_q+1, B/h0 r_q+2, B/h1 r_q) mod 3: every model used once, the four
   seeds of a quadruple all different, so the three values are independent
   draws; their spread and a t interval (2 df) are the training-variance
   uncertainty. Read (b) before the bootstrap CI.

3. Support. Disagreement where A has never seen such inputs is
   extrapolation, not a different conditional. A cross-fitted logistic
   classifier on 4 x 4 block means of the dynamic inputs (physical units)
   gives P(A | x) for B's test samples (folds are whole chunks; each fold's
   training set is subsampled to equal n per domain, so P(A | x) = 0.5 means
   no information). R is reported on B's samples with P(A | x) >= 0.1, 0.3
   and 0.5: if R falls towards 1 as the overlap tightens, the excess is
   extrapolation. dyn7 (no sp, tisr, which locate the patch; see P19)
   defines the overlap; dyn9 is reported for its AUC. Statics, sp and tisr
   are never inside the overlap; the optional masked arm (--masked_suffix,
   half models trained with --mask_inputs lsm z sp tisr, kind "g") removes
   them from the models instead.

4. Power. Positive control on --control_domain B: B's own half models with
   wet predictions (>= 0.1 mm/h) multiplied by --control_factor play "A".
   The conditional difference is then known: b2 should be close to the
   mean of d(T f, f) over B's half models (exact for a log-space shift far
   above the floor, approximate near it), and R's CI should exclude 1.

Reading. R_half ~ 1 on the overlap is evidence for a shared conditional
there. R_half >> 1 is necessary for conditional shift but not sufficient: a
network's approximation error is weighted towards its own input
distribution, and unmasked models still see statics, sp, tisr and sub-block
structure the overlap does not condition on. The test is asymmetric in the
same way as the binned one. b2_half is the bias difference of HALF-data
models, not of the full models. Uncertainty: bootstrap over blocks of
--boot_block consecutive test chunks (4 = ~33 days), plus the replicate
spread. Crash safety: per_sample_<B>.npz is written as each domain finishes;
--from_npz DIR reruns the analysis only.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.dataset_zarr import (
    _LOG_EPS,
    _PASSTHROUGH_VARS,
    DEFAULT_INPUT_VARS,
    STATIC_VARS,
    TARGET_VAR,
)

DOMAINS = ["europe_west", "horn-of-africa", "melanesia"]
FEATURE_SETS = {
    "dyn9": list(DEFAULT_INPUT_VARS),
    "dyn7": [v for v in DEFAULT_INPUT_VARS if v not in ("sp", "tisr")],
}
POOL = 20  # 80 x 80 inputs -> 4 x 4 block means
_CHUNK = 200
OVERLAPS = (0.1, 0.3, 0.5)
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def half_seed(half, rep, base=42, n_rep=3):
    """Seed of half model (half, rep): half 0 -> base..base+n_rep-1, half 1 the next n_rep."""
    return base + half * n_rep + rep


# ---------------------------------------------------------------------------
#  Spaces
# ---------------------------------------------------------------------------


def _ms(stats, var):
    m, s = stats[var]
    if isinstance(m, list):  # same quirk as data.dataset.inverse_transform
        m, s = m
    return float(m), float(s) + 1e-8


def to_phys(z, stats, names):
    """Undo each channel's z-score. The log of tp, cp and z stays: it is the
    same in every domain, so this space is shared. lsm passes through."""
    out = z.clone()
    for c, v in enumerate(names):
        if v not in _PASSTHROUGH_VARS:
            m, s = _ms(stats, v)
            out[:, c] = z[:, c] * s + m
    return out


def from_phys(a, stats, names):
    out = a.clone()
    for c, v in enumerate(names):
        if v not in _PASSTHROUGH_VARS:
            m, s = _ms(stats, v)
            out[:, c] = (a[:, c] - m) / s
    return out


def to_mm(z, stats):
    """Model output (std space of `stats`) to mm, as inverse_transform."""
    m, s = _ms(stats, TARGET_VAR)
    return (torch.exp(z * s + m) - _LOG_EPS).clamp_(min=0.0)


def pairwise(L, pairs):
    """L: (K, n, ...) log fields. Per-sample mean squared difference for each
    pair of models, (len(pairs), n)."""
    red = tuple(range(1, L.ndim - 1))
    return torch.stack([((L[a] - L[b]) ** 2).mean(red) for a, b in pairs])


# ---------------------------------------------------------------------------
#  Support: cross-fitted domain classifier
# ---------------------------------------------------------------------------


def auc(score, label):
    """Mann-Whitney AUC with tied scores given their average rank."""
    order = np.argsort(score, kind="mergesort")
    _, start, counts = np.unique(score[order], return_index=True, return_counts=True)
    ranks = np.empty(len(score))
    ranks[order] = np.repeat(start + (counts + 1) / 2.0, counts)
    pos = label == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logistic(X, y, l2):
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(X @ w + b, y) + l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def oof_propensity(Fa, Fb, ga, gb, k=5, l2=1e-3, seed=0, balance=True):
    """Out-of-fold P(A | x) for A's and B's samples, and the OOF AUC. Folds
    are whole chunks, assigned within each domain, so a fold never trains on
    the autocorrelated neighbours of the samples it scores. With `balance`,
    each fold's training set is subsampled to equal n per domain, so the
    prior is 1/2 and the thresholds mean the same for every pair."""
    rng = np.random.default_rng(seed)

    def folds(g):
        u = np.unique(g)
        f = dict(zip(rng.permutation(u), np.arange(len(u)) % k))
        return np.array([f[x] for x in g])

    X = np.concatenate([Fa, Fb]).astype(np.float64)
    y = np.r_[np.ones(len(Fa)), np.zeros(len(Fb))]
    fold = np.r_[folds(ga), folds(gb)]
    p = np.empty(len(X))
    for f in range(k):
        tr, te = np.flatnonzero(fold != f), fold == f
        if balance:
            ia, ib = tr[y[tr] == 1], tr[y[tr] == 0]
            n = min(len(ia), len(ib))
            tr = np.r_[rng.choice(ia, n, replace=False), rng.choice(ib, n, replace=False)]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        w, b = fit_logistic((X[tr] - mu) / sd, y[tr], l2)
        p[te] = torch.sigmoid(torch.as_tensor((X[te] - mu) / sd) @ w + b).numpy()
    return p[: len(Fa)], p[len(Fa):], auc(p, y)


# ---------------------------------------------------------------------------
#  Pair selection and decomposition
# ---------------------------------------------------------------------------


def _members(keys, dom, kind, half):
    """Keys '<dom>/<kind><half>r<rep>', ordered by rep."""
    return sorted(k for k in keys if k.startswith(f"{dom}/{kind}{half}r"))


def half_pairs(keys, A, B, kind="h"):
    """All pairs of the half-model null: D from cross-year pairs only
    (A/h0 with B/h1, A/h1 with B/h0), N_A and N_B from (h0, h1) pairs."""
    a0, a1, b0, b1 = (_members(keys, d, kind, h) for d, h in ((A, 0), (A, 1), (B, 0), (B, 1)))
    return {"D": list(itertools.product(a0, b1)) + list(itertools.product(a1, b0)),
            "NA": list(itertools.product(a0, a1)), "NB": list(itertools.product(b0, b1))}


def latin_quadruples(keys, A, B, kind="h"):
    """Disjoint quadruples (A/h0[q], A/h1[q+1], B/h0[q+2], B/h1[q]), indices
    mod R: every model is used once, so the R values are independent, and
    with R = 3 the four models of a quadruple have four different seeds."""
    a0, a1, b0, b1 = (_members(keys, d, kind, h) for d, h in ((A, 0), (A, 1), (B, 0), (B, 1)))
    R = len(a0)
    if not R or not (len(a1) == len(b0) == len(b1) == R):
        raise ValueError(f"unequal replicate counts for {A}/{B} kind {kind}: "
                         f"{len(a0)} {len(a1)} {len(b0)} {len(b1)}")
    out = []
    for q in range(R):
        ma0, ma1, mb0, mb1 = a0[q], a1[(q + 1) % R], b0[(q + 2) % R], b1[q]
        out.append({"D": [(ma0, mb1), (ma1, mb0)], "NA": [(ma0, ma1)], "NB": [(mb0, mb1)],
                    "models": [ma0, ma1, mb0, mb1]})
    return out


def seed_pairs(keys, A, B):
    """Seed null: D from A x B pairs of different seeds, N within a domain."""
    ka = sorted(k for k in keys if k.startswith(f"{A}/s"))
    kb = sorted(k for k in keys if k.startswith(f"{B}/s"))
    return {"D": [(a, b) for a in ka for b in kb if a.split("/s")[1] != b.split("/s")[1]],
            "NA": list(itertools.combinations(ka, 2)), "NB": list(itertools.combinations(kb, 2))}


def vectors(keys, pairs, dis, groups):
    """Per-sample D_AB, N_A, N_B: means of the pairwise rows named in groups."""
    pidx = {frozenset((keys[a], keys[b])): i for i, (a, b) in enumerate(pairs)}
    return tuple(np.mean([dis[pidx[frozenset(p)]] for p in groups[g]], axis=0) for g in ("D", "NA", "NB"))


def summarise(D, NA, NB, block, mask, n_boot=1000, seed=0):
    """R and b2 over the samples in `mask`, with a bootstrap over `block`
    ids (blocks of consecutive chunks; every block of the domain is drawn,
    so the overlap count varies). Holds the models fixed."""
    out = {"n": int(mask.sum())}
    if out["n"] == 0:
        return out
    u, inv = np.unique(block, return_inverse=True)
    w = mask.astype(np.float64)
    sums = np.stack([np.bincount(inv, weights=w * q, minlength=len(u)) for q in (D, NA, NB)])
    cnt = np.bincount(inv, weights=w, minlength=len(u))

    def stats(s, n):
        d, na, nb = s / n
        null = (na + nb) / 2
        return d, na, nb, d / null, d - null

    d, na, nb, r, b2 = stats(sums.sum(1), cnt.sum())
    out.update(D=d, N_A=na, N_B=nb, R=r, b2=b2)
    counts = np.random.default_rng(seed).multinomial(len(u), np.full(len(u), 1 / len(u)), size=n_boot)
    n_b = counts @ cnt
    ok = n_b > 0
    bs = np.array(stats(sums @ counts[ok].T, n_b[ok]))  # (5, n_ok)
    out["R_ci"] = np.percentile(bs[3], [2.5, 97.5]).tolist()
    out["b2_ci"] = np.percentile(bs[4], [2.5, 97.5]).tolist()
    return out


def replicate_summary(rows):
    """Spread of R and b2 over independent quadruples: values, mean, sd and
    a t interval on the mean (n - 1 df)."""
    out = {}
    for k in ("R", "b2"):
        v = np.array([r[k] for r in rows if k in r], float)
        if v.size == 0:
            continue
        e = {"values": v.tolist(), "mean": float(v.mean())}
        if v.size >= 2:
            sd = float(v.std(ddof=1))
            h = _T975.get(v.size - 1, 1.96) * sd / np.sqrt(v.size)
            e.update(sd=sd, t_ci=[e["mean"] - h, e["mean"] + h])
        out[k] = e
    return out


def arm_summary(keys, pairs, dis, A, B, kind, block, mask, n_boot, seed):
    """Pooled half-model decomposition plus the per-quadruple values."""
    pooled = summarise(*vectors(keys, pairs, dis, half_pairs(keys, A, B, kind)), block, mask, n_boot, seed)
    quads = [summarise(*vectors(keys, pairs, dis, q), block, mask, n_boot, seed)
             for q in latin_quadruples(keys, A, B, kind)]
    return {"pooled": pooled, "quadruples": quads, "replicates": replicate_summary(quads)}


# ---------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------


def model_trees(args):
    """{kind/rep key suffix: tree} of every model the run needs."""
    root = Path(args.results_root)
    trees = {f"s{s}": root / f"unet_seed{s}" for s in args.seeds}
    if not args.skip_halves:
        for h in (0, 1):
            for r in range(args.half_reps):
                s = half_seed(h, r, args.half_base_seed, args.half_reps)
                trees[f"h{h}r{r}"] = root / f"unet_seed{s}_half{h}"
                if args.masked_suffix:
                    trees[f"g{h}r{r}"] = root / f"unet_seed{s}_half{h}{args.masked_suffix}"
    return trees


def load_models(args, dev):
    from evaluate import load_model  # keep vectors of masked checkpoints

    models = {}
    for D in DOMAINS:
        for t, tree in model_trees(args).items():
            ck = tree / f"{D}__to__{D}__none" / "best.pt"
            if not ck.exists():
                raise SystemExit(f"Missing checkpoint {ck}")
            models[f"{D}/{t}"] = (D, load_model("unet", ck, dev))
            print(f"  {D}/{t}: {ck}")
    return models


@torch.no_grad()
def score_domain(B, models, stats, args, dev):
    """Run every model on B's test chunks. Per sample: each model's error
    against the truth and every pair's disagreement, both per floor, plus
    the pooled input features. On the control domain, B's half models with
    wet predictions scaled are added as 'ctrl/h<h>r<r>'."""
    from data.dataset_zarr import ClimateSRDatasetZarr
    from evaluate import FLOORS_MM
    from finetune_unet import spread_chunks

    ds = ClimateSRDatasetZarr(str(Path(args.data_root) / B), "test", stats=stats[B])
    chunks = spread_chunks(ds.chunk_ids, args.n_chunks, args.seed)
    base = list(models)
    ctrl = ([f"ctrl/{k.split('/')[1]}" for k in base if k.startswith(f"{B}/h")]
            if B == args.control_domain else [])
    keys = base + ctrl
    pairs = list(itertools.combinations(range(len(keys)), 2))
    s_phys = to_phys(ds.static[None].to(dev, torch.float64), stats[B], STATIC_VARS)
    s_in = {D: from_phys(s_phys, stats[D], STATIC_VARS).float() for D in DOMAINS}
    feat_idx = {n: [DEFAULT_INPUT_VARS.index(v) for v in vs] for n, vs in FEATURE_SETS.items()}
    rec = {"time": [], "chunk": [], **{f"feat_{n}": [] for n in FEATURE_SETS},
           **{f"err_{f}": [] for f in FLOORS_MM}, **{f"dis_{f}": [] for f in FLOORS_MM}}

    for j, c in enumerate(chunks):
        x_np, y_np = ds._read_chunk(c)
        for i in range(0, len(x_np), args.batch_size):
            x = torch.from_numpy(x_np[i:i + args.batch_size]).to(dev, torch.float64)
            y = torch.from_numpy(y_np[i:i + args.batch_size]).to(dev, torch.float64)
            n = len(x)
            x_phys = to_phys(x, stats[B], DEFAULT_INPUT_VARS)
            pooled = F.avg_pool2d(x_phys, POOL)
            for name, idx in feat_idx.items():
                rec[f"feat_{name}"].append(pooled[:, idx].flatten(1).float().cpu().numpy())
            preds = {}
            for k in base:
                D, model = models[k]
                z = model(from_phys(x_phys, stats[D], DEFAULT_INPUT_VARS).float(),
                          s_in[D].expand(n, -1, -1, -1).contiguous())
                preds[k] = to_mm(z.double(), stats[D])
            for k in ctrl:
                p = preds[f"{B}/{k.split('/')[1]}"]
                preds[k] = torch.where(p >= args.control_wet, p * args.control_factor, p)
            y_mm = to_mm(y, stats[B])
            for f, v in FLOORS_MM.items():
                L = torch.log(torch.stack([preds[k] for k in keys]) + v)
                rec[f"err_{f}"].append(((L - torch.log(y_mm + v)) ** 2).mean((2, 3, 4)).float().cpu().numpy())
                rec[f"dis_{f}"].append(pairwise(L, pairs).float().cpu().numpy())
            rec["time"].append(c * _CHUNK + i + np.arange(n))
            rec["chunk"].append(np.full(n, c))
        if j % 10 == 0 or j == len(chunks) - 1:
            print(f"  {B}: chunk {j + 1}/{len(chunks)}", flush=True)

    out = {k: np.concatenate(v, axis=-1 if k.startswith(("err_", "dis_")) else 0) for k, v in rec.items()}
    out["keys"], out["pairs"] = np.array(keys), np.array(pairs)
    return out


def load_npz(d):
    rec = {}
    for B in DOMAINS:
        z = np.load(Path(d) / f"per_sample_{B}.npz")
        rec[B] = {k: z[k] for k in z.files}
    return rec


# ---------------------------------------------------------------------------
#  Analysis
# ---------------------------------------------------------------------------


def _kinds(keys):
    return [k for k in ("h", "g") if any(f"/{k}0r" in x for x in keys)]


def positive_control(r, B, floors, args):
    """ctrl (B's half models, wet x factor) against B's half models."""
    keys, pairs = list(map(str, r["keys"])), [tuple(p) for p in r["pairs"]]
    block = r["chunk"] // args.boot_block
    allm = np.ones(len(block), bool)
    pidx = {frozenset((keys[a], keys[b])): i for i, (a, b) in enumerate(pairs)}
    res = {}
    for f in floors:
        dis = r[f"dis_{f}"]
        arm = arm_summary(keys, pairs, dis, "ctrl", B, "h", block, allm, args.n_boot, args.seed)
        same = [pidx[frozenset((k, f"{B}/{k.split('/')[1]}"))] for k in keys if k.startswith("ctrl/")]
        b2_exp = float(dis[same].mean())
        p = arm["pooled"]
        ok = bool(p["R_ci"][0] > 1 and 0.5 < p["b2"] / b2_exp < 2)
        res[f] = {**arm, "b2_expected": b2_exp, "pass": ok}
        print(f"  control {B} x{args.control_factor} {f:5s}: R {p['R']:.2f} [{p['R_ci'][0]:.2f}, {p['R_ci'][1]:.2f}]"
              f"  b2 {p['b2']:.4f} vs expected {b2_exp:.4f}  {'PASS' if ok else 'FAIL'}")
    return res


def analyse(rec, args, out):
    floors = [k[4:] for k in rec[DOMAINS[0]] if k.startswith("dis_")]
    results = {"args": vars(args), "keys": {B: list(map(str, rec[B]["keys"])) for B in DOMAINS},
               "pairs": {}}
    print(f"\n{'A model -> B inputs':34s} {'AUC9':>5} {'AUC7':>5} set     floor "
          f"{'R_half pooled [95% block CI]':>28} {'R per quadruple':>20} {'b2':>8} {'R_seed':>7}")
    for A, B in itertools.permutations(DOMAINS, 2):
        r = rec[B]
        keys, pairs = list(map(str, r["keys"])), [tuple(p) for p in r["pairs"]]
        block = r["chunk"] // args.boot_block
        info = {}
        for name in FEATURE_SETS:
            pa, pb, a = oof_propensity(rec[A][f"feat_{name}"], rec[B][f"feat_{name}"],
                                       rec[A]["chunk"], rec[B]["chunk"], seed=args.seed)
            info[f"auc_{name}"] = a
            if name == "dyn7":
                prop_b = pb
        np.save(out / f"propensity_dyn7_{A}__on__{B}.npy", prop_b)
        sets = {"all": np.ones(len(prop_b), bool), **{f"ovl{t}": prop_b >= t for t in args.overlaps}}
        info["overlap_frac"] = {s: float(m.mean()) for s, m in sets.items()}
        has_seed = any(k.startswith(f"{A}/s") for k in keys) and any(k.startswith(f"{B}/s") for k in keys)
        for f in floors:
            dis, err = r[f"dis_{f}"], r[f"err_{f}"]
            if has_seed:
                ka = [i for i, k in enumerate(keys) if k.startswith(f"{A}/s")]
                kb = [i for i, k in enumerate(keys) if k.startswith(f"{B}/s")]
                gap_ps = err[ka].mean(0) - err[kb].mean(0)
                sv = vectors(keys, pairs, dis, seed_pairs(keys, A, B))
            for sname, mask in sets.items():
                row = {}
                if has_seed:
                    row["gap_seedmean"] = float(gap_ps[mask].mean()) if mask.any() else None
                    row["err_B_seedmean"] = float(err[kb].mean(0)[mask].mean()) if mask.any() else None
                    row["s"] = summarise(*sv, block, mask, args.n_boot, args.seed)
                for kind in _kinds(keys):
                    row[kind] = arm_summary(keys, pairs, dis, A, B, kind, block, mask, args.n_boot, args.seed)
                info[f"{f}/{sname}"] = row
                h = row.get("h", {}).get("pooled", {})
                hs = (f"{h['R']:7.2f} [{h['R_ci'][0]:5.2f}, {h['R_ci'][1]:5.2f}] n={h['n']:5d}"
                      if "R" in h else f"{'-':>28}")
                q = row.get("h", {}).get("replicates", {}).get("R", {}).get("values", [])
                qs = " ".join(f"{v:5.2f}" for v in q) or "-"
                b2 = f"{h['b2']:8.4f}" if "b2" in h else f"{'-':>8}"
                rs = row.get("s", {}).get("R", float("nan"))
                print(f"{A + ' -> ' + B:34s} {info['auc_dyn9']:5.3f} {info['auc_dyn7']:5.3f} {sname:7s} "
                      f"{f:5s} {hs:>28} {qs:>20} {b2} {rs:7.2f}")
        results["pairs"][f"{A}__on__{B}"] = info

    C = args.control_domain
    if C in rec and any(str(k).startswith("ctrl/") for k in rec[C]["keys"]):
        results["positive_control"] = positive_control(rec[C], C, floors, args)
    else:
        print("  positive control: no ctrl models (halves skipped or control domain absent)")

    (out / "summary.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {out}")
    print(
        "\nReading: R = D_AB / mean(N_A, N_B), b2 = D_AB - mean(N_A, N_B), in log(P + floor) MSE.\n"
        "  D_AB from cross-year half-model pairs only. 'R per quadruple' are three independent draws\n"
        "  (disjoint models); their spread is the training variance the block CI ignores, so a\n"
        "  pooled CI that excludes 1 means little unless the quadruples agree. R ~ 1 on the overlap:\n"
        "  no conditional difference beyond two disjoint sets of years within one domain. R >> 1:\n"
        "  necessary for conditional shift, not sufficient. Watch R across ovl0.1/0.3/0.5: falling\n"
        "  towards 1 means extrapolation. R_seed is initialisation-only and liberal. Check the\n"
        "  positive control passed before reading an R ~ 1 as a shared conditional."
    )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data_root", required=False)
    p.add_argument("--results_root", required=False, help="holds unet_seed<S>/ and unet_seed<S>_half<h>/")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--from_npz", default=None, help="analysis only, from DIR/per_sample_<B>.npz")
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    p.add_argument("--half_base_seed", type=int, default=42)
    p.add_argument("--half_reps", type=int, default=3)
    p.add_argument("--skip_halves", action="store_true", help="seed null only (not the primary test)")
    p.add_argument("--masked_suffix", default="", help="also load unet_seed<S>_half<h><suffix>/ (e.g. _mask_geo)")
    p.add_argument("--n_chunks", type=int, default=88, help="test chunks per domain, spread evenly (88 = all)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--overlaps", type=float, nargs="+", default=list(OVERLAPS))
    p.add_argument("--boot_block", type=int, default=4, help="consecutive test chunks per bootstrap block")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--control_domain", default="europe_west")
    p.add_argument("--control_factor", type=float, default=1.5)
    p.add_argument("--control_wet", type=float, default=0.1, help="mm/h; predictions >= this are scaled")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    if out.exists():
        raise SystemExit(f"{out} exists. Not overwriting.")
    if args.from_npz:
        rec = load_npz(args.from_npz)
        out.mkdir(parents=True)
        analyse(rec, args, out)
        return

    if not (args.data_root and args.results_root):
        p.error("--data_root and --results_root are required unless --from_npz")
    from data.dataset import load_domain_stats

    trees = model_trees(args)  # fail on a missing checkpoint before any work
    for D in DOMAINS:
        for t in trees.values():
            if not (t / f"{D}__to__{D}__none" / "best.pt").exists():
                raise SystemExit(f"Missing checkpoint {t / f'{D}__to__{D}__none' / 'best.pt'}")
    out.mkdir(parents=True)
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False

    stats = {D: load_domain_stats(str(Path(args.data_root) / D)) for D in DOMAINS}
    print("Models:")
    models = load_models(args, dev)
    rec = {}
    for B in DOMAINS:
        rec[B] = score_domain(B, models, stats, args, dev)
        np.savez_compressed(out / f"per_sample_{B}.npz", **rec[B])
        print(f"  wrote per_sample_{B}.npz", flush=True)
    analyse(rec, args, out)


if __name__ == "__main__":
    main()
