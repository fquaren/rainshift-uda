"""
Tests for the AFM training changes of 2026-10-29 (audit C), on a tiny
AFMModel (base 4, 32x32) with fixed batches.

Run: python test_afm_uda.py      (inside a job, through scripts/python_env.sh)
Exits non-zero on any failure. No GPU required (CPU, a few seconds).

Checks:
  1. uda_method none is unchanged: one train_one_epoch step equals a
     hand-written total_loss step, and the target loader is never iterated.
  2. DANN with --uda_grad_ratio 0 equals the hand-written
     task + dann_weight * BCE step; the discriminator is in its own group at
     the encoder's LR and wd 1e-3, and is updated.
  3. DANN with grad_ratio > 0: the achieved share on the encoder is
     grad_ratio * alpha, and the matched weight does not depend on alpha.
  4. MMD with grad_ratio > 0 and --mmd_pool grid4 / --mmd_unbiased: share =
     grad_ratio.
  5. AdaBN is label-free and re-estimates the flow-net statistics on the
     sampler's trajectories: 2 * steps calls per batch, max_batches batches.
  6. --uda_grad_ratio on a method without an alignment term is refused.
  7. compute_stats.py refuses to write under raw_data or over a file;
     compute_shift.py refuses --mode normalized.
"""

import copy
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

import train_afm
from models.afm import AFMModel
from train_afm import _adabn_afm, build_optimizer, build_uda, train_one_epoch
from uda import dann_grl_schedule, dann_loss

DEV = torch.device("cpu")
B, HW, F = 4, 32, 4          # encoder bottleneck: 16 F = 64 channels


def check(name, cond):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        raise AssertionError(name)


class _NeverIterated:
    def __iter__(self):
        raise AssertionError("target loader iterated for a method that needs no target batch")


def _batches(seed=1, n=1, nan_labels=False):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        y = torch.randn(B, 1, HW, HW, generator=g)
        out.append((torch.randn(B, 9, HW, HW, generator=g), torch.randn(B, 2, HW, HW, generator=g),
                    y * float("nan") if nan_labels else y))
    return out


def _setup(method, seed=0, lr=0.0, ratio=1.0, mmd_pool="gap", mmd_unbiased=False):
    torch.manual_seed(seed)
    model = AFMModel(9, 2, 1, F)
    uc, ep = build_uda(method, [], DEV, F, mmd_pool, mmd_unbiased)
    if "disc" in uc:
        uc["disc"].eval()   # no dropout: the steps below must be deterministic
    return model, uc, build_optimizer(model, lr, ep, ratio)


def _run(model, uc, opt, method, tgt=None, epoch=5, total=25, grad_ratio=0.0, lam_state=None,
         dann_weight=5e-4, lam_eff=0.1, seed=7):
    # The flow loss draws z and t from the global RNG, and sigma_z is an EMA
    # buffer updated by every forward: fix both so repeated calls match.
    model.sigma_z.fill_(1.0)
    torch.manual_seed(seed)
    tgt = tgt if tgt is not None else _batches(seed=2)
    return train_one_epoch(
        model, _batches(), tgt, opt, uc, method, lam_eff, 0.01, DEV, epoch, total,
        dann_weight=dann_weight, grad_ratio=grad_ratio, lam_state=lam_state,
    )


def _manual_step(model, uc, opt, method, epoch=5, total=25, dann_weight=5e-4, seed=7):
    """The pre-change train_one_epoch step, written out."""
    (xs, ss, ys), = _batches()
    (xt, st, _), = _batches(seed=2)
    model.sigma_z.fill_(1.0)
    torch.manual_seed(seed)
    model.train()
    out = model(xs, ss, x_target=ys, extract_features=(method == "dann"))
    loss = out["total_loss"]
    if method == "dann":
        alpha = dann_grl_schedule(epoch, total)
        _, tf = model(xt, st, extract_features=True)
        loss = loss + dann_weight * dann_loss(uc["disc"], out["features"]["bottleneck"], tf["bottleneck"], alpha)
    else:
        loss = loss + 0.1 * torch.tensor(0.0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def _same(a, b):
    same = all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    if not same:
        d = max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))
        print(f"    max |param difference| = {d:.3e}")
    return same


def _twin(method):
    m1, u1, o1 = _setup(method, lr=1e-3)
    m2, u2 = copy.deepcopy(m1), copy.deepcopy(u1)
    o2 = build_optimizer(m2, 1e-3, list(u2["disc"].parameters()) if "disc" in u2 else [], 1.0)
    u0 = copy.deepcopy(u1)
    tgt = _NeverIterated() if method == "none" else None
    out = _run(m1, u1, o1, method, tgt=tgt)
    _manual_step(m2, u2, o2, method)
    return out, (m1, u1), (m2, u2), u0


def _bn(module):
    return [m for m in module.modules() if isinstance(m, nn.BatchNorm2d)]


def main():
    print("test 1: uda_method none is unchanged and draws no target batch")
    out, (m1, _), (m2, _), _ = _twin("none")
    check("model parameters equal the hand-written step", _same(m1, m2))
    check("no lam / share keys (phase-1 history.csv columns unchanged)", not {"lam", "share"} & set(out))

    print("test 2: DANN, grad_ratio 0, keeps the fixed dann_weight step")
    out, (m1, u1), (m2, u2), u0 = _twin("dann")
    check("model parameters equal the hand-written step", _same(m1, m2))
    check("discriminator parameters equal the hand-written step", _same(u1["disc"], u2["disc"]))
    check("discriminator parameters were updated", not _same(u1["disc"], u0["disc"]))
    m, u, opt = _setup("dann", lr=1e-4, ratio=0.5)
    g = opt.param_groups[2]
    check(f"discriminator group lr {g['lr']:.1e} = encoder lr, wd {g['weight_decay']}",
          g["lr"] == opt.param_groups[0]["lr"] and g["weight_decay"] == 1e-3)

    print("test 3: DANN, grad_ratio > 0: share = ratio * alpha, weight independent of alpha")
    ratio, epoch = 0.2, 5
    alpha = dann_grl_schedule(epoch, 25)
    model, uc, opt = _setup("dann")      # lr 0: the second call sees the same network
    state = {}
    first = _run(model, uc, opt, "dann", epoch=epoch, grad_ratio=ratio, lam_state=state)
    check("first probe logs share 0 (no weight in force yet)", first["share"] == 0.0)
    lam1 = state["lam"]
    second = _run(model, uc, opt, "dann", epoch=epoch, grad_ratio=ratio, lam_state=state)
    rel = abs(second["share"] - ratio * alpha) / (ratio * alpha)
    check(f"share {second['share']:.5f} vs ratio*alpha {ratio * alpha:.5f} (rel {rel:.1e})", rel < 1e-4)
    model, uc, opt = _setup("dann")
    state2 = {}
    _run(model, uc, opt, "dann", epoch=2, grad_ratio=ratio, lam_state=state2)
    rel = abs(state2["lam"] - lam1) / lam1
    check(f"lam at epoch 2 {state2['lam']:.5g} = lam at epoch 5 {lam1:.5g} (rel {rel:.1e})", rel < 1e-4)

    print("test 4: MMD (grid4, unbiased), grad_ratio > 0: share = ratio")
    model, uc, opt = _setup("mmd", mmd_pool="grid4", mmd_unbiased=True)
    state = {}
    _run(model, uc, opt, "mmd", grad_ratio=ratio, lam_state=state)
    second = _run(model, uc, opt, "mmd", grad_ratio=ratio, lam_state=state)
    rel = abs(second["share"] - ratio) / ratio
    check(f"share {second['share']:.5f} vs ratio {ratio} (rel {rel:.1e})", rel < 1e-4)
    check("history keys lam / share / capped present", {"lam", "share", "capped"} <= set(second))

    print("test 5: AdaBN on sampler trajectories, label-free")
    model, _, _ = _setup("adabn")
    steps, max_b, n_b = 3, 2, 3
    _adabn_afm(model, _batches(seed=3, n=n_b, nan_labels=True), DEV, steps=steps, max_batches=max_b)
    enc, flow = _bn(model.encoder), _bn(model.flow_net)
    check("all running statistics finite (NaN labels never read)",
          all(torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all() for m in enc + flow))
    check(f"encoder BN saw every batch ({n_b})", all(int(m.num_batches_tracked) == n_b for m in enc))
    check(f"flow BN saw 2 x {steps} calls x {max_b} batches",
          all(int(m.num_batches_tracked) == 2 * steps * max_b for m in flow))
    check("model left in eval mode", not model.training)

    print("test 6: --uda_grad_ratio refused where there is no alignment term")
    argv = sys.argv
    try:
        sys.argv = ["train_afm.py", "--source_path", "a", "--target_path", "b",
                    "--uda_method", "fda", "--uda_grad_ratio", "0.1"]
        try:
            train_afm.parse_args()
            refused = False
        except SystemExit:
            refused = True
        check("fda + --uda_grad_ratio exits", refused)
    finally:
        sys.argv = argv

    print("test 7: compute_stats.py and compute_shift.py guards")
    from data import compute_stats
    sys.path.insert(0, str(Path(__file__).resolve().parent / "covariate_shift_analysis"))
    import compute_shift

    def _exits(fn, args):
        old = sys.argv
        sys.argv = ["x"] + args
        try:
            fn()
            return False
        except SystemExit:
            return True
        finally:
            sys.argv = old

    raw_out = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift/europe_west/normalization_stats.json"
    check("compute_stats refuses an --out under raw_data",
          _exits(compute_stats.main, ["--domain_path", "/nonexistent", "--out", raw_out]))
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "s.json"
        f.write_text("{}")
        check("compute_stats refuses an existing --out",
              _exits(compute_stats.main, ["--domain_path", "/nonexistent", "--out", str(f)]))
        check("existing --out left untouched", f.read_text() == "{}")
    check("compute_shift refuses --mode normalized",
          _exits(compute_shift.parse_args, ["--data_root", "/nonexistent", "--mode", "normalized"]))

    print("\nAll AFM UDA tests passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nAFM TEST FAILED: {e}", file=sys.stderr)
        sys.exit(1)
