"""
Tests for the DANN weighting in train_unet.train_one_epoch (audit A2), on a
tiny DualEncoderUNet (base_features 4, 32x32) with one fixed batch.

Run: python test_dann_ratio.py
Exits non-zero on any failure. No GPU required (CPU, a few seconds).

Checks:
  1. --uda_grad_ratio 0 keeps the old DANN step exactly: the parameters after
     one train_one_epoch step equal those after a hand-written
     task + dann_weight * BCE step, the discriminator is optimised, and the
     logged weight is dann_weight.
  2. uda_method none is unchanged: same comparison against task-only.
  3. grad_ratio > 0 on DANN: the achieved encoder share, measured with the
     weight in force, is grad_ratio * alpha (the GRL ramp applies on top).
  4. The matched weight does not depend on the GRL alpha (it is divided out).
  5. The cap: an absurd ratio hits _LAM_CAP and is counted.
"""

import copy
import sys

import torch
import torch.nn as nn

from models.unet import DualEncoderUNet
from train_unet import _LAM_CAP, train_one_epoch
from uda import DomainDiscriminator, dann_grl_schedule, dann_loss

DEV = torch.device("cpu")
B, HW, F = 4, 32, 4          # bottleneck: 16 F = 64 channels at 2x2


def check(name, cond):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        raise AssertionError(name)


def _setup(seed=0, lr=0.0):
    torch.manual_seed(seed)
    model = DualEncoderUNet(dynamic_channels=9, static_channels=2, out_channels=1, base_features=F)
    disc = DomainDiscriminator(in_dim=16 * F, hidden_dim=16)
    disc.eval()  # no dropout: the steps below must be deterministic
    opt = torch.optim.AdamW(list(model.parameters()) + list(disc.parameters()), lr=lr, weight_decay=1e-3)
    return model, disc, opt


def _batches(seed=1):
    g = torch.Generator().manual_seed(seed)
    src = (torch.randn(B, 9, HW, HW, generator=g), torch.randn(B, 2, HW, HW, generator=g),
           torch.randn(B, 1, HW, HW, generator=g))
    tgt = (torch.randn(B, 9, HW, HW, generator=g) + 1.0, torch.randn(B, 2, HW, HW, generator=g),
           torch.randn(B, 1, HW, HW, generator=g))
    return [src], [tgt]


def _run(model, disc, opt, method, epoch=5, total=25, grad_ratio=0.0, lam_state=None,
         dann_weight=5e-4, lambda_eff=0.1):
    src, tgt = _batches()
    return train_one_epoch(
        model, src, tgt, opt, nn.MSELoss(), {"disc": disc}, method, lambda_eff, 0.01, DEV,
        epoch, total, dann_weight=dann_weight, grad_ratio=grad_ratio, lam_state=lam_state,
    )


def _manual_step(model, disc, opt, method, epoch=5, total=25, dann_weight=5e-4, lambda_eff=0.1):
    """The pre-fix train_one_epoch step, written out."""
    (x_s, s_s, y_s), = _batches()[0]
    (x_t, s_t, _), = _batches()[1]
    model.train()
    crit = nn.MSELoss()
    if method == "dann":
        alpha = dann_grl_schedule(epoch, total)
        pred_s, fs = model(x_s, s_s, extract_features=True)
        ft = model(x_t, s_t, extract_features=True)[1]
        loss = crit(pred_s, y_s) + dann_weight * dann_loss(disc, fs["bottleneck"], ft["bottleneck"], alpha)
    else:
        pred_s = model(x_s, s_s)
        loss = crit(pred_s, y_s) + lambda_eff * torch.tensor(0.0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def _same_params(a, b):
    # Exact: after one Adam step the update is ~lr * sign(g), so only an exact
    # comparison sees a changed weight on the adversarial term.
    same = all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    if not same:
        d = max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))
        print(f"    max |param difference| = {d:.3e}")
    return same


def _twin_step(method):
    m1, d1, o1 = _setup(lr=1e-3)
    m2, d2 = copy.deepcopy(m1), copy.deepcopy(d1)
    o2 = torch.optim.AdamW(list(m2.parameters()) + list(d2.parameters()), lr=1e-3, weight_decay=1e-3)
    d0 = copy.deepcopy(d1)
    out = _run(m1, d1, o1, method)
    _manual_step(m2, d2, o2, method)
    return out, (m1, d1), (m2, d2), d0


def main():
    print("test 1: grad_ratio 0 keeps the fixed dann_weight step")
    out, (m1, d1), (m2, d2), d0 = _twin_step("dann")
    check("model parameters equal the hand-written step", _same_params(m1, m2))
    check("discriminator parameters equal the hand-written step", _same_params(d1, d2))
    check("discriminator parameters were updated", not _same_params(d1, d0))
    check(f"logged lam is dann_weight ({out['lam']:.2e})", abs(out["lam"] - 5e-4) < 1e-12)
    check("no share logged when not matched", "share" not in out)

    print("test 2: uda_method none is unchanged")
    out, (m1, _), (m2, _), _ = _twin_step("none")
    check("model parameters equal the task-only step", _same_params(m1, m2))
    check(f"logged lam is lambda_eff ({out['lam']})", out["lam"] == 0.1)
    check("no share logged", "share" not in out)

    print("test 3: grad_ratio > 0 on DANN, achieved share = ratio * alpha")
    ratio, epoch = 0.2, 5
    alpha = dann_grl_schedule(epoch, 25)
    model, disc, opt = _setup(lr=0.0)   # lr 0: the second step sees the same network
    state = {}
    first = _run(model, disc, opt, "dann", epoch=epoch, grad_ratio=ratio, lam_state=state)
    check("first probe logs share 0 (no weight in force yet)", first["share"] == 0.0)
    lam1 = state["lam"]
    second = _run(model, disc, opt, "dann", epoch=epoch, grad_ratio=ratio, lam_state=state)
    rel = abs(second["share"] - ratio * alpha) / (ratio * alpha)
    check(f"share {second['share']:.5f} vs ratio*alpha {ratio * alpha:.5f} (rel {rel:.1e})", rel < 1e-4)
    check(f"logged lam is the matched weight ({second['lam']:.4g})", abs(second["lam"] - lam1) < 1e-6 * lam1)
    check("not capped", second["capped"] == 0)

    print("test 4: the matched weight does not depend on alpha")
    model, disc, opt = _setup(lr=0.0)
    state2 = {}
    _run(model, disc, opt, "dann", epoch=2, grad_ratio=ratio, lam_state=state2)
    rel = abs(state2["lam"] - lam1) / lam1
    check(f"lam at epoch 2 {state2['lam']:.5g} = lam at epoch 5 {lam1:.5g} (rel {rel:.1e})", rel < 1e-4)

    print("test 5: cap")
    model, disc, opt = _setup(lr=0.0)
    state3 = {}
    out = _run(model, disc, opt, "dann", grad_ratio=1e12, lam_state=state3)
    check("capped counted", out["capped"] == 1)
    check(f"lam clipped to _LAM_CAP ({state3['lam']})", state3["lam"] == _LAM_CAP)

    print("\nAll DANN weighting tests passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nDANN TEST FAILED: {e}", file=sys.stderr)
        sys.exit(1)
