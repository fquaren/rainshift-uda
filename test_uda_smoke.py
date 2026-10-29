"""
Smoke test for the multi-scale MMD loss and the lambda schedules in uda.py.

Run: python test_uda_smoke.py
Exits non-zero on any failure. No GPU required.

Checks:
  1. mmd_multiscale_loss runs over all four encoder levels and is positive
     for shifted features.
  2. It runs on an arbitrary subset of levels.
  3. It raises KeyError when no requested level is present.
  4. It silently drops absent levels when at least one is present.
  5. Gradients flow back to every requested source level.
  6. The median-heuristic bandwidth is positive at every level and carries
     no gradient (computed under torch.no_grad).
  7. The three lambda schedules behave correctly: fixed is constant,
     linear is monotone from ~0 to lambda_max, sigmoid starts at 0 and
     saturates near lambda_max.
  8. A combined task + lambda * uda loss backpropagates through both terms.
  9. Sinkhorn (audit A8): the defaults reproduce the original fixed
     50-iteration loop exactly; column marginals hold, the row-marginal error
     falls with more iterations; tol stops early below tol; jdot_loss returns
     the same loss with and without diagnostics, and reports the errors.
"""

import sys

import torch

import numpy as np

from uda import (
    jdot_loss,
    lambda_uda_schedule,
    mmd_multiscale_loss,
    _median_heuristic_bandwidth,
    _sinkhorn_log,
)


def _sinkhorn_reference(cost, reg, n_iter=50):
    """The Sinkhorn loop as every JDOT run used it, before audit A8."""
    n, m = cost.shape
    a = torch.full((n,), -np.log(n), dtype=cost.dtype)
    b = torch.full((m,), -np.log(m), dtype=cost.dtype)
    f, g = torch.zeros_like(a), torch.zeros_like(b)
    K = -cost / reg
    for _ in range(n_iter):
        f = a - torch.logsumexp(K + g[None, :], dim=1)
        g = b - torch.logsumexp(K + f[:, None], dim=0)
    return torch.exp(K + f[:, None] + g[None, :])


# Encoder feature shapes for DualEncoderUNet with base_features=32:
#   enc2 -> 64ch @ 100x100, enc3 -> 128ch @ 50x50,
#   enc4 -> 256ch @ 25x25, bottleneck -> 512ch @ 12x12
FEAT_SHAPES = {
    "enc2": (64, 100, 100),
    "enc3": (128, 50, 50),
    "enc4": (256, 25, 25),
    "bottleneck": (512, 12, 12),
}
B = 8


def _make_feats(seed, shift=0.5):
    """Source centred at 0, target shifted by `shift` to inject a real gap."""
    g = torch.Generator().manual_seed(seed)
    src = {k: torch.randn(B, *s, generator=g, requires_grad=True)
           for k, s in FEAT_SHAPES.items()}
    tgt = {k: (torch.randn(B, *s, generator=g) + shift).requires_grad_(True)
           for k, s in FEAT_SHAPES.items()}
    return src, tgt


def check(name, cond):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        raise AssertionError(name)


def main():
    print("test 1-2: multi-scale MMD over all and subset of levels")
    src, tgt = _make_feats(0)
    loss_all = mmd_multiscale_loss(src, tgt)
    check("all four levels, positive loss", loss_all.item() > 0)
    check("all four levels, has grad_fn", loss_all.grad_fn is not None)
    loss_sub = mmd_multiscale_loss(src, tgt, levels=("enc3", "bottleneck"))
    check("subset of levels, positive loss", loss_sub.item() > 0)

    print("test 3: missing-level KeyError")
    raised = False
    try:
        mmd_multiscale_loss(src, tgt, levels=("enc99",))
    except KeyError:
        raised = True
    check("KeyError on fully-absent levels", raised)

    print("test 4: silent drop of absent level when others present")
    loss_drop = mmd_multiscale_loss(
        src, tgt, levels=("enc2", "not_a_level", "bottleneck"),
    )
    check("positive loss with one absent level dropped", loss_drop.item() > 0)

    print("test 5: gradient flows to every requested source level")
    src, tgt = _make_feats(1)
    loss = mmd_multiscale_loss(src, tgt)
    loss.backward()
    for lvl in FEAT_SHAPES:
        g = src[lvl].grad
        gnorm = 0.0 if g is None else g.abs().mean().item()
        check(f"gradient reaches {lvl} (mean|grad|={gnorm:.2e})",
              g is not None and gnorm > 0)

    print("test 6: median-heuristic bandwidth positive and grad-free")
    src, tgt = _make_feats(2)
    for lvl in FEAT_SHAPES:
        bw = _median_heuristic_bandwidth(src[lvl], tgt[lvl])
        check(f"bandwidth>0 at {lvl} (bw={bw:.3f})", bw > 0)
    # bandwidth is a python float, so it cannot carry grad by construction
    check("bandwidth returned as python float",
          isinstance(_median_heuristic_bandwidth(src["enc2"], tgt["enc2"]),
                     float))

    print("test 7: lambda schedules")
    n_epochs, lam_max = 25, 0.1
    fixed = [lambda_uda_schedule(e, n_epochs, lam_max, "fixed")
             for e in (1, 10, 25)]
    check("fixed schedule constant", all(abs(v - lam_max) < 1e-9 for v in fixed))

    lin = [lambda_uda_schedule(e, n_epochs, lam_max, "linear")
           for e in range(1, n_epochs + 1)]
    check("linear monotone non-decreasing",
          all(lin[i] <= lin[i + 1] + 1e-12 for i in range(len(lin) - 1)))
    check("linear starts near 0", lin[0] < 0.01)
    check("linear ends at lambda_max", abs(lin[-1] - lam_max) < 1e-9)

    sig0 = lambda_uda_schedule(0, n_epochs, lam_max, "sigmoid")
    sig_end = lambda_uda_schedule(n_epochs, n_epochs, lam_max, "sigmoid")
    sig = [lambda_uda_schedule(e, n_epochs, lam_max, "sigmoid")
           for e in range(0, n_epochs + 1)]
    check("sigmoid is 0 at p=0", abs(sig0) < 1e-9)
    check("sigmoid saturates near lambda_max at p=1", sig_end > 0.099)
    check("sigmoid monotone non-decreasing",
          all(sig[i] <= sig[i + 1] + 1e-12 for i in range(len(sig) - 1)))

    print("test 8: combined task + lambda*uda backprops through both terms")
    src, tgt = _make_feats(3)
    task = src["bottleneck"].mean() ** 2          # dummy task loss
    uda = mmd_multiscale_loss(src, tgt)
    total = task + lam_max * uda
    total.backward()
    # both the task path (bottleneck) and the uda path (all levels) must
    # have received gradient
    check("task path has gradient at bottleneck",
          src["bottleneck"].grad is not None
          and src["bottleneck"].grad.abs().mean().item() > 0)
    check("uda path has gradient at enc2",
          src["enc2"].grad is not None
          and src["enc2"].grad.abs().mean().item() > 0)

    print("test 9: Sinkhorn defaults, marginal errors, tolerance")
    g = torch.Generator().manual_seed(9)
    cost = torch.rand(32, 32, generator=g)
    for reg in (0.1, 0.01, 0.001):
        check(f"reg={reg}: defaults equal the original loop exactly",
              torch.equal(_sinkhorn_log(cost, reg=reg), _sinkhorn_reference(cost, reg)))
    errs = {}
    for it in (10, 50, 1000):
        _, e = _sinkhorn_log(cost, reg=0.01, n_iter=it, return_err=True)
        errs[it] = e["marg_err_row"]
        check(f"n_iter={it}: column marginals hold (err {e['marg_err_col']:.1e})", e["marg_err_col"] < 1e-5)
        check(f"n_iter={it}: iterations run = {e['iters']}", e["iters"] == it)
    check(f"row error falls with iterations {errs[10]:.2e} > {errs[1000]:.2e}",
          errs[1000] <= errs[50] + 1e-7 and errs[1000] < errs[10])
    _, e = _sinkhorn_log(cost, reg=0.05, n_iter=5000, tol=1e-4, return_err=True)
    check(f"tol stops early ({e['iters']} iterations, row err {e['marg_err_row']:.1e})",
          e["iters"] < 5000 and e["marg_err_row"] < 1e-4)

    fs, ft = torch.randn(8, 16, 3, 3, generator=g), torch.randn(8, 16, 3, 3, generator=g)
    ys, pt = torch.randn(8, 1, 10, 10, generator=g), torch.randn(8, 1, 10, 10, generator=g)
    plain = jdot_loss(fs, ft, ys, pt, reg=0.01)
    with_diag, d = jdot_loss(fs, ft, ys, pt, reg=0.01, return_diag=True)
    check("jdot_loss: same loss with and without diagnostics", torch.equal(plain, with_diag))
    check("jdot_loss: diagnostics report marginal errors and feature share",
          all(k in d for k in ("marg_err_row", "marg_err_col", "sinkhorn_iters", "feat_share_transport")))
    check(f"jdot_loss: feature share in [0, 1] ({d['feat_share_transport']:.3f})",
          0.0 <= d["feat_share_transport"] <= 1.0 + 1e-6)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nSMOKE TEST FAILED: {e}", file=sys.stderr)
        sys.exit(1)