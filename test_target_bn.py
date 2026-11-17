"""
Test of --target_bn eval (train_unet._bn_eval, B4.9). Synthetic tensors only,
no data. Runs in scripts/run_tests.sh.

One training step from the same initialisation, the same source batch and two
different target batches (A: standard normal, B: shifted and scaled, a strong
covariate shift). For each method with a target forward:
  eval mode:  BatchNorm running statistics after the step are identical for A
              and B (the target never touches them), while the weights differ
              (the target still reaches the gradient through the loss);
  train mode: the running statistics differ (control: the test can fail).
Also: every BatchNorm layer is back in train mode after the step, and the
running statistics in eval mode equal those of a source-only forward.
"""

import copy

import torch
import torch.nn as nn

from models.unet import DualEncoderUNet
from train_unet import _bn_eval, build_uda, train_one_epoch

METHODS = ["mmd", "dann", "joint_ot", "spectral", "coral"]
B, H = 4, 200


def _bn_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if "running_mean" in k or "running_var" in k}


def _weights(model):
    return {k: v.detach().clone() for k, v in model.named_parameters()}


def _same(a, b, atol=0.0):
    return all(torch.allclose(a[k], b[k], atol=atol, rtol=0) for k in a)


def one_step(init, method, target_bn, x_t, src, device):
    model = copy.deepcopy(init)
    torch.manual_seed(0)
    uda_comp, extra = build_uda(method, device, None)
    opt = torch.optim.SGD(list(model.parameters()) + extra, lr=1e-2)
    train_one_epoch(
        model, [src], [(x_t, src[1], src[2])], opt, nn.MSELoss(), uda_comp, method,
        lambda_eff=1.0, fda_beta=0.01, device=device, epoch=10, total_epochs=10,
        dann_weight=1.0, jdot_reg=0.1, target_bn=target_bn,
    )
    assert all(m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d)), \
        "BatchNorm layers not restored to train mode"
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1)
    init = DualEncoderUNet(dynamic_channels=9, static_channels=2, out_channels=1, base_features=32).to(device)
    init.train()
    x_s = torch.randn(B, 9, 80, 80, device=device)
    s_s = torch.randn(B, 2, H, H, device=device)
    y_s = torch.randn(B, 1, H, H, device=device)
    src = (x_s, s_s, y_s)
    x_a = torch.randn(B, 9, 80, 80, device=device)
    x_b = 3.0 * torch.randn(B, 9, 80, 80, device=device) + 2.0

    # Reference: running statistics after a source-only forward.
    ref = copy.deepcopy(init)
    with torch.no_grad():
        ref(x_s, s_s)
    ref_bn = _bn_state(ref)

    # The context manager alone: eval-mode forward leaves stats and modes intact.
    m = copy.deepcopy(init)
    before = _bn_state(m)
    with _bn_eval(m):
        m(x_b, s_s)
    assert _same(before, _bn_state(m)), "_bn_eval forward changed running stats"
    assert all(mm.training for mm in m.modules() if isinstance(mm, nn.BatchNorm2d))

    failed = []
    for method in METHODS:
        ea = one_step(init, method, "eval", x_a, src, device)
        eb = one_step(init, method, "eval", x_b, src, device)
        ta = one_step(init, method, "train", x_a, src, device)
        tb = one_step(init, method, "train", x_b, src, device)
        checks = {
            "eval: stats independent of target": _same(_bn_state(ea), _bn_state(eb), atol=1e-6),
            "eval: stats equal source-only forward": _same(_bn_state(ea), ref_bn, atol=1e-5),
            "eval: target reaches the weights": not _same(_weights(ea), _weights(eb), atol=1e-7),
            "train (control): stats depend on target": not _same(_bn_state(ta), _bn_state(tb), atol=1e-6),
        }
        for name, ok in checks.items():
            print(f"{method:9s} {name:42s} {'PASS' if ok else 'FAIL'}")
            if not ok:
                failed.append(f"{method}: {name}")
    if failed:
        raise SystemExit("FAILED:\n  " + "\n  ".join(failed))
    print("all target_bn checks passed")


if __name__ == "__main__":
    main()
