"""
Tests for the AFM arm's C6 prerequisites (PLAN_COSMO.md C6). Synthetic tensors
only, no data. Runs in scripts/run_tests.sh.

1. AFM default unchanged: the flow net takes 1 + 9 + 2 = 12 channels, the
   encoder outputs 1, the state-dict keys are those of the AFM checkpoints
   already on disk (no new parameters or buffers), and the training forward
   still returns mu.
2. --target_bn eval in train_afm.train_one_epoch, for AFM (mmd, coral, dann,
   joint_ot, spectral) and CFM (mmd, coral, dann). One step from the same
   initialisation and source batch, two target batches (A: N(0,1),
   B: 3 N(0,1) + 2):
     eval:  BatchNorm running statistics identical for A and B;
            weights differ (the target still reaches the gradient);
     train (control): running statistics differ.
   BatchNorm layers are back in train mode after the step.
3. CFM model: encoder outputs cond_ch channels; flow net takes
   1 + 9 + 2 + cond_ch; no "mu" in the training output and encoder_loss = 0;
   the sampler starts from N(0, I) (with a zeroed flow net the samples ARE
   the starting noise: mean ~ 0, sd ~ 1, independent of the inputs), whereas
   AFM's start is mu + sigma_z eps; deterministic_predict is reproducible and
   has the target's shape; start_state matches sample's first draw.
4. evaluate.py: a cfm_* run directory is inferred as "cfm", parsed like afm_*,
   and load_model rebuilds it from config.json's cond_ch with identical outputs.
"""

import copy
import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from evaluate import _infer_model_type, load_model, parse_exp_name
from models.afm import AFMModel
from train_afm import build_uda, train_one_epoch

B, H, BASE = 2, 200, 8
FAILED = []


def check(name, ok):
    print(f"{name:70s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        FAILED.append(name)


def _bn_state(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()
            if "running_mean" in k or "running_var" in k}


def _weights(m):
    return {k: v.detach().clone() for k, v in m.named_parameters()}


def _same(a, b, atol=0.0):
    return all(torch.allclose(a[k], b[k], atol=atol, rtol=0) for k in a)


def one_step(init, method, target_bn, x_t, src, device):
    model = copy.deepcopy(init)
    torch.manual_seed(0)
    uc, extra = build_uda(method, ["enc2", "enc3", "enc4", "bottleneck"], device, BASE)
    opt = torch.optim.SGD(list(model.parameters()) + extra, lr=1e-2)
    train_one_epoch(model, [src], [(x_t, src[1], src[2])], opt, uc, method, 1.0, 0.01, device,
                    10, 10, dann_weight=1.0, target_bn=target_bn)
    check(f"  {model.mode} {method} {target_bn}: BN back in train mode",
          all(m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d)))
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1)
    x_s = torch.randn(B, 9, 80, 80, device=device)
    s_s = torch.randn(B, 2, H, H, device=device)
    y_s = torch.randn(B, 1, H, H, device=device)
    src = (x_s, s_s, y_s)
    x_a = torch.randn(B, 9, 80, 80, device=device)
    x_b = 3.0 * torch.randn(B, 9, 80, 80, device=device) + 2.0

    # 1. AFM default unchanged.
    afm = AFMModel(9, 2, 1, BASE).to(device)
    check("afm: flow net in_channels 12", afm.flow_net.enc1.conv1.in_channels == 12)
    check("afm: encoder out_channels 1", afm.encoder(x_s, s_s).shape[1] == 1)
    keys = set(afm.state_dict())
    check("afm: no cond/det buffers in the state dict",
          not any(k.startswith(("det_", "cond")) for k in keys) and "sigma_z" in keys)
    afm.train()
    out = afm(x_s, s_s, x_target=y_s)
    check("afm: training forward returns mu and a positive encoder loss",
          "mu" in out and float(out["encoder_loss"]) > 0)

    # 2. --target_bn eval.
    for mode, methods in (("afm", ["mmd", "coral", "dann", "joint_ot", "spectral"]),
                          ("cfm", ["mmd", "coral", "dann"])):
        torch.manual_seed(2)
        init = AFMModel(9, 2, 1, BASE, mode=mode, cond_ch=4).to(device).train()
        for method in methods:
            ea = one_step(init, method, "eval", x_a, src, device)
            eb = one_step(init, method, "eval", x_b, src, device)
            ta = one_step(init, method, "train", x_a, src, device)
            tb = one_step(init, method, "train", x_b, src, device)
            check(f"{mode} {method}: eval, BN stats independent of target",
                  _same(_bn_state(ea), _bn_state(eb), atol=1e-6))
            check(f"{mode} {method}: eval, target reaches the weights",
                  not _same(_weights(ea), _weights(eb), atol=1e-7))
            check(f"{mode} {method}: train (control), BN stats depend on target",
                  not _same(_bn_state(ta), _bn_state(tb), atol=1e-6))

    # 3. CFM model.
    torch.manual_seed(3)
    cfm = AFMModel(9, 2, 1, BASE, mode="cfm", cond_ch=4).to(device)
    check("cfm: encoder out_channels = cond_ch", cfm.encoder(x_s, s_s).shape[1] == 4)
    check("cfm: flow net in_channels 1+9+2+4", cfm.flow_net.enc1.conv1.in_channels == 16)
    cfm.train()
    out = cfm(x_s, s_s, x_target=y_s)
    check("cfm: no mu, encoder_loss 0, total = flow",
          "mu" not in out and float(out["encoder_loss"]) == 0.0
          and torch.equal(out["total_loss"], out["flow_loss"]))
    with torch.no_grad():
        for m in (cfm, afm):
            m.flow_net.final.weight.zero_()
            m.flow_net.final.bias.zero_()
    g = torch.Generator(device=device)
    g.manual_seed(0)
    big_s = torch.randn(4, 2, H, H, device=device)
    smp = cfm.sample(x_s.repeat(2, 1, 1, 1), big_s, n_samples=2, steps=3, generator=g)
    check("cfm: sample shape (B, n, 1, H, W)", tuple(smp.shape) == (4, 2, 1, H, H))
    check("cfm: zero flow -> samples are N(0, I) noise",
          abs(float(smp.mean())) < 0.02 and abs(float(smp.std()) - 1.0) < 0.02)
    g.manual_seed(0)
    smp_b = cfm.sample(3.0 * x_s.repeat(2, 1, 1, 1) + 2.0, big_s, n_samples=2, steps=3, generator=g)
    check("cfm: zero flow -> start does not depend on inputs", torch.allclose(smp, smp_b))
    afm.eval()
    with torch.no_grad():
        g.manual_seed(0)
        sa = afm.sample(x_s, s_s, n_samples=1, steps=3, generator=g)[:, 0]
        g.manual_seed(0)
        noise = torch.randn((B, 1, H, H), generator=g, device=device)
        expect = afm.encoder(afm._match_dyn(x_s, s_s), s_s).float() + afm.sigma_z * noise
    check("afm: zero flow -> samples are mu + sigma_z eps", torch.allclose(sa, expect, atol=1e-5))
    cfm.eval()
    d1 = cfm.deterministic_predict(x_s, s_s)
    d2 = cfm.deterministic_predict(x_s, s_s)
    check("cfm: deterministic_predict reproducible, shape (B, 1, H, W)",
          torch.equal(d1, d2) and tuple(d1.shape) == (B, 1, H, H))
    check("cfm: deterministic_predict leaves the model in eval", not cfm.training)
    with torch.no_grad():
        g.manual_seed(0)
        f_dyn, z0 = cfm.start_state(cfm._match_dyn(x_s, s_s), s_s, generator=g)
        g.manual_seed(0)
        ref0 = torch.randn((B, 1, H, H), generator=g, device=device)
    check("cfm: start_state z0 is the sampler's noise, flow input 9 + cond_ch",
          torch.equal(z0, ref0) and f_dyn.shape[1] == 13)

    # 4. evaluate.py round trip.
    check("evaluate: cfm_ prefix -> cfm", _infer_model_type("cfm_alps__to__poland__none") == "cfm")
    check("evaluate: afm_ prefix still afm", _infer_model_type("afm_alps__to__poland__none") == "afm")
    check("evaluate: cfm_ parsed like afm_",
          parse_exp_name("cfm_alps__to__poland__mmd") == ("alps", "poland", "mmd"))
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "cfm_alps__to__poland__none"
        run.mkdir()
        torch.save(cfm.state_dict(), run / "best.pt")
        (run / "config.json").write_text(json.dumps({"cond_ch": 4, "base_features": BASE}))
        m2 = load_model("cfm", str(run / "best.pt"), device, BASE)
        check("evaluate: load_model rebuilds cfm with cond_ch from config",
              m2.mode == "cfm" and m2.cond_ch == 4
              and torch.equal(m2.deterministic_predict(x_s, s_s), cfm.deterministic_predict(x_s, s_s)))

    if FAILED:
        raise SystemExit("FAILED:\n  " + "\n  ".join(FAILED))
    print("all afm/cfm checks passed")


if __name__ == "__main__":
    main()
