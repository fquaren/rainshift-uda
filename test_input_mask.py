"""Check of the DualEncoderUNet input mask (PLAN.md P19) on random inputs.
Run inside a job (scripts/run_input_ablation.sh runs it in the SMOKE task)."""

import tempfile
from pathlib import Path

import torch

from evaluate import load_model
from models.unet import DualEncoderUNet
from train_unet import input_keep

MASK = ["lsm", "z", "tisr", "sp"]
keep = input_keep(MASK)
assert keep["dyn_keep"] == [1, 1, 0, 1, 1, 0, 1, 1, 1] and keep["stat_keep"] == [0, 0], keep
assert input_keep([]) == {}

# 1. Same seed, same initial weights with and without the mask: the ablation
#    changes the inputs only.
torch.manual_seed(42)
full = DualEncoderUNet(9, 2, 1, 32)
torch.manual_seed(42)
masked = DualEncoderUNet(9, 2, 1, 32, **keep)
sd_full, sd_masked = full.state_dict(), masked.state_dict()
assert set(sd_masked) - set(sd_full) == {"dyn_keep", "stat_keep"}
assert all(torch.equal(sd_full[k], sd_masked[k]) for k in sd_full)

# 2. The masked model's output does not depend on the masked channels.
masked.eval()
x = torch.randn(2, 9, 80, 80)
s = torch.randn(2, 2, 200, 200)
x2, s2 = x.clone(), s.clone()
x2[:, [2, 5]] = torch.randn(2, 2, 80, 80) * 10
s2[:] = torch.randn(2, 2, 200, 200) * 10
with torch.no_grad():
    assert torch.equal(masked(x, s), masked(x2, s2))
    x3 = x.clone()
    x3[:, 4] += 1.0   # tcw is kept
    assert not torch.equal(masked(x, s), masked(x3, s))

# 3. evaluate.load_model restores the mask; an unmasked model refuses it.
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "best.pt"
    torch.save(masked.state_dict(), p)
    m = load_model("unet", str(p), torch.device("cpu"))
    assert torch.equal(m.dyn_keep.flatten(), torch.tensor(keep["dyn_keep"]))
    with torch.no_grad():
        assert torch.equal(m(x, s), masked(x, s))
    try:
        DualEncoderUNet(9, 2, 1, 32).load_state_dict(torch.load(p, weights_only=True))
        raise AssertionError("masked checkpoint loaded into an unmasked model")
    except RuntimeError as e:
        assert "dyn_keep" in str(e)

print("test_input_mask: all checks passed")
