"""Check of evaluate.py's new metrics and of `batch --only` / `merge` (PLAN.md
3.6 step 5) on random inputs and an untrained width-32 AFM. Run inside a job
(scripts/eval_array.sh runs it before every task)."""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from data.dataset import inverse_transform
from evaluate import FLOORS_MM, append_csv, cmd_merge, evaluate_model, run_finished, save_results
from models.afm import AFMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
model = AFMModel(9, 2, 1, 32, encoder_loss_weight=0.1).to(device).eval()
stats = {"precipitation": [-6.0, 1.0]}   # std 1: an untrained net cannot overflow exp()
batches = [(torch.randn(2, 9, 80, 80), torch.randn(2, 2, 200, 200), torch.randn(2, 1, 200, 200))
           for _ in range(2)]
M = 3

met, pred, true, ps = evaluate_model(model, "afm", batches, device, stats,
                                     n_ensemble=M, sample_steps=2, desc="test")

# mae_std and the floored log MSE against a direct computation.
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    mu = torch.cat([model.deterministic_predict(x.to(device), s.to(device)) for x, s, _ in batches])
y = torch.cat([b[2] for b in batches]).to(device)
assert abs(met["mae_std"] - (mu.float() - y).abs().mean().item()) < 1e-4, met["mae_std"]
p_mm = inverse_transform(mu.float().cpu().numpy(), "precipitation", stats)
t_mm = inverse_transform(y.cpu().numpy(), "precipitation", stats)
assert np.allclose(true, t_mm)
for f, v in FLOORS_MM.items():
    ref = np.mean((np.log(p_mm + v) - np.log(t_mm + v)) ** 2, axis=(1, 2, 3))
    assert np.allclose(ps[f"mse_log_{f}"], ref, rtol=1e-4), f
    assert abs(met[f"mse_log_{f}"] - ref.mean()) < 1e-4 * ref.mean(), f

# Ensemble outputs: per-sample floored CRPS, rank histograms over every pixel.
for k in ("crps_std", "spread_std", "crps_mm", *(f"crps_log_{f}" for f in FLOORS_MM)):
    assert ps[k].shape == (4,) and np.isfinite(ps[k]).all() and k in met, k
assert ps["rank_hist"].shape == (4, M + 1) and (ps["rank_hist"].sum(1) == 200 * 200).all()
assert "rank_hist" not in met

# Spread-skill pair and ratio in every space, M recorded.
for sp in ("std", *(f"log_{f}" for f in FLOORS_MM)):
    for k in (f"ens_var_{sp}", f"ensmean_mse_{sp}"):
        assert ps[k].shape == (4,) and np.isfinite(ps[k]).all() and (ps[k] >= 0).all() and k in met, k
    r = met[f"ssr_{sp}"]
    assert np.isfinite(r) and abs(r - np.sqrt((M + 1) / M * met[f"ens_var_{sp}"] / met[f"ensmean_mse_{sp}"])) < 1e-6, sp
assert met["n_members"] == M

# Floored rank histograms: every pixel once; the wet one counts exactly the
# pixels whose truth is >= the floor.
for f, v in FLOORS_MM.items():
    h, hw = ps[f"rank_hist_{f}"], ps[f"rank_hist_wet_{f}"]
    assert h.shape == hw.shape == (4, M + 1) and (h.sum(1) == 200 * 200).all(), f
    assert np.array_equal(hw.sum(1), (t_mm >= v).reshape(4, -1).sum(1)), f
    assert (hw <= h).all() and f"rank_hist_{f}" not in met, f

# The fixed generators make a re-score identical.
met2, *_ = evaluate_model(model, "afm", batches, device, stats, n_ensemble=M, sample_steps=2, desc="test")
assert all(abs(met2[k] - v) <= 1e-5 * abs(v) + 1e-7 for k, v in met.items() if isinstance(v, float)), \
    "re-scoring the same checkpoint changed the metrics"

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    save_results(met, pred, true, pred, true, d / "res", n_save=0, src_per_sample=ps, tgt_per_sample=ps)
    z = np.load(d / "res" / "per_sample_loss.npz")
    assert z["tgt_rank_hist"].dtype == np.int32 and z["tgt_mae_std"].dtype == np.float32
    assert z["tgt_rank_hist_wet_e01"].dtype == np.int32

    # Completion marker: config.json present and not older than any .pt.
    run = d / "run"
    run.mkdir()
    (run / "best.pt").write_bytes(b"")
    assert not run_finished(run), "no config.json"
    (run / "config.json").write_text("{}")
    os.utime(run / "best.pt", (1e9, 1e9))
    os.utime(run / "config.json", (1e9 + 10, 1e9 + 10))
    assert run_finished(run)
    os.utime(run / "best.pt", (1e9 + 20, 1e9 + 20))
    assert not run_finished(run), "checkpoint newer than config.json"

    # merge: one file per task, refuses duplicates and an existing results.csv.
    row = lambda tgt: {"model": "afm", "source": "europe_west", "target": tgt, "method": "none",
                       "transform": "none", "tgt_crps_std": "0.2600"}
    (d / "rows").mkdir()
    for tgt in ("horn-of-africa", "melanesia"):
        append_csv(d / "rows" / f"afm_europe_west__to__{tgt}__none.csv", row(tgt))
    try:
        cmd_merge(argparse.Namespace(output_dir=str(d), expect=3))
        raise AssertionError("merge accepted 2 rows with --expect 3")
    except SystemExit:
        pass
    assert not (d / "results.csv").exists()
    cmd_merge(argparse.Namespace(output_dir=str(d), expect=2))
    lines = (d / "results.csv").read_text().splitlines()
    assert len(lines) == 3 and lines[0].startswith("model,source"), lines
    try:
        cmd_merge(argparse.Namespace(output_dir=str(d)))
        raise AssertionError("merge overwrote results.csv")
    except SystemExit:
        pass
    (d / "results.csv").unlink()
    append_csv(d / "rows" / "dup.csv", row("melanesia"))
    try:
        cmd_merge(argparse.Namespace(output_dir=str(d)))
        raise AssertionError("merge accepted a duplicate row")
    except SystemExit:
        pass

print("test_evaluate_metrics: all checks passed", {k: round(v, 4) for k, v in met.items()
                                                   if isinstance(v, float)})
