"""
Model-based conditional-shift diagnostic for RainShift domains.

Uses the already-trained source-only models as estimates of each domain's
conditional P(Y|X), and measures whether that mapping differs between domains
in the FULL input space (no binning, no curse of dimensionality).

Two complementary measurements per ordered pair (A, B):

1. Transfer-error decomposition (model vs ground truth)
   - E(A->B): A's model on B's test inputs vs B's ground truth (std-MSE).
   - E(B->B): B's own model on B's inputs (the irreducible floor).
   - conditional penalty = E(A->B) - E(B->B).
   A large penalty means A's mapping fails on B even though B's own model
   succeeds -> the conditional differs. Restricting to B-inputs supported in
   A removes the extrapolation confound (see support note below).

2. Model-disagreement (model vs model, the direct conditional discrepancy)
   - On B's inputs, compare A's predictions to B's predictions.
   - Both models see identical inputs, so any disagreement is purely a
     difference in the learned mapping P_A(Y|X) vs P_B(Y|X) -- no ground-truth
     noise, no target-side extrapolation confound.
   - Reported as std-MSE between the two prediction fields, and as a
     per-sample distribution.

Interpretation
--------------
- Shared conditional (covariate shift): A's model transfers to B with small
  penalty, and the two models agree on B's inputs. Disagreement ~ 0.
- Different conditional (conditional shift): A's model fails on B beyond B's
  floor, and the two models disagree even on the same inputs.

Support note
------------
Prediction error / disagreement can be inflated where B's inputs fall outside
A's training support (extrapolation), which is covariate shift, not conditional
shift. We flag this by also reporting the transfer error of B's model on A's
inputs and vice versa; if disagreement is high but concentrated where one model
extrapolates, treat with caution. For a projection-free support proxy we report
the fraction of B samples whose per-sample disagreement exceeds a robust
threshold -- concentrated disagreement suggests a support edge, diffuse
disagreement suggests a genuine mapping difference.

All predictions are in source-normalized std space (the model's native space),
so std-MSE is directly comparable across pairs. Normalization uses each model's
OWN source stats (the space it was trained in).
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def _load_stats(domain_path: Path):
    import json as _json
    p = domain_path / "normalization_stats.json"
    return _json.loads(p.read_text())


@torch.no_grad()
def _predict_field(model, loader, device, max_batches=None):
    """Return stacked predictions and targets (std space) for a loader."""
    model.eval()
    preds, trues = [], []
    for i, (x, s, y) in enumerate(loader):
        x, s, y = x.to(device), s.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p = model(x, s)
        preds.append(p.float().cpu())
        trues.append(y.float().cpu())
        if max_batches and i + 1 >= max_batches:
            break
    return torch.cat(preds), torch.cat(trues)


@torch.no_grad()
def _predict_on_inputs(model, loader, device, max_batches=None):
    """Return only predictions (std space) for a loader's inputs."""
    model.eval()
    preds = []
    for i, (x, s, _) in enumerate(loader):
        x, s = x.to(device), s.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p = model(x, s)
        preds.append(p.float().cpu())
        if max_batches and i + 1 >= max_batches:
            break
    return torch.cat(preds)


def _std_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a - b) ** 2).item())


def run(args):
    # Repo imports (available on the cluster at runtime).
    sys.path.insert(0, args.code_root)
    from models.unet import DualEncoderUNet  # noqa: F401
    from data.dataset_zarr import ClimateSRDatasetZarr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    exp_root = Path(args.exp_root)

    def load_source_model(domain):
        """Load the source-only (uda_method=none) checkpoint for `domain` as
        source. Its tag is '{domain}__to__{any}__none' -- any target works
        since the source-only model never used the target. Pick the first."""
        cand = sorted(exp_root.glob(f"{domain}__to__*__none/best.pt"))
        if not cand:
            raise FileNotFoundError(f"No source-only checkpoint for {domain} under {exp_root}")
        m = DualEncoderUNet(9, 2, 1, args.base_features).to(device)
        m.load_state_dict(torch.load(cand[0], map_location=device, weights_only=True))
        m.eval()
        return m, cand[0]

    def loader_for(domain, stats):
        ds = ClimateSRDatasetZarr(str(data_root / domain), "test", stats=stats)
        return DataLoader(ds, args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    results = {}
    print(f"{'pair (A->B)':40s} {'E(A->B)':>9} {'E(B->B)':>9} {'penalty':>9} {'disagree':>9}  flag")

    for a, b in itertools.permutations(args.domains, 2):
        model_a, _ = load_source_model(a)
        model_b, _ = load_source_model(b)
        stats_a = _load_stats(data_root / a)   # A's model lives in A's std space
        stats_b = _load_stats(data_root / b)

        # B's inputs, normalized in A's space for A's model, B's space for B's.
        loader_b_in_a = loader_for(b, stats_a)   # B inputs, A-normalized
        loader_b_in_b = loader_for(b, stats_b)   # B inputs, B-normalized

        # 1. Transfer error: A's model on B (A-normalized inputs & target).
        pa_on_b, y_b_a = _predict_field(model_a, loader_b_in_a, device, args.max_batches)
        e_a_to_b = _std_mse(pa_on_b, y_b_a)

        # B's own model on B (B-normalized) -> irreducible floor.
        pb_on_b, y_b_b = _predict_field(model_b, loader_b_in_b, device, args.max_batches)
        e_b_to_b = _std_mse(pb_on_b, y_b_b)

        penalty = e_a_to_b - e_b_to_b

        # 2. Model disagreement on B's inputs. To compare two mappings on the
        # SAME inputs we must feed identically-normalized inputs; use B's own
        # normalization for both models' inputs, then compare prediction fields.
        # (Both models then operate on the same tensor; differences are purely
        # mapping differences. Predictions are in each model's own std space,
        # which is the same B-space here since inputs are B-normalized.)
        pa_on_binputs = _predict_on_inputs(model_a, loader_b_in_b, device, args.max_batches)
        pb_on_binputs = pb_on_b  # already computed on B-normalized B inputs
        # align lengths (in case max_batches rounding differs)
        n = min(pa_on_binputs.shape[0], pb_on_binputs.shape[0])
        disagree = _std_mse(pa_on_binputs[:n], pb_on_binputs[:n])

        # per-sample disagreement distribution -> concentration diagnostic
        per_sample = torch.mean((pa_on_binputs[:n] - pb_on_binputs[:n]) ** 2,
                                dim=(1, 2, 3)).numpy()
        med = float(np.median(per_sample))
        p90 = float(np.percentile(per_sample, 90))
        concentration = p90 / (med + 1e-8)  # high => disagreement in a tail (support edge)

        # Flag
        floor = max(e_b_to_b, 1e-6)
        if penalty / floor < 0.15 and disagree < 0.15 * floor:
            flag = "shared conditional (covariate shift)"
        elif concentration > 8.0:
            flag = "disagreement concentrated (possible support edge, treat with caution)"
        else:
            flag = "different conditional (conditional shift)"

        key = f"{a}__to__{b}"
        results[key] = {
            "e_a_to_b": e_a_to_b, "e_b_to_b": e_b_to_b,
            "conditional_penalty": penalty,
            "model_disagreement": disagree,
            "disagreement_median": med, "disagreement_p90": p90,
            "concentration": concentration, "flag": flag,
        }
        print(f"{key:40s} {e_a_to_b:9.4f} {e_b_to_b:9.4f} {penalty:9.4f} {disagree:9.4f}  {flag}")

        del model_a, model_b
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")
    print(
        "\nInterpretation:\n"
        "  penalty = E(A->B) - E(B->B): how much worse A's mapping is on B than\n"
        "            B's own model. Large => A's conditional fails on B.\n"
        "  disagreement = std-MSE between A's and B's predictions on identical B\n"
        "            inputs: pure mapping difference, no ground-truth term.\n"
        "  Both small  -> shared P(Y|X) (covariate shift, UDA-addressable).\n"
        "  Both large  -> different P(Y|X) (conditional shift, UDA cannot bridge).\n"
        "  High concentration (p90/median) => disagreement lives in a tail of B's\n"
        "  inputs, i.e. a support edge (extrapolation), not a diffuse mapping\n"
        "  difference -- report as low-support rather than conditional shift."
    )


def main():
    p = argparse.ArgumentParser(description="Model-based conditional-shift diagnostic.")
    p.add_argument("--code_root", required=True, help="repo root (for imports)")
    p.add_argument("--data_root", required=True)
    p.add_argument("--exp_root", required=True,
                   help="dir with {domain}__to__*__none/best.pt source checkpoints")
    p.add_argument("--domains", nargs="+",
                   default=["europe_west", "horn-of-africa", "melanesia"])
    p.add_argument("--base_features", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_batches", type=int, default=None,
                   help="cap batches per evaluation for speed (None = full test set)")
    p.add_argument("--out", default="conditional_shift_model.json")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()