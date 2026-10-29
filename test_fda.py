"""
Tests for uda.fda_transfer (audit A1).

Run: python test_fda.py
Exits non-zero on any failure. No GPU required.

Checks:
  1. Every output channel mean equals the target's channel mean.
  2. A sample with a negative mean adapted towards a positive-mean target
     comes out with the target's positive mean (the old amplitude-only DC
     swap gave -|target mean|).
  3. The swap window is symmetric, so the adapted spectrum is Hermitian and
     its inverse FFT is real up to round-off, at several beta and at even
     and odd sizes.
  4. With target == source the output equals the input.
  5. Frequencies outside the window keep the source spectrum.
"""

import sys

import torch

from uda import _fda_spectrum, fda_transfer


def check(name, cond):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        raise AssertionError(name)


def _fields(seed, B=4, C=3, H=80, W=80, mean=0.0):
    g = torch.Generator().manual_seed(seed)
    # Smooth-ish fields plus a per-sample, per-channel offset, as z-scored inputs.
    x = torch.randn(B, C, H, W, generator=g)
    x = torch.nn.functional.avg_pool2d(x, 5, stride=1, padding=2)
    return x + mean + torch.randn(B, C, 1, 1, generator=g)


# fda_transfer computes in float32, so the tolerances are float32 round-off.
def main():
    print("test 1: output channel means equal the target's")
    src, tgt = _fields(0), _fields(1, mean=2.0)
    for beta in (0.0, 0.01, 0.05, 0.2):
        out = fda_transfer(src, tgt, beta=beta)
        err = (out.mean(dim=(-2, -1)) - tgt.mean(dim=(-2, -1))).abs().max().item()
        check(f"beta={beta}: max |mean(out) - mean(tgt)| = {err:.2e}", err < 1e-4)

    print("test 2: negative-mean sample towards a positive-mean target")
    src = _fields(2, B=1, C=1) - _fields(2, B=1, C=1).mean() - 0.3   # mean -0.3
    tgt = _fields(3, B=1, C=1) - _fields(3, B=1, C=1).mean() + 4.4   # mean +4.4
    out = fda_transfer(src, tgt, beta=0.01)
    m = out.mean().item()
    check(f"output mean {m:.4f} is +4.4, not -4.4", abs(m - 4.4) < 1e-4)

    print("test 3: symmetric window -> inverse FFT is real")
    for H, W in ((80, 80), (81, 79), (200, 200)):
        src, tgt = _fields(4, H=H, W=W), _fields(5, H=H, W=W, mean=-1.0)
        for beta in (0.0, 0.01, 0.05, 0.3, 0.6):
            z = torch.fft.ifft2(_fda_spectrum(src, tgt, beta), dim=(-2, -1))
            rel = (z.imag.abs().max() / z.real.abs().max()).item()
            check(f"{H}x{W} beta={beta}: max|imag|/max|real| = {rel:.1e}", rel < 1e-5)

    print("test 4: target == source returns the input")
    src = _fields(6, mean=-0.5)
    for beta in (0.01, 0.1, 0.6):
        err = (fda_transfer(src, src.clone(), beta=beta) - src).abs().max().item()
        check(f"beta={beta}: max |out - in| = {err:.2e}", err < 1e-4)

    print("test 5: frequencies outside the window keep the source spectrum")
    src, tgt = _fields(7), _fields(8, mean=1.0)
    beta, H = 0.05, 80
    h = max(int(H * beta), 1)
    spec_out = torch.fft.fftshift(torch.fft.fft2(fda_transfer(src, tgt, beta)), dim=(-2, -1))
    spec_src = torch.fft.fftshift(torch.fft.fft2(src), dim=(-2, -1))
    spec_tgt = torch.fft.fftshift(torch.fft.fft2(tgt), dim=(-2, -1))
    c = H // 2
    inside = torch.zeros(H, H, dtype=torch.bool)
    inside[c - h:c + h + 1, c - h:c + h + 1] = True
    scale = spec_src.abs().max().item()
    err_out = (spec_out - spec_src)[..., ~inside].abs().max().item() / scale
    check(f"outside window: spectrum unchanged (rel err {err_out:.1e})", err_out < 1e-5)
    amp_err = (spec_out.abs() - spec_tgt.abs())[..., inside].abs().max().item() / scale
    check(f"inside window: target amplitude (rel err {amp_err:.1e})", amp_err < 1e-5)

    print("\nAll FDA tests passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFDA TEST FAILED: {e}", file=sys.stderr)
        sys.exit(1)
