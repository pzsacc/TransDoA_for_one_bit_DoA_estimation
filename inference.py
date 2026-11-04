"""One-bit DOA estimation demo: single-file end-to-end inference.

Pipeline (identical to the evaluation protocol of the paper):
  coherent source signal generation (random phase + fixed amplitudes + AWGN
  with per-realization SNR calibration)
  -> one-bit quantization, sign(Re) + j*sign(Im)
  -> forward pass through the packaged v1.3 network
  -> spec_ip peak search (find_peaks + log-domain parabolic sub-grid
     interpolation)
  -> per-source angle error against ground truth, MSE in dB

Usage (from this directory):
  python inference.py                          # K=3 default scenario, SNR=-6 dB, 100 trials
  python inference.py --snr 0                  # change SNR
  python inference.py --angles "10,25" --trials 50
  python inference.py --device cpu
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import doa

# ---- Physics (matches training config: M=60/N=64 ULA, [-45, 45] deg grid at 0.1 deg, lambda/2 spacing)
M, N = 60, 64
WAVELENGTH = 0.15
AMPS = [1.4, 1.0, 1.8]          # per-source amplitudes (evaluation protocol)


def generate_signal(theta, snr_db, m=M, n=N):
    """Coherent sources + AWGN + one-bit quantization. Returns (m, n) complex matrix."""
    k = len(theta)
    spacing = torch.arange(m, dtype=torch.float32) * (0.5 * WAVELENGTH)
    A = doa.steering_vector(theta, spacing, WAVELENGTH)
    S = (torch.randn(n) + 1j * torch.randn(n)) / (2 ** 0.5)
    S = torch.stack([S] * k, dim=0)              # coherent: shared baseband waveform
    S *= torch.tensor(AMPS)[:k].view(-1, 1)
    X = A @ S
    sig_pow = torch.mean(torch.abs(X) ** 2)
    noise_std = torch.sqrt(sig_pow / (10 ** (snr_db / 10)) / 2)
    X = X + noise_std * (torch.randn_like(X) + 1j * torch.randn_like(X))
    return torch.sign(X.real) + 1j * torch.sign(X.imag)


def run(theta, snr_db, trials, seed=42, device=None, verbose=True):
    """Monte-Carlo evaluation. Returns per-trial angle errors (trials, K) and MSE in dB."""
    torch.manual_seed(seed)
    theta_t = torch.tensor(theta, dtype=torch.float32)
    k = len(theta)
    rows, shown = [], 0
    for i in range(0, trials, 10):
        n = min(10, trials - i)
        Xs = [generate_signal(theta, snr_db) for _ in range(n)]
        preds = doa.predict_batch([x.real.tolist() for x in Xs],
                                  [x.imag.tolist() for x in Xs],
                                  k=k, device=device)
        rows.append(torch.abs(theta_t.unsqueeze(0) - torch.tensor(preds)))
        if verbose and shown < 3:
            for t in range(min(3 - shown, n)):
                true = [f"{a:6.1f}" for a in theta]
                est = [f"{v:7.2f}" for v in preds[t]]
                print(f"  [trial {i + t}] true {true} deg -> estimated {est} deg")
            shown += min(3 - shown, n)
        del Xs
    errs = torch.cat(rows)
    mse_db = 10 * np.log10(float((errs ** 2).mean()) + 1e-12)
    miss = int((errs.max(dim=1).values > 5).sum())
    return errs, mse_db, miss


def main():
    p = argparse.ArgumentParser(
        description="One-bit DOA inference demo (physics-informed transformer, v1.3)")
    p.add_argument("--angles", default="-30.5,5.1,40.2",
                   help="ground-truth angles in degrees, comma separated; "
                        "the count sets the source number K (default: K=3)")
    p.add_argument("--snr", type=float, default=-6.0, help="SNR in dB (default: -6)")
    p.add_argument("--trials", type=int, default=100,
                   help="number of Monte-Carlo trials (default: 100)")
    p.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="torch device (default: cuda if available)")
    args = p.parse_args()

    theta = [float(a) for a in args.angles.split(",")]
    print(f"Config: M={M} ULA (lambda/2)  N={N} snapshots  K={len(theta)} sources  "
          f"SNR={args.snr} dB  grid [-45,45] deg @ 0.1  device={args.device}")

    enc = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "doa_model.enc")
    if not os.path.exists(enc):
        print(f"!! model package not found: {enc} "
              f"(it must sit next to inference.py)")
        sys.exit(1)
    meta = doa.info(device=args.device)
    print(f"Model: {meta['arch']}  params {meta['params']:,}  "
          f"active {meta['active_params']:,}")

    errs, mse_db, miss = run(theta, args.snr, args.trials,
                              seed=args.seed, device=args.device)
    print(f"\nResult: MSE = {mse_db:.2f} dB  ({args.trials} trials / seed {args.seed}; "
          f"10-seed x 100 reference: -24.35 dB)")
    print("Per-source MAE: " + "  ".join(f"{v:.3f} deg" for v in errs.mean(dim=0).tolist()))
    print(f"Miss (any source error > 5 deg): {miss}/{args.trials}")


if __name__ == "__main__":
    main()
