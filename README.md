# One-Bit DOA Estimation with a Physics-Informed Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Inference demo for direction-of-arrival (DoA) estimation from **one-bit quantized** array measurements under **coherent sources** — the challenging regime where classical subspace methods degrade severely.

The network is a physics-informed transformer: the overcomplete array manifold matrix builds the query matrix (one query token per candidate angle), while the one-bit measurements build the key/value tokens, making cross-attention a learnable generalization of matched filtering.

This repository is self-contained: an inference script, a spectrum peak searcher, and a packaged model.

## Highlights

- **Physics-informed attention** — each query token is explicitly associated with a candidate steering vector; attention weights admit a matched-filter interpretation
- **Complex-valued algebra** implemented via explicit real–imaginary split (no complex dtypes, plain autograd)
- **Pyramid mixture-of-experts** FFN: 4 experts in layer 2, 8 experts in layer 3, top-1 routing
- **dB-domain power head** + log-domain parabolic sub-grid interpolation for sub-0.1° peak accuracy

| | |
|---|---|
| Total parameters | 2,278,037 |
| Active parameters (top-1) | 695,945 |
| FLOPs per forward | 2.70 G |
| Single-sample latency (fwd + peak search) | 9.4 ms on RTX PRO 6000 / 119 ms on CPU |

## Requirements

- Python >= 3.10
- PyTorch >= 2.0 (CUDA build recommended)
- `pip install -r requirements.txt`

## Quick Start

```bash
pip install -r requirements.txt

python inference.py                       # K=3 default scenario, SNR=-6 dB, 100 trials
python inference.py --snr 0               # change SNR
python inference.py --angles 10 25        # change source number / angles (K=2)
python inference.py --device cpu          # CPU only
```

Expected output (default configuration, seed 42):

```
Config: M=60 ULA (lambda/2)  N=64 snapshots  K=3 sources  SNR=-6.0 dB  grid [-45,45] deg @ 0.1  device=...
Model: physics-informed transformer v1.3 (3L/h4/d128, MoE [L2:S4, L3:S8] top-1)  params 2,278,037  active 695,945
  [trial 0] true ['-30.5', '  5.1', ' 40.2'] deg -> estimated ['-30.49', '  5.09', ' 40.21'] deg

Result: MSE = -24.36 dB  (100 trials / seed 42; 10-seed x 100 reference: -24.35 dB)
Per-source MAE: 0.045 deg  0.050 deg  0.046 deg
Miss (any source error > 5 deg): 0/100
```

## Usage as a Library

```python
import doa

angles = doa.predict(X_re, X_im, k=3)           # single sample: (M, N) -> [K angles]
angles = doa.predict_batch(Xs_re, Xs_im, k=3)   # batched: (B, M, N) -> B x [K angles]
grid = doa.steering_vector(...)                 # array geometry helper
meta = doa.info()                               # {"arch", "params", "active_params"}
```

The interface is intentionally minimal: the packaged model is encapsulated behind `predict` / `predict_batch`; only angle estimates leave the interface.

## Files

```
doa.py            model loading interface
doa_model.enc     packaged model (8.2 MB)
inference.py      end-to-end demo (signal generation -> forward -> peak search -> MSE)
searcher.py       spectrum peak search (find_peaks + parabolic interpolation)
```

## Reproduced Performance

Measured with this exact package (10 fixed seeds x 100 trials per seed).

**Note: all MSE results below were measured with CPU inference on the local machine — not on the RTX PRO 6000** (the GPU latency quoted in the table above comes from the original benchmark). Numerical results are device-independent up to floating-point reduction order.

| Scenario | MSE (dB) | Note |
|---|---|---|
| K=3, SNR=-6 dB, M=60, N=64 | **-24.37 ± 0.43** | in-distribution (default config) |
| K=2, SNR=-6 dB | -24.56 ± 0.58 | cross-source-number generalization |
| K=3, SNR=0 dB | -22.91 ± 0.41 | cross-SNR generalization |
| K=1, SNR=-6 dB | +1.91 ± 1.10 | out-of-distribution (single-source spectra) |

The packaged model was trained on a single condition (K=3, SNR=-6 dB, 230,400 samples); changing `--angles` / `--snr` therefore probes generalization. MSE = 10·log10(mean squared per-source angle error).

## Package Integrity

```
file    doa_model.enc
size    8,504,778 bytes (~8.2 MB)
source  run_pc_pyr_l1s4_l2s8 (final epoch 80, training set S=230400)
sha256  08fe3f35110051721db37d8e4edd9c2d79868a26b8697ba87f1ed9f38d786202
```

## Full Reproducibility Release

The model in this repository is distributed as a packaged inference demo. **Upon publication of the paper, we will open-source the complete network implementation (including training code) together with the model weights of all ablation and main experiments, enabling full reproduction of every reported result.**

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{li2026onebit,
  title   = {Efficient One-Bit DoA Estimation using Physics-Informed Transformer Network},
  author  = {Li, Xiao-Peng and Zheng, Ping-Ze and Huang, Lei and Shi, Zhang-Lei and Xu, Jingwei and So, Hing Cheung},
  journal = {to be updated upon acceptance},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
