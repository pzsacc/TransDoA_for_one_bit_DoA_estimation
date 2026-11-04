"""Peak search and error metrics (explicitly parameterized, no global state)."""

import numpy as np
import torch
from scipy.signal import find_peaks


def _select_spec_peaks(spec, source_num):
    """Core peak selection (find_peaks, distance=10, pads with the strongest
    peak when fewer peaks are found). Returns selected grid indices."""
    padded = np.pad(spec, (1, 1), mode="constant", constant_values=spec.min() - 100)
    pk, _ = find_peaks(padded, distance=10, prominence=0)
    pk = (pk - 1)[(pk - 1 >= 0) & (pk - 1 < len(spec))]
    if len(pk) >= source_num:
        return np.sort(pk[np.argsort(spec[pk])[-source_num:]])
    if len(pk) > 0:
        pad = np.full(source_num - len(pk), pk[np.argmax(spec[pk])])
        return np.sort(np.concatenate((pk, pad)))
    return np.zeros(source_num, dtype=int)


def _parabolic_interp(spec, pk):
    """Three-point parabolic sub-grid interpolation (XQIFFT style, log domain).

    The log of a Gaussian peak is exactly a parabola whose vertex sits at the
    true angle -- matching the spectral shape induced by the Gaussian soft
    labels (sigma = 0.3 deg) used in training.
    delta = 0.5*(alpha - gamma)/(alpha - 2*beta + gamma), clipped to +-0.5.
    """
    s = np.log(spec + 1e-12)
    out = pk.astype(float)
    for j, i in enumerate(pk):
        if 0 < i < len(spec) - 1:
            a_, b_, c_ = s[i - 1], s[i], s[i + 1]
            denom = a_ - 2 * b_ + c_
            if abs(denom) > 1e-12:
                out[j] = i + float(np.clip(0.5 * (a_ - c_) / denom, -0.5, 0.5))
    return out


def peaks_searching(prediction, source_num, method, resolution, angle_max, device):
    """Search K peak angles (sorted) from the spatial spectrum prediction (B, G).

    method: 'topk' (batched top-K, fast) | 'spec' (find_peaks, per-sample,
    as in the original implementation) | 'spec_ip' (spec + log-domain parabolic
    sub-grid interpolation).
    """
    if method == "topk":
        idx = torch.topk(prediction, k=source_num, dim=1)[1]          # (B, K)
        angles = idx.float() / resolution - angle_max
        return torch.sort(angles, dim=1)[0]
    if method in ("spec", "spec_ip"):
        out = []
        for i in range(prediction.shape[0]):
            spec = prediction[i].detach().cpu().numpy()
            sel = _select_spec_peaks(spec, source_num)
            if method == "spec_ip":
                out.append(_parabolic_interp(spec, sel))
            else:
                out.append(sel)
        idx = torch.from_numpy(np.stack(out)).to(device)
        return idx.float() / resolution - angle_max
    raise ValueError(f"unknown peaks method: {method}")
