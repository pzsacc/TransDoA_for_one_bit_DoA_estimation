"""One-bit DOA inference interface.

    import doa
    angles = doa.predict(X_re, X_im, k=3)           # single sample
    angles = doa.predict_batch(Xs_re, Xs_im, k=3)  # batched, (B, M, N) inputs
    doa.steering_vector                              # array geometry helper
    doa.info()                                        # package metadata
"""
import base64
import io
import os
import sys
import tempfile
import zipfile

import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from searcher import peaks_searching

ENC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doa_model.enc")
MAGIC = b"DOAENC1\x00"
PAD = bytes.fromhex(
    "5f8b13279d416ce03a770fc291542abd7e1860af23d94b861cf530ea670b9244")
M, N, ANGLE_MAX, RES = 60, 64, 45, 10


def _engine():
    st = {}

    def _key() -> bytes:
        parts = ["7u4vqZGVd/J4M38=", "bYqFYgVWvwMqT70=", "+U7kxE98DXTNyQ=="]
        out = bytearray()
        for i, p in enumerate(parts):
            raw = base64.b64decode(p)
            out.extend(a ^ b for a, b in zip(raw, PAD[i * 11:i * 11 + len(raw)]))
        return bytes(out)

    def _cleanup(d):
        if d and os.path.isdir(d):
            for root, dirs, files in os.walk(d, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                for dd in dirs:
                    os.rmdir(os.path.join(root, dd))
            os.rmdir(d)
            if d in sys.path:
                sys.path.remove(d)

    def _ensure(device):
        if st.get("dev") == device:
            return
        if not os.path.exists(ENC):
            raise FileNotFoundError(f"model package not found: {ENC}")
        blob = open(ENC, "rb").read()
        if not blob.startswith(MAGIC):
            raise ValueError("invalid package")
        data = AESGCM(_key()).decrypt(blob[8:20], blob[20:], None)
        tmp = tempfile.mkdtemp(prefix=".doa_")
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(tmp)
        sys.path.insert(0, tmp)
        import model
        net = model.from_pretrained(
            os.path.join(tmp, "weights", "anchor_pyr.pth"), device=device)
        sv = model.steering_vector
        _cleanup(tmp)
        spacing = torch.arange(M, dtype=torch.float32) * (0.5 * 0.15)
        grid = torch.linspace(-ANGLE_MAX, ANGLE_MAX, ANGLE_MAX * 2 * RES + 1)
        A_G = sv(grid, spacing, 0.15).transpose(0, 1)
        a_stack = torch.cat([A_G.real, A_G.imag], dim=-1).to(device)
        st.update(net=net, sv=sv, a_stack=a_stack, dev=device,
                  params=sum(p.numel() for p in net.parameters()))

    def _forward(Xs_re, Xs_im, k, device):
        Xr = torch.tensor(Xs_re, dtype=torch.float32)
        Xi = torch.tensor(Xs_im, dtype=torch.float32)
        X = Xr + 1j * Xi
        single = X.dim() == 2
        if single:
            X = X.unsqueeze(0)
        outs = []
        for i in range(0, X.shape[0], 8):
            xb = X[i:i + 64]
            Xt = xb.transpose(1, 2)
            enc = torch.cat([Xt.real, Xt.imag], dim=-1).to(device)
            with torch.no_grad():
                spec = st["net"](st["a_stack"].unsqueeze(0)
                                 .expand(xb.shape[0], -1, -1), enc)
                peaks = peaks_searching(spec, k, "spec_ip", RES, ANGLE_MAX, device)
            outs.extend(peaks.tolist())
        return outs[0] if single else outs

    def predict(X_re, X_im, k=3, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _ensure(device)
        return _forward(X_re, X_im, k, device)

    def predict_batch(Xs_re, Xs_im, k=3, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _ensure(device)
        return _forward(Xs_re, Xs_im, k, device)

    def info(device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _ensure(device)
        n = st["params"]
        per_ffn = (n - 695_945) // 12
        return {"arch": "physics-informed transformer v1.3 "
                        "(3L/h4/d128, MoE [L2:S4, L3:S8] top-1)",
                "params": n, "active_params": n - (4 + 8) * per_ffn}

    def steering_vector(angles_deg, spacing_range, wavelength):
        return st["sv"](angles_deg, spacing_range, wavelength)

    return predict, predict_batch, info, steering_vector


predict, predict_batch, info, steering_vector = _engine()
