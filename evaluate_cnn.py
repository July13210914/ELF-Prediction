#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evaluate_cnn.py

Evaluate trained 3D CNN on patch dataset.
Writes: i j k y_true y_pred err abs_err

Optional: residual correlation length for 2D slices (xy/xz/yz) using FFT.
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Small3DCNN(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 16, drop: float = 0.0):
        super().__init__()
        b = int(base)
        self.conv1 = nn.Conv3d(in_ch, b, 3, padding=1)
        self.conv2 = nn.Conv3d(b, 2 * b, 3, padding=1)
        self.conv3 = nn.Conv3d(2 * b, 4 * b, 3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(4 * b, 4 * b),
            nn.ReLU(),
            nn.Linear(4 * b, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.mean(dim=(-1, -2, -3))
        x = self.drop(x)
        x = self.head(x)
        return torch.sigmoid(x).view(-1)


def autocorr2d_fft(E: np.ndarray) -> np.ndarray:
    E0 = E - np.mean(E)
    Fk = np.fft.fftn(E0)
    C = np.fft.ifftn(Fk * np.conjugate(Fk)).real
    if C[0, 0] != 0:
        C = C / C[0, 0]
    return C


def radial_average_2d(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ny, nx = C.shape
    y = np.minimum(np.arange(ny), ny - np.arange(ny))
    x = np.minimum(np.arange(nx), nx - np.arange(nx))
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r = np.sqrt(xx * xx + yy * yy)
    r_int = r.astype(np.int32)
    r_max = r_int.max()
    Cr = np.zeros(r_max + 1, dtype=np.float64)
    Nr = np.zeros(r_max + 1, dtype=np.int64)
    np.add.at(Cr, r_int.ravel(), C.ravel())
    np.add.at(Nr, r_int.ravel(), 1)
    Nr = np.maximum(Nr, 1)
    return np.arange(r_max + 1, dtype=np.float64), Cr / Nr


def predict_batched(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    n = X.shape[0]
    y_pred = np.empty(n, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            xb = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
            yb = model(xb)

            y_pred[start:end] = yb.detach().cpu().numpy().astype(np.float32)

            del xb, yb
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if start == 0 or end == n or ((start // batch_size) % 100 == 0):
                print(f"[predict] {end}/{n}")

    return y_pred


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate 3D CNN ELF regressor.")
    p.add_argument("--npz", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Inference batch size to avoid GPU OOM.")
    p.add_argument("--out", default="cnn_eval.dat")
    p.add_argument("--corr", action="store_true")
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--corr-out", default=None)
    p.add_argument("--corr-threshold", type=float, default=float(np.exp(-1)))
    args = p.parse_args(argv)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )

    data = np.load(args.npz, allow_pickle=True)
    X = data["X"]
    y_true = data["y"].astype(np.float32).reshape(-1)
    ijk = data["ijk"].astype(np.int64) if "ijk" in data else -np.ones((y_true.size, 3), dtype=np.int64)

    print(f"[data] X.shape={X.shape} X.dtype={X.dtype} y.shape={y_true.shape}")
    print(f"[device] using {device}")

    model = Small3DCNN(in_ch=int(X.shape[1]), base=args.base, drop=args.drop).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    y_pred = predict_batched(model, X, device, batch_size=args.batch_size)

    print(
        f"[pred_stats] y_true: mean={float(np.mean(y_true)):.6f} "
        f"std={float(np.std(y_true)):.6f} min={float(np.min(y_true)):.6f} max={float(np.max(y_true)):.6f}"
    )
    print(
        f"[pred_stats] y_pred: mean={float(np.mean(y_pred)):.6f} "
        f"std={float(np.std(y_pred)):.6f} min={float(np.min(y_pred)):.6f} max={float(np.max(y_pred)):.6f}"
    )

    err = (y_pred - y_true).astype(np.float32)
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1.0 - (np.sum(err * err) / denom)) if denom > 0 else float("nan")
    print(f"[metrics] MAE={mae:.6f}  MSE={mse:.6f}  RMSE={rmse:.6f}  R2={r2:.6f}")

    if "src" in data.files:
        src = data["src"].astype(np.int64).reshape(-1)
        for s in np.unique(src):
            msk = (src == s)
            if not np.any(msk):
                continue
            e = err[msk]
            yt = y_true[msk]
            mae_s = float(np.mean(np.abs(e)))
            mse_s = float(np.mean(e * e))
            rmse_s = float(np.sqrt(mse_s))
            denom_s = float(np.sum((yt - float(np.mean(yt))) ** 2))
            r2_s = float(1.0 - (np.sum(e * e) / denom_s)) if denom_s > 0 else float("nan")
            print(f"[metrics][src{s}] MAE={mae_s:.6f}  MSE={mse_s:.6f}  RMSE={rmse_s:.6f}  R2={r2_s:.6f}")

    out_tbl = np.column_stack([ijk, y_true, y_pred, err, np.abs(err)]).astype(np.float64)
    np.savetxt(args.out, out_tbl, fmt="%.6f", header="i j k y_true y_pred err abs_err")
    print(f"[evaluate_cnn] wrote: {args.out}")

    if args.corr:
        dim = int(args.dim) if args.dim is not None else int(np.max(ijk[:, :3]) + 1)
        k_unique = np.unique(ijk[:, 2])
        j_unique = np.unique(ijk[:, 1])
        i_unique = np.unique(ijk[:, 0])

        if k_unique.size == 1:
            E = np.zeros((dim, dim), dtype=np.float64)
            E[ijk[:, 1], ijk[:, 0]] = err
            kind = "xy"
        elif j_unique.size == 1:
            E = np.zeros((dim, dim), dtype=np.float64)
            E[ijk[:, 2], ijk[:, 0]] = err
            kind = "xz"
        elif i_unique.size == 1:
            E = np.zeros((dim, dim), dtype=np.float64)
            E[ijk[:, 2], ijk[:, 1]] = err
            kind = "yz"
        else:
            print("[corr] not a single-plane slice; skipping.")
            return

        C = autocorr2d_fft(E)
        r, Cr = radial_average_2d(C)
        thr = float(args.corr_threshold)
        idx = np.where(Cr < thr)[0]
        rc = float(r[idx[0]]) if idx.size > 0 else float(r[-1])
        print(f"[corr] ({kind}): corr_len={rc:.2f} vox (threshold={thr:.3f})")

        if args.corr_out:
            np.savetxt(
                args.corr_out,
                np.column_stack([r, Cr]),
                header=f"r_vox  C(r)\n# corr_len_vox={rc:.6f} threshold={thr:.6f} slice={kind}",
            )
            print(f"[corr] wrote: {args.corr_out}")


if __name__ == "__main__":
    main()
