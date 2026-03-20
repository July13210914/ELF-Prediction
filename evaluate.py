#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_sh.py

Evaluate voxel-wise ELF regressor (linear/MLP) trained on SH features.

Writes: i j k y_true y_pred err abs_err
No autocorrelation.

Safe dtype behavior:
  - default: float32 input + float32 model
  - optional: --dtype float16 (CUDA only) casts BOTH model and inputs to float16
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearELF(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(x)).view(-1)


class MLP_ELF(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 2, drop: float = 0.0, act: str = "silu"):
        super().__init__()
        hidden = int(hidden)
        layers = int(layers)
        drop = float(drop)

        if act.lower() == "relu":
            act_fn = nn.ReLU
        elif act.lower() == "gelu":
            act_fn = nn.GELU
        else:
            act_fn = nn.SiLU

        mods = []
        d = int(in_dim)
        for _ in range(max(layers, 1)):
            mods.append(nn.Linear(d, hidden))
            mods.append(act_fn())
            if drop > 0:
                mods.append(nn.Dropout(drop))
            d = hidden
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).view(-1)


@torch.no_grad()
def metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.float64, copy=False).reshape(-1)
    y_pred = y_pred.astype(np.float64, copy=False).reshape(-1)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1.0 - (np.sum(err * err) / denom)) if denom > 0 else float("nan")
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model_from_ckpt(ckpt: dict, in_dim: int) -> nn.Module:
    model_type = ckpt.get("model_type", None)
    args = ckpt.get("args", {}) or {}

    if model_type == "linear":
        model = LinearELF(in_dim)
    elif model_type == "mlp":
        model = MLP_ELF(
            in_dim=in_dim,
            hidden=int(args.get("hidden", 128)),
            layers=int(args.get("layers", 2)),
            drop=float(args.get("drop", 0.0)),
            act=str(args.get("act", "silu")),
        )
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")

    sd = ckpt.get("state_dict", None)
    if sd is None:
        raise ValueError("Checkpoint missing 'state_dict'.")
    model.load_state_dict(sd, strict=True)
    return model


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate SH-feature ELF regressor (linear/MLP).")
    p.add_argument("--npz", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--out", default="sh_eval.dat")

    p.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                   help="Compute dtype. float16 requires CUDA and casts BOTH model and inputs.")
    args = p.parse_args(argv)

    device = get_device(args.device)
    data = np.load(args.npz, allow_pickle=True)
    X = data["X"]
    y_true = data["y"].astype(np.float32).reshape(-1)
    N = int(y_true.size)

    if "ijk" in data.files:
        ijk = data["ijk"].astype(np.int64)
    else:
        ijk = -np.ones((N, 3), dtype=np.int64)

    ckpt = torch.load(args.model, map_location="cpu")
    in_dim = int(X.shape[1])

    # dtype policy
    if args.dtype == "float16":
        if device.type != "cuda":
            raise ValueError("float16 evaluation requires CUDA. Use --dtype float32 on CPU.")
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    model = build_model_from_ckpt(ckpt, in_dim=in_dim).to(device=device, dtype=model_dtype)
    model.eval()

    # standardization (store as float32; cast to model_dtype at runtime)
    use_std = bool(ckpt.get("standardize", False))
    if use_std:
        x_mean = ckpt.get("x_mean", None)
        x_std = ckpt.get("x_std", None)
        if x_mean is None or x_std is None:
            raise ValueError("Checkpoint indicates standardize=True but x_mean/x_std not found.")
        x_mean_t = x_mean.to(device=device, dtype=torch.float32).view(1, -1)
        x_std_t = x_std.to(device=device, dtype=torch.float32).view(1, -1)

    y_pred = np.empty((N,), dtype=np.float32)
    bs = max(1, int(args.batch_size))

    with torch.no_grad():
        for s0 in range(0, N, bs):
            s1 = min(N, s0 + bs)

            xb = torch.tensor(X[s0:s1], device=device, dtype=torch.float32)
            if use_std:
                xb = (xb - x_mean_t) / x_std_t

            xb = xb.to(dtype=model_dtype)
            pred = model(xb).detach().to("cpu", dtype=torch.float32).numpy()
            y_pred[s0:s1] = pred

    print(f"[eval] npz={args.npz}  model={args.model}")
    print(f"[eval] N={N} F={in_dim} device={device} compute_dtype={args.dtype} standardize={use_std}")
    print(f"[pred_stats] y_true: mean={float(np.mean(y_true)):.6f} std={float(np.std(y_true)):.6f} "
          f"min={float(np.min(y_true)):.6f} max={float(np.max(y_true)):.6f}")
    print(f"[pred_stats] y_pred: mean={float(np.mean(y_pred)):.6f} std={float(np.std(y_pred)):.6f} "
          f"min={float(np.min(y_pred)):.6f} max={float(np.max(y_pred)):.6f}")

    m = metrics_np(y_true, y_pred)
    print(f"[metrics] MAE={m['mae']:.6f}  MSE={m['mse']:.6f}  RMSE={m['rmse']:.6f}  R2={m['r2']:.6f}")

    err = (y_pred - y_true).astype(np.float32)
    out_tbl = np.column_stack([ijk, y_true, y_pred, err, np.abs(err)]).astype(np.float64)
    np.savetxt(args.out, out_tbl, fmt="%.6f", header="i j k y_true y_pred err abs_err")
    print(f"[evaluate_sh] wrote: {args.out}")


if __name__ == "__main__":
    main()

