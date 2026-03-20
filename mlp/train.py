#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_sh.py

Train a voxel-wise regressor for ELF in [0,1] using SH power-spectrum features.

Input .npz:
  X : (N,F) float16/float32 (features)
  y : (N,) float32          (ELF target in [0,1])

Core design:
  - Keep dataset on CPU; move minibatches to device (CPU/GPU) during training.
  - Supports:
      * linear: single affine layer + sigmoid
      * mlp   : MLP + sigmoid
  - Optional feature standardization computed on train split only.
  - Logs training/test metrics and saves model + training history.

Outputs:
  --model-out  : torch checkpoint (.pt)
  --hist-out   : npz with losses/metrics + indices + standardization stats
  --log-out    : human-readable log
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Models
# -----------------------------
class LinearELF(nn.Module):
    """Affine map + sigmoid -> [0,1]."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(x)).view(-1)


class MLP_ELF(nn.Module):
    """MLP + sigmoid -> [0,1]."""
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
            act_fn = nn.SiLU  # default

        mods: List[nn.Module] = []
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


# -----------------------------
# Utilities
# -----------------------------
def split_indices(N: int, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(N)
    n_test = max(1, int(round(float(test_frac) * N)))
    test_idx = idx[:n_test].astype(np.int64)
    train_idx = idx[n_test:].astype(np.int64)
    return train_idx, test_idx


@dataclass
class Standardizer:
    mean: torch.Tensor  # (F,)
    std: torch.Tensor   # (F,)

    def to(self, device: torch.device) -> "Standardizer":
        return Standardizer(self.mean.to(device), self.std.to(device))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def compute_standardizer(X_cpu: np.ndarray, train_idx: np.ndarray) -> Standardizer:
    # Compute in float64 for stability, store float32.
    Xtr = X_cpu[train_idx].astype(np.float64, copy=False)
    mu = Xtr.mean(axis=0)
    var = Xtr.var(axis=0)
    std = np.sqrt(var + 1e-12)
    return Standardizer(mean=torch.tensor(mu, dtype=torch.float32), std=torch.tensor(std, dtype=torch.float32))


@torch.no_grad()
def metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    e = (y_pred - y_true)
    mae = float(torch.mean(torch.abs(e)).item())
    mse = float(torch.mean(e * e).item())
    rmse = float(np.sqrt(mse))
    ybar = torch.mean(y_true)
    denom = float(torch.sum((y_true - ybar) ** 2).item())
    r2 = float(1.0 - (torch.sum(e * e).item() / denom)) if denom > 0 else float("nan")
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def loss_fn_builder(loss_name: str, huber_beta: float):
    if loss_name == "mse":
        def _loss(pred, target):
            return F.mse_loss(pred, target)
        return _loss
    beta = float(huber_beta)
    def _loss(pred, target):
        return F.smooth_l1_loss(pred, target, beta=beta)
    return _loss


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Train ELF regressor (linear or MLP) on SH features (.npz).")

    p.add_argument("--npz", required=True, help="Input dataset .npz with X (N,F) and y (N,).")

    # outputs
    p.add_argument("--model-out", default="sh_model.pt", help="Output PyTorch checkpoint.")
    p.add_argument("--hist-out", default="sh_history.npz", help="Output training history .npz.")
    p.add_argument("--log-out", default="sh_train.log", help="Output log file.")

    # training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=5678)
    p.add_argument("--device", default="auto")

    # split
    p.add_argument("--test-frac", type=float, default=0.2)

    # loss
    p.add_argument("--loss", choices=["huber", "mse"], default="huber")
    p.add_argument("--huber-beta", type=float, default=0.03)

    # feature preprocessing
    p.add_argument("--standardize", action="store_true",
                   help="Standardize X using mean/std computed on train split only.")

    # model selection
    p.add_argument("--model", choices=["linear", "mlp"], default="mlp")
    p.add_argument("--hidden", type=int, default=128, help="MLP hidden width.")
    p.add_argument("--layers", type=int, default=2, help="MLP number of hidden layers.")
    p.add_argument("--drop", type=float, default=0.0, help="MLP dropout.")
    p.add_argument("--act", choices=["silu", "relu", "gelu"], default="silu")

    args = p.parse_args()

    # logging
    with open(args.log_out, "w") as f:
        def log(msg: str):
            print(msg, end="")
            f.write(msg)
            f.flush()

        start = time.time()
        log(f"start_time: {time.ctime()}\n")
        log(f"args: {vars(args)}\n")

        device = get_device(args.device)
        log(f"device: {device}\n")

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        data = np.load(args.npz, allow_pickle=True)
        X_cpu = data["X"]
        y_cpu = data["y"].astype(np.float32).reshape(-1)
        N = int(y_cpu.size)
        Fdim = int(X_cpu.shape[1])

        log(f"npz: {args.npz}\n")
        log(f"N={N} F={Fdim} X_dtype={X_cpu.dtype} y_dtype={y_cpu.dtype}\n")

        train_idx, test_idx = split_indices(N, args.test_frac, args.seed)
        log(f"split: train={train_idx.size} test={test_idx.size} test_frac={args.test_frac}\n")

        # Standardization
        stdzr: Optional[Standardizer] = None
        if args.standardize:
            stdzr = compute_standardizer(X_cpu, train_idx)
            log("standardize: enabled (train-split mean/std)\n")
            log(f"  mean: mean={float(stdzr.mean.mean()):.6e} std={float(stdzr.mean.std()):.6e}\n")
            log(f"  std : mean={float(stdzr.std.mean()):.6e} std={float(stdzr.std.std()):.6e}\n")
        else:
            log("standardize: disabled\n")

        # Create CPU datasets (no device transfer here)
        Xtr = torch.tensor(X_cpu[train_idx], dtype=torch.float32, device="cpu")
        ytr = torch.tensor(y_cpu[train_idx], dtype=torch.float32, device="cpu")
        Xte = torch.tensor(X_cpu[test_idx], dtype=torch.float32, device="cpu")
        yte = torch.tensor(y_cpu[test_idx], dtype=torch.float32, device="cpu")

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtr, ytr),
            batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=(device.type == "cuda")
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xte, yte),
            batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=(device.type == "cuda")
        )

        # Model
        if args.model == "linear":
            model: nn.Module = LinearELF(Fdim)
        else:
            model = MLP_ELF(Fdim, hidden=args.hidden, layers=args.layers, drop=args.drop, act=args.act)
        model = model.to(device)
        log(f"model: {args.model}\n{model}\n")

        # Optim / loss
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = loss_fn_builder(args.loss, args.huber_beta)

        # Move standardizer to device (used batch-wise)
        stdzr_dev: Optional[Standardizer] = stdzr.to(device) if stdzr is not None else None

        hist: Dict[str, List[float]] = {
            "train_loss": [], "test_loss": [],
            "train_mae": [], "test_mae": [],
            "train_rmse": [], "test_rmse": [],
            "train_r2": [], "test_r2": [],
        }

        def forward_batch(xb: torch.Tensor) -> torch.Tensor:
            if stdzr_dev is not None:
                xb = stdzr_dev(xb)
            return model(xb)

        # Training loop
        for ep in range(1, args.epochs + 1):
            model.train()
            tl = 0.0
            nb = 0
            for xb_cpu, yb_cpu in train_loader:
                xb = xb_cpu.to(device, non_blocking=True)
                yb = yb_cpu.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = forward_batch(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                tl += float(loss.item())
                nb += 1
            tl /= max(nb, 1)

            # Evaluate (full-batch metrics)
            model.eval()
            with torch.no_grad():
                # Train metrics
                preds_tr = []
                ys_tr = []
                for xb_cpu, yb_cpu in train_loader:
                    xb = xb_cpu.to(device, non_blocking=True)
                    yb = yb_cpu.to(device, non_blocking=True)
                    preds_tr.append(forward_batch(xb))
                    ys_tr.append(yb)
                yhat_tr = torch.cat(preds_tr, dim=0)
                y_tr = torch.cat(ys_tr, dim=0)
                m_tr = metrics(y_tr, yhat_tr)

                # Test loss + metrics
                vl = 0.0
                nb2 = 0
                preds_te = []
                ys_te = []
                for xb_cpu, yb_cpu in test_loader:
                    xb = xb_cpu.to(device, non_blocking=True)
                    yb = yb_cpu.to(device, non_blocking=True)
                    pred = forward_batch(xb)
                    loss = loss_fn(pred, yb)
                    vl += float(loss.item())
                    nb2 += 1
                    preds_te.append(pred)
                    ys_te.append(yb)
                vl /= max(nb2, 1)
                yhat_te = torch.cat(preds_te, dim=0)
                y_te = torch.cat(ys_te, dim=0)
                m_te = metrics(y_te, yhat_te)

            # Record history
            hist["train_loss"].append(tl)
            hist["test_loss"].append(vl)
            hist["train_mae"].append(m_tr["mae"])
            hist["test_mae"].append(m_te["mae"])
            hist["train_rmse"].append(m_tr["rmse"])
            hist["test_rmse"].append(m_te["rmse"])
            hist["train_r2"].append(m_tr["r2"])
            hist["test_r2"].append(m_te["r2"])

            if ep == 1 or ep % 5 == 0 or ep == args.epochs:
                log(
                    f"epoch {ep:4d}/{args.epochs}  "
                    f"loss train={tl:.6f} test={vl:.6f}  "
                    f"MAE train={m_tr['mae']:.6f} test={m_te['mae']:.6f}  "
                    f"RMSE train={m_tr['rmse']:.6f} test={m_te['rmse']:.6f}  "
                    f"R2 train={m_tr['r2']:.6f} test={m_te['r2']:.6f}\n"
                )

        # Save checkpoint
        ckpt = {
            "model_type": args.model,
            "state_dict": model.state_dict(),
            "in_dim": Fdim,
            "args": vars(args),
            "standardize": bool(args.standardize),
        }
        if stdzr is not None:
            ckpt["x_mean"] = stdzr.mean.cpu()
            ckpt["x_std"] = stdzr.std.cpu()

        torch.save(ckpt, args.model_out)
        log(f"saved model: {args.model_out}\n")

        # Save history + indices + standardizer stats
        out = {
            "train_loss": np.asarray(hist["train_loss"], dtype=np.float32),
            "test_loss": np.asarray(hist["test_loss"], dtype=np.float32),
            "train_mae": np.asarray(hist["train_mae"], dtype=np.float32),
            "test_mae": np.asarray(hist["test_mae"], dtype=np.float32),
            "train_rmse": np.asarray(hist["train_rmse"], dtype=np.float32),
            "test_rmse": np.asarray(hist["test_rmse"], dtype=np.float32),
            "train_r2": np.asarray(hist["train_r2"], dtype=np.float32),
            "test_r2": np.asarray(hist["test_r2"], dtype=np.float32),
            "train_idx": train_idx,
            "test_idx": test_idx,
        }
        if stdzr is not None:
            out["x_mean"] = stdzr.mean.cpu().numpy().astype(np.float32)
            out["x_std"] = stdzr.std.cpu().numpy().astype(np.float32)

        np.savez_compressed(args.hist_out, **out)
        log(f"saved history: {args.hist_out}\n")

        log(f"elapsed_total={time.time()-start:.2f}s\n")


if __name__ == "__main__":
    main()

