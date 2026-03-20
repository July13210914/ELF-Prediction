#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_cnn.py

Train a compact 3D CNN regressor for ELF in [0,1] from patch datasets.

No prefix. All outputs explicit via --model-out/--hist-out/--log-out.
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Small3DCNN(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 16, drop: float = 0.0):
        super().__init__()
        b = int(base)
        self.conv1 = nn.Conv3d(in_ch, b, 3, padding=1)
        self.conv2 = nn.Conv3d(b, 2*b, 3, padding=1)
        self.conv3 = nn.Conv3d(2*b, 4*b, 3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.head = nn.Sequential(nn.Linear(4*b, 4*b), nn.ReLU(), nn.Linear(4*b, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x)); x = self.pool(x)
        x = F.relu(self.conv2(x)); x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.mean(dim=(-1, -2, -3))
        x = self.drop(x)
        x = self.head(x)
        return torch.sigmoid(x).view(-1)

def split_indices(N: int, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(N)
    n_test = max(1, int(round(float(test_frac) * N)))
    return idx[n_test:].astype(np.int64), idx[:n_test].astype(np.int64)

def main():
    p = argparse.ArgumentParser(description="Train 3D CNN on ELF density patches.")
    p.add_argument("--npz", required=True)
    p.add_argument("--model-out", default="cnn_model.pt")
    p.add_argument("--hist-out", default="cnn_history.npz")
    p.add_argument("--log-out", default="cnn_train.log")

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=5678)
    p.add_argument("--device", default="auto")

    p.add_argument("--base", type=int, default=16)
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--loss", choices=["huber","mse"], default="huber")
    p.add_argument("--huber-beta", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.2)
    args = p.parse_args()

    device = torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else args.device)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    data = np.load(args.npz, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(np.float32).reshape(-1)
    N = y.size
    train_idx, test_idx = split_indices(N, args.test_frac, args.seed)

    Xtr = torch.tensor(X[train_idx], device=device, dtype=torch.float32)
    ytr = torch.tensor(y[train_idx], device=device, dtype=torch.float32)
    Xte = torch.tensor(X[test_idx], device=device, dtype=torch.float32)
    yte = torch.tensor(y[test_idx], device=device, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, ytr),
                                               batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xte, yte),
                                              batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = Small3DCNN(in_ch=int(Xtr.shape[1]), base=args.base, drop=args.drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def loss_fn(pred, target):
        if args.loss == "mse":
            return F.mse_loss(pred, target)
        return F.smooth_l1_loss(pred, target, beta=float(args.huber_beta))

    hist: Dict[str, List[float]] = {"train_loss": [], "test_loss": []}

    with open(args.log_out, "w") as f:
        f.write(f"start_time: {time.ctime()}\n")
        f.write(f"npz: {args.npz}\nN={N} train={train_idx.size} test={test_idx.size}\n")
        f.write(f"X shape={X.shape} dtype={X.dtype}\n")
        f.write(f"device: {device}\nmodel: {model}\n")
        f.flush()

        for ep in range(1, args.epochs+1):
            model.train()
            tl=0.0; nb=0
            for xb, yb in train_loader:
                opt.zero_grad()
                pred=model(xb)
                loss=loss_fn(pred, yb)
                loss.backward()
                opt.step()
                tl += float(loss.item()); nb += 1
            tl /= max(nb, 1)

            model.eval()
            vl=0.0; nb2=0
            with torch.no_grad():
                for xb, yb in test_loader:
                    pred=model(xb)
                    loss=loss_fn(pred, yb)
                    vl += float(loss.item()); nb2 += 1
            vl /= max(nb2, 1)

            hist["train_loss"].append(tl)
            hist["test_loss"].append(vl)

            if ep == 1 or ep % 5 == 0:
                msg=f"epoch {ep:4d}/{args.epochs} train={tl:.6f} test={vl:.6f}\n"
                print(msg, end=""); f.write(msg); f.flush()

    torch.save(model.state_dict(), args.model_out)
    np.savez_compressed(args.hist_out,
                        train_loss=np.asarray(hist["train_loss"], dtype=np.float32),
                        test_loss=np.asarray(hist["test_loss"], dtype=np.float32),
                        train_idx=train_idx, test_idx=test_idx)
    print(f"saved model: {args.model_out}")
    print(f"saved history: {args.hist_out}")
    print(f"saved log: {args.log_out}")

if __name__=="__main__":
    main()
