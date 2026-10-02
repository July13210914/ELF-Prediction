#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a variable-element voxel ELF MLP.

The checkpoint is self-contained: model weights, architecture, species list,
descriptor hyperparameters, feature slices, and X standardizer are all stored in
best_model.pt. No external scaler_x.npz is written or required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, "/projects/academic/ezurek/xiaoyu/elf/")

from variable_elf_common import (
    MLPRegressor,
    build_file_to_structure_map,
    expected_pair_soap_nfeatures,
    fit_standardizer,
    log,
    pick_device,
    r2_np,
    run_epoch,
    set_seed,
    transform_standardizer,
)


def build_structure_split(structure_labels: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Tuple[set, set, set]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")
    uniq = sorted(set(structure_labels))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = max(1, int(round(train_ratio * n)))
    n_val = max(1, int(round(val_ratio * n)))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
    while n_train + n_val + n_test > n:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
    return set(uniq[:n_train]), set(uniq[n_train:n_train+n_val]), set(uniq[n_train+n_val:])


def masks_from_file_index(file_index, file_to_structure, train_structs, val_structs, test_structs):
    struct_labels = np.array([file_to_structure[int(i)] for i in file_index], dtype=object)
    return (
        np.isin(struct_labels, list(train_structs)),
        np.isin(struct_labels, list(val_structs)),
        np.isin(struct_labels, list(test_structs)),
        struct_labels,
    )


def evaluate_per_structure(y_true, y_pred, struct_labels):
    from variable_elf_common import mae_np, rmse_np
    out = {}
    for s in sorted(set(struct_labels.tolist())):
        mask = struct_labels == s
        yt, yp = y_true[mask], y_pred[mask]
        out[s] = {"n": int(mask.sum()), "mae": mae_np(yt, yp), "rmse": rmse_np(yt, yp), "r2": r2_np(yt, yp)}
    return out


def main():
    ap = argparse.ArgumentParser(description="Train variable-element voxel ELF MLP regressor with Huber loss.")
    ap.add_argument("dataset", help="Input dataset .npz from make_training_data_variable_elements.py")
    ap.add_argument("--meta", required=True, help="Input metadata .json")
    ap.add_argument("--outdir", default="train_variable_mlp_huber")
    ap.add_argument("--label-mode", choices=["auto", "parent", "grandparent", "structure-pressure", "pressure-structure"], default="auto")

    ap.add_argument("--train-ratio", type=float, default=0.80)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--huber-delta", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-out", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    log_fh = open(args.log_out, "w", encoding="utf-8") if args.log_out else None

    try:
        log(f"Using device: {device}", log_fh)
        data = np.load(args.dataset, allow_pickle=True)
        with open(args.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)
        file_index = data["file_index"].astype(np.int32)
        dataset_species = [str(x) for x in data["species_list"].tolist()]
        meta_species = list(meta.get("species_list", []))
        if dataset_species != meta_species:
            raise ValueError(f"Dataset/meta species mismatch: dataset={dataset_species}, meta={meta_species}")
        expected = expected_pair_soap_nfeatures(len(dataset_species), int(meta["n_radial"]), int(meta["lmax"]))
        if X.shape[1] != expected or int(meta["nfeatures"]) != expected:
            raise ValueError(f"Feature dimension mismatch: X={X.shape[1]}, meta={meta.get('nfeatures')}, expected={expected}")
        log(f"Species channels ({len(dataset_species)}): {' '.join(dataset_species)}", log_fh)
        log(f"Loaded dataset: X={X.shape}, y={y.shape}", log_fh)

        file_to_structure = build_file_to_structure_map(meta, args.label_mode)
        all_structures = [file_to_structure[int(i)] for i in sorted(file_to_structure.keys())]
        train_structs, val_structs, test_structs = build_structure_split(all_structures, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
        train_mask, val_mask, test_mask, structure_per_sample = masks_from_file_index(file_index, file_to_structure, train_structs, val_structs, test_structs)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        log(f"Train: {X_train.shape[0]} samples from {len(train_structs)} structures", log_fh)
        log(f"Val:   {X_val.shape[0]} samples from {len(val_structs)} structures", log_fh)
        log(f"Test:  {X_test.shape[0]} samples from {len(test_structs)} structures", log_fh)

        x_mean, x_std = fit_standardizer(X_train)
        X_train_s = transform_standardizer(X_train, x_mean, x_std)
        X_val_s = transform_standardizer(X_val, x_mean, x_std)
        X_test_s = transform_standardizer(X_test, x_mean, x_std)

        pin_memory = device.type == "cuda"
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train)), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val)), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_s), torch.from_numpy(y_test)), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

        model = MLPRegressor(X.shape[1], args.hidden_dims, args.dropout, True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.HuberLoss(delta=args.huber_delta)
        history = []
        best_val_rmse = float("inf"); best_epoch = -1; patience_counter = 0
        best_ckpt_path = outdir / "best_model.pt"

        descriptor_meta = {
            "species_list": dataset_species,
            "nfeatures": int(X.shape[1]),
            "r_cut": float(meta["r_cut"]),
            "n_radial": int(meta["n_radial"]),
            "lmax": int(meta["lmax"]),
            "sigma_r": float(meta["sigma_r"]),
            "feature_slices": meta.get("feature_slices"),
            "representation": meta.get("representation", "variable_element_pair_channel_SOAP_power_spectrum"),
            "invariants": meta.get("invariants", "all same/cross species-pair channels; no explicit three-element invariant"),
        }

        for epoch in range(1, args.epochs + 1):
            train_metrics, _, _ = run_epoch(model, train_loader, optimizer, loss_fn, device)
            val_metrics, _, _ = run_epoch(model, val_loader, None, loss_fn, device)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"], "train_mae": train_metrics["mae"], "train_rmse": train_metrics["rmse"], "train_r2": train_metrics["r2"],
                "val_loss": val_metrics["loss"], "val_mae": val_metrics["mae"], "val_rmse": val_metrics["rmse"], "val_r2": val_metrics["r2"],
            }
            history.append(row)
            log(f"Epoch {epoch:4d} | train rmse {train_metrics['rmse']:.6f} r2 {train_metrics['r2']:.4f} | val rmse {val_metrics['rmse']:.6f} r2 {val_metrics['r2']:.4f}", log_fh)
            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]; best_epoch = epoch; patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "in_dim": int(X.shape[1]),
                    "hidden_dims": list(args.hidden_dims),
                    "dropout": float(args.dropout),
                    "use_sigmoid_output": True,
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "descriptor": descriptor_meta,
                    "best_epoch": int(best_epoch),
                    "best_val_rmse": float(best_val_rmse),
                    "args": vars(args),
                }, best_ckpt_path)
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                log(f"Early stopping at epoch {epoch} (best epoch = {best_epoch})", log_fh)
                break

        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model = MLPRegressor(ckpt["in_dim"], ckpt["hidden_dims"], ckpt["dropout"], ckpt["use_sigmoid_output"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        train_metrics, ytr, ptr = run_epoch(model, train_loader, None, loss_fn, device)
        val_metrics, yva, pva = run_epoch(model, val_loader, None, loss_fn, device)
        test_metrics, yte, pte = run_epoch(model, test_loader, None, loss_fn, device)

        log("", log_fh)
        log("Best checkpoint evaluation", log_fh)
        log(f"Train | MAE {train_metrics['mae']:.6f} RMSE {train_metrics['rmse']:.6f} R2 {train_metrics['r2']:.4f}", log_fh)
        log(f"Val   | MAE {val_metrics['mae']:.6f} RMSE {val_metrics['rmse']:.6f} R2 {val_metrics['r2']:.4f}", log_fh)
        log(f"Test  | MAE {test_metrics['mae']:.6f} RMSE {test_metrics['rmse']:.6f} R2 {test_metrics['r2']:.4f}", log_fh)

        np.savez_compressed(outdir / "predictions.npz", y_train_true=ytr.astype(np.float32), y_train_pred=ptr.astype(np.float32), y_val_true=yva.astype(np.float32), y_val_pred=pva.astype(np.float32), y_test_true=yte.astype(np.float32), y_test_pred=pte.astype(np.float32))
        split_info = {
            "species_list": dataset_species,
            "train_structures": sorted(train_structs), "val_structures": sorted(val_structs), "test_structures": sorted(test_structs),
            "best_epoch": best_epoch, "best_val_rmse": best_val_rmse,
            "final_metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
            "standardizer_stored_in_checkpoint": True,
            "checkpoint": str(best_ckpt_path.resolve()),
            "per_structure": {
                "train": evaluate_per_structure(ytr, ptr, structure_per_sample[train_mask]),
                "val": evaluate_per_structure(yva, pva, structure_per_sample[val_mask]),
                "test": evaluate_per_structure(yte, pte, structure_per_sample[test_mask]),
            },
        }
        with open(outdir / "split_and_metrics.json", "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2)
        with open(outdir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        log(f"\nSaved outputs in: {outdir.resolve()}", log_fh)
        log("No external scaler_x.npz was written; the standardizer is stored in best_model.pt.", log_fh)
    finally:
        if log_fh:
            log_fh.close()


if __name__ == "__main__":
    main()
