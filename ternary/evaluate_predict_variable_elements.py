#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate or predict ELF from ELFCAR geometry using a variable-element model.

This script does not require a precomputed dataset or an external scaler file.
It reads descriptor hyperparameters, species channels, and the standardizer from
best_model.pt. For input structures containing species absent from the model,
those atoms are ignored and a warning is reported.

Modes:
  Evaluation: input ELFCARs have DFT ELF values; metrics are computed on sampled or full grids.
  Prediction: same input format is used for geometry; predictions can be written as ELFCAR-like files.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, "/projects/academic/ezurek/xiaoyu/elf/")

from variable_elf_common import (
    MLPRegressor,
    build_descriptor_batch_torch,
    check_geometry_consistency,
    expand_species,
    log,
    mae_np,
    pick_device,
    read_vasp_scalar_field,
    rmse_np,
    r2_np,
    sample_voxel_indices,
    set_seed,
    str_to_torch_dtype,
    transform_standardizer,
    write_vasp_scalar_field_like,
)


def predict_array(model, Xs, batch_size, device, num_workers=0):
    dummy = np.zeros((Xs.shape[0],), dtype=np.float32)
    ds = TensorDataset(torch.from_numpy(Xs.astype(np.float32)), torch.from_numpy(dummy))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Evaluate/predict ELFCAR directly using variable-element pair-channel model.")
    ap.add_argument("elfcars", nargs="+", help="Input ELFCAR-like files. In prediction mode, the scalar values are only used to read grid dimensions unless --no-metrics is omitted.")
    ap.add_argument("--checkpoint", required=True, help="Self-contained best_model.pt from train_variable_elements.py")
    ap.add_argument("--outdir", default="eval_predict_variable")
    ap.add_argument("--no-metrics", action="store_true", help="Do not compare against input ELF grid values; prediction only.")
    ap.add_argument("--num-samples-per-file", type=int, default=None, help="Evaluate/predict only sampled voxels. Default: full grid.")
    ap.add_argument("--write-pred-elfcar", action="store_true", help="Write full predicted ELFCAR-like files. Requires full-grid prediction, so it ignores --num-samples-per-file for writing.")
    ap.add_argument("--write-sampled-npz", action="store_true", help="Write per-file sampled prediction npz files.")

    ap.add_argument("--geometry-source", choices=["elfcar", "poscar", "strict"], default="elfcar")
    ap.add_argument("--lattice-tol", type=float, default=1e-5)
    ap.add_argument("--frac-tol", type=float, default=1e-5)
    ap.add_argument("--descriptor-batch", type=int, default=1024)
    ap.add_argument("--eval-batch-size", type=int, default=4096)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-out", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    dtype = str_to_torch_dtype(args.dtype)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    log_fh = open(args.log_out, "w", encoding="utf-8") if args.log_out else None

    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        descriptor = ckpt["descriptor"]
        species_list = list(descriptor["species_list"])
        x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
        x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
        model = MLPRegressor(ckpt["in_dim"], ckpt["hidden_dims"], ckpt["dropout"], ckpt["use_sigmoid_output"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        rcut = float(descriptor["r_cut"])
        nrad = int(descriptor["n_radial"])
        lmax = int(descriptor["lmax"])
        sigma_r = float(descriptor["sigma_r"])
        log(f"Using device: {device}", log_fh)
        log("Model species channels: " + " ".join(species_list), log_fh)
        log(f"Descriptor: r_cut={rcut}, n_radial={nrad}, lmax={lmax}, sigma_r={sigma_r}, in_dim={ckpt['in_dim']}", log_fh)

        summary_rows = []
        all_unknown = set()
        csv_path = outdir / "evaluation_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=["file", "n", "mae", "rmse", "r2", "unknown_species_ignored", "written_pred_elfcar"])
            writer.writeheader()

            for ifile, elfcar in enumerate(args.elfcars):
                elfcar = str(Path(elfcar).resolve())
                lattice_e, species_e, counts_e, atom_frac_e, grid = read_vasp_scalar_field(elfcar)
                lattice, species, counts, atom_frac, poscar_path = check_geometry_consistency(
                    elfcar, lattice_e, species_e, counts_e, atom_frac_e,
                    args.geometry_source, args.lattice_tol, args.frac_tol, log_fh,
                )
                atom_species = expand_species(species, counts)
                nx, ny, nz = grid.shape
                unknown = sorted(set(atom_species) - set(species_list))
                if unknown:
                    all_unknown.update(unknown)
                    log(f"[WARN] {elfcar}: species not in model and ignored: {' '.join(unknown)}", log_fh)

                # Sampled/full evaluation path.
                nsample = None if args.num_samples_per_file is None else int(args.num_samples_per_file)
                _, ijk, voxel_frac = sample_voxel_indices(nx, ny, nz, nsample, args.seed + 1009 * (ifile + 1))
                X_chunks = []
                for start in range(0, len(voxel_frac), args.descriptor_batch):
                    end = min(start + args.descriptor_batch, len(voxel_frac))
                    Xb, _, unk_batch = build_descriptor_batch_torch(
                        voxel_frac_np=voxel_frac[start:end], lattice_np=lattice, atom_frac_np=atom_frac,
                        atom_species=atom_species, species_list=species_list, rcut=rcut, nrad=nrad,
                        lmax=lmax, sigma_r=sigma_r, device=device, dtype=dtype, unknown_policy="ignore",
                    )
                    X_chunks.append(Xb.numpy().astype(np.float32))
                X = np.vstack(X_chunks)
                if X.shape[1] != ckpt["in_dim"]:
                    raise ValueError(f"Feature dimension mismatch for {elfcar}: got {X.shape[1]}, checkpoint expects {ckpt['in_dim']}")
                Xs = transform_standardizer(X, x_mean, x_std)
                y_pred = predict_array(model, Xs, args.eval_batch_size, device, args.num_workers)
                y_true = grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.float32)

                if args.no_metrics:
                    metrics = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
                else:
                    metrics = {"mae": mae_np(y_true, y_pred), "rmse": rmse_np(y_true, y_pred), "r2": r2_np(y_true, y_pred)}

                stem = f"file_{ifile:04d}_{Path(elfcar).parent.name}"
                if args.write_sampled_npz:
                    np.savez_compressed(outdir / f"{stem}_sampled_predictions.npz", ijk=ijk.astype(np.int32), y_true=y_true.astype(np.float32), y_pred=y_pred.astype(np.float32), species_list=np.array(species_list, dtype=object), file_species=np.array(species, dtype=object))

                written = ""
                if args.write_pred_elfcar:
                    # Full-grid prediction for output. If user also requested sampling, this second pass covers all voxels.
                    pred_grid_flat = np.empty(nx * ny * nz, dtype=np.float32)
                    for start_flat in range(0, nx * ny * nz, args.descriptor_batch):
                        end_flat = min(start_flat + args.descriptor_batch, nx * ny * nz)
                        flat = np.arange(start_flat, end_flat, dtype=np.int64)
                        ijk_full = np.array(np.unravel_index(flat, (nx, ny, nz), order="C")).T
                        frac = np.zeros((len(flat), 3), dtype=float)
                        frac[:, 0] = ijk_full[:, 0] / nx
                        frac[:, 1] = ijk_full[:, 1] / ny
                        frac[:, 2] = ijk_full[:, 2] / nz
                        Xb, _, _ = build_descriptor_batch_torch(
                            voxel_frac_np=frac, lattice_np=lattice, atom_frac_np=atom_frac,
                            atom_species=atom_species, species_list=species_list, rcut=rcut, nrad=nrad,
                            lmax=lmax, sigma_r=sigma_r, device=device, dtype=dtype, unknown_policy="ignore",
                        )
                        Xbs = transform_standardizer(Xb.numpy().astype(np.float32), x_mean, x_std)
                        pred_grid_flat[start_flat:end_flat] = predict_array(model, Xbs, args.eval_batch_size, device, args.num_workers)
                    pred_grid = pred_grid_flat.reshape((nx, ny, nz), order="C")
                    out_elf = outdir / f"{stem}_PRED_ELFCAR"
                    write_vasp_scalar_field_like(elfcar, str(out_elf), pred_grid)
                    written = str(out_elf)

                row = {"file": elfcar, "n": int(len(y_pred)), "mae": metrics["mae"], "rmse": metrics["rmse"], "r2": metrics["r2"], "unknown_species_ignored": " ".join(unknown), "written_pred_elfcar": written}
                writer.writerow(row); fcsv.flush()
                summary_rows.append(row)
                log(f"[{ifile+1:4d}/{len(args.elfcars)}] n={len(y_pred):8d} MAE {metrics['mae']:.6f} RMSE {metrics['rmse']:.6f} R2 {metrics['r2']:.4f} file={elfcar}", log_fh)

        with open(outdir / "evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump({"checkpoint": str(Path(args.checkpoint).resolve()), "species_list": species_list, "unknown_species_ignored": sorted(all_unknown), "files": summary_rows}, f, indent=2)
        log(f"\nSaved summary: {csv_path.resolve()}", log_fh)
    finally:
        if log_fh:
            log_fh.close()


if __name__ == "__main__":
    main()
