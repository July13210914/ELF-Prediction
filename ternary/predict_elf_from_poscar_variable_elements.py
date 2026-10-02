#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict a full-grid ELF-like VASP volumetric file from POSCAR geometry using a
variable-element pair-channel ELF checkpoint.

Requires:
  variable_elf_common.py
  best_model.pt from train_variable_elements.py

Example:
  python predict_elf_from_poscar_variable_elements.py POSCAR \
    --checkpoint train_mlp/best_model.pt \
    --grid 80 80 80 \
    --out PRED_ELFCAR

You may also copy the grid dimensions from an existing ELFCAR/CHGCAR-like file:
  python predict_elf_from_poscar_variable_elements.py POSCAR \
    --checkpoint train_mlp/best_model.pt \
    --grid-from ELFCAR \
    --out PRED_ELFCAR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from variable_elf_common import (
    MLPRegressor,
    build_descriptor_batch_torch,
    expand_species,
    log,
    pick_device,
    read_poscar,
    read_vasp_scalar_field,
    set_seed,
    str_to_torch_dtype,
    transform_standardizer,
)


def read_grid_shape_from_vasp_vol(path: str):
    lattice, species, counts, frac, grid = read_vasp_scalar_field(path)
    return tuple(int(x) for x in grid.shape)


def write_vasp_scalar_field_from_poscar(
    output_path: str,
    comment: str,
    lattice: np.ndarray,
    species: list[str],
    counts: list[int],
    frac: np.ndarray,
    grid: np.ndarray,
):
    """Write a VASP ELFCAR/CHGCAR-like scalar volumetric file from POSCAR data."""
    nx, ny, nz = grid.shape
    flat = grid.reshape(nx * ny * nz, order="F")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write((comment if comment else "ML predicted ELF from POSCAR") + "\n")
        f.write("   1.00000000000000\n")
        for v in lattice:
            f.write("  " + "  ".join(f"{x:20.12f}" for x in v) + "\n")
        f.write("  " + "  ".join(species) + "\n")
        f.write("  " + "  ".join(str(int(c)) for c in counts) + "\n")
        f.write("Direct\n")
        for x in frac:
            f.write("  " + "  ".join(f"{float(xx) % 1.0:18.10f}" for xx in x[:3]) + "\n")
        f.write("\n")
        f.write(f"   {nx:d}   {ny:d}   {nz:d}\n")
        for i in range(0, len(flat), 5):
            f.write("".join(f" {float(v):16.10E}" for v in flat[i:i + 5]) + "\n")


def predict_batch(model, Xs: np.ndarray, batch_size: int, device, num_workers: int):
    dummy = np.zeros((Xs.shape[0],), dtype=np.float32)
    ds = TensorDataset(torch.from_numpy(Xs.astype(np.float32)), torch.from_numpy(dummy))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    out = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            yp = model(xb).detach().cpu().numpy()
            out.append(yp)
    return np.concatenate(out).astype(np.float32)


def make_voxel_frac_from_flat(nx: int, ny: int, nz: int, start_flat: int, end_flat: int):
    flat = np.arange(start_flat, end_flat, dtype=np.int64)
    ijk = np.array(np.unravel_index(flat, (nx, ny, nz), order="C")).T
    frac = np.zeros((len(flat), 3), dtype=float)
    frac[:, 0] = ijk[:, 0] / nx
    frac[:, 1] = ijk[:, 1] / ny
    frac[:, 2] = ijk[:, 2] / nz
    return flat, frac


def main():
    ap = argparse.ArgumentParser(
        description="Predict full-grid ELF from a VASP POSCAR using a variable-element pair-channel checkpoint."
    )
    ap.add_argument("poscar", help="Input POSCAR/CONTCAR")
    ap.add_argument("--checkpoint", required=True, help="Self-contained best_model.pt")
    ap.add_argument("--out", default="PRED_ELFCAR", help="Output VASP volumetric file")
    ap.add_argument("--grid", nargs=3, type=int, default=None, metavar=("NX", "NY", "NZ"),
                    help="Output grid dimensions")
    ap.add_argument("--grid-from", default=None,
                    help="Optional ELFCAR/CHGCAR-like file whose grid dimensions are reused")
    ap.add_argument("--descriptor-batch", type=int, default=1024,
                    help="Voxel batch size for descriptor construction")
    ap.add_argument("--eval-batch-size", type=int, default=4096,
                    help="MLP batch size")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-out", default=None)
    args = ap.parse_args()

    if (args.grid is None) == (args.grid_from is None):
        raise ValueError("Provide exactly one of --grid NX NY NZ or --grid-from ELFCAR/CHGCAR.")

    set_seed(args.seed)
    device = pick_device(args.device)
    dtype = str_to_torch_dtype(args.dtype)

    log_fh = open(args.log_out, "w", encoding="utf-8") if args.log_out else None

    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        descriptor = ckpt["descriptor"]
        species_list = list(descriptor["species_list"])
        x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
        x_std = np.asarray(ckpt["x_std"], dtype=np.float32)

        model = MLPRegressor(
            ckpt["in_dim"],
            ckpt["hidden_dims"],
            ckpt["dropout"],
            ckpt["use_sigmoid_output"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        rcut = float(descriptor["r_cut"])
        nrad = int(descriptor["n_radial"])
        lmax = int(descriptor["lmax"])
        sigma_r = float(descriptor["sigma_r"])

        comment, lattice, species, counts, atom_frac = read_poscar(args.poscar)
        atom_species = expand_species(species, counts)

        if args.grid_from is not None:
            nx, ny, nz = read_grid_shape_from_vasp_vol(args.grid_from)
        else:
            nx, ny, nz = [int(x) for x in args.grid]

        unknown = sorted(set(atom_species) - set(species_list))
        if unknown:
            log("[WARN] species not in model and ignored: " + " ".join(unknown), log_fh)

        log(f"Using device: {device}", log_fh)
        log("Model species channels: " + " ".join(species_list), log_fh)
        log("POSCAR species: " + " ".join(species), log_fh)
        log(f"Grid: {nx} x {ny} x {nz} = {nx*ny*nz} voxels", log_fh)
        log(f"Descriptor: r_cut={rcut}, n_radial={nrad}, lmax={lmax}, sigma_r={sigma_r}, in_dim={ckpt['in_dim']}", log_fh)

        pred_flat = np.empty(nx * ny * nz, dtype=np.float32)

        for start in range(0, nx * ny * nz, args.descriptor_batch):
            end = min(start + args.descriptor_batch, nx * ny * nz)
            flat, voxel_frac = make_voxel_frac_from_flat(nx, ny, nz, start, end)

            Xb, _, _ = build_descriptor_batch_torch(
                voxel_frac_np=voxel_frac,
                lattice_np=lattice,
                atom_frac_np=atom_frac,
                atom_species=atom_species,
                species_list=species_list,
                rcut=rcut,
                nrad=nrad,
                lmax=lmax,
                sigma_r=sigma_r,
                device=device,
                dtype=dtype,
                unknown_policy="ignore",
            )
            X = Xb.numpy().astype(np.float32)

            if X.shape[1] != int(ckpt["in_dim"]):
                raise ValueError(
                    f"Feature dimension mismatch: got {X.shape[1]}, checkpoint expects {ckpt['in_dim']}"
                )

            Xs = transform_standardizer(X, x_mean, x_std)
            pred_flat[start:end] = predict_batch(
                model,
                Xs,
                batch_size=args.eval_batch_size,
                device=device,
                num_workers=args.num_workers,
            )

            if start == 0 or end == nx * ny * nz or (end // max(args.descriptor_batch, 1)) % 100 == 0:
                log(f"Predicted {end}/{nx*ny*nz} voxels", log_fh)

        pred_grid = pred_flat.reshape((nx, ny, nz), order="C")
        write_vasp_scalar_field_from_poscar(
            output_path=args.out,
            comment=f"ML predicted ELF from {Path(args.poscar).name}",
            lattice=lattice,
            species=species,
            counts=counts,
            frac=atom_frac,
            grid=pred_grid,
        )

        log(f"Wrote predicted ELF: {Path(args.out).resolve()}", log_fh)

    finally:
        if log_fh:
            log_fh.close()


if __name__ == "__main__":
    main()
