#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common utilities for variable-element voxel ELF pair-channel models.

Representation:
  species-resolved SOAP-like power spectrum with all same/cross pair channels
  for a user-specified or dataset-detected global species list. No explicit
  three-element invariant is used.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg: str, fh=None):
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    if ss_tot < 1e-20:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# ----------------------------- VASP IO -----------------------------

def read_poscar(poscar_path: str):
    with open(poscar_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 8:
        raise ValueError(f"POSCAR too short: {poscar_path}")
    comment = lines[0]
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)], dtype=float) * scale
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    natoms = sum(counts)
    idx = 7
    if lines[idx].lower().startswith("s"):
        idx += 1
    mode = lines[idx].lower(); idx += 1
    coords = []
    for i in range(natoms):
        parts = lines[idx + i].split()
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
    coords = np.array(coords, dtype=float)
    frac = coords @ np.linalg.inv(lattice) if (mode.startswith("c") or mode.startswith("k")) else coords
    return comment, lattice, species, counts, frac


def read_vasp_scalar_field(vol_path: str):
    with open(vol_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]
    if len(lines) < 10:
        raise ValueError(f"Volumetric file too short: {vol_path}")
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)], dtype=float) * scale
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    natoms = sum(counts)
    idx = 7
    if lines[idx].strip().lower().startswith("s"):
        idx += 1
    mode = lines[idx].strip().lower(); idx += 1
    coords = []
    for _ in range(natoms):
        parts = lines[idx].split()
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        idx += 1
    coords = np.array(coords, dtype=float)
    frac = coords @ np.linalg.inv(lattice) if (mode.startswith("c") or mode.startswith("k")) else coords
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    dims = [int(x) for x in lines[idx].split()]
    if len(dims) != 3:
        raise ValueError(f"Malformed grid dimensions line in {vol_path}: {lines[idx]}")
    nx, ny, nz = dims; idx += 1
    ngrid = nx * ny * nz
    vals = []
    while idx < len(lines) and len(vals) < ngrid:
        line = lines[idx].strip()
        if line:
            vals.extend(float(x) for x in line.split())
        idx += 1
    if len(vals) < ngrid:
        raise ValueError(f"Expected {ngrid} grid values but found {len(vals)} in {vol_path}")
    grid = np.array(vals[:ngrid], dtype=float).reshape((nx, ny, nz), order="F")
    return lattice, species, counts, frac, grid


def write_vasp_scalar_field_like(input_path: str, output_path: str, grid: np.ndarray):
    """Write a VASP volumetric file by reusing the header of input_path."""
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    scale = float(lines[1].split()[0])
    counts = [int(x) for x in lines[6].split()]
    natoms = sum(counts)
    idx = 7
    if lines[idx].strip().lower().startswith("s"):
        idx += 1
    idx += 1 + natoms
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    header = lines[:idx]
    nx, ny, nz = grid.shape
    flat = grid.reshape(nx * ny * nz, order="F")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in header:
            f.write(line + "\n")
        f.write(f"   {nx:d}   {ny:d}   {nz:d}\n")
        for i in range(0, len(flat), 5):
            f.write("".join(f" {v:16.10E}" for v in flat[i:i+5]) + "\n")


def expand_species(species: Sequence[str], counts: Sequence[int]) -> List[str]:
    out: List[str] = []
    for s, n in zip(species, counts):
        out.extend([s] * int(n))
    return out


def detect_species_from_elfcars(elfcars: Sequence[str], species_order: Optional[Sequence[str]] = None) -> List[str]:
    if species_order:
        order = list(species_order)
        if len(order) != len(set(order)):
            raise ValueError(f"Duplicate entries in --species-order: {order}")
        return order
    seen = []
    for path in elfcars:
        _, species, _, _, _ = read_vasp_scalar_field(path)
        for sp in species:
            if sp not in seen:
                seen.append(sp)
    if not seen:
        raise ValueError("No species detected.")
    return seen


def check_geometry_consistency(elfcar_path: str, lattice_e, species_e, counts_e, atom_frac_e,
                               geometry_source: str, lattice_tol: float, frac_tol: float, log_fh=None):
    poscar_path_obj = Path(elfcar_path).parent / "POSCAR"
    if not poscar_path_obj.is_file():
        if geometry_source in ("poscar", "strict"):
            raise FileNotFoundError(f"Could not find POSCAR next to {elfcar_path}, but --geometry-source {geometry_source} requires it.")
        return lattice_e, species_e, counts_e, atom_frac_e, None
    poscar_path = str(poscar_path_obj.resolve())
    _, lattice_p, species_p, counts_p, atom_frac_p = read_poscar(poscar_path)
    if species_p != species_e or counts_p != counts_e:
        raise ValueError(
            f"POSCAR/ELFCAR species/count mismatch in {elfcar_path}\n"
            f"  POSCAR species/counts = {species_p} / {counts_p}\n"
            f"  ELFCAR species/counts = {species_e} / {counts_e}"
        )
    max_latt_diff = float(np.max(np.abs(lattice_p - lattice_e)))
    frac_delta = atom_frac_p - atom_frac_e
    frac_delta -= np.rint(frac_delta)
    max_frac_diff = float(np.max(np.abs(frac_delta)))
    if geometry_source == "strict":
        if max_latt_diff > lattice_tol or max_frac_diff > frac_tol:
            raise ValueError(
                f"POSCAR/ELFCAR geometry mismatch in {elfcar_path}: "
                f"max|dL|={max_latt_diff:.6e} Å, max|dfrac|={max_frac_diff:.6e}"
            )
    elif max_latt_diff > lattice_tol or max_frac_diff > frac_tol:
        log(
            f"[WARN] POSCAR/ELFCAR geometry differs for {elfcar_path}: "
            f"max|dL|={max_latt_diff:.3e} Å, max|dfrac|={max_frac_diff:.3e}; "
            f"using --geometry-source {geometry_source}", log_fh,
        )
    if geometry_source == "poscar":
        return lattice_p, species_p, counts_p, atom_frac_p, poscar_path
    return lattice_e, species_e, counts_e, atom_frac_e, poscar_path


# ----------------------------- sampling -----------------------------

def allocate_even_quotas(nfiles: int, total_samples: int, seed: int) -> np.ndarray:
    base = total_samples // nfiles
    rem = total_samples % nfiles
    quotas = np.full(nfiles, base, dtype=int)
    if rem > 0:
        rng = np.random.default_rng(seed)
        extra = rng.choice(nfiles, size=rem, replace=False)
        quotas[extra] += 1
    return quotas


def sample_voxel_indices(nx: int, ny: int, nz: int, nsample: Optional[int], seed: int):
    ngrid = nx * ny * nz
    if nsample is None or nsample >= ngrid:
        flat = np.arange(ngrid, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        flat = np.sort(rng.choice(ngrid, size=nsample, replace=False)).astype(np.int64)
    ijk = np.array(np.unravel_index(flat, (nx, ny, nz), order="C")).T
    frac = np.zeros((len(flat), 3), dtype=float)
    frac[:, 0] = ijk[:, 0] / nx
    frac[:, 1] = ijk[:, 1] / ny
    frac[:, 2] = ijk[:, 2] / nz
    return flat, ijk, frac


def make_grid_frac(nx: int, ny: int, nz: int, start_flat: int, end_flat: int):
    flat = np.arange(start_flat, end_flat, dtype=np.int64)
    ijk = np.array(np.unravel_index(flat, (nx, ny, nz), order="C")).T
    frac = np.zeros((len(flat), 3), dtype=float)
    frac[:, 0] = ijk[:, 0] / nx
    frac[:, 1] = ijk[:, 1] / ny
    frac[:, 2] = ijk[:, 2] / nz
    return flat, ijk, frac


# ----------------------------- descriptors -----------------------------

def str_to_torch_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s == "float32":
        return torch.float32
    if s == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {s}")


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def make_translation_shifts(lattice_np: np.ndarray, rcut: float) -> np.ndarray:
    lens = np.linalg.norm(lattice_np, axis=1)
    nrep = np.ceil(rcut / np.maximum(lens, 1e-12)).astype(int) + 1
    return np.array([[i, j, k]
                     for i in range(-nrep[0], nrep[0] + 1)
                     for j in range(-nrep[1], nrep[1] + 1)
                     for k in range(-nrep[2], nrep[2] + 1)], dtype=float)


def cosine_cutoff_torch(r: torch.Tensor, rcut: float) -> torch.Tensor:
    out = torch.zeros_like(r)
    mask = r <= rcut
    out[mask] = 0.5 * (torch.cos(math.pi * r[mask] / rcut) + 1.0)
    return out


def gaussian_radial_torch(r: torch.Tensor, mus: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-0.5 * ((r[:, None, :] - mus[None, :, None]) / sigma) ** 2)


def real_sph_harmonics_l012(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, eps: float = 1e-12):
    """Low-order real angular basis on the unit sphere.

    The original implementation used polynomial real-harmonic-like channels up to
    l=2.  This patched version adds the seven l=3 real cubic-harmonic channels.
    The channels are intentionally unnormalized; downstream standardization removes
    scale differences, and the SOAP-like power spectrum only requires a consistent
    angular basis.
    """
    r = torch.sqrt(x * x + y * y + z * z + eps)
    xr, yr, zr = x / r, y / r, z / r
    Y = {0: torch.ones_like(r)[..., None]}
    Y[1] = torch.stack([xr, yr, zr], dim=-1)
    Y[2] = torch.stack([
        xr * yr,
        yr * zr,
        zr * xr,
        xr * xr - yr * yr,
        3.0 * zr * zr - 1.0,
    ], dim=-1)
    # Seven independent real cubic harmonics, proportional to the usual l=3
    # tesseral harmonics on the unit sphere.  Exact normalization constants are
    # omitted because feature standardization is applied before training.
    Y[3] = torch.stack([
        xr * (5.0 * zr * zr - 1.0),
        yr * (5.0 * zr * zr - 1.0),
        zr * (5.0 * zr * zr - 3.0),
        zr * (xr * xr - yr * yr),
        2.0 * xr * yr * zr,
        xr * (xr * xr - 3.0 * yr * yr),
        yr * (3.0 * xr * xr - yr * yr),
    ], dim=-1)
    return Y


def expected_pair_soap_nfeatures(nspecies: int, nrad: int, lmax: int) -> int:
    same = nspecies * (nrad * (nrad + 1) // 2)
    cross = (nspecies * (nspecies - 1) // 2) * (nrad * nrad)
    return (lmax + 1) * (same + cross)


def build_descriptor_batch_torch(
    voxel_frac_np: np.ndarray,
    lattice_np: np.ndarray,
    atom_frac_np: np.ndarray,
    atom_species: Sequence[str],
    species_list: Sequence[str],
    rcut: float,
    nrad: int,
    lmax: int,
    sigma_r: float,
    device: torch.device,
    dtype: torch.dtype,
    unknown_policy: str = "ignore",
):
    """Build all global pair channels; atoms whose species are absent from species_list are ignored."""
    if lmax > 3:
        raise NotImplementedError("This implementation currently supports lmax <= 3.")
    if unknown_policy not in ("ignore", "error"):
        raise ValueError("unknown_policy must be ignore or error")
    model_species = list(species_list)
    unknown = sorted(set(atom_species) - set(model_species))
    if unknown and unknown_policy == "error":
        raise ValueError(f"Encountered species not in model species list: {unknown}")

    lattice = torch.tensor(lattice_np, dtype=dtype, device=device)
    atom_frac = torch.tensor(atom_frac_np, dtype=dtype, device=device)
    voxel_frac = torch.tensor(voxel_frac_np, dtype=dtype, device=device)
    B = voxel_frac.shape[0]
    A = atom_frac.shape[0]
    shifts_np = make_translation_shifts(lattice_np, rcut)
    shifts = torch.tensor(shifts_np, dtype=dtype, device=device)
    S = shifts.shape[0]
    atom_img_frac = (atom_frac[:, None, :] + shifts[None, :, :]).reshape(A * S, 3)
    atom_img_cart = atom_img_frac @ lattice
    voxel_cart = voxel_frac @ lattice
    rel = atom_img_cart[None, :, :] - voxel_cart[:, None, :]
    x, y, z = rel[..., 0], rel[..., 1], rel[..., 2]
    r = torch.sqrt(x * x + y * y + z * z + 1e-12)
    weight = cosine_cutoff_torch(r, rcut) * (r <= rcut).to(dtype)
    mus = torch.linspace(0.0, rcut, nrad, dtype=dtype, device=device)
    radial = gaussian_radial_torch(r, mus, sigma_r) * weight[:, None, :]
    Y = real_sph_harmonics_l012(x, y, z)

    atom_species_expanded = []
    for sp in atom_species:
        atom_species_expanded.extend([sp] * S)

    coeffs = {}
    for sp in model_species:
        spmask_np = np.array([1.0 if s == sp else 0.0 for s in atom_species_expanded], dtype=np.float32)
        spmask = torch.tensor(spmask_np, dtype=dtype, device=device)
        coeffs[sp] = {}
        for l in range(lmax + 1):
            term = radial * spmask[None, None, :]
            coeffs[sp][l] = torch.einsum("bnm,bmk->bnk", term, Y[l])

    feats = []
    feature_slices = []
    start = 0
    for l in range(lmax + 1):
        for sp in model_species:
            c = coeffs[sp][l]
            block = torch.einsum("bnk,bmk->bnm", c, c)
            tri_vals = torch.stack([block[:, n, np_] for n in range(nrad) for np_ in range(n, nrad)], dim=1)
            feats.append(tri_vals)
            end = start + tri_vals.shape[1]
            feature_slices.append({"l": l, "species_pair": [sp, sp], "triangular": True, "start": start, "end": end})
            start = end
        for i in range(len(model_species)):
            for j in range(i + 1, len(model_species)):
                sp1, sp2 = model_species[i], model_species[j]
                block = torch.einsum("bnk,bmk->bnm", coeffs[sp1][l], coeffs[sp2][l])
                vals = block.reshape(B, nrad * nrad)
                feats.append(vals)
                end = start + vals.shape[1]
                feature_slices.append({"l": l, "species_pair": [sp1, sp2], "triangular": False, "start": start, "end": end})
                start = end
    return torch.cat(feats, dim=1).detach().cpu(), feature_slices, unknown


# ----------------------------- model -----------------------------

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], dropout: float = 0.0, use_sigmoid_output: bool = True):
        super().__init__()
        layers = []
        prev = int(in_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
        self.use_sigmoid_output = bool(use_sigmoid_output)
    def forward(self, x):
        y = self.head(self.backbone(x))
        if self.use_sigmoid_output:
            y = torch.sigmoid(y)
        return y.squeeze(-1)


def fit_standardizer(X: np.ndarray):
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-12, 1.0, std).astype(np.float32)
    return mean, std


def transform_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((X - mean) / std).astype(np.float32)


def run_epoch(model, loader, optimizer, loss_fn, device):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0; total_n = 0
    preds_all = []; ys_all = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if training:
                loss.backward(); optimizer.step()
        bs = xb.shape[0]
        total_loss += float(loss.item()) * bs; total_n += bs
        preds_all.append(pred.detach().cpu().numpy()); ys_all.append(yb.detach().cpu().numpy())
    preds = np.concatenate(preds_all) if preds_all else np.array([], dtype=np.float32)
    ys = np.concatenate(ys_all) if ys_all else np.array([], dtype=np.float32)
    if len(ys) == 0:
        return {"loss": float("nan"), "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}, ys, preds
    return {"loss": total_loss / max(total_n, 1), "mae": mae_np(ys, preds), "rmse": rmse_np(ys, preds), "r2": r2_np(ys, preds)}, ys, preds


def infer_structure_label_from_elfcar(elfcar_path: str, label_mode: str = "auto") -> str:
    p = Path(elfcar_path).resolve()
    parent = p.parent.name
    grandparent = p.parent.parent.name
    greatgrandparent = p.parent.parent.parent.name
    if label_mode == "parent":
        return parent
    if label_mode == "grandparent":
        return grandparent
    if label_mode == "auto":
        return grandparent if parent.isdigit() and grandparent else parent
    if label_mode in ("structure-pressure", "pressure-structure"):
        if parent.isdigit() and grandparent:
            structure, pressure = grandparent, greatgrandparent
        else:
            structure, pressure = parent, grandparent
        return f"{structure}-{pressure}" if label_mode == "structure-pressure" else f"{pressure}-{structure}"
    raise ValueError(f"Unknown label_mode: {label_mode}")


def build_file_to_structure_map(meta: Dict, label_mode: str = "auto") -> Dict[int, str]:
    return {int(entry["file_index"]): infer_structure_label_from_elfcar(entry["elfcar"], label_mode) for entry in meta["files"]}
