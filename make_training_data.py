#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_training_data_lda.py

Clean "LDA-like" dataset builder for ELF prediction:
  - H-only geometry
  - voxel-centered SH power-spectrum features (rotation-invariant)
  - NO GGA channels, NO tau channels, no extra knobs

Outputs .npz:
  X   : (N,F) float16/float32
  y   : (N,) float32
  ijk : (N,3) int64
  src : (N,) int64
  meta: dict (pickle)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Dict

import numpy as np
import torch


# -----------------------------
# Logging
# -----------------------------
def setup_logger(log_out: str) -> logging.Logger:
    logger = logging.getLogger("make_training_data_lda")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    if log_out:
        fh = logging.FileHandler(log_out, mode="w")
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    return logger


# -----------------------------
# ELFCAR reader (POSCAR-like header)
# -----------------------------
@dataclass
class ElfGrid:
    lattice: np.ndarray      # (3,3) in Angstrom
    pos_frac: np.ndarray     # (Nat,3) fractional
    species: List[str]
    counts: List[int]
    elf: np.ndarray          # (dim,dim,dim) float32 in [0,1]
    dim: int

    @property
    def pos_h_frac(self) -> np.ndarray:
        if "H" not in self.species:
            raise ValueError("No hydrogen in this ELFCAR species list.")
        h0 = sum(self.counts[: self.species.index("H")])
        nh = self.counts[self.species.index("H")]
        return self.pos_frac[h0 : h0 + nh]


def _read_poscar_like_header(lines: List[str]) -> Tuple[np.ndarray, List[str], List[int], np.ndarray]:
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)],
                       dtype=np.float64) * scale

    tokens5 = lines[5].split()
    tokens6 = lines[6].split()

    def _is_int_list(toks):
        try:
            _ = [int(x) for x in toks]
            return True
        except Exception:
            return False

    if _is_int_list(tokens5):
        counts = [int(x) for x in tokens5]
        species = [f"X{i}" for i in range(len(counts))]
        coord_line = 6
    else:
        species = tokens5
        counts = [int(x) for x in tokens6]
        coord_line = 7

    coord_type = lines[coord_line].strip().lower()
    nat = int(sum(counts))
    pos_start = coord_line + 1
    pos = np.array([[float(x) for x in lines[pos_start + i].split()[:3]] for i in range(nat)],
                   dtype=np.float64)

    if coord_type.startswith("c"):
        pos_frac = pos @ np.linalg.inv(lattice)
    else:
        pos_frac = pos
    pos_frac = pos_frac - np.floor(pos_frac)
    return lattice, species, counts, pos_frac


def read_elfcar(path: str | Path) -> ElfGrid:
    path = Path(path)
    lines = path.read_text().splitlines()
    lattice, species, counts, pos_frac = _read_poscar_like_header(lines)

    nat = int(sum(counts))
    i = 8 + nat
    while i < len(lines) and len(lines[i].split()) < 3:
        i += 1
    if i >= len(lines):
        raise ValueError("Failed to locate grid dimension line in ELFCAR")

    dims = [int(x) for x in lines[i].split()[:3]]
    if dims[0] != dims[1] or dims[1] != dims[2]:
        raise ValueError(f"Non-cubic grid not supported: dims={dims}")
    dim = int(dims[0])

    vals: List[float] = []
    for j in range(i + 1, len(lines)):
        parts = lines[j].split()
        if parts:
            vals.extend([float(x) for x in parts])

    expected = dim * dim * dim
    if len(vals) < expected:
        raise ValueError(f"ELF values truncated: got {len(vals)} < expected {expected}")
    vals = vals[:expected]
    elf = np.array(vals, dtype=np.float32).reshape((dim, dim, dim), order="F")
    return ElfGrid(lattice=lattice, pos_frac=pos_frac, species=species, counts=counts, elf=elf, dim=dim)


# -----------------------------
# Sampling
# -----------------------------
def sample_random_voxels(dim: int, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(min(max(n, 1), dim**3))
    rng = np.random.default_rng(int(seed))
    vox = rng.choice(dim**3, size=n, replace=False).astype(np.int64)

    k = vox // (dim * dim)
    rem = vox - k * (dim * dim)
    j = rem // dim
    i = rem - j * dim
    ijk = np.stack([i, j, k], axis=1).astype(np.int64)
    return vox, ijk


# -----------------------------
# Math utilities (PBC, voxel centers)
# -----------------------------
def minimal_image_frac(dfrac: torch.Tensor) -> torch.Tensor:
    return dfrac - torch.round(dfrac)

def voxel_frac_from_ijk(ijk: torch.Tensor, dim: int) -> torch.Tensor:
    return ijk.to(torch.float32) / float(dim)

def cosine_cutoff(r: torch.Tensor, r_cut: float) -> torch.Tensor:
    x = r / float(r_cut)
    out = 0.5 * (torch.cos(torch.pi * x) + 1.0)
    return torch.where(r <= r_cut, out, torch.zeros_like(out))


# -----------------------------
# Real spherical harmonics up to l=3 (closed forms)
# -----------------------------
def real_sph_harm_lmax3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, lmax: int) -> torch.Tensor:
    lmax = int(lmax)
    if lmax < 0 or lmax > 3:
        raise ValueError("lmax must be in [0,3].")

    c0 = 0.28209479177387814
    c1 = 0.4886025119029199
    c2_0 = 0.31539156525252005
    c2_1 = 1.0925484305920792
    c2_2 = 0.5462742152960396
    c3_0 = 0.3731763325901154
    c3_1 = 0.4570457994644658
    c3_2 = 1.445305721320277
    c3_3 = 0.5900435899266435

    outs = []
    outs.append(c0 * torch.ones_like(x))

    if lmax >= 1:
        outs.append(c1 * y)
        outs.append(c1 * z)
        outs.append(c1 * x)

    if lmax >= 2:
        outs.append(c2_2 * (2.0 * x * y))
        outs.append(c2_1 * (y * z))
        outs.append(c2_0 * (3.0 * z * z - 1.0))
        outs.append(c2_1 * (x * z))
        outs.append(c2_2 * (x * x - y * y))

    if lmax >= 3:
        outs.append(c3_3 * (y * (3.0 * x * x - y * y)))
        outs.append(c3_2 * (2.0 * x * y * z))
        outs.append(c3_1 * (y * (5.0 * z * z - 1.0)))
        outs.append(c3_0 * (z * (5.0 * z * z - 3.0)))
        outs.append(c3_1 * (x * (5.0 * z * z - 1.0)))
        outs.append(c3_2 * (z * (x * x - y * y)))
        outs.append(c3_3 * (x * (x * x - 3.0 * y * y)))

    return torch.stack(outs, dim=-1)


def lm_slices(lmax: int) -> List[slice]:
    s = []
    start = 0
    for l in range(int(lmax) + 1):
        n = 2 * l + 1
        s.append(slice(start, start + n))
        start += n
    return s


# -----------------------------
# Feature builder (LDA-like)
# -----------------------------
@torch.no_grad()
def build_lda_features(
    Hf: torch.Tensor,          # (Nh,3) frac
    lattice: torch.Tensor,     # (3,3) cart
    ijk: np.ndarray,           # (N,3)
    dim: int,
    r_cut: float,
    n_radial: int,
    lmax: int,
    radial_sigma: float,
    device: torch.device,
    batch_size: int,
    dtype_out: torch.dtype,
    logger: Optional[logging.Logger] = None,
) -> torch.Tensor:
    r_cut = float(r_cut)
    n_radial = int(n_radial)
    lmax = int(lmax)
    if lmax > 3:
        raise ValueError("lmax<=3 required in this script.")

    mu = torch.linspace(0.0, r_cut, steps=n_radial, device=device, dtype=torch.float32)
    sig = float(radial_sigma)
    if sig <= 0:
        sig = (r_cut / max(n_radial - 1, 1)) * 0.75
    inv_denom_R = 1.0 / (2.0 * sig * sig + 1e-12)

    lm_blocks = lm_slices(lmax)
    tri = n_radial * (n_radial + 1) // 2
    F = (lmax + 1) * tri

    ijk_t = torch.tensor(ijk, device=device, dtype=torch.int64)
    vf = voxel_frac_from_ijk(ijk_t, dim)
    N = int(vf.shape[0])

    out_cpu = torch.empty((N, F), device="cpu", dtype=dtype_out)

    t0 = time.time()
    for s0 in range(0, N, batch_size):
        s1 = min(N, s0 + batch_size)
        vfb = vf[s0:s1]
        B = int(vfb.shape[0])

        dfrac = minimal_image_frac(Hf.unsqueeze(0) - vfb.unsqueeze(1))      # (B,Nh,3)
        dcart = torch.einsum("bnj,jk->bnk", dfrac, lattice)                 # (B,Nh,3)
        r = torch.linalg.norm(dcart, dim=-1)                                # (B,Nh)
        w = cosine_cutoff(r, r_cut)                                         # (B,Nh)

        inv_r = torch.where(r > 1e-12, 1.0 / r, torch.zeros_like(r))
        ux = dcart[..., 0] * inv_r
        uy = dcart[..., 1] * inv_r
        uz = dcart[..., 2] * inv_r

        Y = real_sph_harm_lmax3(ux, uy, uz, lmax=lmax)                      # (B,Nh,Nlm)
        dr = r.unsqueeze(-1) - mu.view(1, 1, -1)
        R = torch.exp(-(dr * dr) * inv_denom_R)                             # (B,Nh,Nr)
        Rw = R * w.unsqueeze(-1)

        c = torch.einsum("bhn,bhm->bnm", Rw, Y)                             # (B,Nr,Nlm)

        feats_b = []
        iu = torch.triu_indices(n_radial, n_radial, offset=0, device=device)
        for sl in lm_blocks:
            cl = c[:, :, sl]                                                # (B,Nr,2l+1)
            Pl = torch.einsum("bnm,bkm->bnk", cl, cl)                       # (B,Nr,Nr)
            feats_b.append(Pl[:, iu[0], iu[1]])                             # (B,tri)

        fb = torch.cat(feats_b, dim=1)                                      # (B,F)
        out_cpu[s0:s1] = fb.to(device="cpu", dtype=dtype_out)

        if logger and (s0 == 0 or (s0 // batch_size) % 20 == 0):
            dt = time.time() - t0
            rate = (s1) / max(dt, 1e-9)
            logger.info(f"features: {s1:>8d}/{N}  |  {rate:,.1f} vox/s")

    if logger:
        logger.info(f"feature build done: N={N} F={F} elapsed={time.time()-t0:.2f}s")
    return out_cpu


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Build clean LDA-like SH power-spectrum features for ELF learning (H-only).")

    p.add_argument("elfcars", nargs="+")
    p.add_argument("--out", default="lda_train.npz")
    p.add_argument("--log-out", default="make_training_data_lda.log")

    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-samples-per-file", type=int, default=50000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")

    p.add_argument("--r-cut", type=float, default=3.0)
    p.add_argument("--n-radial", type=int, default=10)
    p.add_argument("--lmax", type=int, default=2)
    p.add_argument("--radial-sigma", type=float, default=0.0)

    args = p.parse_args(argv)

    logger = setup_logger(args.log_out)
    logger.info("=== make_training_data_lda ===")
    logger.info(f"start_time: {time.ctime()}")
    logger.info(f"args: {vars(args)}")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dtype_out = torch.float16 if args.dtype == "float16" else torch.float32

    X_all, y_all, ijk_all, src_all = [], [], [], []
    t_all0 = time.time()

    for fi, fpath in enumerate(args.elfcars):
        t0 = time.time()
        grid = read_elfcar(fpath)
        dim = int(grid.dim)

        logger.info(f"--- file[{fi}] {fpath} ---")
        logger.info(f"dim={dim} species={grid.species} counts={grid.counts}")

        vox, ijk = sample_random_voxels(dim, int(args.num_samples_per_file), int(args.seed) + fi)
        elf_flat = grid.elf.reshape(-1, order="F")
        y = elf_flat[vox].astype(np.float32)

        Hf = torch.tensor(grid.pos_h_frac, device=device, dtype=torch.float32)
        lattice = torch.tensor(grid.lattice, device=device, dtype=torch.float32)

        logger.info(f"Nh(H)={Hf.shape[0]} samples={ijk.shape[0]} r_cut={args.r_cut} Nr={args.n_radial} lmax={args.lmax}")
        logger.info(f"compute_device={device} batch={args.batch} out_dtype={args.dtype}")

        X_cpu = build_lda_features(
            Hf=Hf,
            lattice=lattice,
            ijk=ijk,
            dim=dim,
            r_cut=args.r_cut,
            n_radial=args.n_radial,
            lmax=args.lmax,
            radial_sigma=args.radial_sigma,
            device=device,
            batch_size=args.batch,
            dtype_out=dtype_out,
            logger=logger,
        )

        X_all.append(X_cpu.numpy())
        y_all.append(y)
        ijk_all.append(ijk.astype(np.int64))
        src_all.append(np.full(y.shape[0], fi, dtype=np.int64))

        logger.info(f"done file[{fi}] elapsed={time.time()-t0:.2f}s X={tuple(X_cpu.shape)} y={y.shape}")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0).astype(np.float32)
    ijk = np.vstack(ijk_all).astype(np.int64)
    src = np.concatenate(src_all).astype(np.int64)

    meta: Dict[str, object] = dict(
        mode="random_voxel_sh_power_spectrum_LDA_clean",
        r_cut=float(args.r_cut),
        n_radial=int(args.n_radial),
        lmax=int(args.lmax),
        radial_sigma=float(args.radial_sigma),
        num_samples_per_file=int(args.num_samples_per_file),
        seed=int(args.seed),
        batch=int(args.batch),
        dtype=str(args.dtype),
        device=str(device),
        files=[str(x) for x in args.elfcars],
        feature_dim=int(X.shape[1]),
    )

    np.savez_compressed(args.out, X=X, y=y, ijk=ijk, src=src, meta=meta)

    logger.info("=== summary ===")
    logger.info(f"total: N={y.size} F={X.shape[1]} X_dtype={X.dtype}")
    logger.info(f"y: mean={float(np.mean(y)):.6f} std={float(np.std(y)):.6f} min={float(np.min(y)):.6f} max={float(np.max(y)):.6f}")
    X64 = X.astype(np.float64, copy=False)
    logger.info(f"X: mean={float(np.mean(X64)):.6e} std={float(np.std(X64)):.6e} min={float(np.min(X64)):.6e} max={float(np.max(X64)):.6e}")
    logger.info(f"wrote: {args.out}")
    logger.info(f"elapsed_total={time.time()-t_all0:.2f}s")


if __name__ == "__main__":
    main()

