#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_xy_slice.py

Build SH power-spectrum ("LDA-like") features for a single XY slice (z = k)
from an ELFCAR for visual error inspection.

LDA-only in this context means:
  - use ONLY local geometry -> SH power-spectrum features
  - NO GGA channels (rho/grad/laplacian/s), no tau, no extra knobs

Output .npz:
  X   : (N, F) float16/float32
  y   : (N,) float32
  ijk : (N,3) int64
  meta: dict
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch


# -----------------------------
# ELFCAR reader (consistent with your SH pipeline)
# -----------------------------
class ElfGrid:
    def __init__(self, lattice, pos_frac, species, counts, elf, dim):
        self.lattice = lattice
        self.pos_frac = pos_frac
        self.species = species
        self.counts = counts
        self.elf = elf
        self.dim = dim

    @property
    def pos_h_frac(self) -> np.ndarray:
        if "H" not in self.species:
            raise ValueError("No hydrogen in this ELFCAR")
        h0 = sum(self.counts[: self.species.index("H")])
        nh = self.counts[self.species.index("H")]
        return self.pos_frac[h0 : h0 + nh]


def _read_poscar_like_header(lines: List[str]):
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
    pos = np.array([[float(x) for x in lines[pos_start + i].split()[:3]]
                    for i in range(nat)], dtype=np.float64)

    if coord_type.startswith("c"):
        pos_frac = pos @ np.linalg.inv(lattice)
    else:
        pos_frac = pos
    pos_frac -= np.floor(pos_frac)
    return lattice, species, counts, pos_frac


def read_elfcar(path: str | Path) -> ElfGrid:
    lines = Path(path).read_text().splitlines()
    lattice, species, counts, pos_frac = _read_poscar_like_header(lines)

    nat = sum(counts)
    i = 8 + nat
    while i < len(lines) and len(lines[i].split()) < 3:
        i += 1
    dims = [int(x) for x in lines[i].split()[:3]]
    dim = int(dims[0])
    if dim != int(dims[1]) or dim != int(dims[2]):
        raise ValueError(f"Non-cubic grid not supported: dims={dims}")

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
# SH power-spectrum features (same math as your SH slice builder; LDA-only)
# -----------------------------
def minimal_image_frac(dfrac: torch.Tensor) -> torch.Tensor:
    return dfrac - torch.round(dfrac)

def voxel_frac_from_ijk(ijk: torch.Tensor, dim: int) -> torch.Tensor:
    return ijk.to(torch.float32) / float(dim)

def cosine_cutoff(r: torch.Tensor, r_cut: float) -> torch.Tensor:
    x = r / float(r_cut)
    out = 0.5 * (torch.cos(torch.pi * x) + 1.0)
    return torch.where(r <= r_cut, out, torch.zeros_like(out))

def lm_slices(lmax: int):
    s = []
    start = 0
    for l in range(int(lmax) + 1):
        n = 2 * l + 1
        s.append(slice(start, start + n))
        start += n
    return s

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

    outs = [c0 * torch.ones_like(x)]
    if lmax >= 1:
        outs += [c1 * y, c1 * z, c1 * x]
    if lmax >= 2:
        outs += [
            c2_2 * (2.0 * x * y),
            c2_1 * (y * z),
            c2_0 * (3.0 * z * z - 1.0),
            c2_1 * (x * z),
            c2_2 * (x * x - y * y),
        ]
    if lmax >= 3:
        outs += [
            c3_3 * (y * (3.0 * x * x - y * y)),
            c3_2 * (2.0 * x * y * z),
            c3_1 * (y * (5.0 * z * z - 1.0)),
            c3_0 * (z * (5.0 * z * z - 3.0)),
            c3_1 * (x * (5.0 * z * z - 1.0)),
            c3_2 * (z * (x * x - y * y)),
            c3_3 * (x * (x * x - 3.0 * y * y)),
        ]
    return torch.stack(outs, dim=-1)

@torch.no_grad()
def build_xy_slice_features(
    Hf: torch.Tensor,
    lattice: torch.Tensor,
    ijk: np.ndarray,
    dim: int,
    r_cut: float,
    n_radial: int,
    lmax: int,
    radial_sigma: float,
    device: torch.device,
    batch: int,
    dtype_out: torch.dtype,
) -> np.ndarray:
    r_cut = float(r_cut)
    n_radial = int(n_radial)
    lmax = int(lmax)
    if lmax > 3:
        raise ValueError("lmax>3 not supported in this compact slice tool (use <=3).")

    mu = torch.linspace(0.0, r_cut, steps=n_radial, device=device, dtype=torch.float32)
    sig = float(radial_sigma)
    if sig <= 0:
        sig = (r_cut / max(n_radial - 1, 1)) * 0.75
    inv_denom = 1.0 / (2.0 * sig * sig + 1e-12)

    lm_blocks = lm_slices(lmax)
    tri = n_radial * (n_radial + 1) // 2
    Fdim = (lmax + 1) * tri

    ijk_t = torch.tensor(ijk, device=device, dtype=torch.int64)
    vf = voxel_frac_from_ijk(ijk_t, dim)
    N = int(vf.shape[0])

    out_cpu = torch.empty((N, Fdim), device="cpu", dtype=dtype_out)

    for s0 in range(0, N, batch):
        s1 = min(N, s0 + batch)
        vfb = vf[s0:s1]
        dfrac = minimal_image_frac(Hf.unsqueeze(0) - vfb.unsqueeze(1))   # (B,Nh,3)
        dcart = torch.einsum("bnj,jk->bnk", dfrac, lattice)              # (B,Nh,3)
        r = torch.linalg.norm(dcart, dim=-1)                              # (B,Nh)
        w = cosine_cutoff(r, r_cut)

        inv_r = torch.where(r > 1e-12, 1.0 / r, torch.zeros_like(r))
        ux = dcart[..., 0] * inv_r
        uy = dcart[..., 1] * inv_r
        uz = dcart[..., 2] * inv_r

        Y = real_sph_harm_lmax3(ux, uy, uz, lmax=lmax)                   # (B,Nh,Nlm)
        dr = r.unsqueeze(-1) - mu.view(1, 1, -1)
        R = torch.exp(-(dr * dr) * inv_denom)                            # (B,Nh,Nr)
        Rw = R * w.unsqueeze(-1)

        c = torch.einsum("bhn,bhm->bnm", Rw, Y)                          # (B,Nr,Nlm)

        feats = []
        iu = torch.triu_indices(n_radial, n_radial, offset=0, device=device)
        for sl in lm_blocks:
            cl = c[:, :, sl]                                             # (B,Nr,2l+1)
            Pl = torch.einsum("bnm,bkm->bnk", cl, cl)                    # (B,Nr,Nr)
            feats.append(Pl[:, iu[0], iu[1]])                            # (B,tri)

        fb = torch.cat(feats, dim=1)                                     # (B,Fdim)
        out_cpu[s0:s1] = fb.to("cpu", dtype=dtype_out)

    return out_cpu.numpy()


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Make XY slice dataset (LDA-only SH power spectrum).")
    p.add_argument("elfcar")
    p.add_argument("--k", type=int, default=0, help="z-index slice (0..dim-1)")
    p.add_argument("--stride", type=int, default=1, help="voxel stride in x/y for the slice")

    # LDA feature params
    p.add_argument("--r-cut", type=float, default=3.0)
    p.add_argument("--n-radial", type=int, default=10)
    p.add_argument("--lmax", type=int, default=2)
    p.add_argument("--radial-sigma", type=float, default=0.0)

    # compute
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="lda_xy_slice.npz")
    args = p.parse_args(argv)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dtype_out = torch.float16 if (args.dtype == "float16") else torch.float32

    grid = read_elfcar(args.elfcar)
    dim = int(grid.dim)
    k = int(args.k) % dim

    xs = np.arange(0, dim, int(args.stride), dtype=np.int64)
    ys = np.arange(0, dim, int(args.stride), dtype=np.int64)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    ijk = np.stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, k, dtype=np.int64)], axis=1)

    elf_flat = grid.elf.reshape(-1, order="F")
    vox = ijk[:, 2] * (dim * dim) + ijk[:, 1] * dim + ijk[:, 0]
    y = elf_flat[vox].astype(np.float32)

    Hf = torch.tensor(grid.pos_h_frac, device=device, dtype=torch.float32)
    lattice = torch.tensor(grid.lattice, device=device, dtype=torch.float32)

    X = build_xy_slice_features(
        Hf, lattice, ijk, dim,
        r_cut=float(args.r_cut),
        n_radial=int(args.n_radial),
        lmax=int(args.lmax),
        radial_sigma=float(args.radial_sigma),
        device=device,
        batch=int(args.batch),
        dtype_out=dtype_out,
    )

    meta = dict(
        elfcar=str(args.elfcar),
        dim=dim,
        k=k,
        stride=int(args.stride),
        r_cut=float(args.r_cut),
        n_radial=int(args.n_radial),
        lmax=int(args.lmax),
        radial_sigma=float(args.radial_sigma),
        dtype=str(args.dtype),
        mode="LDA_only_SH_power_spectrum",
    )

    np.savez_compressed(args.out, X=X, y=y, ijk=ijk.astype(np.int64), meta=meta)
    print(f"[make_xy_slice_lda] wrote: {args.out}  N={y.size}  F={X.shape[1]}  device={device}")


if __name__ == "__main__":
    main()

