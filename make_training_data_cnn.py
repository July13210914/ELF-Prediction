#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_training_data_cnn.py

Build CNN-ready training data from VASP ELFCAR files.

For each sampled voxel center, construct a local 3D geometry-only density patch
by Gaussian-splatting nearby H atoms onto a patch grid centered at the voxel.

Output .npz:
  X   : (N, 1, P, P, P) float16/float32
  y   : (N,) float32
  ijk : (N,3) int64
  src : (N,) int64 (file index)
  cluster_id : (N,) int64 (cluster id, else -1)
  meta : dict

Designed for pure hydrogen; requires H in ELFCAR species list.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch

@dataclass
class ElfGrid:
    lattice: np.ndarray
    pos_frac: np.ndarray
    species: List[str]
    counts: List[int]
    elf: np.ndarray
    dim: int

    @property
    def pos_h_frac(self) -> np.ndarray:
        if "H" not in self.species:
            raise ValueError("No hydrogen in this ELFCAR")
        h0 = sum(self.counts[: self.species.index("H")])
        nh = self.counts[self.species.index("H")]
        return self.pos_frac[h0 : h0 + nh]

def _read_poscar_like_header(lines: List[str]) -> Tuple[np.ndarray, List[str], List[int], np.ndarray, str]:
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=np.float64) * scale

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
    pos = np.array([[float(x) for x in lines[pos_start + i].split()[:3]] for i in range(nat)], dtype=np.float64)

    if coord_type.startswith("c"):
        pos_frac = pos @ np.linalg.inv(lattice)
    else:
        pos_frac = pos
    pos_frac = pos_frac - np.floor(pos_frac)
    return lattice, species, counts, pos_frac, coord_type

def read_elfcar(path: str | Path) -> ElfGrid:
    path = Path(path)
    lines = path.read_text().splitlines()
    lattice, species, counts, pos_frac, _ = _read_poscar_like_header(lines)

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

def make_region_indices(dim:int, mode:str, k:int, thickness:int, x0:int,x1:int,y0:int,y1:int,z0:int,z1:int, stride:int):
    stride = max(int(stride), 1)
    if mode == "xy":
        k0 = int(k); ks = [(k0+t)%dim for t in range(int(thickness))]
        xs = np.arange(0, dim, stride, dtype=np.int64)
        ys = np.arange(0, dim, stride, dtype=np.int64)
        ijk = np.vstack([np.stack(np.meshgrid(xs, ys, indexing="xy") + (np.full((ys.size,xs.size), kk),), axis=-1).reshape(-1,3) for kk in ks])
    elif mode == "xz":
        j0 = int(k); js = [(j0+t)%dim for t in range(int(thickness))]
        xs = np.arange(0, dim, stride, dtype=np.int64)
        zs = np.arange(0, dim, stride, dtype=np.int64)
        ijk_list=[]
        for jj in js:
            X,Z=np.meshgrid(xs,zs,indexing="xy"); J=np.full_like(X,jj)
            ijk_list.append(np.stack([X.ravel(),J.ravel(),Z.ravel()],axis=1))
        ijk=np.vstack(ijk_list)
    elif mode == "yz":
        i0=int(k); is_=[(i0+t)%dim for t in range(int(thickness))]
        ys=np.arange(0,dim,stride,dtype=np.int64)
        zs=np.arange(0,dim,stride,dtype=np.int64)
        ijk_list=[]
        for ii in is_:
            Y,Z=np.meshgrid(ys,zs,indexing="xy"); I=np.full_like(Y,ii)
            ijk_list.append(np.stack([I.ravel(),Y.ravel(),Z.ravel()],axis=1))
        ijk=np.vstack(ijk_list)
    elif mode == "block":
        xs = np.arange(int(x0), int(x1), stride, dtype=np.int64)%dim
        ys = np.arange(int(y0), int(y1), stride, dtype=np.int64)%dim
        zs = np.arange(int(z0), int(z1), stride, dtype=np.int64)%dim
        X,Y,Z=np.meshgrid(xs,ys,zs,indexing="xy")
        ijk=np.stack([X.ravel(),Y.ravel(),Z.ravel()],axis=1)
    else:
        raise ValueError(f"Unknown mode {mode}")
    flat = (ijk[:,2]*(dim*dim) + ijk[:,1]*dim + ijk[:,0]).astype(np.int64)
    return flat, ijk.astype(np.int64)

def make_cluster_indices(dim:int, num_samples:int, cluster_size:int, cluster_stride:int, seed:int, min_sep:int=0, max_tries:int=200000):
    rng=np.random.default_rng(int(seed))
    cluster_size=int(cluster_size); cluster_stride=max(int(cluster_stride),1)
    h=cluster_size//2
    offs=np.arange(-h, -h+cluster_size, cluster_stride, dtype=np.int64)
    ox,oy,oz=np.meshgrid(offs,offs,offs,indexing="xy")
    local=np.stack([ox.reshape(-1),oy.reshape(-1),oz.reshape(-1)],axis=1)

    centers=[]; vox_all=[]; ijk_all=[]; cid_all=[]
    def pbc_linf_sep(a,b):
        dx=min((a[0]-b[0])%dim,(b[0]-a[0])%dim)
        dy=min((a[1]-b[1])%dim,(b[1]-a[1])%dim)
        dz=min((a[2]-b[2])%dim,(b[2]-a[2])%dim)
        return max(dx,dy,dz)

    n_acc=0; tries=0; cid=0
    while n_acc < num_samples and tries < max_tries:
        tries += 1
        cx,cy,cz = rng.integers(0, dim, size=3)
        cand=(int(cx),int(cy),int(cz))
        if min_sep>0 and any(pbc_linf_sep(cand,c0)<min_sep for c0 in centers):
            continue
        centers.append(cand)
        ijk=(local + np.array([cx,cy,cz],dtype=np.int64))%dim
        flat=(ijk[:,2]*(dim*dim)+ijk[:,1]*dim+ijk[:,0]).astype(np.int64)
        vox_all.append(flat); ijk_all.append(ijk); cid_all.append(np.full(flat.shape[0], cid, dtype=np.int64))
        n_acc += flat.size; cid += 1

    if not vox_all:
        raise RuntimeError("Failed to generate clusters; adjust parameters.")
    vox=np.concatenate(vox_all); ijk=np.vstack(ijk_all); cid_arr=np.concatenate(cid_all)
    uniq, ui=np.unique(vox, return_index=True)
    vox=uniq; ijk=ijk[ui]; cid_arr=cid_arr[ui]
    if vox.size>num_samples:
        sel=rng.choice(vox.size, size=int(num_samples), replace=False)
        vox=vox[sel]; ijk=ijk[sel]; cid_arr=cid_arr[sel]
    return vox.astype(np.int64), ijk.astype(np.int64), cid_arr.astype(np.int64)

def _minimal_image_frac(dfrac: torch.Tensor) -> torch.Tensor:
    return dfrac - torch.round(dfrac)

def _voxel_frac_from_ijk(ijk: torch.Tensor, dim: int) -> torch.Tensor:
    return (ijk.to(torch.float32) + 0.5) / float(dim)

def build_density_patches(Hf: torch.Tensor, lattice: torch.Tensor, ijk: np.ndarray, dim:int, patch_size:int, r_cut:float, sigma:float,
                          device: torch.device, batch:int=512, dtype_out: torch.dtype=torch.float16) -> np.ndarray:
    P=int(patch_size); r_cut=float(r_cut); sigma=float(sigma)
    lin=torch.linspace(-r_cut, r_cut, steps=P, device=device, dtype=torch.float32)
    X,Y,Z=torch.meshgrid(lin,lin,lin,indexing="ij")
    grid_xyz=torch.stack([X,Y,Z],dim=-1).reshape(-1,3)  # (P^3,3)

    ijk_t=torch.tensor(ijk, device=device, dtype=torch.int64)
    vf=_voxel_frac_from_ijk(ijk_t, dim)
    N=vf.shape[0]
    patches=torch.empty((N,1,P,P,P), device=device, dtype=dtype_out)

    r_incl = r_cut + 3.0*sigma
    denom = 2.0*sigma*sigma + 1e-12

    for s0 in range(0, N, batch):
        s1=min(N, s0+batch)
        vfb=vf[s0:s1]
        dfrac=_minimal_image_frac(Hf.unsqueeze(0) - vfb.unsqueeze(1))
        dcart=torch.einsum("bnj,jk->bnk", dfrac, lattice)
        dist=torch.linalg.norm(dcart, dim=-1)
        mask = dist <= r_incl

        B=s1-s0
        out_b=torch.empty((B, P*P*P), device=device, dtype=torch.float32)
        for bi in range(B):
            sel=torch.where(mask[bi])[0]
            if sel.numel()==0:
                out_b[bi].zero_(); continue
            atoms=dcart[bi, sel, :]  # (Na,3)
            diff=grid_xyz[:,None,:] - atoms[None,:,:]
            r2=torch.sum(diff*diff, dim=-1)
            out_b[bi]=torch.exp(-r2/denom).sum(dim=1)
        patches[s0:s1]=out_b.reshape(B,1,P,P,P).to(dtype_out)

    return patches.detach().cpu().numpy()

def main(argv: Optional[Sequence[str]] = None) -> None:
    p=argparse.ArgumentParser(description="Build CNN density-patch dataset from ELFCAR.")
    p.add_argument("elfcars", nargs="+")

    p.add_argument("--out", default="elf_cnn_train.npz")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--mode", default="cluster", choices=["random","cluster","block","xy","xz","yz"])
    p.add_argument("--num-samples-per-file", type=int, default=50000)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--thickness", type=int, default=1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--x0", type=int, default=0); p.add_argument("--x1", type=int, default=0)
    p.add_argument("--y0", type=int, default=0); p.add_argument("--y1", type=int, default=0)
    p.add_argument("--z0", type=int, default=0); p.add_argument("--z1", type=int, default=0)
    p.add_argument("--cluster-size", type=int, default=32)
    p.add_argument("--cluster-stride", type=int, default=2)
    p.add_argument("--cluster-min-sep", type=int, default=0)

    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--r-cut", type=float, default=3.0)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--dtype", choices=["float16","float32"], default="float16")
    args=p.parse_args(argv)

    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else args.device)
    dtype_out=torch.float16 if args.dtype=="float16" else torch.float32

    X_all=[]; y_all=[]; ijk_all=[]; src_all=[]; cid_all=[]
    for fi,fpath in enumerate(args.elfcars):
        grid=read_elfcar(fpath); dim=grid.dim

        if args.mode=="random":
            rng=np.random.default_rng(args.seed+fi)
            n=min(int(args.num_samples_per_file), dim**3)
            vox=rng.choice(dim**3, size=n, replace=False).astype(np.int64)
            k_=vox//(dim*dim); rem=vox-k_*(dim*dim)
            j_=rem//dim; i_=rem-j_*dim
            ijk=np.stack([i_,j_,k_],axis=1).astype(np.int64)
            cid=-np.ones(n,dtype=np.int64)
        elif args.mode=="cluster":
            vox,ijk,cid=make_cluster_indices(dim, int(args.num_samples_per_file), args.cluster_size, args.cluster_stride,
                                             args.seed+fi, min_sep=args.cluster_min_sep)
        else:
            vox,ijk=make_region_indices(dim, args.mode, args.k, args.thickness, args.x0,args.x1,args.y0,args.y1,args.z0,args.z1,args.stride)
            if vox.size>args.num_samples_per_file and args.num_samples_per_file>0:
                rng=np.random.default_rng(args.seed+fi)
                sel=rng.choice(vox.size, size=int(args.num_samples_per_file), replace=False)
                vox=vox[sel]; ijk=ijk[sel]
            cid=-np.ones(vox.shape[0],dtype=np.int64)

        elf_flat=grid.elf.reshape(-1, order="F")
        y=elf_flat[vox].astype(np.float32)

        Hf=torch.tensor(grid.pos_h_frac, device=device, dtype=torch.float32)
        lattice=torch.tensor(grid.lattice, device=device, dtype=torch.float32)

        X=build_density_patches(Hf, lattice, ijk, dim, args.patch_size, args.r_cut, args.sigma, device, batch=args.batch, dtype_out=dtype_out)

        X_all.append(X); y_all.append(y)
        ijk_all.append(ijk.astype(np.int64))
        src_all.append(np.full(y.shape[0], fi, dtype=np.int64))
        cid_all.append(cid.astype(np.int64))

        print(f"[make_training_data_cnn] {fpath}: dim={dim}, Nh={grid.pos_h_frac.shape[0]}, samples={y.shape[0]}, X={X.shape}, dtype={X.dtype}")

    X=np.concatenate(X_all,axis=0)
    y=np.concatenate(y_all,axis=0).astype(np.float32)
    ijk=np.vstack(ijk_all).astype(np.int64)
    src=np.concatenate(src_all).astype(np.int64)
    cluster_id=np.concatenate(cid_all).astype(np.int64)
    meta=dict(mode=args.mode, patch_size=int(args.patch_size), r_cut=float(args.r_cut), sigma=float(args.sigma),
              dtype=args.dtype, num_samples_per_file=int(args.num_samples_per_file),
              cluster_size=int(args.cluster_size), cluster_stride=int(args.cluster_stride), cluster_min_sep=int(args.cluster_min_sep),
              stride=int(args.stride))
    np.savez_compressed(args.out, X=X, y=y, ijk=ijk, src=src, cluster_id=cluster_id, meta=meta)
    print(f"[make_training_data_cnn] wrote: {args.out} (N={X.shape[0]}, Xshape={X.shape})")

if __name__=="__main__":
    main()
