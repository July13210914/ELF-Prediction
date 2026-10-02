#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build variable-element pair-channel SOAP-like voxel ELF dataset.

This generalizes the fixed ternary builder:
  * no hard-coded number of elements;
  * global species list is detected from all ELFCARs unless --species-order is supplied;
  * every file is encoded into the same global same/cross species-pair channel space;
  * missing species in a file simply give zero channels;
  * user controls the species axis through --species-order, --include-elements, and --exclude-elements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "/projects/academic/ezurek/xiaoyu/elf/")

from variable_elf_common import (
    allocate_even_quotas,
    build_descriptor_batch_torch,
    check_geometry_consistency,
    detect_species_from_elfcars,
    expected_pair_soap_nfeatures,
    expand_species,
    log,
    pick_device,
    read_vasp_scalar_field,
    sample_voxel_indices,
    str_to_torch_dtype,
)


def filter_species(species_list, include=None, exclude=None):
    out = list(species_list)
    if include:
        include = list(include)
        missing = [s for s in include if s not in out]
        if missing:
            raise ValueError(f"--include-elements contains species not detected/specified: {missing}")
        out = [s for s in out if s in include]
    if exclude:
        out = [s for s in out if s not in set(exclude)]
    if not out:
        raise ValueError("Global species list is empty after include/exclude filtering.")
    return out


def main():
    ap = argparse.ArgumentParser(description="Build variable-element pair-channel SOAP-like voxel ELF training data.")
    ap.add_argument("elfcars", nargs="+", help="Input ELFCAR files")
    ap.add_argument("--total-samples", type=int, default=None, help="Total sampled voxels across all files")
    ap.add_argument("--num-samples-per-file", type=int, default=None, help="Fixed sampled voxels per file")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--species-order", nargs="*", default=None,
                    help="Optional global species order of arbitrary length, e.g. --species-order Ca B H Mg Al. If omitted, detect first appearance across files.")
    ap.add_argument("--include-elements", nargs="*", default=None,
                    help="Optional subset to keep, preserving global order.")
    ap.add_argument("--exclude-elements", nargs="*", default=None,
                    help="Optional species to drop from the descriptor. Atoms of these species are ignored.")

    ap.add_argument("--geometry-source", choices=["elfcar", "poscar", "strict"], default="elfcar")
    ap.add_argument("--lattice-tol", type=float, default=1e-5)
    ap.add_argument("--frac-tol", type=float, default=1e-5)

    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--r-cut", type=float, default=3.0)
    ap.add_argument("--n-radial", type=int, default=10)
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--sigma-r", type=float, default=0.35)
    ap.add_argument("--out", default="train_variable.npz")
    ap.add_argument("--meta-out", default=None)
    ap.add_argument("--log-out", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if (args.total_samples is None) == (args.num_samples_per_file is None):
        raise ValueError("Provide exactly one of --total-samples or --num-samples-per-file")

    dtype = str_to_torch_dtype(args.dtype)
    device = pick_device(args.device)
    log_fh = open(args.log_out, "w", encoding="utf-8") if args.log_out else None

    try:
        elfcars = [str(Path(p).resolve()) for p in args.elfcars]
        species_detected = detect_species_from_elfcars(elfcars, args.species_order)
        species_list = filter_species(species_detected, args.include_elements, args.exclude_elements)
        species_ignored_by_design = sorted(set(species_detected) - set(species_list))

        log(f"Using device: {device}", log_fh)
        log(f"Using dtype: {dtype}", log_fh)
        log(f"Number of input files: {len(elfcars)}", log_fh)
        log("Global species channels: " + " ".join(species_list), log_fh)
        if species_ignored_by_design:
            log("[WARN] These detected species are intentionally not represented and will be ignored: " + " ".join(species_ignored_by_design), log_fh)

        nfiles = len(elfcars)
        quotas = allocate_even_quotas(nfiles, args.total_samples, args.seed) if args.total_samples is not None else np.full(nfiles, args.num_samples_per_file, dtype=int)
        total_samples = int(np.sum(quotas))

        X_all, y_all, frac_all, ijk_all, file_index_all = [], [], [], [], []
        metadata_files = []
        feature_slices_global = None
        unknown_species_seen = set()

        for ifile, elfcar_path in enumerate(elfcars):
            quota = int(quotas[ifile])
            local_seed = args.seed + 1009 * (ifile + 1)
            lattice_e, species_e, counts_e, atom_frac_e, grid = read_vasp_scalar_field(elfcar_path)
            lattice, species, counts, atom_frac, poscar_path = check_geometry_consistency(
                elfcar_path, lattice_e, species_e, counts_e, atom_frac_e,
                args.geometry_source, args.lattice_tol, args.frac_tol, log_fh,
            )
            atom_species = expand_species(species, counts)
            nx, ny, nz = grid.shape
            _, ijk, voxel_frac = sample_voxel_indices(nx, ny, nz, quota, local_seed)
            y = grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.float32)

            X_chunks = []
            file_unknown = set()
            for start in range(0, quota, args.batch):
                end = min(start + args.batch, quota)
                Xb, feature_slices, unknown = build_descriptor_batch_torch(
                    voxel_frac_np=voxel_frac[start:end],
                    lattice_np=lattice,
                    atom_frac_np=atom_frac,
                    atom_species=atom_species,
                    species_list=species_list,
                    rcut=args.r_cut,
                    nrad=args.n_radial,
                    lmax=args.lmax,
                    sigma_r=args.sigma_r,
                    device=device,
                    dtype=dtype,
                    unknown_policy="ignore",
                )
                X_chunks.append(Xb.numpy().astype(np.float32))
                file_unknown.update(unknown)
                if feature_slices_global is None:
                    feature_slices_global = feature_slices

            unknown_species_seen.update(file_unknown)
            X_file = np.vstack(X_chunks)
            X_all.append(X_file); y_all.append(y)
            frac_all.append(voxel_frac.astype(np.float32)); ijk_all.append(ijk.astype(np.int32))
            file_index_all.append(np.full(quota, ifile, dtype=np.int32))

            metadata_files.append({
                "file_index": ifile,
                "elfcar": elfcar_path,
                "poscar": poscar_path,
                "quota": quota,
                "grid_shape": [int(nx), int(ny), int(nz)],
                "file_species": species,
                "counts": counts,
                "ignored_species_in_file": sorted(file_unknown),
            })
            msg = f"[{ifile+1:4d}/{nfiles}] quota={quota:6d} grid={nx}x{ny}x{nz} file_species={species} file={elfcar_path}"
            if file_unknown:
                msg += " | ignored=" + ",".join(sorted(file_unknown))
            log(msg, log_fh)

        X = np.vstack(X_all)
        expected = expected_pair_soap_nfeatures(len(species_list), args.n_radial, args.lmax)
        if X.shape[1] != expected:
            raise RuntimeError(f"Unexpected descriptor size: got {X.shape[1]}, expected {expected}.")
        y = np.concatenate(y_all)
        frac_coords = np.vstack(frac_all)
        ijk = np.vstack(ijk_all)
        file_index = np.concatenate(file_index_all)

        np.savez_compressed(
            args.out,
            X=X,
            y=y,
            frac_coords=frac_coords,
            ijk=ijk,
            file_index=file_index,
            species_list=np.array(species_list, dtype=object),
        )
        meta_out = args.meta_out or (str(Path(args.out).with_suffix("")) + "_meta.json")
        meta = {
            "total_samples": int(total_samples),
            "nfiles": int(nfiles),
            "nfeatures": int(X.shape[1]),
            "species_list": species_list,
            "species_detected": species_detected,
            "species_ignored_by_design": species_ignored_by_design,
            "r_cut": float(args.r_cut),
            "n_radial": int(args.n_radial),
            "lmax": int(args.lmax),
            "sigma_r": float(args.sigma_r),
            "dtype": args.dtype,
            "device": str(device),
            "feature_slices": feature_slices_global,
            "representation": "variable_element_pair_channel_SOAP_power_spectrum",
            "invariants": "all same/cross species-pair channels over global species_list; no explicit three-element invariant",
            "expected_nfeatures": int(expected),
            "species_order_enforced": args.species_order is not None,
            "unknown_species_seen_and_ignored": sorted(unknown_species_seen),
            "geometry_source": args.geometry_source,
            "files": metadata_files,
        }
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        log(f"Wrote dataset: {args.out}", log_fh)
        log(f"Wrote metadata: {meta_out}", log_fh)
        log(f"Final shapes: X={X.shape}, y={y.shape}", log_fh)
    finally:
        if log_fh:
            log_fh.close()


if __name__ == "__main__":
    main()
