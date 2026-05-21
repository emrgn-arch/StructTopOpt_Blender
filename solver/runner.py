"""
3-D SIMP+OC topology optimisation runner.

Yields one IterResult per iteration so the Blender modal operator can
update the viewport between iterations without blocking the UI.
"""

from dataclasses import dataclass
from typing import Generator

import numpy as np

from .fe3d import (
    element_stiffness_3d, build_dof_maps_3d,
    assemble_K_3d, apply_supports_3d, solve_system_3d,
    compute_sensitivities_3d,
)
from .oc import sensitivity_filter, oc_update, RHO_MIN


@dataclass
class IterResult:
    iteration: int
    density: np.ndarray   # (nx, ny, nz)
    compliance: float
    vol_frac: float
    converged: bool
    change: float


def solve_3d(
    problem,
    penal: float = 3.0,
    filter_radius: float = 1.5,
    max_iter: int = 80,
    conv_tol: float = 0.01,
    move_limit: float = 0.2,
) -> Generator[IterResult, None, None]:
    """Full 3-D SIMP+OC generator — yields one IterResult per iteration."""

    nx, ny, nz = problem.shape
    E        = problem.youngs_modulus_GPa
    nu       = problem.poissons_ratio
    vs       = problem.voxel_size
    volfrac  = problem.target_volume_fraction

    Ke   = element_stiffness_3d(E, nu)
    edof = build_dof_maps_3d(nx, ny, nz)
    nxy  = (nx + 1) * (ny + 1)
    n_dofs = 3 * (nx + 1) * (ny + 1) * (nz + 1)

    # --- Build load vector (all 3 force components) ---
    f_global = np.zeros(n_dofs)
    for lc in problem.loads:
        n_vox = int(lc.mask.sum())
        if n_vox == 0:
            continue
        # Distribute total force equally across all load voxels, then
        # split each voxel's share to its 8 corner nodes.
        force_per_vox = lc.total_force_kN / n_vox * lc.direction  # kN per voxel
        fx, fy, fz = float(force_per_vox[0]), float(force_per_vox[1]), float(force_per_vox[2])
        if abs(fx) + abs(fy) + abs(fz) < 1e-30:
            continue
        ix, iy, iz = np.where(lc.mask)
        n0 = iz * nxy + iy * (nx+1) + ix
        for n_arr in [n0, n0+1, n0+(nx+1)+1, n0+(nx+1),
                      n0+nxy, n0+nxy+1, n0+nxy+(nx+1)+1, n0+nxy+(nx+1)]:
            np.add.at(f_global, 3*n_arr,   fx / 8)
            np.add.at(f_global, 3*n_arr+1, fy / 8)
            np.add.at(f_global, 3*n_arr+2, fz / 8)

    if np.linalg.norm(f_global) < 1e-30:
        print("[Struct Topo] No force — solver skipped.")
        yield IterResult(0, problem.domain_mask.astype(float) * volfrac,
                         0.0, volfrac, True, 0.0)
        return

    # --- Fixed DOFs from support mask ---
    fixed_set = set()
    sup_ix, sup_iy, sup_iz = np.where(problem.support_mask)
    for si, sj, sk in zip(sup_ix, sup_iy, sup_iz):
        n0 = sk * nxy + sj * (nx+1) + si
        for n in [n0, n0+1, n0+(nx+1)+1, n0+(nx+1),
                  n0+nxy, n0+nxy+1, n0+nxy+(nx+1)+1, n0+nxy+(nx+1)]:
            fixed_set.add(3*n); fixed_set.add(3*n+1); fixed_set.add(3*n+2)
    fixed_dofs = np.array(sorted(fixed_set), dtype=np.int32)

    if len(fixed_dofs) == 0:
        print("[Struct Topo] No support DOFs — solver skipped.")
        return

    # --- Domain / passive masks (ravel 'F' matches edof k→j→i traversal) ---
    domain_flat = problem.domain_mask.ravel(order='F')
    n_domain    = int(domain_flat.sum())

    ps_flat = (problem.passive_solid_mask.ravel(order='F') & domain_flat
               if problem.passive_solid_mask is not None else None)
    pv_flat = (problem.passive_void_mask.ravel(order='F')  & domain_flat
               if problem.passive_void_mask  is not None else None)

    opt_flat = domain_flat.copy()
    if ps_flat is not None: opt_flat &= ~ps_flat
    if pv_flat is not None: opt_flat &= ~pv_flat
    n_opt = int(opt_flat.sum())

    if n_opt == 0:
        print("[Struct Topo] No optimisable elements.")
        return

    print(f"[Struct Topo] {nx}×{ny}×{nz} grid, {n_opt} opt elements, {n_dofs} DOFs")

    # --- Initial density ---
    rho = np.full(nx * ny * nz, RHO_MIN)
    rho[opt_flat] = volfrac
    if ps_flat is not None: rho[ps_flat] = 1.0
    if pv_flat is not None: rho[pv_flat] = RHO_MIN

    rho_old = rho.copy()

    for iteration in range(max_iter):
        K = assemble_K_3d(rho, Ke, edof, penal)
        K_free, f_free, free_dofs = apply_supports_3d(K, f_global, fixed_dofs)

        try:
            u = solve_system_3d(K_free, f_free, n_dofs, free_dofs)
        except RuntimeError as err:
            print(f"[Struct Topo] ERROR iter {iteration+1}: {err}")
            return

        compliance = float(f_global @ u)
        dc = compute_sensitivities_3d(u, edof, rho, Ke, penal)

        dc_3d  = dc.reshape(nx, ny, nz, order='F')
        rho_3d = rho.reshape(nx, ny, nz, order='F')
        dc_filt = sensitivity_filter(dc_3d, rho_3d, filter_radius).ravel(order='F')

        rho_new_opt = oc_update(
            rho[opt_flat].copy(),
            np.maximum(0.0, dc_filt[opt_flat]),
            np.ones(n_opt),
            volfrac,
            move=move_limit,
        )
        rho[opt_flat] = rho_new_opt
        if ps_flat is not None: rho[ps_flat] = 1.0
        if pv_flat is not None: rho[pv_flat] = RHO_MIN

        change = float(np.max(np.abs(rho[opt_flat] - rho_old[opt_flat])))
        rho_old = rho.copy()

        vol_frac_now = float(rho[opt_flat].mean())
        converged    = change < conv_tol

        yield IterResult(
            iteration=iteration + 1,
            density=rho.reshape(nx, ny, nz, order='F').copy(),
            compliance=compliance,
            vol_frac=vol_frac_now,
            converged=converged,
            change=change,
        )

        if converged:
            break
