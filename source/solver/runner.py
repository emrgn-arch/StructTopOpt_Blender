"""
3-D SIMP+OC topology optimisation runner.

Yields one IterResult per iteration so the Blender modal operator can
update the viewport between iterations without blocking the UI.
"""

from dataclasses import dataclass
from typing import Generator

import numpy as np

from .fe3d import (
    E_MIN, element_stiffness_3d, build_dof_maps_3d, ReducedSystem, compute_sensitivities_3d,
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

    # --- Fixed DOFs: all 3 DOFs of every node of every support voxel ---
    sup_ix, sup_iy, sup_iz = np.where(problem.support_mask)
    n0 = sup_iz * nxy + sup_iy * (nx+1) + sup_ix
    sup_nodes = np.unique(np.concatenate([
        n0, n0+1, n0+(nx+1)+1, n0+(nx+1),
        n0+nxy, n0+nxy+1, n0+nxy+(nx+1)+1, n0+nxy+(nx+1),
    ]))
    fixed_dofs = (3 * sup_nodes[:, None] + np.arange(3)).ravel()

    if len(fixed_dofs) == 0:
        raise RuntimeError("No support voxels — the structure is unconstrained.")

    # --- Domain / passive masks (ravel 'F' matches edof k→j→i traversal) ---
    domain_flat = problem.domain_mask.ravel(order='F')

    ps_flat = (problem.passive_solid_mask.ravel(order='F') & domain_flat
               if problem.passive_solid_mask is not None else None)
    pv_flat = (problem.passive_void_mask.ravel(order='F')  & domain_flat
               if problem.passive_void_mask  is not None else None)

    opt_flat = domain_flat.copy()
    if ps_flat is not None: opt_flat &= ~ps_flat
    if pv_flat is not None: opt_flat &= ~pv_flat
    n_opt = int(opt_flat.sum())

    if n_opt == 0:
        raise RuntimeError("No optimisable voxels — supports and solid/void property "
                           "regions cover the whole domain.")

    # --- Volume groups: one per partial-density property region, the rest at the domain target ---
    groups = []
    rest = opt_flat.copy()
    for vr in problem.volume_regions:
        g = vr.mask.ravel(order='F') & rest
        if g.any():
            groups.append((g, vr.target_density))
            rest &= ~g
    n_local = len(groups)
    if rest.any():
        groups.append((rest, volfrac))

    # --- FE system over the domain elements only ---
    act    = np.flatnonzero(domain_flat)
    system = ReducedSystem(Ke, edof[act], fixed_dofs, n_dofs)
    f_free = f_global[system.free_dofs]

    print(f"[Struct Topo] {nx}×{ny}×{nz} grid, {n_opt} opt elements, {system.n_free} free DOFs"
          + (f", {n_local} local volume region(s)" if n_local else ""))

    # --- Initial density ---
    rho = np.full(nx * ny * nz, RHO_MIN)
    for g, target in groups:
        rho[g] = target
    if ps_flat is not None: rho[ps_flat] = 1.0
    if pv_flat is not None: rho[pv_flat] = RHO_MIN

    rho_old = rho.copy()

    for iteration in range(max_iter):
        rho_act = rho[act]
        u = system.solve(E_MIN + rho_act**penal * (1.0 - E_MIN), f_free)

        compliance = float(f_free @ u)
        dc = np.zeros_like(rho)
        dc[act] = compute_sensitivities_3d(system.element_displacements(u), rho_act, Ke, penal)

        dc_3d  = dc.reshape(nx, ny, nz, order='F')
        rho_3d = rho.reshape(nx, ny, nz, order='F')
        dc_filt = sensitivity_filter(dc_3d, rho_3d, filter_radius).ravel(order='F')

        dc_pos = np.maximum(0.0, dc_filt)
        for g, target in groups:
            rho[g] = oc_update(rho[g], dc_pos[g], np.ones(int(g.sum())), target, move=move_limit)
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
