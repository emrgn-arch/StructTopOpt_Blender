"""
3-D SIMP+OC topology optimisation runner.

Yields one IterResult per iteration so the Blender modal operator can
update the viewport between iterations without blocking the UI.
"""

from dataclasses import dataclass
from typing import Generator

import numpy as np

from .fe3d import (
    E_MIN, NODE_OFFSETS, GridSystem, element_stiffness_3d, compute_sensitivities_3d,
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
    cancel=None,
) -> Generator[IterResult, None, None]:
    """Full 3-D SIMP+OC generator — yields one IterResult per iteration.

    `cancel` is an optional threading.Event; setting it aborts a running
    FE solve with fe3d.SolveCancelled.
    """

    nx, ny, nz = problem.shape
    volfrac    = problem.target_volume_fraction
    Ke         = element_stiffness_3d(problem.youngs_modulus_GPa, problem.poissons_ratio)

    # --- Load vector: each load's total force is spread equally over the
    # corner nodes of its voxels ---
    f = np.zeros((nx + 1, ny + 1, nz + 1, 3))
    for lc in problem.loads:
        n_vox = int(lc.mask.sum())
        if n_vox == 0:
            continue
        force = lc.total_force_kN / n_vox * lc.direction / 8
        ix, iy, iz = np.nonzero(lc.mask)
        for a, b, c in NODE_OFFSETS:
            f[ix + a, iy + b, iz + c] += force

    if np.linalg.norm(f) < 1e-30:
        print("[Struct Topo] No force — solver skipped.")
        yield IterResult(0, problem.domain_mask.astype(float) * volfrac,
                         0.0, volfrac, True, 0.0)
        return

    if not problem.support_mask.any():
        raise RuntimeError("No support voxels — the structure is unconstrained.")

    # --- Domain / passive masks ---
    domain = problem.domain_mask
    ps = problem.passive_solid_mask & domain if problem.passive_solid_mask is not None else None
    pv = problem.passive_void_mask  & domain if problem.passive_void_mask  is not None else None

    opt = domain.copy()
    if ps is not None: opt &= ~ps
    if pv is not None: opt &= ~pv
    n_opt = int(opt.sum())

    if n_opt == 0:
        raise RuntimeError("No optimisable voxels — supports and solid/void property "
                           "regions cover the whole domain.")

    # --- Volume groups: one per partial-density property region, the rest at the domain target ---
    groups = []
    rest = opt.copy()
    for vr in problem.volume_regions:
        g = vr.mask & rest
        if g.any():
            groups.append((g, vr.target_density))
            rest &= ~g
    n_local = len(groups)
    if rest.any():
        groups.append((rest, volfrac))

    system = GridSystem(Ke, domain, problem.support_mask)

    print(f"[Struct Topo] {nx}×{ny}×{nz} grid, {n_opt} opt elements, {system.n_free} free DOFs, "
          f"{system.n_levels} multigrid levels"
          + (f", {n_local} local volume region(s)" if n_local else ""))

    # --- Initial density ---
    rho = np.full((nx, ny, nz), RHO_MIN)
    for g, target in groups:
        rho[g] = target
    if ps is not None: rho[ps] = 1.0
    if pv is not None: rho[pv] = RHO_MIN

    rho_old = rho.copy()
    u = None

    for iteration in range(max_iter):
        E_e = np.where(domain, E_MIN + rho**penal * (1.0 - E_MIN), 0.0)
        u = system.solve(E_e, f, x0=u, cancel=cancel)  # warm start from the last iteration

        compliance = float(np.vdot(f, u))
        dc = compute_sensitivities_3d(system.element_displacements(u), rho, Ke, penal)
        dc = np.where(domain, dc, 0.0)
        dc_pos = np.maximum(0.0, sensitivity_filter(dc, rho, filter_radius))

        for g, target in groups:
            rho[g] = oc_update(rho[g], dc_pos[g], np.ones(int(g.sum())), target, move=move_limit)
        if ps is not None: rho[ps] = 1.0
        if pv is not None: rho[pv] = RHO_MIN

        change = float(np.max(np.abs(rho[opt] - rho_old[opt])))
        rho_old = rho.copy()

        vol_frac_now = float(rho[opt].mean())
        converged    = change < conv_tol

        yield IterResult(
            iteration=iteration + 1,
            density=rho.copy(),
            compliance=compliance,
            vol_frac=vol_frac_now,
            converged=converged,
            change=change,
        )

        if converged:
            break
