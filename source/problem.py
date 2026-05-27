"""
Problem gathering — walk the scene, voxelize tagged objects, and produce a
ProblemData for the solver. Raises ProblemError with a clear message when
the scene isn't valid.  Overlap priority: support > load > property > domain.
"""

from dataclasses import dataclass, field

import numpy as np

from . import properties as props
from . import voxelizer as vox


@dataclass
class LoadCase:
    name: str
    mask: np.ndarray        # bool (nx, ny, nz)
    direction: np.ndarray   # unit vector (3,)
    total_force_kN: float


@dataclass
class PropertyRegion:
    name: str
    mask: np.ndarray
    target_density: float


@dataclass
class ProblemData:
    """Everything the solver needs. Contains no bpy references."""
    voxel_size: float
    shape: tuple[int, int, int]
    grid_offset_local: np.ndarray

    domain_mask: np.ndarray
    target_volume_fraction: float
    youngs_modulus_GPa: float
    poissons_ratio: float

    support_mask: np.ndarray
    loads: list[LoadCase] = field(default_factory=list)
    property_regions: list[PropertyRegion] = field(default_factory=list)

    passive_solid_mask: np.ndarray = None
    passive_void_mask: np.ndarray = None
    full_load_mask: np.ndarray = None

    @property
    def n_design_voxels(self) -> int:
        return int(self.domain_mask.sum())

    @property
    def n_loaded_voxels(self) -> int:
        return int(sum(lc.mask.sum() for lc in self.loads))

    @property
    def n_support_voxels(self) -> int:
        return int(self.support_mask.sum())


class ProblemError(Exception):
    """Raised when the scene doesn't define a valid problem."""


def gather_problem(context) -> ProblemData:
    """Walk the scene, voxelize, return ProblemData. May raise ProblemError."""
    scene = context.scene
    depsgraph = context.evaluated_depsgraph_get()

    by_role = {role[0]: [] for role in props.ROLE_ITEMS}
    for obj in scene.objects:
        if obj.type == 'MESH' and obj.topopt.role != props.ROLE_NONE:
            by_role[obj.topopt.role].append(obj)

    domains = by_role[props.ROLE_DOMAIN]
    if len(domains) == 0:
        raise ProblemError("No boundary domain set. Tag a mesh as 'Boundary Domain' in the panel.")
    if len(domains) > 1:
        raise ProblemError(f"Multiple boundary domains ({', '.join(o.name for o in domains)}). There must be exactly one.")
    domain = domains[0]

    if scene.topopt.voxel_size <= 0:
        raise ProblemError("Voxel size must be positive.")
    if len(by_role[props.ROLE_LOAD]) == 0:
        raise ProblemError("No loads defined. Tag at least one mesh as 'Load'.")
    if len(by_role[props.ROLE_SUPPORT]) == 0:
        raise ProblemError("No supports defined. Tag at least one mesh as 'Support'. "
                           "Without supports the structure is unconstrained and the FE problem is singular.")

    voxel_size = scene.topopt.voxel_size
    grid = vox.voxelize_domain(domain, voxel_size, depsgraph)

    if grid.n_voxels == 0 or not grid.domain_mask.any():
        raise ProblemError(
            f"Voxelization produced no domain voxels. Grid is {grid.shape}; "
            f"voxel size {voxel_size:g} may be too large for this mesh's bounds."
        )

    support_mask = np.zeros(grid.shape, dtype=bool)
    for obj in by_role[props.ROLE_SUPPORT]:
        support_mask |= vox.voxelize_role(grid, obj, depsgraph)

    if not support_mask.any():
        raise ProblemError(
            "Supports do not intersect any domain voxels. Check that your support meshes "
            "overlap the boundary domain mesh, and that voxel size is small enough to "
            "resolve the overlap."
        )

    # Voxelize each load object once (unclipped), then derive both the
    # clipped LoadCase mask and the full_load_mask from the same result.
    full_load_mask = np.zeros(grid.shape, dtype=bool)
    loads: list[LoadCase] = []
    for obj in by_role[props.ROLE_LOAD]:
        full_mask = vox.voxelize_role(grid, obj, depsgraph, clip_to_domain=False)
        full_load_mask |= full_mask
        mask = full_mask & grid.domain_mask
        dir_vec = np.array(obj.topopt.load_direction, dtype=np.float64)
        n = np.linalg.norm(dir_vec)
        dir_vec = dir_vec / n if n >= 1e-12 else np.array([0.0, 0.0, -1.0])
        loads.append(LoadCase(
            name=obj.name,
            mask=mask,
            direction=dir_vec,
            total_force_kN=obj.topopt.load_total_force_kN,
        ))

    property_regions: list[PropertyRegion] = []
    for obj in by_role[props.ROLE_PROPERTY]:
        mask = vox.voxelize_role(grid, obj, depsgraph)
        property_regions.append(PropertyRegion(
            name=obj.name,
            mask=mask,
            target_density=obj.topopt.property_target_density,
        ))

    # Priority: support > load > property
    for lc in loads:
        lc.mask &= ~support_mask

    combined_load_mask = np.zeros(grid.shape, dtype=bool)
    for lc in loads:
        combined_load_mask |= lc.mask

    for pr in property_regions:
        pr.mask &= ~support_mask
        pr.mask &= ~combined_load_mask

    passive_solid = np.zeros(grid.shape, dtype=bool)
    passive_void  = np.zeros(grid.shape, dtype=bool)
    for pr in property_regions:
        if pr.target_density >= 1.0 - 1e-6:
            passive_solid |= pr.mask
        elif pr.target_density <= 1e-6:
            passive_void |= pr.mask

    passive_solid |= support_mask

    p = ProblemData(
        voxel_size=voxel_size,
        shape=grid.shape,
        grid_offset_local=grid.grid_offset_local,
        domain_mask=grid.domain_mask,
        target_volume_fraction=domain.topopt.domain_target_density,
        youngs_modulus_GPa=domain.topopt.domain_youngs_modulus,
        poissons_ratio=domain.topopt.domain_poissons_ratio,
        support_mask=support_mask,
        loads=loads,
        property_regions=property_regions,
        passive_solid_mask=passive_solid,
        passive_void_mask=passive_void,
        full_load_mask=full_load_mask,
    )

    nx, ny, nz = grid.shape
    scene.topopt.grid_info = f"{nx}×{ny}×{nz} = {grid.n_voxels} voxels ({p.n_design_voxels} inside domain)"
    return p


def summarize(problem: ProblemData) -> str:
    nx, ny, nz = problem.shape
    lines = [
        f"Grid: {nx}×{ny}×{nz} ({nx*ny*nz} voxels)",
        f"  Domain voxels:  {problem.n_design_voxels}",
        f"  Support voxels: {problem.n_support_voxels}",
        f"  Loaded voxels:  {problem.n_loaded_voxels}",
        f"Volume fraction target: {problem.target_volume_fraction:.2f}",
        f"Material: E={problem.youngs_modulus_GPa:g} GPa, ν={problem.poissons_ratio:.2f}",
        f"Load cases: {len(problem.loads)}",
    ]
    for lc in problem.loads:
        d = lc.direction
        lines.append(
            f"  • {lc.name}: {lc.mask.sum()} voxels, "
            f"dir=({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f}), "
            f"{lc.total_force_kN:.4g} kN total"
        )
    if problem.property_regions:
        lines.append(f"Property regions: {len(problem.property_regions)}")
        for pr in problem.property_regions:
            lines.append(f"  • {pr.name}: {pr.mask.sum()} voxels, target={pr.target_density:.2f}")
    if min(problem.shape) < 10:
        lines.append(f"\n⚠ Grid is coarse ({min(problem.shape)} voxels on shortest axis). "
                     f"Topology optimization typically needs ≥20 voxels per axis. Decrease voxel size.")
    return "\n".join(lines)
