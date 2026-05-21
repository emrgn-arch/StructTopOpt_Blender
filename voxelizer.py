"""
Voxelizer — turn Blender meshes into boolean masks on a regular voxel grid.

Grid axes align with the domain object's local axes. Inside-test uses 3-ray
majority voting, which is robust against meshes that aren't perfectly watertight.
"""

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


_RAY_EPS = 1e-6

# Slightly off-axis directions to avoid degenerate alignment with axis-aligned mesh edges.
_RAY_DIRS = [
    Vector((1.0,   0.013, 0.007)).normalized(),
    Vector((0.011, 1.0,   0.005)).normalized(),
    Vector((0.009, 0.003, 1.0  )).normalized(),
]


class VoxelGrid:
    def __init__(self, voxel_size, shape, host_matrix):
        self.voxel_size  = voxel_size
        self.shape       = tuple(shape)
        self.host_matrix = host_matrix.copy()
        self.domain_mask = np.zeros(shape, dtype=bool)

    @property
    def n_voxels(self):
        return self.shape[0] * self.shape[1] * self.shape[2]

    def voxel_centers_local(self):
        """(nx, ny, nz, 3) array of voxel centers in host-local space."""
        nx, ny, nz = self.shape
        i  = (np.arange(nx) + 0.5) * self.voxel_size
        j  = (np.arange(ny) + 0.5) * self.voxel_size
        k  = (np.arange(nz) + 0.5) * self.voxel_size
        ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')
        return np.stack([ii, jj, kk], axis=-1)


def _bvh_from_object(obj, depsgraph):
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        verts = [v.co.copy() for v in mesh.vertices]
        polys = [tuple(p.vertices) for p in mesh.polygons]
        bvh = BVHTree.FromPolygons(verts, polys, epsilon=0.0)
    finally:
        eval_obj.to_mesh_clear()
    return bvh


def _inside_test_batch(points_local, bvh, bbox_min=None, bbox_max=None):
    """3-ray majority-vote inside test. Returns bool array of shape (N,)."""
    n = points_local.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    if bbox_min is not None and bbox_max is not None:
        slop = 1e-6
        candidate = np.all(
            (points_local >= bbox_min - slop) & (points_local <= bbox_max + slop),
            axis=1,
        )
    else:
        candidate = np.ones(n, dtype=bool)

    result   = np.zeros(n, dtype=bool)
    cand_idx = np.flatnonzero(candidate)
    if cand_idx.size == 0:
        return result

    votes = np.zeros(cand_idx.size, dtype=np.int32)
    for ray_dir in _RAY_DIRS:
        for k, p_idx in enumerate(cand_idx):
            origin = Vector(points_local[p_idx]) + ray_dir * _RAY_EPS
            count  = 0
            cur    = origin
            while True:
                hit, _, _, _ = bvh.ray_cast(cur, ray_dir)
                if hit is None:
                    break
                count += 1
                cur = hit + ray_dir * 1e-5  # step past hit triangle
            if count % 2 == 1:
                votes[k] += 1

    result[cand_idx] = votes >= 2
    return result


def compute_grid_shape(domain_obj, voxel_size):
    """Return (shape, grid_offset_local, bbox_max) from the domain's local AABB."""
    corners = np.array([list(c) for c in domain_obj.bound_box])
    bmin    = corners.min(axis=0)
    bmax    = corners.max(axis=0)

    grid_offset_local = np.floor(bmin / voxel_size) * voxel_size
    grid_max_local    = np.ceil(bmax  / voxel_size) * voxel_size
    shape = tuple(max(1, int(round((grid_max_local[ax] - grid_offset_local[ax]) / voxel_size)))
                  for ax in range(3))
    return shape, grid_offset_local, bmax


def voxelize_domain(domain_obj, voxel_size, depsgraph):
    """Build a VoxelGrid for the boundary domain and fill its domain_mask."""
    shape, grid_offset_local, _ = compute_grid_shape(domain_obj, voxel_size)

    grid = VoxelGrid(voxel_size, shape, domain_obj.matrix_world)
    grid.grid_offset_local = np.array(grid_offset_local, dtype=np.float64)

    centers = grid.voxel_centers_local() + grid.grid_offset_local
    grid._centers_shifted = centers  # cache for voxelize_role

    pts_flat = centers.reshape(-1, 3)
    bvh      = _bvh_from_object(domain_obj, depsgraph)
    corners  = np.array([list(c) for c in domain_obj.bound_box])
    inside   = _inside_test_batch(pts_flat, bvh, corners.min(axis=0), corners.max(axis=0))
    grid.domain_mask = inside.reshape(shape)

    return grid


def voxelize_role(grid, obj, depsgraph, clip_to_domain=True):
    """Voxelize a non-domain object onto the existing grid.

    Returns a bool mask of shape grid.shape. clip_to_domain=False returns the
    full extent of the object, ignoring the domain boundary.
    """
    centers = getattr(grid, '_centers_shifted', None)
    if centers is None:
        centers = grid.voxel_centers_local() + grid.grid_offset_local

    transform_mat  = obj.matrix_world.inverted() @ grid.host_matrix
    transform      = np.array([list(row) for row in transform_mat])
    pts_flat       = centers.reshape(-1, 3)
    pts_homog      = np.concatenate([pts_flat, np.ones((pts_flat.shape[0], 1))], axis=1)
    pts_obj_local  = (transform @ pts_homog.T).T[:, :3]

    bvh     = _bvh_from_object(obj, depsgraph)
    corners = np.array([list(c) for c in obj.bound_box])
    inside  = _inside_test_batch(pts_obj_local, bvh, corners.min(axis=0), corners.max(axis=0))
    mask    = inside.reshape(grid.shape)

    if clip_to_domain:
        mask &= grid.domain_mask
    return mask
