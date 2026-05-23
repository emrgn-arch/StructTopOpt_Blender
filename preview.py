"""
Preview mesh generation — visualize the voxelized problem as colored cubes.

We build a single mesh containing one small cube per "interesting" voxel
(supports = red, loads = blue, domain-only = grey). Cubes are smaller than
the voxel size so adjacent voxels are visually distinguishable.

Why one big mesh rather than instancing? Because:
  1. Instancing in Blender via geometry nodes would require setting up a node
     tree at runtime — fragile across Blender versions.
  2. A single mesh with vertex colors renders instantly and is portable.
  3. For Phase A grids (<100³ with most voxels empty) the cube count is small
     enough that mesh size is a non-issue. A 60³ grid with 30% filled = 65k
     cubes = 520k verts. Blender handles this comfortably.

For huge grids in Phase C+ we can switch to a Volume object or geometry nodes,
but Phase A keeps it simple.
"""

import bmesh
import bpy
import numpy as np
from mathutils import Vector


# Preview cube size as a fraction of voxel size. 0.85 leaves visible gaps.
_CUBE_FRACTION = 0.85

# Role colors (RGBA, linear).
_COLOR_DOMAIN  = (0.70, 0.70, 0.70, 0.05)  # near-ghost grey — just enough to see the boundary
_COLOR_SUPPORT = (0.90, 0.15, 0.15, 1.0)   # red
_COLOR_LOAD    = (0.15, 0.30, 0.90, 1.0)   # blue
_COLOR_KEEP    = (0.95, 0.85, 0.10, 1.0)   # yellow (property regions / passive solid)

# Name we use for the preview object. Re-running the voxelizer overwrites it.
PREVIEW_NAME = "TopOpt_Preview"


def _domain_shell(mask):
    """Return only the boundary voxels of a boolean mask — those with at least
    one face-adjacent neighbour that is False (outside the domain).
    Interior voxels are excluded so role voxels inside remain visible."""
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1, 1:-1] = (
        mask[1:-1, 1:-1, 1:-1] &
        mask[:-2,  1:-1, 1:-1] & mask[2:,   1:-1, 1:-1] &
        mask[1:-1, :-2,  1:-1] & mask[1:-1, 2:,   1:-1] &
        mask[1:-1, 1:-1, :-2]  & mask[1:-1, 1:-1, 2:]
    )
    return mask & ~interior


def _make_cube_bmesh(bm, center, size, color):
    """Add a unit cube to bmesh `bm` at `center`, scaled to `size`, painted `color`.

    `bm` must have a "Col" color layer already created on its loops.
    """
    # 8 corners of the cube
    h = size * 0.5
    verts = []
    for dz in (-h, +h):
        for dy in (-h, +h):
            for dx in (-h, +h):
                verts.append(bm.verts.new((center[0] + dx, center[1] + dy, center[2] + dz)))

    # 6 faces, by vertex index in the order we created them.
    # Indices:
    #   0: (-,-,-)  1: (+,-,-)  2: (-,+,-)  3: (+,+,-)
    #   4: (-,-,+)  5: (+,-,+)  6: (-,+,+)  7: (+,+,+)
    face_indices = [
        (2, 3, 1, 0),  # -Z  (normal → −Z)
        (5, 7, 6, 4),  # +Z  (normal → +Z)
        (4, 6, 2, 0),  # -X  (normal → −X)
        (3, 7, 5, 1),  # +X  (normal → +X)
        (1, 5, 4, 0),  # -Y  (normal → −Y)
        (6, 7, 3, 2),  # +Y  (normal → +Y)
    ]
    color_layer = bm.loops.layers.color.get("Col")
    for fi in face_indices:
        face = bm.faces.new([verts[i] for i in fi])
        for loop in face.loops:
            loop[color_layer] = color


def build_preview_mesh(context, problem, show_domain=True):
    """Build (or replace) the preview mesh in the scene.

    `problem` is a problem.ProblemData. `show_domain` controls whether
    plain domain voxels (no role) are visualized as grey cubes; toggling
    this off helps see loads/supports clearly in dense domains.

    Returns the created bpy Object.
    """
    grid_shape = problem.shape
    vs = problem.voxel_size


    nx, ny, nz = grid_shape
    role_color = np.zeros((nx, ny, nz, 4), dtype=np.float32)
    role_show = np.zeros((nx, ny, nz), dtype=bool)

    if show_domain:
        m = problem.domain_mask
        shell = _domain_shell(m)
        role_color[shell] = _COLOR_DOMAIN
        role_show |= shell


    for pr in problem.property_regions:
        m = pr.mask & ~problem.support_mask
        role_color[m] = _COLOR_KEEP
        role_show |= m
    load_combined = np.zeros(grid_shape, dtype=bool)
    for lc in problem.loads:
        load_combined |= lc.mask
    role_color[load_combined] = _COLOR_LOAD
    role_show |= load_combined

    # Supports (red) — highest priority
    role_color[problem.support_mask] = _COLOR_SUPPORT
    role_show |= problem.support_mask

    # Find/create the preview object
    if PREVIEW_NAME in bpy.data.objects:
        old = bpy.data.objects[PREVIEW_NAME]
        old_mesh = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)

    mesh = bpy.data.meshes.new(PREVIEW_NAME + "_mesh")
    obj = bpy.data.objects.new(PREVIEW_NAME, mesh)
    context.collection.objects.link(obj)

    bm = bmesh.new()
    # Create the color layer up front so _make_cube_bmesh can write to it.
    bm.loops.layers.color.new("Col")

    cube_size = vs * _CUBE_FRACTION
    offset = problem.grid_offset_local

    # Iterate only over visible voxels — much faster than full grid.
    idx_i, idx_j, idx_k = np.where(role_show)
    for i, j, k in zip(idx_i, idx_j, idx_k):
        cx = offset[0] + (i + 0.5) * vs
        cy = offset[1] + (j + 0.5) * vs
        cz = offset[2] + (k + 0.5) * vs
        _make_cube_bmesh(bm, (cx, cy, cz), cube_size, tuple(role_color[i, j, k]))

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    # Parent to the domain so the preview moves with the domain mesh.
    # Find the domain (re-scanning is fine; this is once per voxelize).
    from . import properties as props
    domain = next((o for o in context.scene.objects
                   if o.type == 'MESH' and o.topopt.role == props.ROLE_DOMAIN), None)
    if domain is not None:
        obj.parent = domain
        obj.matrix_parent_inverse.identity()  # let parent transform propagate correctly

    # Set viewport shading to use vertex color by adding a simple material.
    _ensure_vertex_color_material(obj)

    return obj


def _ensure_vertex_color_material(obj):
    """Material that reads 'Col' vertex color (RGBA) with alpha transparency.

    Uses Emission + Transparent BSDF mixed by vertex alpha so that
    domain voxels (low alpha) appear ghost-like while supports/loads
    (alpha=1) are fully opaque.
    """
    mat_name = "TopOpt_PreviewMat"
    # Always rebuild so colour/alpha changes take effect after hot-reload.
    old = bpy.data.materials.get(mat_name)
    if old:
        bpy.data.materials.remove(old)

    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'

    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    attr   = nt.nodes.new('ShaderNodeAttribute');  attr.attribute_name = "Col";  attr.location = (-400, 0)
    emit   = nt.nodes.new('ShaderNodeEmission');                                 emit.location  = (-100, 80)
    transp = nt.nodes.new('ShaderNodeBsdfTransparent');                          transp.location = (-100, -80)
    mix    = nt.nodes.new('ShaderNodeMixShader');                                mix.location   = (150, 0)
    out    = nt.nodes.new('ShaderNodeOutputMaterial');                           out.location   = (380, 0)

    nt.links.new(attr.outputs['Color'], emit.inputs['Color'])
    nt.links.new(attr.outputs['Alpha'], mix.inputs['Fac'])   # alpha → opaque blend
    nt.links.new(transp.outputs['BSDF'],       mix.inputs[1])
    nt.links.new(emit.outputs['Emission'],     mix.inputs[2])
    nt.links.new(mix.outputs['Shader'],        out.inputs['Surface'])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def build_result_preview(context, problem, density, threshold=0.3):
    """Rebuild the preview showing only voxels above `threshold`.

    Called every iteration so the user sees the structure forming without
    black void voxels. Vectorised color assignment keeps it fast.
    Support and load voxels are always included regardless of threshold.
    """
    vs     = problem.voxel_size
    offset = problem.grid_offset_local

    # Precompute combined load mask (vectorised, done once).
    load_mask = np.zeros(problem.shape, dtype=bool)
    for lc in problem.loads:
        load_mask |= lc.mask

    # Voxels to show: above threshold OR support OR load.
    show = (density > threshold) | problem.support_mask | load_mask

    # Precompute per-voxel colors (vectorised) before the bmesh loop.
    idx_i, idx_j, idx_k = np.where(show)
    n_show = len(idx_i)

    rho_vals = np.clip(density[idx_i, idx_j, idx_k], 0.0, 1.0).astype(np.float32)
    colors_arr = np.stack([rho_vals, rho_vals, rho_vals,
                           np.ones(n_show, dtype=np.float32)], axis=1)

    # Role overrides (priority: support > load > passive-solid > density).
    is_load = load_mask[idx_i, idx_j, idx_k]
    colors_arr[is_load] = _COLOR_LOAD

    if problem.passive_solid_mask is not None:
        ps = (problem.passive_solid_mask[idx_i, idx_j, idx_k]
              & ~problem.support_mask[idx_i, idx_j, idx_k])
        colors_arr[ps] = _COLOR_KEEP

    is_support = problem.support_mask[idx_i, idx_j, idx_k]
    colors_arr[is_support] = _COLOR_SUPPORT

    # Replace preview mesh geometry.
    if PREVIEW_NAME in bpy.data.objects:
        old = bpy.data.objects[PREVIEW_NAME]
        old_mesh = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)

    mesh = bpy.data.meshes.new(PREVIEW_NAME + "_mesh")
    obj  = bpy.data.objects.new(PREVIEW_NAME, mesh)
    context.collection.objects.link(obj)

    bm = bmesh.new()
    bm.loops.layers.color.new("Col")
    cube_size = vs * _CUBE_FRACTION

    for n in range(n_show):
        i, j, k = int(idx_i[n]), int(idx_j[n]), int(idx_k[n])
        cx = offset[0] + (i + 0.5) * vs
        cy = offset[1] + (j + 0.5) * vs
        cz = offset[2] + (k + 0.5) * vs
        _make_cube_bmesh(bm, (cx, cy, cz), cube_size, tuple(colors_arr[n]))

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    from . import properties as props
    domain = next((o for o in context.scene.objects
                   if o.type == 'MESH' and o.topopt.role == props.ROLE_DOMAIN), None)
    if domain is not None:
        obj.parent = domain
        obj.matrix_parent_inverse.identity()  # let parent transform propagate correctly

    _ensure_vertex_color_material(obj)
    return obj


def clear_preview():
    """Remove the preview object if it exists."""
    if PREVIEW_NAME in bpy.data.objects:
        old = bpy.data.objects[PREVIEW_NAME]
        old_mesh = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if old_mesh and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
