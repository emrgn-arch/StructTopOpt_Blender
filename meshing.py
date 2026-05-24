"""Topology mesh generation via voxel boundary extraction on the optimised density field."""

import bmesh
import bpy
import numpy as np

from . import properties as props

MESH_NAME = "TopOpt_Mesh"


def _extract_surface(vol, threshold, vs, offset):
    """
    Vectorized voxel-boundary surface extraction using only NumPy + bmesh.

    For every pair of adjacent voxels that straddle `threshold`, emit a quad
    at the shared boundary face.  Winding is chosen so outward normals point
    away from the solid (above-threshold) side.

    Returns (verts, faces) arrays ready for from_pydata, or (None, None).
    """
    inside = vol >= threshold
    quads = []

    # X-axis crossings — boundary quad at x = (i+1)*vs
    ix, jx, kx = np.where(inside[:-1, :, :] ^ inside[1:, :, :])
    if len(ix):
        x = (ix + 1) * vs
        # CCW from +X gives a +X-pointing normal
        corners = np.stack([
            np.column_stack([x, jx       * vs, kx       * vs]),
            np.column_stack([x, (jx + 1) * vs, kx       * vs]),
            np.column_stack([x, (jx + 1) * vs, (kx + 1) * vs]),
            np.column_stack([x, jx       * vs, (kx + 1) * vs]),
        ], axis=1)
        # Flip when the right (i+1) side is inside → outward normal becomes -X
        flip = ~inside[ix, jx, kx]
        corners[flip] = corners[flip, ::-1]
        quads.append(corners)

    # Y-axis crossings — boundary quad at y = (j+1)*vs
    iy, jy, ky = np.where(inside[:, :-1, :] ^ inside[:, 1:, :])
    if len(iy):
        y = (jy + 1) * vs
        corners = np.stack([
            np.column_stack([iy       * vs, y, ky       * vs]),
            np.column_stack([(iy + 1) * vs, y, ky       * vs]),
            np.column_stack([(iy + 1) * vs, y, (ky + 1) * vs]),
            np.column_stack([iy       * vs, y, (ky + 1) * vs]),
        ], axis=1)
        flip = ~inside[iy, jy, ky]
        corners[flip] = corners[flip, ::-1]
        quads.append(corners)

    # Z-axis crossings — boundary quad at z = (k+1)*vs
    iz, jz, kz = np.where(inside[:, :, :-1] ^ inside[:, :, 1:])
    if len(iz):
        z = (kz + 1) * vs
        corners = np.stack([
            np.column_stack([iz       * vs, jz       * vs, z]),
            np.column_stack([(iz + 1) * vs, jz       * vs, z]),
            np.column_stack([(iz + 1) * vs, (jz + 1) * vs, z]),
            np.column_stack([iz       * vs, (jz + 1) * vs, z]),
        ], axis=1)
        flip = ~inside[iz, jz, kz]
        corners[flip] = corners[flip, ::-1]
        quads.append(corners)

    if not quads:
        return None, None

    all_corners = np.concatenate(quads, axis=0)  # (F, 4, 3)
    n_faces = len(all_corners)
    verts = all_corners.reshape(-1, 3).astype(np.float32) + offset
    faces = np.arange(n_faces * 4).reshape(n_faces, 4)
    return verts, faces


def generate(
    context,
    problem,
    density_3d,
    threshold=0.5,
    include_loads=True,
    include_supports=True,
    close_holes=False,
    smooth_factor=0.5,
    smooth_iterations=5,
):
    """Run boundary extraction → hole-fill → smooth → return Blender mesh object."""

    vs     = problem.voxel_size
    offset = problem.grid_offset_local

    vol = density_3d.copy().astype(np.float32)
    vol[~problem.domain_mask] = 0.0
    if include_supports:
        vol[problem.support_mask] = 1.0
    else:
        vol[problem.support_mask] = 0.0

    if include_loads:
        for lc in problem.loads:
            vol[lc.mask] = 1.0

    verts, faces = _extract_surface(vol, threshold, vs, offset)
    if verts is None or len(verts) == 0 or len(faces) == 0:
        return None

    if MESH_NAME in bpy.data.objects:
        old = bpy.data.objects[MESH_NAME]
        old_data = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if old_data and old_data.users == 0:
            bpy.data.meshes.remove(old_data)

    mesh = bpy.data.meshes.new(MESH_NAME)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=vs * 0.01)
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.update()

    obj = bpy.data.objects.new(MESH_NAME, mesh)
    context.collection.objects.link(obj)

    domain = next((o for o in context.scene.objects
                   if o.type == 'MESH' and o.topopt.role == props.ROLE_DOMAIN), None)
    if domain:
        obj.parent = domain
        obj.matrix_parent_inverse.identity()

    if close_holes:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        boundary_edges = [e for e in bm.edges if e.is_boundary]
        if boundary_edges:
            bmesh.ops.holes_fill(bm, edges=boundary_edges, sides=0)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

    # Make obj active so modifier_apply works
    prev_active = context.view_layer.objects.active
    obj.select_set(True)
    context.view_layer.objects.active = obj

    mod = obj.modifiers.new("TopOpt_Remesh_Smooth", 'REMESH')
    mod.mode = 'SMOOTH'
    mod.octree_depth = 8
    bpy.ops.object.modifier_apply(modifier=mod.name)

    mod = obj.modifiers.new("TopOpt_Remesh_Voxel", 'REMESH')
    mod.mode = 'VOXEL'
    mod.voxel_size = vs * 0.6
    bpy.ops.object.modifier_apply(modifier=mod.name)

    context.view_layer.objects.active = prev_active

    if smooth_iterations > 0 and smooth_factor > 0:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        for _ in range(smooth_iterations):
            bmesh.ops.smooth_vert(
                bm,
                verts=bm.verts,
                factor=smooth_factor,
                mirror_clip_x=False, mirror_clip_y=False, mirror_clip_z=False,
                clip_dist=0.0,
                use_axis_x=True, use_axis_y=True, use_axis_z=True,
            )
        bm.normal_update()
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

    obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))
    obj.data.update()

    return obj
