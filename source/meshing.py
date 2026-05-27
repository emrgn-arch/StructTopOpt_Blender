"""Topology mesh generation via marching cubes on the optimised density field."""

import bmesh
import bpy
import numpy as np

from . import properties as props
from .MarchingNumPy import marching_cubes_lorensen

MESH_NAME = "TopOpt_Mesh"


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
    """Run marching cubes → hole-fill → smooth → return Blender mesh object."""

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

    raw_verts, faces = marching_cubes_lorensen(vol, level=threshold)
    if raw_verts is None or len(raw_verts) == 0 or len(faces) == 0:
        return None

    # marching_cubes_lorensen returns float16 positions in voxel-index space;
    # scale to world space here.
    verts = raw_verts.astype(np.float32) * vs + offset

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



    with bpy.context.temp_override(active_object=obj, selected_objects=[obj], object=obj):
        mod = obj.modifiers.new("TopOpt_Remesh_Smooth", 'REMESH')
        mod.mode = 'SMOOTH'
        mod.octree_depth = 7
        bpy.ops.object.modifier_apply(modifier=mod.name)

    with bpy.context.temp_override(active_object=obj, selected_objects=[obj], object=obj):
        mod = obj.modifiers.new("TopOpt_Remesh_Voxel", 'REMESH')
        mod.mode = 'VOXEL'
        mod.voxel_size = vs / 2
        bpy.ops.object.modifier_apply(modifier=mod.name)

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