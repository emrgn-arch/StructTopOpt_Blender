"""
Per-object and per-scene properties for the topology optimization problem.
Objects are tagged with a role (domain / load / support / property region)
and role-specific numeric parameters via obj.topopt and scene.topopt.
"""

import bpy
from bpy.props import (
    BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty,
    IntProperty, PointerProperty, StringProperty,
)
from bpy.types import PropertyGroup


ROLE_NONE     = 'NONE'
ROLE_DOMAIN   = 'DOMAIN'
ROLE_LOAD     = 'LOAD'
ROLE_SUPPORT  = 'SUPPORT'
ROLE_PROPERTY = 'PROPERTY'

ROLE_ITEMS = [
    (ROLE_NONE,     "None",            "Not part of the topology optimization problem", 'X', 0),
    (ROLE_DOMAIN,   "Boundary Domain", "The design space — voxels inside this mesh are the universe", 'MESH_CUBE', 1),
    (ROLE_LOAD,     "Load",            "Voxels inside this mesh receive a force", 'FORCE_FORCE', 2),
    (ROLE_SUPPORT,  "Support",         "Voxels inside this mesh are fixed (Dirichlet BC)", 'PINNED', 3),
    (ROLE_PROPERTY, "Property Region", "Voxels inside this mesh are constrained to a target density", 'MOD_MASK', 4),
]

# Colors match the voxel preview so source meshes and preview cubes look consistent.
_ROLE_STYLE = {
    ROLE_DOMAIN:   ((0.70, 0.70, 0.70), 0.75),
    ROLE_SUPPORT:  ((0.90, 0.15, 0.15), 0.75),
    ROLE_LOAD:     ((0.15, 0.30, 0.90), 0.75),
    ROLE_PROPERTY: ((0.95, 0.85, 0.10), 0.75),
}


def _get_role_material(role):
    if role not in _ROLE_STYLE:
        return None
    mat_name = f"TopOpt_{role}"
    mat = bpy.data.materials.get(mat_name)
    if mat is not None:
        return mat

    (r, g, b), alpha = _ROLE_STYLE[role]
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    try:
        mat.blend_method = 'BLEND'
    except AttributeError:
        pass

    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (-200, 0)
    bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
    try:
        bsdf.inputs['Alpha'].default_value = alpha
    except KeyError:
        pass

    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (100, 0)
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def _apply_role_material(obj, role):
    if obj is None or obj.type != 'MESH' or obj.data is None:
        return
    mat = _get_role_material(role)
    if mat is None:
        slots = list(obj.data.materials)
        obj.data.materials.clear()
        for m in slots:
            if m is None or not m.name.startswith("TopOpt_"):
                obj.data.materials.append(m)
        return
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def _on_role_change(self, context):
    obj = self.id_data  # reliable owner lookup; 'is' comparison on wrappers always fails
    if obj is not None and obj.type == 'MESH':
        _apply_role_material(obj, self.role)
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces.active.shading.type = 'MATERIAL'
            break


def _apply_threshold(scene_props, context):
    if scene_props.is_solving:
        return
    import bpy as _bpy
    from . import results  as _res
    from . import preview  as _prev
    from . import problem  as _prob
    from . import meshing  as _mesh

    density = _res.get_cached_density()
    if density is None:
        return
    try:
        p = _prob.gather_problem(context)
    except Exception:
        return

    if _mesh.MESH_NAME in _bpy.data.objects:
        if _prev.PREVIEW_NAME in _bpy.data.objects:
            _bpy.data.objects[_prev.PREVIEW_NAME].hide_set(True)
        _mesh.generate(
            context, p, density,
            threshold         = scene_props.density_threshold,
            include_supports  = scene_props.mesh_include_supports,
            include_loads     = scene_props.mesh_include_loads,
            close_holes       = scene_props.mesh_close_holes,
            smooth_factor     = scene_props.mesh_smooth_factor,
            smooth_iterations = scene_props.mesh_smooth_iterations,
        )
    else:
        _prev.build_result_preview(context, p, density, scene_props.density_threshold)


class TopOptObjectProps(PropertyGroup):

    role: EnumProperty(
        name="Role",
        description="What this mesh represents in the topology optimization problem",
        items=ROLE_ITEMS,
        default=ROLE_NONE,
        update=_on_role_change,
    )

    domain_target_density: FloatProperty(
        name="Target Density",
        description="Fraction of the domain to keep as material (volume fraction)",
        default=0.25, min=0.01, max=0.99, subtype='FACTOR',
    )
    domain_youngs_modulus: FloatProperty(
        name="Young's Modulus (GPa)",
        description="Material stiffness. Steel ~200, aluminum ~70, plastic ~2",
        default=60.0, min=0.01, soft_max=500.0,
    )
    domain_poissons_ratio: FloatProperty(
        name="Poisson's Ratio",
        description="Lateral contraction ratio. Steel ~0.3, rubber ~0.49",
        default=0.3, min=0.0, max=0.49, subtype='FACTOR',
    )

    load_direction: FloatVectorProperty(
        name="Direction",
        description="Direction of the load (will be normalized)",
        default=(0.0, 0.0, -1.0), subtype='XYZ', size=3,
    )
    load_total_force_kN: FloatProperty(
        name="Total Force (kN)",
        description="Total force applied to the load region, distributed equally across all load voxels",
        default=1.0, min=0.001, soft_max=1000.0,
    )

    property_target_density: FloatProperty(
        name="Target Density",
        description="1.0 = keep solid, 0.0 = remove",
        default=1.0, min=0.0, max=1.0, subtype='FACTOR',
    )


class TopOptSceneProps(PropertyGroup):

    voxel_size: FloatProperty(
        name="Voxel Size",
        description="Edge length of each voxel in scene units",
        default=0.20, min=0.001, soft_min=0.01, soft_max=1.0,
        precision=3, unit='LENGTH',
    )
    penalty: FloatProperty(
        name="SIMP Penalty",
        description="Penalization exponent. Higher = more 0/1 result, less smooth",
        default=3.0, min=1.0, max=6.0,
    )
    filter_radius_voxels: FloatProperty(
        name="Filter Radius (voxels)",
        description="Sensitivity filter radius in voxel units. Prevents checkerboards",
        default=1.5, min=1.0, soft_max=5.0,
    )
    max_iterations: IntProperty(
        name="Max Iterations",
        description="Maximum optimization iterations before stopping",
        default=80, min=1, soft_max=500,
    )
    convergence_tol: FloatProperty(
        name="Convergence Tol",
        description="Stop when max density change falls below this",
        default=0.01, min=1e-6, soft_max=0.1, precision=4,
    )
    iter_timeout_secs: IntProperty(
        name="Iter Timeout (s)",
        description="Cancel the solver if a single iteration takes longer than this many seconds",
        default=60, min=10, max=3600,
    )
    oc_move_limit: FloatProperty(
        name="OC Move Limit",
        description="Maximum density change per OC step. Lower = more stable, higher = faster",
        default=0.2, min=0.05, max=0.5, precision=2,
    )

    grid_info: StringProperty(
        name="Grid Info",
        default="(no grid yet — click Voxelize)",
    )
    solve_status:        StringProperty(default="")
    is_solving:          BoolProperty(default=False)
    solve_cancel_requested: BoolProperty(default=False)
    solve_confirm_pending:  BoolProperty(default=False)

    density_threshold: FloatProperty(
        name="Threshold",
        description="Hide voxels below this density in the result view",
        default=0.7, min=0.0, max=1.0, subtype='FACTOR',
        update=lambda self, ctx: _apply_threshold(self, ctx),
    )
    grid_domain_voxels: IntProperty(default=0)

    mesh_include_supports: BoolProperty(
        name="Include Supports",
        description="Force support voxels solid in the extracted mesh",
        default=True,
    )
    mesh_include_loads: BoolProperty(
        name="Include Loads",
        description="Force load voxels solid in the extracted mesh",
        default=True,
    )
    mesh_close_holes: BoolProperty(
        name="Close Holes",
        description="Fill open boundary edges for a watertight mesh",
        default=False,
    )
    mesh_smooth_factor: FloatProperty(
        name="Smooth",
        description="Laplacian smoothing strength per pass (0 = off)",
        default=0.5, min=0.0, max=1.0, subtype='FACTOR',
    )
    mesh_smooth_iterations: IntProperty(
        name="Passes",
        description="Number of smoothing passes",
        default=3, min=0, max=30,
    )

    solve_iter_info:        StringProperty(default="")
    solve_compliance_info:  StringProperty(default="")
    solve_volume_info:      StringProperty(default="")
    solve_change_info:      StringProperty(default="")
    solve_time_info:        StringProperty(default="")
    solve_total_time_info:  StringProperty(default="")


CLASSES = (TopOptObjectProps, TopOptSceneProps)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.topopt = PointerProperty(type=TopOptObjectProps)
    bpy.types.Scene.topopt  = PointerProperty(type=TopOptSceneProps)


def unregister():
    del bpy.types.Scene.topopt
    del bpy.types.Object.topopt
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
