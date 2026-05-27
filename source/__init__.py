"""
Structural Topology Optimization for Blender
=============================================

Tag meshes as domain / load / support / property region, voxelize, solve
with 3-D SIMP+OC, and extract a smooth mesh of the optimised topology.

Install via Blender's Extensions Manager (Edit → Preferences → Get Extensions).
Panel appears in the 3D viewport N-panel under the 'Struct Topo' tab.
"""

bl_info = {
    "name":        "Structural Topology Optimization",
    "author":      "Emre Ergin",
    "version":     (0, 4, 3),
    "blender":     (4, 2, 0),
    "location":    "View3D > N-panel > Struct Topo",
    "description": "SIMP+OC structural topology optimisation with voxelized meshes",
    "doc_url":     "https://github.com/emrgn-arch/StructTopOpt_Blender",
    "tracker_url": "https://github.com/emrgn-arch/StructTopOpt_Blender/issues",
    "category":    "Physics",
}


from . import meshing    # noqa: F401
from . import properties
from . import voxelizer  # noqa: F401
from . import problem    # noqa: F401
from . import preview    # noqa: F401
from . import results    # noqa: F401
from . import ui


def register():
    properties.register()
    ui.register()

    import bpy as _bpy
    try:
        for scene in _bpy.data.scenes:
            scene.topopt.is_solving             = False
            scene.topopt.solve_cancel_requested = False
            scene.topopt.solve_confirm_pending  = False
    except AttributeError:
        pass  # bpy.data not yet available during install-time registration


def unregister():
    ui.unregister()
    properties.unregister()


if __name__ == "__main__":
    register()
