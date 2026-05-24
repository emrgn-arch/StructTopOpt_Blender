"""
Structural Topology Optimization for Blender
=============================================

Define a structural problem by tagging meshes as domain / load / support /
property region, voxelize, solve with 3-D SIMP+OC, and extract a smooth
isosurface mesh of the optimised topology.

Install: zip this folder, Edit → Preferences → Add-ons → Install...
Then enable "Object: Structural Topology Optimization".
Panel appears in the 3D viewport N-panel under the 'Struct Topo' tab.
"""

bl_info = {
    "name":        "Structural Topology Optimization",
    "author":      "Emre Ergin",
    "version":     (0, 3, 7),
    "blender":     (4, 2, 0),
    "location":    "View3D > N-panel > Struct Topo",
    "description": "SIMP+OC structural topology optimisation with voxelized meshes",
    "doc_url":     "https://github.com/emrgn-arch/StructTopOpt_Blender",
    "tracker_url": "https://github.com/emrgn-arch/StructTopOpt_Blender/issues",
    "category":    "Object",
}


# Ensure user site-packages is on sys.path so pip-installed packages
# (scipy, etc.) are importable inside Blender's Python.
import site as _site, sys as _sys
try:
    _user_site = _site.getusersitepackages()
    if _user_site not in _sys.path:
        _sys.path.insert(0, _user_site)
except Exception:
    pass

from . import dependencies
from . import meshing    # noqa: F401
from . import properties
from . import voxelizer  # noqa: F401
from . import problem    # noqa: F401
from . import preview    # noqa: F401
from . import results    # noqa: F401
from . import ui


def register():
    already_ok, err = dependencies.ensure_dependencies()
    if err:
        print(f"[Struct Topo] WARNING: dependency install failed: {err}")
        print("[Struct Topo] Solver unavailable until dependencies are installed.")
    elif not already_ok:
        print("[Struct Topo] Dependencies installed. Reload the addon (run dev_loader again).")

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
