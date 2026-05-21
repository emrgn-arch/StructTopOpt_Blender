# Structural Topology Optimization for Blender

A Blender addon for early-stage structural topology optimization, intended as a tool for **geometry discovery** rather than structural verification.

Inspired by [TopOpt_teach](https://github.com/MCM-QMUL/TopOpt_teach/tree/main), the addon implements a simplified 3D SIMP (Solid Isotropic Material with Penalization) solver and wraps it in Blender's 3D viewport — so you can sketch a design space, tag loads and supports on regular meshes, and quickly visualize where material "wants" to be in a voxelized manner. The aim is to give designers and curious tinkerers a low-friction way to *see* candidate shapes early in a process, without relying on closed or commercially available topology optimization software.

To be clear upfront regarding Blender implementation: this is *structural* topology optimization — finding where material should go inside a design domain to carry given loads — not **mesh topology** editing in the Blender sense (edge flow, retopology, etc.). Blender isn't traditionally a place where this kind of FEA-adjacent solver lives, which is exactly what made it an interesting experiment: its python scripting and viewport make it easy to sketch a design space and visualize results, even if it was never built with solvers in mind.

> 💡 Structural mechanics isn't my domain, so treat results as visual sketches rather than verified designs — verify anything load-bearing with proper FEA tools. The solver also runs on CPU for now, so larger grids can get slow depending on the hardware.