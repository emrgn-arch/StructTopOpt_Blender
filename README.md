# TopOpt for Blender — Phase A

Define topology optimization problems by tagging meshes in your Blender scene.
This is the **problem-definition + voxelization preview** phase. The solver
itself is not yet wired up — that's Phase B/C.

## What this gives you right now

- A panel in the 3D viewport's N-panel under the **TopOpt** tab.
- The ability to tag any mesh in your scene as one of:
  - **Boundary Domain** — the design space (exactly one per scene). Carries
    target density, Young's modulus, and Poisson's ratio.
  - **Load** — a region that receives a volume force, with direction and
    magnitude in kN/m³.
  - **Support** — a region of fixed DOFs (Dirichlet BCs).
  - **Property Region** — a region where density is constrained (1.0 = keep,
    0.0 = remove, intermediate = bias hint, reserved for Phase C).
- A "Voxelize & Preview" button that:
  - Reads the scene, validates the problem, and reports errors clearly.
  - Voxelizes every tagged mesh onto a shared grid defined by the domain.
  - Resolves overlapping roles using the priority rule
    `support > property > load > domain`.
  - Creates a `TopOpt_Preview` mesh in the scene with one small cube per
    interesting voxel, colored by role:
    - Grey: plain domain voxel
    - Yellow: passive solid (keep)
    - Blue: load
    - Red: support
  - Parents the preview to the domain so it follows the domain when you
    move/rotate it.
- A "Print Problem Summary" button that prints voxel counts, load directions,
  material parameters, etc. to the system console.

## What's coming

- **Phase B**: 2D SIMP+OC solver running on a single Z-slice of the grid,
  used to validate the FE assembly machinery in an easily-debuggable setting.
- **Phase C**: Full 3D SIMP+OC solver with live per-iteration density update.
- **Phase D**: Marching cubes output mesh, PyAMG for big grids, save/load
  problem definitions.

## Installation

1. Zip the `topopt_blender/` folder (the folder itself, not its contents).
2. In Blender: `Edit → Preferences → Add-ons → Install...` → pick the zip.
3. Enable "Object: TopOpt for Blender".
4. Open the N-panel in the 3D viewport (press N), find the **TopOpt** tab.

## Quick workflow

1. Make a mesh that will be your design space (e.g. a box). Select it,
   in the TopOpt panel set its role to **Boundary Domain**.
2. Make smaller meshes representing where you want loads and supports.
   Set each to **Load** or **Support**, fill in load direction/magnitude.
3. Set **Voxel Size** in the Voxel Grid section (try 0.1 for a 1m³ box).
4. Click **Voxelize & Preview**.

If the preview looks wrong (e.g. a support didn't show up), check that
your support mesh actually overlaps the boundary domain — only the
intersection counts.

## Tested on

- Blender 3.6 LTS and later. Should work on 4.x.

## Files

- `__init__.py` — addon entry, bl_info, register/unregister
- `properties.py` — data schema (per-object and per-scene property groups)
- `voxelizer.py` — mesh inside-test, voxel grid construction
- `problem.py` — scene → ProblemData (typed, solver-ready data)
- `preview.py` — colored cubes mesh generation
- `ui.py` — panel and operator definitions
