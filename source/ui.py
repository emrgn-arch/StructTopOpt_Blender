"""N-panel UI for the Structural Topology Optimization addon (3D viewport › Struct Topo tab)."""

import time

import bpy
import numpy as np
from bpy.types import Operator, Panel

from . import meshing
from . import problem
from . import preview
from . import properties as props
from . import results as res

try:
    import scipy as _scipy_check  # noqa: F401
    _scipy_ok = True
except ImportError:
    _scipy_ok = False

def _apply_transforms(context):
    """Apply rotation and scale on every tagged mesh before voxelization."""
    prev_active   = context.view_layer.objects.active
    prev_selected = {o: o.select_get() for o in context.scene.objects}

    for o in context.scene.objects:
        o.select_set(False)

    for o in context.scene.objects:
        if o.type != 'MESH' or o.topopt.role == props.ROLE_NONE:
            continue
        o.select_set(True)
        context.view_layer.objects.active = o
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        o.select_set(False)

    for o, was in prev_selected.items():
        o.select_set(was)
    context.view_layer.objects.active = prev_active

class TOPOPT_OT_set_role(Operator):
    """Assign a topology optimization role to all selected mesh objects."""
    bl_idname = "topopt.set_role"
    bl_label = "Set Role"
    bl_options = {'REGISTER', 'UNDO'}

    role: bpy.props.EnumProperty(items=props.ROLE_ITEMS)

    def execute(self, context):
        count = 0
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                obj.topopt.role = self.role
                count += 1
        if count == 0:
            self.report({'WARNING'}, "No mesh objects selected.")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Set {count} object(s) to role: {self.role}")
        return {'FINISHED'}


class TOPOPT_OT_voxelize_preview(Operator):
    """Voxelize the scene and build the colored-cube preview mesh."""
    bl_idname = "topopt.voxelize_preview"
    bl_label = "Voxelize & Preview"
    bl_options = {'REGISTER', 'UNDO'}

    show_domain: bpy.props.BoolProperty(
        name="Show Domain Voxels",
        description="Show plain (no role) domain voxels as grey cubes",
        default=True,
    )

    def execute(self, context):
        _apply_transforms(context)

        try:
            p = problem.gather_problem(context)
        except problem.ProblemError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        context.scene.topopt.grid_domain_voxels = p.n_design_voxels
        preview_obj = preview.build_preview_mesh(context, p, show_domain=self.show_domain)

        hidden_count = 0
        for obj in context.scene.objects:
            if obj.type != 'MESH':
                continue
            if obj is preview_obj:
                continue  # don't hide the preview itself
            if obj.topopt.role != props.ROLE_NONE:
                obj.hide_set(True)
                hidden_count += 1

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.shading.type = 'MATERIAL'
                break

        summary_lines = problem.summarize(p).split("\n")
        for line in summary_lines:
            print("[Struct Topo] " + line)
        msg = summary_lines[0]
        if hidden_count:
            msg += f"  ({hidden_count} source meshes hidden)"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class TOPOPT_OT_toggle_sources(Operator):
    """Toggle visibility of all tagged source meshes (domain/load/support/property)."""
    bl_idname = "topopt.toggle_sources"
    bl_label = "Show/Hide Source Meshes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        tagged = [o for o in context.scene.objects
                  if o.type == 'MESH' and o.topopt.role != props.ROLE_NONE]
        if not tagged:
            self.report({'WARNING'}, "No tagged meshes found.")
            return {'CANCELLED'}
        any_hidden = any(o.hide_get() for o in tagged)
        new_state = not any_hidden
        for obj in tagged:
            obj.hide_set(new_state)
        return {'FINISHED'}


class TOPOPT_OT_solve_3d(Operator):
    """Run the full 3-D SIMP+OC solver on the voxelized problem."""
    bl_idname = "topopt.solve_3d"
    bl_label  = "Solve"
    bl_options = {'REGISTER'}

    confirmed: bpy.props.BoolProperty(default=False, options={'SKIP_SAVE'})

    _timer       = None
    _gen         = None
    _problem     = None
    _solve_start = None

    def modal(self, context, event):
        sp = context.scene.topopt
        if event.type == 'TIMER':
            if sp.solve_cancel_requested:
                self._finish(context)
                sp.solve_status = "Cancelled"
                context.workspace.status_text_set(
                    f"TopOpt  Cancelled  {sp.solve_iter_info}  {sp.solve_total_time_info}"
                )
                return {'CANCELLED'}

            t0 = time.time()
            try:
                result = next(self._gen)
            except StopIteration:
                self._finish(context)
                sp.solve_status = "Max iterations reached"
                context.workspace.status_text_set(
                    f"TopOpt  Max iterations reached  {sp.solve_iter_info}  "
                    f"{sp.solve_compliance_info}  {sp.solve_total_time_info}"
                )
                self._show_result(context)
                return {'FINISHED'}
            except RuntimeError as err:
                self._finish(context)
                sp.solve_status = f"Error: {err}"
                context.workspace.status_text_set(f"TopOpt  Error: {err}")
                self.report({'ERROR'}, str(err))
                return {'CANCELLED'}

            elapsed = time.time() - t0
            total   = time.time() - self._solve_start

            def _fmt(s):
                return f"{int(s//60)}m {s%60:.0f}s" if s >= 60 else f"{s:.1f}s"

            timeout = sp.iter_timeout_secs
            if elapsed > timeout:
                self._finish(context)
                sp.solve_status = (
                    f"Timed out: iteration took {_fmt(elapsed)} "
                    f"(limit {timeout}s). Reduce grid resolution."
                )
                context.workspace.status_text_set(
                    f"TopOpt  Timed out  {sp.solve_iter_info}  "
                    f"iter took {_fmt(elapsed)} (limit {timeout}s)"
                )
                return {'CANCELLED'}

            res.cache_density(result.density)
            threshold = context.scene.topopt.density_threshold
            preview.build_result_preview(context, self._problem, result.density, threshold)
            self._redraw(context)

            max_iter = sp.max_iterations
            sp.solve_iter_info       = f"Iter {result.iteration}/{max_iter}"
            sp.solve_time_info       = f"Iter: {_fmt(elapsed)}"
            sp.solve_total_time_info = f"Total: {_fmt(total)}"
            sp.solve_status          = "Converged" if result.converged else ""
            sp.solve_compliance_info = f"Comp={result.compliance:.4g}"
            sp.solve_volume_info     = f"Vol={result.vol_frac:.3f}"
            sp.solve_change_info     = f"Δ={result.change:.5f}"

            context.workspace.status_text_set(
                f"TopOpt  {sp.solve_iter_info}  {sp.solve_compliance_info}  "
                f"{sp.solve_change_info}  {sp.solve_time_info}  {sp.solve_total_time_info}     [ESC] Cancel"
            )

            if result.converged:
                self._finish(context)
                context.workspace.status_text_set(
                    f"TopOpt  Converged  {sp.solve_iter_info}  "
                    f"{sp.solve_compliance_info}  {sp.solve_total_time_info}"
                )
                self._show_result(context)
                self.report({'INFO'}, f"Converged in {result.iteration} iterations.")
                return {'FINISHED'}

        elif event.type in {'ESC', 'RIGHTMOUSE'}:
            self._finish(context)
            sp.solve_status = "Cancelled"
            context.workspace.status_text_set(
                f"TopOpt  Cancelled  {sp.solve_iter_info}  {sp.solve_total_time_info}"
            )
            self.report({'WARNING'}, "Solve cancelled.")
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        n_dom = context.scene.topopt.grid_domain_voxels
        if n_dom > 15000 and not self.confirmed:
            # Flag the panel to show the inline confirmation buttons instead.
            context.scene.topopt.solve_confirm_pending = True
            return {'CANCELLED'}
        context.scene.topopt.solve_confirm_pending = False
        return self._start_modal(context)

    def execute(self, context):
        # Called when the operator is triggered programmatically (e.g. by
        # the "Yes, Solve" confirmation button with confirmed=True).
        context.scene.topopt.solve_confirm_pending = False
        return self._start_modal(context)

    def _start_modal(self, context):
        sp = context.scene.topopt
        if preview.PREVIEW_NAME not in bpy.data.objects:
            self.report({'ERROR'}, "No preview mesh — run 'Voxelize & Preview' first.")
            return {'CANCELLED'}
        try:
            p = problem.gather_problem(context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        self._problem = p
        try:
            from .solver.runner import solve_3d
        except Exception as e:
            self.report({'ERROR'}, f"Solver import error: {type(e).__name__}: {e}")
            return {'CANCELLED'}

        self._gen = solve_3d(
            p,
            penal=sp.penalty,
            filter_radius=sp.filter_radius_voxels,
            max_iter=sp.max_iterations,
            conv_tol=sp.convergence_tol,
            move_limit=sp.oc_move_limit,
        )

        nx, ny, nz = p.shape
        sp.solve_status          = f"Starting solver ({nx}×{ny}×{nz} grid)…"
        sp.solve_iter_info        = ""
        sp.solve_compliance_info  = ""
        sp.solve_volume_info      = ""
        sp.solve_change_info      = ""
        sp.solve_time_info        = ""
        sp.solve_total_time_info  = ""
        sp.is_solving             = True
        sp.solve_cancel_requested = False
        self._solve_start         = time.time()

        wm = context.window_manager
        context.workspace.status_text_set(
            f"TopOpt  Starting ({nx}×{ny}×{nz})     [ESC] Cancel"
        )
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None
        self._gen = None
        context.scene.topopt.is_solving             = False
        context.scene.topopt.solve_cancel_requested = False
        context.scene.topopt.solve_confirm_pending  = False

    def _redraw(self, context):
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

    def _show_result(self, context):
        density = res.get_cached_density()
        if density is None or self._problem is None:
            return
        try:
            threshold = context.scene.topopt.density_threshold
            preview.build_result_preview(context, self._problem, density, threshold)
            self._redraw(context)
        except Exception:
            pass



class TOPOPT_OT_generate_mesh(Operator):
    """Generate a smooth topology mesh from the optimised density field."""
    bl_idname = "topopt.generate_mesh"
    bl_label  = "Generate Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        density = res.get_cached_density()
        if density is None:
            self.report({'WARNING'}, "No solve result — run Solve first.")
            return {'CANCELLED'}
        try:
            p = problem.gather_problem(context)
        except problem.ProblemError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        sp  = context.scene.topopt
        obj = meshing.generate(
            context, p, density,
            threshold         = sp.density_threshold,
            include_supports  = sp.mesh_include_supports,
            include_loads     = sp.mesh_include_loads,
            close_holes       = sp.mesh_close_holes,
            smooth_factor     = sp.mesh_smooth_factor,
            smooth_iterations = sp.mesh_smooth_iterations,
        )
        if obj is None:
            self.report({'WARNING'}, "No voxels above threshold — lower the threshold.")
            return {'CANCELLED'}

        if preview.PREVIEW_NAME in bpy.data.objects:
            bpy.data.objects[preview.PREVIEW_NAME].hide_set(True)

        for o in context.scene.objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        self.report({'INFO'}, f"Mesh created: {obj.name}")
        return {'FINISHED'}


class TOPOPT_OT_cancel_confirm(Operator):
    """Dismiss the large-grid confirmation prompt."""
    bl_idname = "topopt.cancel_confirm"
    bl_label  = "Cancel"
    bl_options = {'REGISTER'}

    def execute(self, context):
        context.scene.topopt.solve_confirm_pending = False
        return {'FINISHED'}


class TOPOPT_OT_cancel_solve(Operator):
    """Stop the running solver after the current iteration finishes."""
    bl_idname = "topopt.cancel_solve"
    bl_label = "Cancel Solver"
    bl_options = {'REGISTER'}

    def execute(self, context):
        context.scene.topopt.solve_cancel_requested = True
        return {'FINISHED'}


class TOPOPT_OT_print_summary(Operator):
    """Print a full problem summary to the system console."""
    bl_idname = "topopt.print_summary"
    bl_label = "Print Problem Summary"

    def execute(self, context):
        try:
            p = problem.gather_problem(context)
        except problem.ProblemError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        print("=" * 60)
        print("TopOpt problem summary")
        print("=" * 60)
        print(problem.summarize(p))
        print("=" * 60)
        self.report({'INFO'}, "Summary printed to console. (Window → Toggle System Console on Windows)")
        return {'FINISHED'}


class TOPOPT_PT_main(Panel):
    bl_label      = "Structural Topology Optimization"
    bl_idname     = "TOPOPT_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category   = "Struct Topo"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.active_object

        box = layout.box()
        box.label(text="Selected Object", icon='OBJECT_DATA')
        if obj is None or obj.type != 'MESH':
            box.label(text="(select a mesh)")
        else:
            box.label(text=obj.name)
            row = box.row()
            row.prop(obj.topopt, "role", text="Role")

            role = obj.topopt.role
            if role == props.ROLE_DOMAIN:
                col = box.column(align=True)
                col.prop(obj.topopt, "domain_target_density")
                col.prop(obj.topopt, "domain_youngs_modulus")
                col.prop(obj.topopt, "domain_poissons_ratio")
            elif role == props.ROLE_LOAD:
                col = box.column(align=True)
                col.prop(obj.topopt, "load_direction")
                col.prop(obj.topopt, "load_total_force_kN")
            elif role == props.ROLE_PROPERTY:
                col = box.column(align=True)
                col.prop(obj.topopt, "property_target_density")
        box = layout.box()
        box.label(text="Voxel Grid", icon='MOD_REMESH')
        box.prop(scene.topopt, "voxel_size")
        if scene.topopt.grid_info:
            box.label(text=scene.topopt.grid_info)

        box = layout.box()
        box.label(text="Model Actions", icon='MESH_CUBE')
        box.operator("topopt.voxelize_preview", icon='MESH_GRID')
        n_dom = scene.topopt.grid_domain_voxels
        est_dof = 3 * n_dom
        if est_dof > 150_000:
            w = box.column()
            w.alert = True
            w.label(text=f"~{est_dof:,} DOFs — 3D solve will be very slow!", icon='ERROR')
            w.label(text="Increase voxel size to reduce DOF count")
        elif est_dof > 45_000:
            w = box.column()
            w.alert = True
            w.label(text=f"~{est_dof:,} DOFs — solve may be slow", icon='INFO')
        row = box.row(align=True)
        row.operator("topopt.toggle_sources", text="Show/Hide Sources", icon='HIDE_OFF')
        row.operator("topopt.print_summary",  text="Summary", icon='TEXT')

        box = layout.box()
        box.label(text="Solver", icon='SETTINGS')

        if not _scipy_ok:
            box.label(text="scipy not found — reload addon to install", icon='ERROR')
            return

        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(scene.topopt, "penalty")
        row.prop(scene.topopt, "filter_radius_voxels")
        row = col.row(align=True)
        row.prop(scene.topopt, "max_iterations")
        row.prop(scene.topopt, "convergence_tol")
        row = col.row(align=True)
        row.prop(scene.topopt, "iter_timeout_secs")
        row.prop(scene.topopt, "oc_move_limit")


        box.separator(factor=0.8)
        sp      = scene.topopt
        has_pre = preview.PREVIEW_NAME in bpy.data.objects

        if sp.is_solving:
            box.operator("topopt.cancel_solve", icon='X')
        elif sp.solve_confirm_pending:
            # Large-grid inline confirmation
            w = box.column(align=True)
            w.alert = True
            w.label(text=f"Large grid ({n_dom} voxels) — continue?", icon='ERROR')
            r = w.row(align=True)
            op = r.operator("topopt.solve_3d", text="Yes, Solve", icon='CHECKMARK')
            op.confirmed = True
            r.operator("topopt.cancel_confirm", text="Cancel", icon='X')
        else:
            row = box.row()
            row.scale_y   = 1.6
            row.enabled   = has_pre
            row.operator("topopt.solve_3d", icon='PLAY')
            if not has_pre:
                box.label(text="Voxelize first", icon='INFO')

        if sp.is_solving and not sp.solve_iter_info:
            box.label(text=sp.solve_status, icon='TIME')
        elif sp.solve_iter_info:
            col2 = box.column(align=True)
            col2.label(text=f"{sp.solve_iter_info}   {sp.solve_time_info}   {sp.solve_total_time_info}")
            col2.label(text=f"{sp.solve_compliance_info}   {sp.solve_volume_info}   {sp.solve_change_info}")

        if not sp.is_solving and sp.solve_status:
            row = box.row()
            row.scale_y = 1.4
            if sp.solve_status == "Converged":
                row.label(text="Converged", icon='CHECKMARK')
            else:
                row.alert = True
                row.label(text=sp.solve_status, icon='CANCEL')

        if res.get_cached_density() is not None:
            box2 = layout.box()
            box2.label(text="Result", icon='OUTLINER_OB_MESH')
            box2.prop(scene.topopt, "density_threshold", slider=True)
            box2.separator(factor=0.4)
            row_m1 = box2.row(align=True)
            row_m1.prop(scene.topopt, "mesh_close_holes",   toggle=True)
            row_m1.prop(scene.topopt, "mesh_include_supports", toggle=True)
            row_m1.prop(scene.topopt, "mesh_include_loads",    toggle=True)
            row_m2 = box2.row(align=True)
            row_m2.prop(scene.topopt, "mesh_smooth_iterations", text="Passes")
            row_m2.prop(scene.topopt, "mesh_smooth_factor",     text="Smooth")
            box2.operator("topopt.generate_mesh", icon='MESH_DATA')


CLASSES = (
    TOPOPT_OT_generate_mesh,
    TOPOPT_OT_solve_3d,
    TOPOPT_OT_cancel_solve,
    TOPOPT_OT_cancel_confirm,
    TOPOPT_OT_set_role,
    TOPOPT_OT_voxelize_preview,
    TOPOPT_OT_toggle_sources,
    TOPOPT_OT_print_summary,
    TOPOPT_PT_main,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
