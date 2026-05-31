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


def _fmt(s):
    return f"{int(s // 60)}m {s % 60:.0f}s" if s >= 60 else f"{s:.1f}s"


# ── minimal 4×6 bitmap font (bit 3 = leftmost column) ───────────────────────
_GLYPHS = {
    '0':[0b0110,0b1010,0b1010,0b1010,0b1010,0b0110],
    '1':[0b0100,0b1100,0b0100,0b0100,0b0100,0b1110],
    '2':[0b0110,0b0010,0b0110,0b1000,0b1000,0b1110],
    '3':[0b1110,0b0010,0b0110,0b0010,0b0010,0b1110],
    '4':[0b1010,0b1010,0b1110,0b0010,0b0010,0b0010],
    '5':[0b1110,0b1000,0b1110,0b0010,0b0010,0b1110],
    '6':[0b0110,0b1000,0b1110,0b1010,0b1010,0b0110],
    '7':[0b1110,0b0010,0b0010,0b0100,0b0100,0b0100],
    '8':[0b0110,0b1010,0b0110,0b1010,0b1010,0b0110],
    '9':[0b0110,0b1010,0b0110,0b0010,0b0010,0b0110],
    '.':[0b0000,0b0000,0b0000,0b0000,0b0110,0b0110],
    '-':[0b0000,0b0000,0b1110,0b0000,0b0000,0b0000],
    'k':[0b1000,0b1010,0b1100,0b1010,0b1010,0b1010],
    'C':[0b0110,0b1000,0b1000,0b1000,0b1000,0b0110],
    'D':[0b1100,0b1010,0b1010,0b1010,0b1010,0b1100],
    'V':[0b1010,0b1010,0b1010,0b1010,0b1010,0b0100],
    'T':[0b1110,0b0100,0b0100,0b0100,0b0100,0b0100],
    'o':[0b0000,0b0000,0b0110,0b1010,0b1010,0b0110],
    'm':[0b0000,0b0000,0b1010,0b1110,0b1010,0b1010],
    'p':[0b0000,0b0000,0b1110,0b1010,0b1110,0b1000],
    'e':[0b0000,0b0000,0b0110,0b1110,0b1000,0b0110],
    'l':[0b1000,0b1000,0b1000,0b1000,0b1000,0b0110],
    't':[0b0100,0b1110,0b0100,0b0100,0b0100,0b0010],
    'a':[0b0000,0b0000,0b0110,0b1110,0b1010,0b0110],
    'i':[0b0100,0b0000,0b0100,0b0100,0b0100,0b0100],
    'n':[0b0000,0b0000,0b1100,0b1010,0b1010,0b1010],
    'r':[0b0000,0b0000,0b1010,0b1100,0b1000,0b1000],
    ' ':[0b0000,0b0000,0b0000,0b0000,0b0000,0b0000],
}
_GW = 4

_GRAPH_IMAGE_NAME = "TopOpt_Metrics"


def _fmt_val(v):
    a = abs(v)
    if a >= 1e5:  return f"{v / 1e3:.0f}k"
    if a >= 1000: return f"{v:.0f}"
    if a >= 100:  return f"{v:.1f}"
    if a >= 10:   return f"{v:.2f}"
    return f"{v:.3f}"


def _draw_str(canvas, text, tx, ty, color, H, W, scale=1):
    """Draw text. tx/ty = top-left in top-down screen coords (ty=0 is image top).
    Each glyph pixel is rendered as a scale×scale block."""
    cx = tx
    for ch in str(text):
        rows = _GLYPHS.get(ch, _GLYPHS[' '])
        for r, bits in enumerate(rows):
            for sr in range(scale):
                cy = H - 1 - (ty + r * scale + sr)
                if 0 <= cy < H:
                    for c in range(_GW):
                        if bits & (1 << (_GW - 1 - c)):
                            for sc in range(scale):
                                px = cx + c * scale + sc
                                if 0 <= px < W:
                                    canvas[cy, px] = color
        cx += (_GW + 1) * scale
    return cx


def _draw_str_right(canvas, text, rx, ty, color, H, W, scale=1):
    """Draw text right-aligned so the last pixel lands at x = rx."""
    tx = rx - len(text) * (_GW + 1) * scale + scale
    _draw_str(canvas, text, max(0, tx), ty, color, H, W, scale)


def _show_compliance_graph(c_hist, d_hist=None, v_hist=None, t_hist=None):
    """Square 4-panel solver metrics graph rendered as a Blender image."""
    n = len(c_hist)
    if n < 2:
        return

    d_hist = d_hist or [0.0] * n
    v_hist = v_hist or [0.0] * n
    t_hist = t_hist or [0.0] * n

    W          = 1024
    FONT_SCALE = 3
    FONT_H     = 6 * FONT_SCALE          # glyph height in pixels
    CHAR_W     = (_GW + 1) * FONT_SCALE  # per-character stride in pixels
    PAD_L   = 6 * CHAR_W + 8  # fits 6-char y-axis labels with margin
    PAD_R   = 10
    PAD_T   = 10
    PAD_B   = 6
    XAXIS_H = FONT_H + 10     # tick labels + breathing room
    GAP     = 5
    n_panels = 4
    # Compute PANEL_H so that H ≈ W (square image)
    PANEL_H = (W - PAD_T - (n_panels - 1) * GAP - XAXIS_H - PAD_B) // n_panels
    H       = PAD_T + n_panels * PANEL_H + (n_panels - 1) * GAP + XAXIS_H + PAD_B
    pw      = W - PAD_L - PAD_R

    panels = [
        (c_hist, np.array([0.18, 0.42, 0.78, 1.0], dtype=np.float32), "Comp"),
        (d_hist, np.array([0.90, 0.50, 0.10, 1.0], dtype=np.float32), "Delt"),
        (v_hist, np.array([0.20, 0.70, 0.30, 1.0], dtype=np.float32), "Vol"),
        (t_hist, np.array([0.65, 0.20, 0.70, 1.0], dtype=np.float32), "Time"),
    ]

    canvas  = np.ones((H, W, 4), dtype=np.float32)
    DARK    = np.array([0.15, 0.15, 0.15, 1.0], dtype=np.float32)
    MUTED   = np.array([0.50, 0.50, 0.50, 1.0], dtype=np.float32)

    def panel_bot(i):          # i=0 → top (Comp), i=3 → bottom (Time)
        return PAD_B + XAXIS_H + (n_panels - 1 - i) * (PANEL_H + GAP)

    for i, (data, color, label) in enumerate(panels):
        yb = panel_bot(i)
        yt = yb + PANEL_H

        canvas[yb:yt, PAD_L:W - PAD_R, :3] = 0.93
        for frac in (0.25, 0.5, 0.75):
            gy = yb + int(frac * PANEL_H)
            if gy < yt:
                canvas[gy, PAD_L:W - PAD_R, :3] = 0.80

        d_min, d_max = min(data), max(data)
        if d_max == d_min:
            d_max = d_min + 1.0

        for j in range(n - 1):
            x0 = PAD_L + int(j       / (n - 1) * pw)
            x1 = PAD_L + int((j + 1) / (n - 1) * pw)
            y0 = yb + int((data[j]     - d_min) / (d_max - d_min) * (PANEL_H - 1))
            y1 = yb + int((data[j + 1] - d_min) / (d_max - d_min) * (PANEL_H - 1))
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for s in range(steps + 1):
                frac = s / steps
                cx = int(x0 + frac * (x1 - x0))
                cy = int(y0 + frac * (y1 - y0))
                for dy in (-2, -1, 0, 1, 2):
                    for dx in (-2, -1, 0, 1, 2):
                        px, py = cx + dx, cy + dy
                        if PAD_L <= px < W - PAD_R and yb <= py < yt:
                            canvas[py, px] = color

        # panel label inside plot (top-left corner)
        _draw_str(canvas, label, PAD_L + 4, H - yt + 1, DARK, H, W, FONT_SCALE)

        # y-axis labels (right-aligned in PAD_L margin)
        rx = PAD_L - 4
        ty_max = H - yt + 1
        ty_min = H - FONT_H - yb - 1
        _draw_str_right(canvas, _fmt_val(d_max), rx, ty_max, MUTED, H, W, FONT_SCALE)
        _draw_str_right(canvas, _fmt_val(d_min), rx, ty_min, MUTED, H, W, FONT_SCALE)

    # ── x axis ────────────────────────────────────────────────────────────────
    tick_y = PAD_B + XAXIS_H - 1
    canvas[tick_y, PAD_L:W - PAD_R, :3] = 0.40

    step = 1 if n <= 10 else 5 if n <= 30 else 10 if n <= 100 else 20
    ty_tick = H - tick_y + 3

    def _draw_tick(it):
        tx = PAD_L + int(it / (n - 1) * pw) if n > 1 else PAD_L
        canvas[max(0, tick_y - 4):tick_y, tx, :3] = 0.40
        lbl = str(it + 1)
        lw  = len(lbl) * CHAR_W - FONT_SCALE
        _draw_str(canvas, lbl, tx - lw // 2, ty_tick, DARK, H, W, FONT_SCALE)
        return it

    drawn = {_draw_tick(it) for it in range(0, n, step)}
    if n - 1 not in drawn:
        _draw_tick(n - 1)

    # ── output (always overwrite; use_fake_user=False so .blend doesn't keep it) ──
    if _GRAPH_IMAGE_NAME in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[_GRAPH_IMAGE_NAME])
    img = bpy.data.images.new(_GRAPH_IMAGE_NAME, width=W, height=H, alpha=False)
    img.use_fake_user = False
    img.pixels.foreach_set(canvas.flatten())
    img.update()

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.spaces.active.image = img
                return
    print(f"[TopOpt] Metrics graph ready — open an Image Editor to view '{_GRAPH_IMAGE_NAME}'.")


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

    _timer              = None
    _gen                = None
    _problem            = None
    _solve_start        = None
    _compliance_history = None
    _delta_history      = None
    _volume_history     = None
    _time_history       = None

    def modal(self, context, event):
        sp = context.scene.topopt
        if event.type == 'TIMER':
            if sp.solve_cancel_requested:
                self._finish(context)
                sp.solve_status = "Cancelled"
                context.workspace.status_text_set(
                    f"TopOpt  Cancelled  {sp.solve_iter_info}  {sp.solve_total_time_info}"
                )
                print(f"[TopOpt] Cancelled  {sp.solve_iter_info}  {sp.solve_total_time_info}")
                _show_compliance_graph(
                    self._compliance_history or [],
                    self._delta_history,
                    self._volume_history,
                    self._time_history,
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
                print(f"[TopOpt] Max iterations reached  {sp.solve_iter_info}  {sp.solve_total_time_info}")
                _show_compliance_graph(
                    self._compliance_history or [],
                    self._delta_history,
                    self._volume_history,
                    self._time_history,
                )
                self._show_result(context)
                return {'FINISHED'}
            except Exception as err:
                self._finish(context)
                sp.solve_status = f"Error: {type(err).__name__}: {err}"
                context.workspace.status_text_set(f"TopOpt  Error: {err}")
                self.report({'ERROR'}, f"{type(err).__name__}: {err}")
                print(f"[TopOpt] ERROR  {type(err).__name__}: {err}")
                _show_compliance_graph(
                    self._compliance_history or [],
                    self._delta_history,
                    self._volume_history,
                    self._time_history,
                )
                return {'CANCELLED'}

            elapsed = time.time() - t0

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
                print(f"[TopOpt] Timed out  {sp.solve_iter_info}  iter took {_fmt(elapsed)} (limit {timeout}s)")
                _show_compliance_graph(
                    self._compliance_history or [],
                    self._delta_history,
                    self._volume_history,
                    self._time_history,
                )
                return {'CANCELLED'}

            res.cache_density(result.density)
            threshold = context.scene.topopt.density_threshold
            preview.build_result_preview(context, self._problem, result.density, threshold)
            self._redraw(context)

            total = time.time() - self._solve_start

            max_iter = sp.max_iterations
            sp.solve_iter_info       = f"Iter {result.iteration}/{max_iter}"
            sp.solve_time_info       = f"Iter: {_fmt(elapsed)}"
            sp.solve_total_time_info = f"Total: {_fmt(total)}"
            sp.solve_status          = "Converged" if result.converged else ""
            sp.solve_compliance_info = f"Comp={result.compliance:.4g}"
            self._compliance_history.append(result.compliance)
            self._delta_history.append(result.change)
            self._volume_history.append(result.vol_frac)
            self._time_history.append(elapsed)
            sp.solve_volume_info     = f"Vol={result.vol_frac:.3f}"
            sp.solve_change_info     = f"Δ={result.change:.5f}"

            context.workspace.status_text_set(
                f"TopOpt  {sp.solve_iter_info}  {sp.solve_compliance_info}  "
                f"{sp.solve_change_info}  {sp.solve_time_info}  {sp.solve_total_time_info}     [ESC] Cancel"
            )
            print(f"[TopOpt] {sp.solve_iter_info}  {sp.solve_compliance_info}"
                  f"  {sp.solve_change_info}  {sp.solve_time_info}  {sp.solve_total_time_info}")

            if result.converged:
                self._finish(context)
                context.workspace.status_text_set(
                    f"TopOpt  Converged  {sp.solve_iter_info}  "
                    f"{sp.solve_compliance_info}  {sp.solve_total_time_info}"
                )
                print(f"[TopOpt] Converged  {sp.solve_iter_info}  {sp.solve_compliance_info}  {sp.solve_total_time_info}")
                _show_compliance_graph(
                    self._compliance_history or [],
                    self._delta_history,
                    self._volume_history,
                    self._time_history,
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
            print(f"[TopOpt] Cancelled (ESC)  {sp.solve_iter_info}  {sp.solve_total_time_info}")
            _show_compliance_graph(
                    self._compliance_history or [],
                    self._delta_history,
                    self._volume_history,
                    self._time_history,
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
        bpy.data.objects[preview.PREVIEW_NAME].hide_set(False)
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
        self._compliance_history = []
        self._delta_history      = []
        self._volume_history     = []
        self._time_history       = []
        print(f"\n[TopOpt] Starting  grid={nx}×{ny}×{nz}  voxels={p.n_design_voxels}"
              f"  target_vol={p.target_volume_fraction:.2f}"
              f"  E={p.youngs_modulus_GPa}GPa  ν={p.poissons_ratio}")

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
    bl_label = "Cancel Solver  [ESC]"
    bl_options = {'REGISTER'}

    def execute(self, context):
        sp = context.scene.topopt
        sp.solve_cancel_requested = True
        # Force-reset in case the modal died without cleaning up.
        sp.is_solving            = False
        sp.solve_confirm_pending = False
        sp.solve_status          = "Cancelled"
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
    img = bpy.data.images.get(_GRAPH_IMAGE_NAME)
    if img:
        bpy.data.images.remove(img)
