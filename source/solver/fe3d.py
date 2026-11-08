"""
3-D finite elements for topology optimisation, solved matrix-free on the voxel grid.

One hex8 (8-node brick) element per voxel. All elements are identical cubes
so Ke is computed once and scaled per element. Node order per element:

  7---6        Nodes 0-3: Z_min face (CCW from bottom-left)
  |   |        Nodes 4-7: Z_max face (same XY order)
  4---5
  3---2
  |   |   Z  Y
  0---1   | /
          |/__X

Node fields (displacements, forces) are arrays of shape (nx+1, ny+1, nz+1, 3).
Element DOFs are ordered [node 0 xyz, node 1 xyz, ...] as in Ke.

K u = f is solved with conjugate gradients, preconditioned by a geometric
multigrid V-cycle: each coarser level merges 2×2×2 elements into one with
Galerkin (Pᵀ K P) element matrices, smooths with damped Jacobi, and the
coarsest level is solved densely. Plain NumPy, no sparse matrices.
"""

import itertools
import numpy as np

# Stiffness of a void element relative to solid.
E_MIN = 1e-9

# Grid offset (i, j, k) of each element node, in Ke's node order.
NODE_OFFSETS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))

_CG_RTOL     = 1e-6   # stop when |r| <= _CG_RTOL * |f|
_CG_MAX_ITER = 500
_SMOOTH      = 2      # damped-Jacobi sweeps before and after each coarse correction
_OMEGA       = 0.6    # Jacobi damping
_COARSEST    = 600    # solve a level directly once it has at most this many DOFs


class SolveCancelled(Exception):
    """Raised inside a solve when its cancel event is set."""


def element_stiffness_3d(E, nu):
    """24×24 stiffness for a unit hex8 element (natural coords ±1).

    Uses 2×2×2 Gauss quadrature.  For a unit element J=I, det(J)=1.
    Topology is scale-invariant so element physical size is irrelevant.
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    C = np.array([
        [lam+2*mu, lam,      lam,      0,  0,  0],
        [lam,      lam+2*mu, lam,      0,  0,  0],
        [lam,      lam,      lam+2*mu, 0,  0,  0],
        [0,        0,        0,        mu, 0,  0],
        [0,        0,        0,        0,  mu, 0],
        [0,        0,        0,        0,  0,  mu],
    ])

    # Node natural coordinates
    ni = np.array([
        [-1,-1,-1],[+1,-1,-1],[+1,+1,-1],[-1,+1,-1],
        [-1,-1,+1],[+1,-1,+1],[+1,+1,+1],[-1,+1,+1],
    ], dtype=float)

    gp = 1.0 / np.sqrt(3)
    gauss = list(itertools.product([-gp, gp], repeat=3))   # 8 Gauss points

    Ke = np.zeros((24, 24))
    for xi, eta, zeta in gauss:
        # Shape function derivatives in natural coords
        dN = np.array([
            [ni[i,0]*(1+ni[i,1]*eta) *(1+ni[i,2]*zeta)/8,
             ni[i,1]*(1+ni[i,0]*xi)  *(1+ni[i,2]*zeta)/8,
             ni[i,2]*(1+ni[i,0]*xi)  *(1+ni[i,1]*eta) /8]
            for i in range(8)
        ])  # (8, 3)

        # Strain-displacement matrix B (6×24)
        B = np.zeros((6, 24))
        for i, (dx, dy, dz) in enumerate(dN):
            c = 3 * i
            B[0, c]   = dx
            B[1, c+1] = dy
            B[2, c+2] = dz
            B[3, c]   = dy;  B[3, c+1] = dx
            B[4, c+1] = dz;  B[4, c+2] = dy
            B[5, c]   = dz;  B[5, c+2] = dx

        Ke += B.T @ C @ B   # det(J)=1, weight=1 each Gauss point

    return Ke


def gather(u, shape):
    """Element DOF values (nx, ny, nz, 24) from a node field (nx+1, ny+1, nz+1, 3)."""
    nx, ny, nz = shape
    return np.concatenate([u[a:a + nx, b:b + ny, c:c + nz] for a, b, c in NODE_OFFSETS], axis=-1)


def scatter(fe, shape):
    """Sum element DOF values (nx, ny, nz, 24) onto a node field (nx+1, ny+1, nz+1, 3)."""
    nx, ny, nz = shape
    y = np.zeros((nx + 1, ny + 1, nz + 1, 3))
    for n, (a, b, c) in enumerate(NODE_OFFSETS):
        y[a:a + nx, b:b + ny, c:c + nz] += fe[..., 3 * n:3 * n + 3]
    return y


def compute_sensitivities_3d(ue, rho_elem, Ke, penal, E_min=E_MIN):
    ue_Ke_ue = ((ue @ Ke) * ue).sum(axis=-1)
    return penal * (rho_elem**(penal - 1)) * (1.0 - E_min) * ue_Ke_ue


# ── multigrid transfer ────────────────────────────────────────────────────────

def _prolong(xc, fine_nodes):
    """Trilinear interpolation of a coarse node field onto the fine node grid
    (fine node 2I sits on coarse node I)."""
    for axis, n in enumerate(fine_nodes):
        xc = np.moveaxis(xc, axis, 0)
        xf = np.empty((2 * xc.shape[0] - 1,) + xc.shape[1:])
        xf[0::2] = xc
        xf[1::2] = 0.5 * (xc[:-1] + xc[1:])
        xc = np.moveaxis(xf[:n], 0, axis)
    return xc


def _restrict(xf, coarse_nodes):
    """Transpose of _prolong."""
    for axis, n in enumerate(coarse_nodes):
        xf = np.moveaxis(xf, axis, 0)
        missing = 2 * n - 1 - xf.shape[0]
        if missing:
            xf = np.concatenate([xf, np.zeros((missing,) + xf.shape[1:])])
        xc = xf[0::2].copy()
        half = 0.5 * xf[1::2]
        xc[:-1] += half
        xc[1:] += half
        xf = np.moveaxis(xc, 0, axis)
    return xf


def _children(x):
    """Group a per-element array (nx, ny, nz, ...) into 2×2×2 blocks (ncx, ncy, ncz, 8, ...),
    padding odd dimensions with empty elements."""
    pad = [(0, s % 2) for s in x.shape[:3]] + [(0, 0)] * (x.ndim - 3)
    x = np.pad(x, pad)
    ncx, ncy, ncz = x.shape[0] // 2, x.shape[1] // 2, x.shape[2] // 2
    x = x.reshape((ncx, 2, ncy, 2, ncz, 2) + x.shape[3:])
    x = np.moveaxis(x, (1, 3, 5), (3, 4, 5))
    return x.reshape((ncx, ncy, ncz, 8) + x.shape[6:])


def _child_prolongations():
    """(8, 24, 24): coarse-element DOFs → DOFs of each child element, in _children order."""
    W = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])  # 1-D: coarse nodes 0,1 → fine nodes 0,1,2
    P = np.zeros((8, 24, 24))
    for c, (ci, cj, ck) in enumerate(itertools.product((0, 1), repeat=3)):
        Pn = np.array([[W[ci + fi, Ci] * W[cj + fj, Cj] * W[ck + fk, Ck]
                        for Ci, Cj, Ck in NODE_OFFSETS]
                       for fi, fj, fk in NODE_OFFSETS])
        P[c] = np.kron(Pn, np.eye(3))
    return P


class _Level:
    """Operator of one grid level: E * Ke per element on the finest level,
    full element matrices K on coarser ones."""

    def __init__(self, shape, free, reg, diag, E=None, K=None):
        self.shape = shape
        self.nodes = tuple(s + 1 for s in shape)
        self.free  = free
        self.reg   = reg * free
        self.dinv  = np.divide(1.0, diag, out=np.zeros_like(diag), where=free)
        self.E, self.K = E, K


class GridSystem:
    """K u = f over the domain elements of a voxel grid, with support nodes fixed.

    Nodes not touched by a domain element and nodes of support voxels are kept
    at zero. The grid hierarchy and everything that doesn't depend on the
    element densities is set up once.
    """

    def __init__(self, Ke, domain_mask, support_mask, E_min=E_MIN):
        self.Ke    = Ke
        self.shape = nx, ny, nz = domain_mask.shape
        active = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
        fixed  = np.zeros_like(active)
        for a, b, c in NODE_OFFSETS:
            active[a:a + nx, b:b + ny, c:c + nz] |= domain_mask
            fixed[a:a + nx, b:b + ny, c:c + nz]  |= support_mask
        self.free   = np.repeat((active & ~fixed)[..., None], 3, axis=-1)
        self.n_free = int(self.free.sum())
        if self.n_free == 0:
            raise RuntimeError("Supports fix every node of the domain — nothing left to solve.")
        # One void element's worth of stiffness on every free DOF keeps hinged or loose parts solvable.
        self.reg = E_min * Ke.diagonal().mean()

        self._shapes = [self.shape]
        while 3 * np.prod([s + 1 for s in self._shapes[-1]]) > _COARSEST and max(self._shapes[-1]) > 1:
            self._shapes.append(tuple((s + 1) // 2 for s in self._shapes[-1]))
        self.n_levels = len(self._shapes)

        self._P = _child_prolongations()
        Pt = np.swapaxes(self._P, 1, 2)
        self._A = Pt @ Ke @ self._P  # each child's Ke seen from the coarse element

        # Elements touching a fixed node contribute their masked Ke (fixed rows and
        # columns zeroed) to the first coarse level: store that difference once.
        el_free = gather(self.free.astype(float), self.shape)
        partial = domain_mask & (el_free.min(axis=-1) < 1.0)
        i, j, k = np.nonzero(partial)
        m = el_free[partial]
        child = (i % 2) * 4 + (j % 2) * 2 + (k % 2)
        self._partial     = (i, j, k)
        self._partial_dA  = Pt[child] @ ((m[:, :, None] * m[:, None, :] - 1.0) * Ke) @ self._P[child]
        self._partial_par = (i // 2, j // 2, k // 2)

        self._levels = None
        self.last_iterations = 0

    # ── per-iteration setup ──────────────────────────────────────────────────

    def _setup(self, E):
        """Build all level operators for element moduli E (nx, ny, nz)."""
        Ke = self.Ke
        diag = (scatter(E[..., None] * Ke.diagonal(), self.shape) + self.reg) * self.free
        levels = [_Level(self.shape, self.free, self.reg, diag, E=E)]

        for n, shape in enumerate(self._shapes[1:]):
            prev = levels[-1]
            if n == 0:
                K = np.tensordot(_children(E), self._A, axes=([3], [0]))
                np.add.at(K, self._partial_par, E[self._partial][:, None, None] * self._partial_dA)
            else:
                K = (np.swapaxes(self._P, 1, 2) @ (_children(prev.K) @ self._P)).sum(axis=3)
            reg  = _restrict(prev.reg, tuple(s + 1 for s in shape))
            diag = scatter(np.diagonal(K, axis1=-2, axis2=-1), shape) + reg
            free = diag > 0
            levels.append(_Level(shape, free, reg, diag * free, K=K))

        last = levels[-1]
        K_el = last.K if last.K is not None else E[..., None, None] * Ke
        idx  = np.arange(np.prod(last.nodes) * 3).reshape(last.nodes + (3,))
        edof = gather(idx, last.shape).reshape(-1, 24)
        Kd   = np.zeros((idx.size, idx.size))
        np.add.at(Kd, (edof[:, :, None], edof[:, None, :]), K_el.reshape(-1, 24, 24))
        Kd[np.diag_indices_from(Kd)] += last.reg.ravel()
        last.fi  = np.flatnonzero(last.free.ravel())
        last.inv = np.linalg.inv(Kd[np.ix_(last.fi, last.fi)])
        self._levels = levels

    # ── operators ────────────────────────────────────────────────────────────

    def _apply(self, lv, u):
        """K u on level `lv` (u must be zero outside lv.free)."""
        ue = gather(u, lv.shape)
        if lv.E is not None:
            fe = (ue.reshape(-1, 24) @ self.Ke).reshape(ue.shape) * lv.E[..., None]
        else:
            fe = np.einsum('...i,...ij->...j', ue, lv.K)
        return (scatter(fe, lv.shape) + lv.reg * u) * lv.free

    def _vcycle(self, n, r):
        lv = self._levels[n]
        if n == len(self._levels) - 1:
            x = np.zeros_like(r)
            x.reshape(-1)[lv.fi] = lv.inv @ r.reshape(-1)[lv.fi]
            return x
        x = _OMEGA * lv.dinv * r
        for _ in range(_SMOOTH - 1):
            x += _OMEGA * lv.dinv * (r - self._apply(lv, x))
        nxt = self._levels[n + 1]
        rc  = _restrict(r - self._apply(lv, x), nxt.nodes) * nxt.free
        x  += _prolong(self._vcycle(n + 1, rc), lv.nodes) * lv.free
        for _ in range(_SMOOTH):
            x += _OMEGA * lv.dinv * (r - self._apply(lv, x))
        return x

    # ── public ───────────────────────────────────────────────────────────────

    def solve(self, E, f, x0=None, cancel=None):
        """Solve K(E) u = f for element moduli E (nx, ny, nz) and node forces f.

        `x0` warm-starts CG (e.g. the previous iteration's u). `cancel` is an
        optional threading.Event; when set, SolveCancelled is raised.
        """
        self._setup(E)
        fine = self._levels[0]
        b = f * self.free
        b_norm = np.linalg.norm(b)
        x = np.zeros_like(b) if x0 is None else x0 * self.free
        if b_norm == 0.0:
            return np.zeros_like(b)

        r  = b - self._apply(fine, x)
        z  = self._vcycle(0, r)
        p  = z.copy()
        rz = np.vdot(r, z)
        it = 0
        while np.linalg.norm(r) > _CG_RTOL * b_norm and it < _CG_MAX_ITER:
            if cancel is not None and cancel.is_set():
                raise SolveCancelled()
            Ap     = self._apply(fine, p)
            alpha  = rz / np.vdot(p, Ap)
            x     += alpha * p
            r     -= alpha * Ap
            z      = self._vcycle(0, r)
            rz_new = np.vdot(r, z)
            p      = z + (rz_new / rz) * p
            rz     = rz_new
            it    += 1
        self.last_iterations = it

        if not np.all(np.isfinite(x)):
            raise RuntimeError(
                "3-D FE solve produced NaN. Ensure supports constrain all 6 "
                "rigid-body modes (3 translations + 3 rotations)."
            )
        return x

    def element_displacements(self, u):
        """(nx, ny, nz, 24) displacements per element."""
        return gather(u, self.shape)
