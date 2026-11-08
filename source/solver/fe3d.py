"""
3-D finite element assembly for topology optimisation.

One hex8 (8-node brick) element per voxel. All elements are identical cubes
so Ke is computed once and reused. DOF ordering per element:

  7---6        Nodes 0-3: Z_min face (CCW from bottom-left)
  |   |        Nodes 4-7: Z_max face (same XY order)
  4---5
  3---2
  |   |   Z  Y
  0---1   | /
          |/__X

  node(i,j,k) = k*(nx+1)*(ny+1) + j*(nx+1) + i
  DOFs per node: [3n, 3n+1, 3n+2] = [u, v, w]
"""

import itertools
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu

# Stiffness of a void element relative to solid (keeps K non-singular).
E_MIN = 1e-9


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


def build_dof_maps_3d(nx, ny, nz):
    """Vectorised DOF map for all (nx × ny × nz) elements.

    Element e = k*(nx*ny) + j*nx + i  matches ravel(order='F') on (nx,ny,nz).
    Returns edof of shape (n_elem, 24).
    """
    nxy = (nx + 1) * (ny + 1)

    # Grid indices for every element, traversal k→j→i
    k_r = np.repeat(np.arange(nz), nx * ny)
    j_r = np.tile(np.repeat(np.arange(ny), nx), nz)
    i_r = np.tile(np.arange(nx), ny * nz)

    n0 = k_r * nxy + j_r * (nx+1) + i_r
    n1 = n0 + 1
    n2 = n0 + (nx+1) + 1
    n3 = n0 + (nx+1)
    n4 = n0 + nxy
    n5 = n1 + nxy
    n6 = n2 + nxy
    n7 = n3 + nxy

    edof = np.column_stack([
        3*n0, 3*n0+1, 3*n0+2,
        3*n1, 3*n1+1, 3*n1+2,
        3*n2, 3*n2+1, 3*n2+2,
        3*n3, 3*n3+1, 3*n3+2,
        3*n4, 3*n4+1, 3*n4+2,
        3*n5, 3*n5+1, 3*n5+2,
        3*n6, 3*n6+1, 3*n6+2,
        3*n7, 3*n7+1, 3*n7+2,
    ]).astype(np.int32)

    return edof


class ReducedSystem:
    """Stiffness system of the given (domain) elements with support DOFs removed.

    The DOF numbering, sparsity pattern and the map from element stiffness
    entries to matrix values are built once; each iteration only recomputes
    the values (one sparse mat-vec) and factorizes.
    """

    def __init__(self, Ke, edof, fixed_dofs, n_dofs, E_min=E_MIN):
        used = np.unique(edof)
        self.free_dofs = np.setdiff1d(used, fixed_dofs, assume_unique=True)
        n_free = self.n_free = len(self.free_dofs)
        if n_free == 0:
            raise RuntimeError("Supports fix every node of the domain — nothing left to solve.")

        # Reduced numbering; fixed DOFs map to n_free, a zero slot past the end.
        to_free = np.full(n_dofs, n_free, dtype=np.int64)
        to_free[self.free_dofs] = np.arange(n_free)
        self.edof_free = to_free[edof]                  # (n_elem, 24)

        # Entry (e, i, j) of the element matrices lands in K[edof_free[e, i], edof_free[e, j]].
        n_elem = len(edof)
        rows = np.repeat(self.edof_free, 24, axis=1).ravel()
        cols = np.tile(self.edof_free, (1, 24)).ravel()
        keep = (rows < n_free) & (cols < n_free)
        keys = rows[keep] * n_free + cols[keep]
        del rows, cols
        uniq, slot = np.unique(keys, return_inverse=True)
        del keys
        elem = np.repeat(np.arange(n_elem), 24 * 24)[keep]
        vals = np.tile(Ke.ravel(), n_elem)[keep]
        self.P = csr_matrix((vals, (slot, elem)), shape=(len(uniq), n_elem))

        k_rows       = uniq // n_free
        self.indices = (uniq % n_free).astype(np.int32)
        self.indptr  = np.searchsorted(k_rows, np.arange(n_free + 1)).astype(np.int32)
        self.diag    = np.flatnonzero(k_rows == self.indices)
        # One void element's worth of stiffness on every DOF keeps hinged or loose parts solvable.
        self.reg = E_min * Ke.diagonal().mean()

    def solve(self, E_e, f_free):
        """Solve K(E_e) u = f_free; returns u in reduced numbering."""
        data = self.P @ E_e
        data[self.diag] += self.reg
        # K is symmetric, so its CSR arrays read as CSC describe the same matrix.
        K = csc_matrix((data, self.indices, self.indptr), shape=(self.n_free, self.n_free))
        u = splu(K, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                 options=dict(SymmetricMode=True)).solve(f_free)
        if not np.all(np.isfinite(u)):
            raise RuntimeError(
                "3-D FE solve produced NaN. Ensure supports constrain all 6 "
                "rigid-body modes (3 translations + 3 rotations)."
            )
        return u

    def element_displacements(self, u_free):
        """(n_elem, 24) displacements per element; fixed DOFs are zero."""
        return np.append(u_free, 0.0)[self.edof_free]


def compute_sensitivities_3d(ue, rho_elem, Ke, penal, E_min=E_MIN):
    ue_Ke_ue = ((ue @ Ke) * ue).sum(axis=1)
    return penal * (rho_elem**(penal - 1)) * (1.0 - E_min) * ue_Ke_ue
