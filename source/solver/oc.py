"""
OC (Optimality Criteria) density update and 3-D sensitivity filter.

Reference: Sigmund (2001) 99-line code, Andreassen et al. (2011) 88-line code.
"""

import numpy as np

RHO_MIN = 1e-3


def _cone_kernel(r_min, ndim=3):
    """Unnormalised cone weight kernel H = max(0, r_min - dist)."""
    r  = int(np.ceil(r_min))
    ki = np.arange(-r, r + 1)
    if ndim == 3:
        kx, ky, kz = np.meshgrid(ki, ki, ki, indexing='ij')
        return np.maximum(0.0, r_min - np.sqrt(kx**2 + ky**2 + kz**2))
    kx, ky = np.meshgrid(ki, ki)
    return np.maximum(0.0, r_min - np.sqrt(kx**2 + ky**2))


def sensitivity_filter(dc, rho, r_min):
    """Rho-weighted sensitivity filter (Sigmund 2001 eq. 7) — works for 2-D or 3-D arrays.

    dc_filt[e] = Σ_f (H_ef · ρ_f · dc_f) / (ρ_e · Σ_f H_ef)
    """
    H = _cone_kernel(r_min, ndim=dc.ndim)
    H_sum = H.sum()  # mirrored edges make Σ_f H_ef the same for every element
    if H_sum < 1e-12:
        return dc
    return _convolve(rho * dc, H) / (np.maximum(RHO_MIN, rho) * H_sum)


def _convolve(x, H):
    """Convolve with a symmetric kernel H, mirroring the edges (half-sample
    symmetric, like scipy.ndimage's 'reflect' mode): one shifted sum per weight."""
    r   = H.shape[0] // 2
    xp  = np.pad(x, r, mode='symmetric')
    out = np.zeros(x.shape)
    for offset in zip(*np.nonzero(H)):
        out += H[offset] * xp[tuple(slice(o, o + n) for o, n in zip(offset, x.shape))]
    return out


def oc_update(rho, dc, dv, volfrac, move=0.2, eta=0.5):
    """Classic OC density update with bisection on the volume Lagrange multiplier λ.

    rho     : (n_elem,) current densities
    dc      : (n_elem,) filtered sensitivities (positive, already clipped)
    dv      : (n_elem,) volume sensitivities (= 1 for uniform elements)
    volfrac : target volume fraction for these elements
    """
    lam_lo, lam_hi = 1e-9, 1e9
    n_target = volfrac * len(rho)

    dc_pos  = np.maximum(0.0, dc)
    rho_new = np.empty_like(rho)

    for _ in range(200):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        B_e     = (dc_pos / (np.maximum(1e-12, dv) * lam_mid)) ** eta
        rho_new = np.maximum(RHO_MIN,
                  np.maximum(rho - move,
                  np.minimum(1.0,
                  np.minimum(rho + move, rho * B_e))))
        if rho_new.sum() > n_target:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
        if (lam_hi - lam_lo) / (lam_hi + lam_lo + 1e-30) < 1e-6:
            break

    return rho_new
