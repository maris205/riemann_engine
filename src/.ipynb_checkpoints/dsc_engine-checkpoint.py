"""
DSC Physics Engine - Shared module for all experiments.

Implements the core Störmer-Verlet symplectic integrator and analysis tools
for the Discrete Symplectic Cosmology (DSC) CMB simulation paper.

Reference: Wang, L. (2026). Discrete Symplectic Cosmology.
           Zenodo. https://doi.org/10.5281/zenodo.19429778
"""

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import uniform_filter
from scipy.stats import skew, kurtosis

# ── Isotropic 2D Laplacian kernel ────────────────────────
LAP_KERNEL = np.array([[1, 2, 1],
                        [2, -12, 2],
                        [1, 2, 1]], dtype=np.float64) / 12.0


# ── Initial field generation ─────────────────────────────

def generate_initial_2d(N, seed=42):
    """
    Generate a 2D initial field with scale-invariant spectrum P(k) ~ 1/k^3.
    This mimics the approximately scale-invariant primordial power spectrum.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((N, N))
    fk = np.fft.fft2(noise)
    kx, ky = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N))
    K2 = kx**2 + ky**2
    K2[0, 0] = 1e-10
    filt = 1.0 / (np.sqrt(K2) * N)**1.5
    filt[0, 0] = 0
    phi = np.real(np.fft.ifft2(fk * filt))
    return (phi - phi.mean()) / phi.std()


def generate_initial_3d(N, seed=42, spectral_index=0.75):
    """
    Generate a 3D initial field with P(k) ~ k^{-2*spectral_index}.
    Default spectral_index=0.75 gives P(k) ~ k^{-1.5}.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((N, N, N))
    fk = np.fft.fftn(noise)
    kx, ky, kz = np.meshgrid(
        np.fft.fftfreq(N), np.fft.fftfreq(N), np.fft.fftfreq(N), indexing='ij'
    )
    K = np.sqrt(kx**2 + ky**2 + kz**2)
    K[0, 0, 0] = 1e-10
    filt = 1.0 / (K * N)**spectral_index
    filt[0, 0, 0] = 0
    phi = np.real(np.fft.ifftn(fk * filt))
    return phi / phi.std() * 0.15


# ── 2D Symplectic evolution (Störmer-Verlet) ─────────────

def evolve_symplectic_2d(phi0, n_steps=45, c2_base=0.45, c0=10.0, drag=0.0):
    """
    Störmer-Verlet symplectic integrator on a 2D lattice.

    phi_{n+1} = 2*phi_n - phi_{n-1} + c2(n)*Lap(phi_n) - drag*(phi_n - phi_{n-1})

    where c2(n) = c2_base / ln^2(n + c0) is the DSC cooling law.

    Parameters
    ----------
    phi0 : (N, N) array - initial field
    n_steps : int - number of evolution steps
    c2_base : float - base sound speed squared
    c0 : float - regularization offset
    drag : float - Hubble drag coefficient (0 = exactly symplectic)

    Returns
    -------
    phi : (N, N) array - final field
    """
    phi = phi0.copy()
    phi_prev = phi0.copy()
    for t in range(1, n_steps + 1):
        c2_t = c2_base / np.log(t + c0)**2
        lap = convolve2d(phi, LAP_KERNEL, mode='same', boundary='wrap')
        phi_new = 2.0 * phi - phi_prev + c2_t * lap - drag * (phi - phi_prev)
        phi_new = np.clip(phi_new, -10.0, 10.0)
        phi_prev = phi.copy()
        phi = phi_new
    return phi


# ── 2D Dissipative evolution (CML baseline) ──────────────

def evolve_dissipative_2d(phi0, n_steps=150, mu_c=1.5437, k_opt=1.248,
                           c0=10.0, eps=0.55):
    """
    Dissipative coupled map lattice (CML) for ablation comparison.

    x_{n+1} = f(x_n) + eps * Lap(f(x_n))
    f(x) = 1 - mu(n) * x^2
    mu(n) = mu_c + k_opt / ln^2(n + c0)

    This model is first-order and dissipative (|det DF| < 1).
    """
    phi = phi0 * 0.1  # CML needs small initial amplitude
    for t in range(1, n_steps + 1):
        mu_t = mu_c + k_opt / np.log(t + c0)**2
        fx = 1.0 - mu_t * phi**2
        lap_fx = convolve2d(fx, LAP_KERNEL, mode='same', boundary='wrap')
        phi = fx + eps * lap_fx
        phi = np.clip(phi, -10.0, 10.0)
    return phi


# ── 3D Symplectic evolution ──────────────────────────────

def laplacian_3d(phi):
    """3D discrete Laplacian with periodic boundary conditions."""
    return (np.roll(phi, 1, 0) + np.roll(phi, -1, 0) +
            np.roll(phi, 1, 1) + np.roll(phi, -1, 1) +
            np.roll(phi, 1, 2) + np.roll(phi, -1, 2) - 6.0 * phi) / 6.0


def evolve_symplectic_3d(phi0, n_steps=45, c2_base=0.45, c0=10.0,
                          drag=0.015, nonlinear=0.005):
    """
    Störmer-Verlet on a 3D lattice with optional nonlinear term.

    Parameters
    ----------
    nonlinear : float - coefficient of -alpha*phi^2 term (gravitational structure)
    """
    phi = phi0.copy()
    phi_prev = phi0.copy()
    for t in range(1, n_steps):
        c2_t = c2_base / np.log(t + c0)**2
        lap = laplacian_3d(phi)
        phi_new = (2.0 * phi - phi_prev + c2_t * lap
                   - drag * (phi - phi_prev)
                   - nonlinear * phi**2)
        phi_new = np.clip(phi_new, -5.0, 5.0)
        phi_prev = phi.copy()
        phi = phi_new
    return phi


# ── Post-processing ──────────────────────────────────────

def apply_silk_damping(phi, k_silk=35.0):
    """Apply Silk damping in Fourier space: exp(-(k/k_silk)^2)."""
    N = phi.shape[0]
    fk = np.fft.fft2(phi)
    kx = np.fft.fftfreq(N, d=1.0 / N)
    ky = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    return np.real(np.fft.ifft2(fk * np.exp(-(K / k_silk)**2)))


def normalize(phi):
    """Zero-mean, unit-variance normalization."""
    return (phi - phi.mean()) / (phi.std() + 1e-15)


# ── Power spectrum analysis ──────────────────────────────

def compute_power_spectrum(phi):
    """
    Compute radially averaged 2D power spectrum P(k) and D(k) = k^2 P(k).

    Returns
    -------
    k_bins : 1D array of wavenumbers
    Dk : D(k) = k^2 P(k) (scaled power spectrum)
    Pk : P(k) (raw power spectrum)
    """
    N = phi.shape[0]
    ps = np.abs(np.fft.fftshift(np.fft.fft2(phi)))**2 / N**4
    kx = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / N))
    ky = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / N))
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    k_bins = np.arange(1, N // 2 + 1, dtype=float)
    Pk = np.zeros(len(k_bins))
    for i in range(len(k_bins)):
        mask = (K >= i + 0.5) & (K < i + 1.5)
        if mask.any():
            Pk[i] = ps[mask].mean()
    Dk = k_bins**2 * Pk
    return k_bins, Dk, Pk


def mock_lcdm_spectrum(k):
    """
    Mock ΛCDM D(k) with acoustic oscillations and Silk damping.
    D(k) = exp(-(k/35)^2) * [1 + 1.2*sin^2(pi*k/12)] * 50/(k+10)
    """
    return (np.exp(-(k / 35)**2) *
            (1.0 + 1.2 * np.sin(np.pi * k / 12)**2) *
            (50.0 / (k + 10)))


def smooth(y, window=3):
    """Simple moving average."""
    return uniform_filter(y.astype(float), size=window)


# ── Gaussianity metrics ──────────────────────────────────

def gaussianity_report(phi_flat):
    """Compute skewness, kurtosis, and standard errors."""
    z = (phi_flat - phi_flat.mean()) / (phi_flat.std() + 1e-15)
    n = len(z)
    return {
        'skewness': float(skew(z)),
        'kurtosis': float(kurtosis(z)),  # excess kurtosis
        'skew_se': float(np.sqrt(6.0 / n)),
        'kurt_se': float(np.sqrt(24.0 / n)),
        'n_samples': n,
    }
