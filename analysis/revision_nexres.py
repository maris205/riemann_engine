#!/usr/bin/env python3
"""Reproducible calculations for the NEXRES-D-26-02183 revision.

Every number quoted in the revised manuscript is printed here and written to
analysis/outputs/revision_results.json. No result is copied from an earlier
draft.

Conventions
-----------
Time anchor for the logarithmic clock:
    t_* = 5.391247e-44 s
    1 yr = 365.25 * 86400 s

Hubble relaxation (the four-anchor fit). The regression variable is
    xi(t) = 1e5 / ln(t/t_*)^2
    H(t)  = H_inf + beta * xi(t)
beta multiplies xi, not 1/ln^2. The equivalent coefficient of 1/ln^2 is
beta * 1e5. H_inf is the t -> infinity asymptote.

Fine-structure history, one convention everywhere:
    delta(t) = [alpha(t) - alpha(t0)] / alpha(t0)
             = Gamma * [L(t)^{-2} - L(t0)^{-2}]
    L(t) = ln(t/t_*)
    D0 = (d alpha / alpha / dt) at t0 = -2 Gamma / (t0_yr * L0^3)
Positive Gamma means alpha was larger in the past. Tabulated Delta alpha/alpha
values are in units of 1e-5 and are converted to fractional delta before the
fit. The redshift-to-age map is the flat matter-plus-Lambda integral with
H0 = 67.4 km s^-1 Mpc^-1 and Omega_m = 0.315. That background is a clock for
the fit, not a claim that the relaxation law is LambdaCDM.

The previous manuscript quoted Gamma = +6.42 and Gamma = +5.56 for the same
Murphy table. Those two numbers used opposite sign relative to the convention
above and, respectively, this LambdaCDM age and the Einstein-de Sitter
approximation t proportional to (1+z)^{-3/2}. This script reports only the
convention in this docstring.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import curve_fit, least_squares
from scipy.stats import chi2 as chi2_dist

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

T_STAR = 5.391247e-44  # s
SEC_PER_YEAR = 365.25 * 86400.0
KM_PER_MPC = 3.0856775814913673e19
H0_FID = 67.4  # km/s/Mpc
OM_FID = 0.315

# Four anchors. Epochs are assigned, not measured. Sources are the published
# H0 inferences named in data comments; this script does not re-reduce them.
ANCHORS = [
    {"name": "Planck CMB", "t_yr": 3.8e5, "H": 67.4, "err": 0.5,
     "cite": "Planck 2018 VI"},
    {"name": "TRGB (CCHP/JWST)", "t_yr": 1.5e9, "H": 70.4, "err": 1.9,
     "cite": "Freedman et al. 2025"},
    {"name": "Miras (HST)", "t_yr": 6.0e9, "H": 72.37, "err": 2.97,
     "cite": "Huang et al. 2024"},
    {"name": "SH0ES Cepheids", "t_yr": 13.8e9, "H": 73.04, "err": 1.04,
     "cite": "Riess et al. 2022"},
]

# 32 cosmic-chronometer H(z) points. Values are unchanged. The source label
# is the paper that published that row. z=0.09 is Jimenez et al. 2003
# (H=69+/-12); the earlier notebook comment that assigned it to Simon 2005
# does not match the published compilations.
# Columns: z, H, sigma. Source key is stored separately so the array stays numeric.
CC = np.array([
    [0.0700, 69.0, 19.6],    # Zhang2014
    [0.0900, 69.0, 12.0],    # Jimenez2003
    [0.1200, 68.6, 26.2],    # Zhang2014
    [0.1700, 83.0, 8.0],     # Simon2005
    [0.1791, 75.0, 4.0],     # Moresco2012
    [0.1993, 75.0, 5.0],     # Moresco2012
    [0.2000, 72.9, 29.6],    # Zhang2014
    [0.2700, 77.0, 14.0],    # Simon2005
    [0.2800, 88.8, 36.6],    # Zhang2014
    [0.3519, 83.0, 14.0],    # Moresco2012
    [0.3802, 83.0, 13.5],    # Moresco2016
    [0.4000, 95.0, 17.0],    # Simon2005
    [0.4004, 77.0, 10.2],    # Moresco2016
    [0.4247, 87.1, 11.2],    # Moresco2016
    [0.4497, 92.8, 12.9],    # Moresco2016
    [0.4783, 80.9, 9.0],     # Moresco2016
    [0.4800, 97.0, 62.0],    # Stern2010
    [0.5929, 104.0, 13.0],   # Moresco2012
    [0.6797, 92.0, 8.0],     # Moresco2012
    [0.7500, 98.8, 33.6],    # Borghi2022
    [0.7812, 105.0, 12.0],   # Moresco2012
    [0.8754, 125.0, 17.0],   # Moresco2012
    [0.8800, 90.0, 40.0],    # Stern2010
    [0.9000, 117.0, 23.0],   # Simon2005
    [1.0370, 154.0, 20.0],   # Moresco2012
    [1.3000, 168.0, 17.0],   # Simon2005
    [1.3630, 160.0, 33.6],   # Moresco2015
    [1.4300, 177.0, 18.0],   # Simon2005
    [1.4650, 150.0, 50.0],   # kept as transcribed; not in Stern2010 Table 2
    [1.5300, 140.0, 14.0],   # Simon2005
    [1.7500, 202.0, 40.0],   # Simon2005
    [1.9650, 186.5, 50.4],   # Moresco2015
], dtype=float)


def H_to_per_year(H_kms):
    return H_kms / KM_PER_MPC * SEC_PER_YEAR


def E_lcdm(z, om):
    return np.sqrt(om * (1.0 + z) ** 3 + (1.0 - om))


def age_yr(z, H0=H0_FID, om=OM_FID):
    """Proper time from the Big Bang to redshift z, flat matter plus Lambda.

    Integrand in the scale factor: dt = da / (a H(a)), with the
    matter-dominated piece from a=0 to 1e-6 added in closed form.
    """
    H_per_yr = H_to_per_year(H0)
    a_max = 1.0 / (1.0 + float(z))
    a_split = 1.0e-6

    def integrand(a):
        # H0 dt = da / sqrt(Omega_m / a + (1-Omega_m) a^2)
        return 1.0 / np.sqrt(om / a + (1.0 - om) * a ** 2)

    def early_piece(a):
        # matter-era integral of a^{1/2} / sqrt(Omega_m)
        return (2.0 / 3.0) * a ** 1.5 / np.sqrt(om)

    if a_max <= a_split:
        return float(early_piece(a_max) / H_per_yr), 0.0
    val_num, err = quad(integrand, a_split, a_max, epsabs=1e-12, limit=200)
    return float((early_piece(a_split) + val_num) / H_per_yr), float(err / H_per_yr)


def xi_of_t_yr(t_yr):
    t_sec = np.asarray(t_yr, dtype=float) * SEC_PER_YEAR
    return 1.0e5 / np.log(t_sec / T_STAR) ** 2


def L_of_t_yr(t_yr):
    t_sec = np.asarray(t_yr, dtype=float) * SEC_PER_YEAR
    return np.log(t_sec / T_STAR)


def wls_line(x, y, yerr):
    """y = a + b x, inverse-variance weighted, absolute sigma."""
    w = 1.0 / np.asarray(yerr, dtype=float) ** 2
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sw = np.sum(w)
    sx = np.sum(w * x)
    sy = np.sum(w * y)
    sxx = np.sum(w * x * x)
    sxy = np.sum(w * x * y)
    delta = sw * sxx - sx * sx
    a = (sxx * sy - sx * sxy) / delta
    b = (sw * sxy - sx * sy) / delta
    var_a = sxx / delta
    var_b = sw / delta
    cov_ab = -sx / delta
    resid = y - (a + b * x)
    chi2 = float(np.sum(w * resid ** 2))
    # Unweighted R^2, reported only as a descriptive statistic.
    r2 = float(1.0 - np.sum(resid ** 2) / np.sum((y - np.mean(y)) ** 2))
    return {
        "a": float(a),
        "b": float(b),
        "sigma_a": float(np.sqrt(var_a)),
        "sigma_b": float(np.sqrt(var_b)),
        "cov_ab": float(cov_ab),
        "chi2": chi2,
        "r2_unweighted": r2,
        "n": int(len(y)),
        "k": 2,
    }


def constant_fit(y, yerr):
    w = 1.0 / np.asarray(yerr, dtype=float) ** 2
    y = np.asarray(y, dtype=float)
    yhat = float(np.sum(w * y) / np.sum(w))
    chi2 = float(np.sum(w * (y - yhat) ** 2))
    sigma = float(1.0 / np.sqrt(np.sum(w)))
    return {"yhat": yhat, "sigma": sigma, "chi2": chi2, "n": int(len(y)), "k": 1}


def aic_bic(chi2, k, n):
    return {"aic": float(chi2 + 2 * k), "bic": float(chi2 + k * np.log(n)),
            "chi2_dof": float(chi2 / (n - k)) if n > k else None,
            "dof": int(n - k)}


def gauss_equiv_one_extra(dchi2):
    """For one extra nested parameter, z = sqrt(Delta chi^2) and the
    chi-square survival probability. z equals |estimate|/sigma when the
    extra parameter is Gaussian."""
    z = float(np.sqrt(max(dchi2, 0.0)))
    p = float(chi2_dist.sf(dchi2, 1))
    return {"z": z, "p_chi2_1dof": p}


def f_de_cpl(z, w0, wa):
    zp1 = 1.0 + z
    return zp1 ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z / zp1)


def fit_curve(model, x, y, yerr, p0, bounds):
    popt, pcov = curve_fit(
        model, x, y, p0=p0, sigma=yerr, absolute_sigma=True,
        bounds=bounds, maxfev=20000,
    )
    resid = (y - model(x, *popt)) / yerr
    chi2 = float(np.sum(resid ** 2))
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, pcov, chi2


def relaxation_H_of_t_yr(t_yr, H_inf, beta):
    return H_inf + beta * xi_of_t_yr(t_yr)


def z_of_t_relaxation(t_yr_grid, H_inf, beta, t0_yr):
    """Redshift on a prescribed H(t), with z(t0)=0, integrating backward."""
    t_grid = np.asarray(t_yr_grid, dtype=float)
    # solve_ivp wants increasing t. Integrate from t0 down by flipping time.
    def ode(t_back, z):
        t = t0_yr - t_back
        H = relaxation_H_of_t_yr(t, H_inf, beta)
        # dz/d t_back = - dz/dt = (1+z) H
        return (1.0 + z) * H_to_per_year(H)

    t_back_max = t0_yr - np.min(t_grid)
    sol = solve_ivp(
        ode, (0.0, t_back_max), [0.0],
        t_eval=t0_yr - t_grid[::-1] if False else None,
        rtol=1e-8, atol=1e-10, dense_output=True, method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    t_back = t0_yr - t_grid
    z = sol.sol(t_back)[0]
    return z


def H_of_z_relaxation(z_query, H_inf, beta, t0_yr):
    """Invert z(t) on a log-spaced time grid. z_query must lie on the grid."""
    t_min = t0_yr / (1.0 + np.max(z_query) + 1.0) ** 2
    t_min = max(t_min, 1.0e5)  # stay far above t_*
    # The relaxation H(t) with a large positive beta is larger in the past,
    # so the lookback to a given z is shorter than in LambdaCDM. A grid down
    # to t_min computed from LambdaCDM-like scaling can undershoot. Extend.
    t_grid = np.geomspace(t_min * 0.05, t0_yr, 4000)
    z_grid = z_of_t_relaxation(t_grid, H_inf, beta, t0_yr)
    # z increases as t decreases. Sort by z for interpolation.
    order = np.argsort(z_grid)
    z_s = z_grid[order]
    H_s = relaxation_H_of_t_yr(t_grid, H_inf, beta)[order]
    # Guard against a non-monotonic glitch at the endpoint.
    keep = np.concatenate([[True], np.diff(z_s) > 0])
    return np.interp(z_query, z_s[keep], H_s[keep])


def load_murphy(path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        star = line.endswith("*")
        parts = line.replace("*", "").split()
        # sample, qso, zem, zabs, transitions, dalpha, err
        zabs = float(parts[3])
        dalpha = float(parts[-2])  # units of 1e-5
        err = float(parts[-1])
        rows.append((zabs, dalpha, err, star))
    return rows


def load_king(path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # seq name zem zabs dalpha err instrument category outlier
        zabs = float(parts[3])
        dalpha = float(parts[4])
        err = float(parts[5])
        inst = parts[6]
        cat = int(parts[7])
        outlier = int(parts[8])
        rows.append((zabs, dalpha, err, inst, cat, outlier))
    return rows


def murphy_sigma(err_1e5, star, budget):
    """err and extra scatter are in units of 1e-5. Return fractional sigma."""
    if budget == "fit":
        extra = 0.0
    elif budget == "fiducial":
        # Murphy et al. 2003 ReadMe: add 2.09e-5 in quadrature to starred
        # high-contrast systems only.
        extra = 2.09 if star else 0.0
    else:
        raise ValueError(budget)
    return 1e-5 * np.sqrt(err_1e5 ** 2 + extra ** 2)


def king_sigma(err_1e5, cat, budget):
    if budget == "fit":
        extra = 0.0
    elif budget == "weighted_mean":
        # King et al. 2012 combined weighted-mean sigma_rand, units of 1e-5.
        extra = {1: 0.0, 2: 1.743, 3: 0.905}[cat]
    else:
        raise ValueError(budget)
    return 1e-5 * np.sqrt(err_1e5 ** 2 + extra ** 2)


def fit_gamma(z, y_frac, sigma_frac, t0):
    """delta = Gamma * (L(t)^{-2} - L0^{-2}). Linear in Gamma."""
    t = np.array([age_yr(float(zz))[0] for zz in z])
    design = L_of_t_yr(t) ** -2 - L_of_t_yr(t0) ** -2
    w = 1.0 / sigma_frac ** 2
    # y = Gamma * design. Through the origin.
    gamma = float(np.sum(w * design * y_frac) / np.sum(w * design ** 2))
    var = float(1.0 / np.sum(w * design ** 2))
    resid = y_frac - gamma * design
    chi2 = float(np.sum(w * resid ** 2))
    chi2_null = float(np.sum(w * y_frac ** 2))
    return {
        "gamma": gamma,
        "sigma": float(np.sqrt(var)),
        "chi2": chi2,
        "chi2_null": chi2_null,
        "delta_chi2": chi2_null - chi2,
        "n": int(len(z)),
        "k": 1,
        **gauss_equiv_one_extra(chi2_null - chi2),
        **{f"ic_{k}": v for k, v in aic_bic(chi2, 1, len(z)).items()},
        "null_aic": aic_bic(chi2_null, 0, len(z))["aic"],
        "null_bic": aic_bic(chi2_null, 0, len(z))["bic"],
    }


def main():
    log = []

    def emit(label, obj):
        log.append((label, obj))
        print(f"\n===== {label} =====")
        print(json.dumps(obj, indent=2, default=float)[:4000])

    t0, t0_err = age_yr(0.0)
    if not (13.0e9 < t0 < 14.5e9):
        raise RuntimeError(f"fiducial age {t0/1e9:.4f} Gyr is not near 13.8 Gyr")
    L0 = float(L_of_t_yr(t0))
    # D0 = kappa * Gamma, kappa in yr^-1.
    kappa = float(-2.0 / (t0 * L0 ** 3))
    clock = {
        "t0_Gyr": t0 / 1e9,
        "t0_age_integral_error_yr": t0_err,
        "L0": L0,
        "kappa_per_year": kappa,
        "note": (
            "D0 = kappa * Gamma. kappa = -2/(t0_yr * L0^3). "
            "Background age uses H0=67.4, Omega_m=0.315."
        ),
    }
    emit("atomic_clock_transfer", clock)

    # Published clock summaries transcribed from the follow-up ledger
    # data/clock_constraints.csv (Filzinger 2023; Rosenband 2008). These are
    # published central values, not fits performed here. The translation to
    # Gamma uses kappa computed above.
    published_clocks = {
        "Filzinger2023": {"D0": 1.8e-19, "sigma": 2.5e-19,
                          "doi": "10.1103/PhysRevLett.130.253001"},
        "Rosenband2008": {"D0": -1.6e-17, "sigma": 2.3e-17,
                          "doi": "10.1126/science.1154622"},
    }
    for name, row in published_clocks.items():
        row["gamma"] = row["D0"] / kappa
        row["sigma_gamma"] = row["sigma"] / abs(kappa)
    emit("published_clocks_as_gamma", published_clocks)

    # ---- four-anchor relaxation ------------------------------------------
    names = [a["name"] for a in ANCHORS]
    t_a = np.array([a["t_yr"] for a in ANCHORS])
    H_a = np.array([a["H"] for a in ANCHORS])
    e_a = np.array([a["err"] for a in ANCHORS])
    xi_a = xi_of_t_yr(t_a)
    fit = wls_line(xi_a, H_a, e_a)
    # In wls_line, y = a + b x, so H_inf = a, beta = b.
    const = constant_fit(H_a, e_a)
    dchi = const["chi2"] - fit["chi2"]
    anchor = {
        "anchors": [
            {**a, "xi": float(xi_of_t_yr(a["t_yr"]))} for a in ANCHORS
        ],
        "H_inf": fit["a"],
        "sigma_H_inf": fit["sigma_a"],
        "beta": fit["b"],
        "sigma_beta": fit["sigma_b"],
        "cov_Hinf_beta": fit["cov_ab"],
        "chi2": fit["chi2"],
        "r2_unweighted": fit["r2_unweighted"],
        "beta_over_ln2": fit["b"] * 1e5,
        "sigma_beta_over_ln2": fit["sigma_b"] * 1e5,
        **aic_bic(fit["chi2"], 2, 4),
        "constant_H0": const["yhat"],
        "constant_sigma": const["sigma"],
        "constant_chi2": const["chi2"],
        **{f"constant_{k}": v for k, v in aic_bic(const["chi2"], 1, 4).items()},
        "delta_chi2_vs_constant": dchi,
        **{f"vs_constant_{k}": v for k, v in gauss_equiv_one_extra(dchi).items()},
    }
    # H at a few assigned times and at redshifts, using the anchor-fitted
    # parameters and the fiducial age map. These are evaluations of the
    # fitted line, not a new fit.
    samples = {}
    for label, t in [("t_recomb_assigned", 3.8e5), ("t_1Gyr", 1e9),
                     ("t_6Gyr", 6e9), ("t_13p8Gyr", 13.8e9),
                     ("t_100Gyr", 1e11), ("t_1Tyr", 1e12)]:
        samples[label] = float(relaxation_H_of_t_yr(t, fit["a"], fit["b"]))
    for zq in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
        tz, _ = age_yr(zq)
        samples[f"H_at_LCDM_age_z_{zq}"] = float(
            relaxation_H_of_t_yr(tz, fit["a"], fit["b"]))
        samples[f"t_Gyr_z_{zq}"] = tz / 1e9
        samples[f"H_LCDM_z_{zq}"] = float(H0_FID * E_lcdm(zq, OM_FID))
    anchor["evaluations"] = samples
    # Present-day value under the Cepheid epoch assignment.
    anchor["H_at_assigned_today"] = samples["t_13p8Gyr"]
    emit("four_anchor", anchor)

    # Leave-one-anchor-out.
    loo = []
    for i, held in enumerate(ANCHORS):
        mask = np.ones(4, dtype=bool)
        mask[i] = False
        sub = wls_line(xi_a[mask], H_a[mask], e_a[mask])
        H_pred = sub["a"] + sub["b"] * xi_a[i]
        # Prediction variance from the parameter covariance.
        # var(a + b x) = var_a + x^2 var_b + 2 x cov
        x = float(xi_a[i])
        var_pred = sub["sigma_a"] ** 2 + (x * sub["sigma_b"]) ** 2 + 2 * x * sub["cov_ab"]
        pull_meas = (H_a[i] - H_pred) / e_a[i]
        pull_pred = (H_a[i] - H_pred) / np.sqrt(e_a[i] ** 2 + var_pred)
        loo.append({
            "held_out": held["name"],
            "H_obs": held["H"],
            "sigma_obs": held["err"],
            "H_pred": float(H_pred),
            "sigma_pred": float(np.sqrt(var_pred)),
            "residual": float(H_a[i] - H_pred),
            "pull_over_measurement_error": float(pull_meas),
            "pull_over_total": float(pull_pred),
            "H_inf": sub["a"],
            "sigma_H_inf": sub["sigma_a"],
            "beta": sub["b"],
            "sigma_beta": sub["sigma_b"],
            "chi2_on_three": sub["chi2"],
            "dof_on_three": 1,
        })
    emit("leave_one_anchor_out", loo)

    # Leverage of xi. Descriptive, not a fit.
    xi_anchor_span = float(xi_a.max() - xi_a.min())
    z_dense = np.linspace(0.0, 2.0, 401)
    t_dense = np.array([age_yr(z)[0] for z in z_dense])
    xi_dense = xi_of_t_yr(t_dense)
    leverage = {
        "delta_xi_anchors": xi_anchor_span,
        "delta_xi_z_0_to_2": float(xi_dense.max() - xi_dense.min()),
        "ratio": float(xi_anchor_span / (xi_dense.max() - xi_dense.min())),
        "xi_z0": float(xi_of_t_yr(age_yr(0.0)[0])),
        "xi_z2": float(xi_of_t_yr(age_yr(2.0)[0])),
    }
    emit("xi_leverage", leverage)

    # ---- LambdaCDM and CPL on the cosmic-chronometer sample --------------
    z_cc, H_cc, e_cc = CC[:, 0], CC[:, 1], CC[:, 2]

    def model_lcdm(z, H0, om):
        return H0 * E_lcdm(z, om)

    def model_cpl(z, H0, om, w0, wa):
        return H0 * np.sqrt(
            om * (1.0 + z) ** 3 + (1.0 - om) * f_de_cpl(z, w0, wa)
        )

    p_l, e_l, cov_l, chi_l = fit_curve(
        model_lcdm, z_cc, H_cc, e_cc, p0=[70.0, 0.3],
        bounds=([40.0, 0.05], [100.0, 0.8]),
    )
    p_c, e_c, cov_c, chi_c = fit_curve(
        model_cpl, z_cc, H_cc, e_cc, p0=[70.0, 0.3, -1.0, 0.0],
        bounds=([40.0, 0.05, -3.0, -8.0], [100.0, 0.8, 1.0, 8.0]),
    )

    # Relaxation refit on the same CC points. H(z) = H_inf + beta * xi(t(z))
    # with t(z) the fiducial LambdaCDM age. The shape is almost flat; this is
    # the same dataset and the same weighting as the two models above.
    xi_cc = xi_of_t_yr(np.array([age_yr(z)[0] for z in z_cc]))
    rel = wls_line(xi_cc, H_cc, e_cc)
    # Self-consistent ODE check at the fiducial-age best fit: compare H(z).
    H_ode = H_of_z_relaxation(z_cc, rel["a"], rel["b"], t0)
    chi_ode = float(np.sum(((H_cc - H_ode) / e_cc) ** 2))

    # Anchor parameters, not refit, scored on the CC points.
    H_anchor_on_cc = relaxation_H_of_t_yr(np.array([age_yr(z)[0] for z in z_cc]),
                                          fit["a"], fit["b"])
    chi_anchor_on_cc = float(np.sum(((H_cc - H_anchor_on_cc) / e_cc) ** 2))

    def pack_model(name, chi2, k, params):
        out = {"model": name, "chi2": chi2, "k": k, "n": 32, "params": params}
        out.update(aic_bic(chi2, k, 32))
        return out

    cc_models = {
        "lcdm": pack_model("flat_LambdaCDM", chi_l, 2, {
            "H0": float(p_l[0]), "sigma_H0": float(e_l[0]),
            "Omega_m": float(p_l[1]), "sigma_Omega_m": float(e_l[1]),
        }),
        "cpl": pack_model("flat_CPL", chi_c, 4, {
            "H0": float(p_c[0]), "sigma_H0": float(e_c[0]),
            "Omega_m": float(p_c[1]), "sigma_Omega_m": float(e_c[1]),
            "w0": float(p_c[2]), "sigma_w0": float(e_c[2]),
            "wa": float(p_c[3]), "sigma_wa": float(e_c[3]),
        }),
        "relaxation_refit": pack_model("relaxation_on_fiducial_age", rel["chi2"], 2, {
            "H_inf": rel["a"], "sigma_H_inf": rel["sigma_a"],
            "beta": rel["b"], "sigma_beta": rel["sigma_b"],
        }),
        "relaxation_ode_at_refit_params_chi2": chi_ode,
        "anchor_parameters_scored_on_cc_chi2": chi_anchor_on_cc,
        "relaxation_refit_H_z0": float(rel["a"] + rel["b"] * xi_of_t_yr(age_yr(0.0)[0])),
        "relaxation_refit_H_z2": float(rel["a"] + rel["b"] * xi_of_t_yr(age_yr(2.0)[0])),
        "delta_chi2_cpl_minus_lcdm": chi_c - chi_l,
        "delta_chi2_relaxation_minus_lcdm": rel["chi2"] - chi_l,
        "cpl_vs_lcdm_p_2dof": float(chi2_dist.sf(max(chi_l - chi_c, 0.0), 2)),
    }
    # Nested CPL vs LambdaCDM is two extra parameters (w0, wa), and only if
    # the LambdaCDM point is inside the CPL family (w0=-1, wa=0).
    cc_models["lcdm_nested_in_cpl_dchi2"] = float(chi_l - chi_c)
    emit("cc_model_comparison", cc_models)

    # ---- quasar exploratory fits ------------------------------------------
    murphy_rows = load_murphy(ROOT / "notebooks/data/Murphy03_table3_128absorbers.dat")
    king_rows = load_king(ROOT / "notebooks/data/King12_VLT_Keck_295absorbers.dat")
    z_m = np.array([r[0] for r in murphy_rows])
    y_m = np.array([r[1] for r in murphy_rows]) * 1e-5
    star = np.array([r[3] for r in murphy_rows])
    err_m = np.array([r[2] for r in murphy_rows])

    quasar = {"n_murphy": len(murphy_rows), "n_starred": int(star.sum()),
              "n_king_file": len(king_rows)}
    for budget in ("fit", "fiducial"):
        sig = np.array([murphy_sigma(er, st, budget) for er, st in zip(err_m, star)])
        quasar[f"murphy_{budget}"] = fit_gamma(z_m, y_m, sig, t0)

    # Diagnostic only: the Einstein-de Sitter age used by the previous
    # abstract and Table 1. Not used in the revised results.
    def fit_gamma_with_age(z, y_frac, sigma_frac, t_yr, t0_yr):
        design = L_of_t_yr(t_yr) ** -2 - L_of_t_yr(t0_yr) ** -2
        w = 1.0 / sigma_frac ** 2
        gamma = float(np.sum(w * design * y_frac) / np.sum(w * design ** 2))
        var = float(1.0 / np.sum(w * design ** 2))
        chi2 = float(np.sum(w * (y_frac - gamma * design) ** 2))
        chi2_null = float(np.sum(w * y_frac ** 2))
        return {"gamma": gamma, "sigma": float(np.sqrt(var)),
                "delta_chi2": chi2_null - chi2, "chi2": chi2, "chi2_null": chi2_null}

    sig_fit = np.array([murphy_sigma(er, st, "fit") for er, st in zip(err_m, star)])
    t_eds = t0 / (1.0 + z_m) ** 1.5
    quasar["murphy_fit_eds_age_diagnostic"] = fit_gamma_with_age(
        z_m, y_m, sig_fit, t_eds, t0)
    quasar["murphy_fit_eds_age_diagnostic"]["note"] = (
        "t(z)=t0/(1+z)^{3/2}. This is the mapping behind the previous "
        "abstract value. It is not the result used in the revision."
    )

    king_kept = [r for r in king_rows if r[5] == 0]
    quasar["n_king_author_retained"] = len(king_kept)
    quasar["n_king_removed_by_author_flag"] = len(king_rows) - len(king_kept)

    def king_subset(rows, budget, selector):
        chosen = [r for r in rows if selector(r)]
        if not chosen:
            return None
        z = np.array([r[0] for r in chosen])
        y = np.array([r[1] for r in chosen]) * 1e-5
        sig = np.array([king_sigma(r[2], r[4], budget) for r in chosen])
        out = fit_gamma(z, y, sig, t0)
        out["n_keck"] = int(sum(r[3] == "Keck" for r in chosen))
        out["n_vlt"] = int(sum(r[3] == "VLT" for r in chosen))
        return out

    for budget in ("fit", "weighted_mean"):
        quasar[f"king293_{budget}"] = king_subset(
            king_kept, budget, lambda r: True)
        quasar[f"king_keck_{budget}"] = king_subset(
            king_kept, budget, lambda r: r[3] == "Keck")
        quasar[f"king_vlt_{budget}"] = king_subset(
            king_kept, budget, lambda r: r[3] == "VLT")

    # Overlap of Murphy absorption redshifts with King Keck rows.
    king_keck_z = np.array([r[0] for r in king_rows if r[3] == "Keck"])
    n_match = 0
    for z in z_m:
        if np.min(np.abs(king_keck_z - z)) < 5e-6:
            n_match += 1
    quasar["murphy_rows_with_king_keck_z_match"] = n_match

    # Clock-only versus Murphy-only, same Gamma. Not a joint spectral likelihood.
    g_m = quasar["murphy_fit"]["gamma"]
    s_m = quasar["murphy_fit"]["sigma"]
    g_c = published_clocks["Filzinger2023"]["gamma"]
    s_c = published_clocks["Filzinger2023"]["sigma_gamma"]
    # Inverse-variance joint under the assumption of one shared Gamma.
    w_m, w_c = 1.0 / s_m ** 2, 1.0 / s_c ** 2
    g_j = (w_m * g_m + w_c * g_c) / (w_m + w_c)
    s_j = (w_m + w_c) ** -0.5
    # Tension between the two Gamma estimates.
    tension = (g_m - g_c) / np.sqrt(s_m ** 2 + s_c ** 2)
    quasar["shared_gamma_assumption"] = {
        "murphy_fit_errors_gamma": g_m,
        "filzinger_gamma": g_c,
        "inverse_variance_joint_gamma": float(g_j),
        "inverse_variance_joint_sigma": float(s_j),
        "tension_murphy_vs_filzinger_sigma": float(tension),
        "D0_implied_by_murphy_fit_errors": float(kappa * g_m),
    }
    emit("quasar_exploratory", quasar)

    # ---- figures ----------------------------------------------------------
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    xg = np.linspace(xi_a.min() - 0.05, xi_a.max() + 0.05, 200)
    ax.plot(xg, fit["a"] + fit["b"] * xg, color="#d62728", lw=1.7,
            label=r"$H=H_\infty+\beta\,\xi$")
    ax.axhline(const["yhat"], color="0.4", ls="--", lw=1.0,
               label=fr"constant ${const['yhat']:.2f}$")
    markers = ["s", "D", "^", "o"]
    colors = ["#1f77b4", "#9467bd", "#ff7f0e", "#2ca02c"]
    for a, m, c in zip(ANCHORS, markers, colors):
        ax.errorbar(xi_of_t_yr(a["t_yr"]), a["H"], yerr=a["err"], fmt=m,
                    color=c, ms=5.5, capsize=2.5, label=a["name"])
    ax.set_xlabel(r"$\xi = 10^{5}/\ln^{2}(t/t_{*})$")
    ax.set_ylabel(r"reported $H$ (km s$^{-1}$ Mpc$^{-1}$)")
    ax.legend(framealpha=0.92, loc="upper right")
    ax.grid(alpha=0.25, ls=":")
    fig.savefig(FIG / "rev_fig_anchors.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    ypos = np.arange(len(loo))
    resid = [row["residual"] for row in loo]
    pred_err = [row["sigma_pred"] for row in loo]
    meas_err = [row["sigma_obs"] for row in loo]
    ax.axvline(0.0, color="0.5", lw=0.8)
    ax.errorbar(resid, ypos, xerr=meas_err, fmt="o", color="#1f77b4",
                ms=5, capsize=2, label="measurement error")
    ax.errorbar(resid, ypos + 0.12, xerr=pred_err, fmt="none",
                ecolor="#d62728", capsize=2, label="prediction error")
    ax.set_yticks(ypos)
    ax.set_yticklabels([row["held_out"] for row in loo])
    ax.set_xlabel(r"held-out residual $H_{\mathrm{obs}}-H_{\mathrm{pred}}$")
    ax.legend(framealpha=0.92, loc="best")
    ax.grid(alpha=0.25, ls=":", axis="x")
    fig.savefig(FIG / "rev_fig_loo.pdf")
    plt.close(fig)

    z_plot = np.linspace(0.0, 2.0, 300)
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    ax.errorbar(z_cc, H_cc, yerr=e_cc, fmt="o", ms=3.2, color="0.35",
                ecolor="0.7", elinewidth=0.7, label="cosmic chronometers", zorder=3)
    ax.plot(z_plot, model_lcdm(z_plot, *p_l), color="#1f77b4", lw=1.6,
            label=r"flat $\Lambda$CDM")
    ax.plot(z_plot, model_cpl(z_plot, *p_c), color="#2ca02c", lw=1.6,
            label="flat CPL")
    xi_plot = xi_of_t_yr(np.array([age_yr(z)[0] for z in z_plot]))
    ax.plot(z_plot, rel["a"] + rel["b"] * xi_plot, color="#d62728", lw=1.6,
            label=r"$1/\ln^{2}$ refit")
    ax.plot(z_plot, fit["a"] + fit["b"] * xi_plot, color="#d62728", lw=1.1,
            ls="--", label=r"anchor parameters, not refit")
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"$H(z)$ (km s$^{-1}$ Mpc$^{-1}$)")
    ax.legend(framealpha=0.92, fontsize=6.6, loc="upper left")
    ax.grid(alpha=0.25, ls=":")
    fig.savefig(FIG / "rev_fig_cc_models.pdf")
    plt.close(fig)

    # Exploratory Gamma. Clock band is far narrower than the quasar errors;
    # draw the quasar points and mark the clock interval in the caption via
    # a second axis annotation rather than squashing the scale.
    labels, gammas, sigmas = [], [], []
    for key, lab in [
        ("murphy_fit", "Murphy, fit errors"),
        ("murphy_fiducial", "Murphy, fiducial"),
        ("king_keck_weighted_mean", "King Keck"),
        ("king_vlt_weighted_mean", "King VLT"),
        ("king293_weighted_mean", "King retained"),
    ]:
        labels.append(lab)
        gammas.append(quasar[key]["gamma"])
        sigmas.append(quasar[key]["sigma"])
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    ypos = np.arange(len(labels))
    ax.axvline(0.0, color="0.4", lw=0.9)
    ax.errorbar(gammas, ypos, xerr=sigmas, fmt="o", color="#1f77b4",
                ms=5, capsize=2.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"exploratory $\Gamma$")
    g_clock = published_clocks["Filzinger2023"]["gamma"]
    s_clock = published_clocks["Filzinger2023"]["sigma_gamma"]
    ax.set_title(
        fr"Filzinger $\Gamma={g_clock:.3f}\pm{s_clock:.3f}$ (off scale)",
        fontsize=8)
    ax.grid(alpha=0.25, ls=":", axis="x")
    fig.savefig(FIG / "rev_fig_gamma.pdf")
    plt.close(fig)

    results = {
        "constants": {
            "t_star_s": T_STAR,
            "sec_per_year": SEC_PER_YEAR,
            "km_per_mpc": KM_PER_MPC,
            "H0_fid": H0_FID,
            "Omega_m_fid": OM_FID,
        },
        "atomic_clock_transfer": clock,
        "published_clocks_as_gamma": published_clocks,
        "four_anchor": anchor,
        "leave_one_anchor_out": loo,
        "xi_leverage": leverage,
        "cc_model_comparison": cc_models,
        "quasar_exploratory": quasar,
    }
    (OUT / "revision_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT / 'revision_results.json'}")


if __name__ == "__main__":
    main()
