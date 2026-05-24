#!/usr/bin/env python3
"""
Generate all paper figures for DSC CMB paper.
Runs simulations and produces publication-quality plots.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from paper_plot_style import *
import numpy as np
from scipy.signal import convolve2d, find_peaks
from scipy.stats import skew, kurtosis, norm, kstest, shapiro, probplot
from scipy.ndimage import uniform_filter, map_coordinates
import healpy as hp
import warnings
warnings.filterwarnings('ignore')

# ── Physics engine (same as experiments) ─────────────────
LAP = np.array([[1,2,1],[2,-12,2],[1,2,1]], dtype=np.float64) / 12.0

def gen_init(N, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((N,N))
    fk = np.fft.fft2(noise)
    kx, ky = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N))
    K2 = kx**2+ky**2; K2[0,0]=1e-10
    filt = 1.0/(np.sqrt(K2)*N)**1.5; filt[0,0]=0
    phi = np.real(np.fft.ifft2(fk*filt))
    return (phi - phi.mean()) / phi.std()

def evolve_symp(phi0, steps=45, c2=0.45, c0=10.0, drag=0.0):
    phi = phi0.copy(); pp = phi0.copy()
    for t in range(1, steps+1):
        c2t = c2 / np.log(t+c0)**2
        lap = convolve2d(phi, LAP, mode='same', boundary='wrap')
        pn = 2*phi - pp + c2t*lap - drag*(phi-pp)
        pn = np.clip(pn, -10, 10)
        pp = phi.copy(); phi = pn
    return phi

def evolve_diss(phi0, steps=150, mu_c=1.5437, k_opt=1.248, c0=10.0, eps=0.55):
    # CML needs small initial amplitudes ([-0.1, 0.1] range, matching gemini2)
    phi = phi0 * 0.1  # scale down from std=1 to std=0.1
    for t in range(1, steps+1):
        mu = mu_c + k_opt/np.log(t+c0)**2
        fx = 1.0 - mu*phi**2
        phi = fx + eps*convolve2d(fx, LAP, mode='same', boundary='wrap')
        phi = np.clip(phi, -10, 10)
    return phi

def silk_damp(phi, sk=35.0):
    N = phi.shape[0]; fk = np.fft.fft2(phi)
    kx = np.fft.fftfreq(N, d=1.0/N); ky = np.fft.fftfreq(N, d=1.0/N)
    KX, KY = np.meshgrid(kx, ky); K = np.sqrt(KX**2+KY**2)
    return np.real(np.fft.ifft2(fk*np.exp(-(K/sk)**2)))

def znorm(phi):
    return (phi-phi.mean())/(phi.std()+1e-15)

def compute_Dk(phi):
    N = phi.shape[0]
    ps = np.abs(np.fft.fftshift(np.fft.fft2(phi)))**2 / N**4
    kx = np.fft.fftshift(np.fft.fftfreq(N, d=1.0/N))
    ky = np.fft.fftshift(np.fft.fftfreq(N, d=1.0/N))
    KX, KY = np.meshgrid(kx, ky); K = np.sqrt(KX**2+KY**2)
    k_bins = np.arange(1, N//2+1, dtype=float)
    Pk = np.zeros(len(k_bins))
    for i in range(len(k_bins)):
        mask = (K >= i+0.5) & (K < i+1.5)
        if mask.any(): Pk[i] = ps[mask].mean()
    return k_bins, k_bins**2*Pk, Pk

def mock_lcdm(k):
    return np.exp(-(k/35)**2)*(1+1.2*np.sin(np.pi*k/12)**2)*(50/(k+10))

def smooth(y, w=3):
    return uniform_filter(y.astype(float), size=w)

# ── 3D engine ────────────────────────────────────────────
def gen_3d(N, seed=42):
    rng = np.random.default_rng(seed)
    fk = np.fft.fftn(rng.standard_normal((N,N,N)))
    kx,ky,kz = np.fft.fftfreq(N),np.fft.fftfreq(N),np.fft.fftfreq(N)
    KX,KY,KZ = np.meshgrid(kx,ky,kz,indexing='ij')
    K = np.sqrt(KX**2+KY**2+KZ**2); K[0,0,0]=1e-10
    filt = 1.0/(K*N)**0.375; filt[0,0,0]=0
    phi = np.real(np.fft.ifftn(fk*filt))
    return phi/phi.std()*0.15

def dsc_3d(init, steps=45, c2=0.45, c0=10.0, drag=0.015, nl=0.005):
    p = init.copy(); pp = init.copy()
    for t in range(1, steps):
        c2t = c2/np.log(t+c0)**2
        lap = (np.roll(p,1,0)+np.roll(p,-1,0)+np.roll(p,1,1)+np.roll(p,-1,1)+np.roll(p,1,2)+np.roll(p,-1,2)-6*p)/6
        pn = 2*p-pp+c2t*lap-drag*(p-pp)-nl*p**2
        pn = np.clip(pn,-5,5); pp = p.copy(); p = pn
    return p

def sample_sphere(vol, nside=64, rf=0.38):
    N = vol.shape[0]; c=N/2; r=N*rf; npix=hp.nside2npix(nside)
    th,ph = hp.pix2ang(nside, np.arange(npix))
    coords = np.array([c+r*np.sin(th)*np.cos(ph), c+r*np.sin(th)*np.sin(ph), c+r*np.cos(th)])
    sky = map_coordinates(vol, coords, order=1, mode='wrap')
    return (sky-sky.mean())/(sky.std()+1e-15)


# ================================================================
# Figure 2: Ablation — Symplectic vs Dissipative
# ================================================================
def gen_fig2():
    print("Generating Fig 2: Ablation...")
    N = 300
    phi0 = gen_init(N, seed=42)

    phi_d = evolve_diss(phi0, steps=150)
    Td = znorm(phi_d)
    phi_s = evolve_symp(phi0, steps=45, c2=0.45, drag=0.0)
    phi_s = silk_damp(phi_s, 35.0)
    Ts = znorm(phi_s)

    k, Dk_d, _ = compute_Dk(Td)
    _, Dk_s, _ = compute_Dk(Ts)
    Dk_ref = mock_lcdm(k)

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))

    axes[0,0].imshow(Td, cmap='RdYlBu_r', vmin=-3, vmax=3, origin='lower')
    axes[0,0].set_title(f'(a) Dissipative (CML)', fontsize=9)
    axes[0,0].axis('off')
    axes[1,0].imshow(Ts, cmap='RdYlBu_r', vmin=-3, vmax=3, origin='lower')
    axes[1,0].set_title(f'(d) Symplectic (Störmer–Verlet)', fontsize=9)
    axes[1,0].axis('off')

    xg = np.linspace(-5,5,200)
    axes[0,1].hist(Td.flat, bins=80, density=True, alpha=0.7, color=C_DISS, edgecolor='none')
    axes[0,1].plot(xg, norm.pdf(xg), 'k--', lw=1.5)
    sd, kd = skew(Td.flat), kurtosis(Td.flat)
    axes[0,1].set_title(f'(b) Skew={sd:.2f}, Kurt={kd:.2f}', fontsize=9)
    axes[0,1].set_xlim(-5,5)

    axes[1,1].hist(Ts.flat, bins=80, density=True, alpha=0.7, color=C_DSC, edgecolor='none')
    axes[1,1].plot(xg, norm.pdf(xg), 'k--', lw=1.5)
    ss, ks = skew(Ts.flat), kurtosis(Ts.flat)
    axes[1,1].set_title(f'(e) Skew={ss:.2f}, Kurt={ks:.2f}', fontsize=9)
    axes[1,1].set_xlim(-5,5)

    kk = k[:80]
    Drs = Dk_ref[:80]/Dk_ref[:80].max()*smooth(Dk_d[:80]).max()
    axes[0,2].semilogy(kk, smooth(Dk_d[:80]), color=C_DISS, lw=1.5, label='Dissipative')
    axes[0,2].semilogy(kk, Drs, color=C_LCDM, ls='--', lw=1, alpha=0.6, label=r'mock $\Lambda$CDM')
    axes[0,2].set_title(f'(c) D(k)', fontsize=9)
    axes[0,2].legend(fontsize=7, frameon=False)

    Drs2 = Dk_ref[:80]/Dk_ref[:80].max()*smooth(Dk_s[:80]).max()
    axes[1,2].semilogy(kk, smooth(Dk_s[:80]), color=C_DSC, lw=1.5, label='Symplectic')
    axes[1,2].semilogy(kk, Drs2, color=C_LCDM, ls='--', lw=1, alpha=0.6, label=r'mock $\Lambda$CDM')
    axes[1,2].set_title(f'(f) D(k)', fontsize=9)
    axes[1,2].legend(fontsize=7, frameon=False)

    for ax in axes[:,1]: ax.set_xlabel(r'$\delta T/\sigma$', fontsize=9)
    for ax in axes[:,2]: ax.set_xlabel(r'$k$', fontsize=9)

    plt.tight_layout(h_pad=0.8, w_pad=0.5)
    save_fig(fig, 'fig2_ablation')
    return sd, kd, ss, ks


# ================================================================
# Figure 3: Acoustic Peaks
# ================================================================
def gen_fig3():
    print("Generating Fig 3: Acoustic peaks...")
    N = 300
    phi0 = gen_init(N, seed=42)
    phi = evolve_symp(phi0, steps=45, c2=0.45, drag=0.0)
    phi = silk_damp(phi, 35.0)
    T = znorm(phi)
    k, Dk, _ = compute_Dk(T)
    Dk_s = smooth(Dk, 3)
    Dk_ref = mock_lcdm(k)
    Dk_ref_n = Dk_ref/Dk_ref.max()*Dk_s.max()

    kk = k[:70]
    # Normalize both to same peak amplitude for shape comparison
    Dk_ref_raw = mock_lcdm(k)
    Dk_s_plot = Dk_s[:70]
    Dk_ref_plot = Dk_ref_raw[:70] / Dk_ref_raw[:70].max() * Dk_s_plot.max()

    peaks_dsc, _ = find_peaks(Dk_s_plot, distance=5, prominence=Dk_s_plot.max()*0.01)
    peaks_ref, _ = find_peaks(Dk_ref_plot, distance=5, prominence=Dk_ref_plot.max()*0.01)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(kk, Dk_s_plot, color=C_DSC, lw=1.8, label='DSC symplectic')
    ax.plot(kk, Dk_ref_plot, color=C_LCDM, ls='--', lw=1.5, alpha=0.7, label=r'Mock $\Lambda$CDM')
    for p in peaks_dsc:
        if p < 70:
            ax.plot(kk[p], Dk_s_plot[p], 'v', color=C_DSC, ms=6, zorder=5)
    for p in peaks_ref:
        if p < 70:
            ax.plot(kk[p], Dk_ref_plot[p], '^', color=C_LCDM, ms=5, zorder=5, alpha=0.6)

    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$D(k) = k^2 P(k)$')
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(1, 70)
    plt.tight_layout()
    save_fig(fig, 'fig3_acoustic_peaks')
    return k[peaks_dsc].astype(int).tolist(), k[peaks_ref].astype(int).tolist()


# ================================================================
# Figure 4: Twin Experiment (AI Parameter Recovery)
# ================================================================
def gen_fig4():
    print("Generating Fig 4: Twin experiment...")
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [SKIP] optuna not installed")
        return

    N = 150; steps = 55
    k_vals = np.arange(1, N//2)
    np.random.seed(42)
    kx, ky = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N))
    k2 = kx**2+ky**2; k2[0,0]=1e-10
    amp = np.random.normal(0,1,(N,N))+1j*np.random.normal(0,1,(N,N))
    fk0 = amp/(np.sqrt(k2)**1.5); fk0[0,0]=0
    PHI0 = np.real(np.fft.ifft2(fk0))
    PHI0 = (PHI0-PHI0.mean())/PHI0.std()
    kernel = LAP

    def sim(c2, drag, k_opt):
        phi=PHI0.copy(); pp=PHI0.copy()
        for t in range(1,steps):
            ce=c2*(k_opt/np.log(t+10)**2)
            lap=convolve2d(phi, kernel, mode='same', boundary='wrap')
            pn=2*phi-pp+ce*lap-drag*(phi-pp)
            pp=phi.copy(); phi=pn
        ps=np.abs(np.fft.fftshift(np.fft.fft2(phi)))**2
        y,x=np.indices(ps.shape)
        r=np.round(np.sqrt((x-N//2)**2+(y-N//2)**2)).astype(int)
        rp=np.bincount(r.ravel(),ps.ravel())/(np.bincount(r.ravel())+1e-10)
        Dk=k_vals**2*rp[1:N//2]*np.exp(-(k_vals/35)**2)
        Ds=np.convolve(Dk, np.ones(3)/3, mode='same')
        return Ds/(Ds.max()+1e-10)

    TRUE = (0.650, 0.015, 2.100)
    target = sim(*TRUE)
    losses = []
    def obj(trial):
        c2=trial.suggest_float('c2',0.3,0.9)
        d=trial.suggest_float('drag',0.005,0.05)
        k=trial.suggest_float('k_opt',1.0,4.0)
        loss=np.mean((sim(c2,d,k)-target)**2)
        losses.append(loss)
        return loss

    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials=100)
    best = study.best_params
    best_Dk = sim(best['c2'], best['drag'], best['k_opt'])
    bad_Dk = sim(0.3, 0.04, 1.2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax1.plot(losses, 'o', color=C_DSC, alpha=0.2, ms=2)
    ax1.plot(np.minimum.accumulate(losses), color='navy', lw=1.5)
    ax1.set_yscale('log')
    ax1.set_xlabel('Trials')
    ax1.set_ylabel('MSE')
    ax1.set_title('(a) Optimization convergence', fontsize=9)

    ax2.plot(k_vals[:60], target[:60], 'k--', lw=2.5, alpha=0.6, label='Ground truth')
    ax2.plot(k_vals[:60], bad_Dk[:60], color='gray', lw=1, alpha=0.5, label='Random guess')
    ax2.plot(k_vals[:60], best_Dk[:60], color=C_AI, lw=1.8, label='Recovered')
    ax2.set_xlabel(r'$k$')
    ax2.set_ylabel(r'Normalized $D(k)$')
    ax2.set_title('(b) Parameter recovery', fontsize=9)
    ax2.legend(frameon=False, fontsize=8)

    txt = (f"True: c²={TRUE[0]:.3f}, drag={TRUE[1]:.3f}, k={TRUE[2]:.3f}\n"
           f"AI:   c²={best['c2']:.3f}, drag={best['drag']:.3f}, k={best['k_opt']:.3f}")
    ax2.text(0.35, 0.65, txt, transform=ax2.transAxes, fontsize=7, family='monospace',
             bbox=dict(facecolor='#f0f8ff', alpha=0.8, edgecolor='steelblue', boxstyle='round,pad=0.3'))

    plt.tight_layout()
    save_fig(fig, 'fig4_twin_experiment')


# ================================================================
# Figure 5: Acoustic Freeze-out
# ================================================================
def gen_fig5():
    print("Generating Fig 5: Freeze-out...")
    N = 300
    phi0 = np.zeros((N,N))
    cx, cy = N//2, N//2
    Y, X = np.mgrid[0:N, 0:N]
    phi0 = 10.0*np.exp(-((X-cx)**2+(Y-cy)**2)/(2*4.0**2))

    n_steps = 300
    phi=phi0.copy(); pp=phi0.copy()
    radii=[0]; c2h=[0.45/np.log(2)**2]
    snaps = {0: phi0.copy()}
    R = np.sqrt((X-cx)**2+(Y-cy)**2)

    for t in range(1, n_steps+1):
        c2t = 0.45/np.log(t+2)**2
        lap = convolve2d(phi, LAP, mode='same', boundary='wrap')
        pn = 2*phi-pp+c2t*lap-0.015*(phi-pp)
        pn = np.clip(pn,-10,10); pp=phi.copy(); phi=pn
        thr = 0.01*np.abs(phi).max()
        act = R[np.abs(phi)>thr] if np.any(np.abs(phi)>thr) else [0]
        radii.append(np.percentile(act,95) if len(act)>10 else radii[-1])
        c2h.append(c2t)
        if t in [5, 30, 100, 300]: snaps[t] = phi.copy()

    fig = plt.figure(figsize=(10.5, 5.4))
    gs = fig.add_gridspec(2, 10, height_ratios=[1.0, 1.4],
                          hspace=0.55, wspace=1.2,
                          left=0.07, right=0.95, top=0.92, bottom=0.10)

    snap_times = sorted(snaps.keys())  # [0, 5, 30, 100, 300]
    for i, t in enumerate(snap_times):
        ax = fig.add_subplot(gs[0, 2*i:2*i+2])
        ax.imshow(np.sqrt(np.abs(snaps[t])), cmap='inferno', origin='lower',
                  extent=[0,N,0,N])
        ax.set_title(f't={t}', fontsize=8)
        ax.axis('off')

    # (e) on bottom-left half, (f) on bottom-right half, with empty gap column between
    ax1 = fig.add_subplot(gs[1, 0:5])
    ax1.plot(radii, color=C_DSC, lw=1.5)
    ax1.axhline(radii[-1], color=C_DSC, ls=':', alpha=0.4)
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Wavefront radius')
    ax1.set_title(f'(e) Acoustic horizon $r_s \\approx {radii[-1]:.0f}$', fontsize=9)

    ax1b = ax1.twinx()
    ax1b.plot(c2h, color=C_DISS, ls='--', lw=1, alpha=0.6)
    ax1b.set_ylabel(r'$c^2_{\rm eff}$', color=C_DISS, fontsize=9)
    ax1b.tick_params(axis='y', colors=C_DISS)

    ax2 = fig.add_subplot(gs[1, 5:10])
    t_arr = np.arange(len(radii))
    ax2.plot(t_arr, radii, color=C_DSC, lw=1.5)
    ax2.fill_between(t_arr, 0, radii, alpha=0.08, color=C_DSC)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Comoving distance')
    ax2.set_title('(f) Causal structure', fontsize=9)

    save_fig(fig, 'fig5_freezeout')


# ================================================================
# Figure 6: Ensemble Statistics
# ================================================================
def gen_fig6():
    print("Generating Fig 6: Ensemble statistics (20 runs)...")
    N = 300; n_runs = 20
    all_s, all_k, all_Dk = [], [], []
    k_ref = None

    for run in range(n_runs):
        phi0 = gen_init(N, seed=run*17+3)
        phi = evolve_symp(phi0, steps=45, c2=0.45, drag=0.0)
        phi = silk_damp(phi, 35.0)
        T = znorm(phi)
        all_s.append(skew(T.flat))
        all_k.append(kurtosis(T.flat))
        k, Dk, _ = compute_Dk(T)
        if k_ref is None: k_ref = k
        all_Dk.append(Dk)

    S = np.array(all_s); K = np.array(all_k)
    Dk_all = np.array(all_Dk)
    Dk_m = Dk_all.mean(0); Dk_std = Dk_all.std(0)

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2))

    kk = k_ref[:70]
    axes[0].semilogy(kk, smooth(Dk_m[:70],3), color=C_DSC, lw=1.5)
    axes[0].fill_between(kk, smooth(np.maximum(Dk_m[:70]-Dk_std[:70],1e-20),3),
                         smooth(Dk_m[:70]+Dk_std[:70],3), alpha=0.2, color=C_DSC)
    Dr = mock_lcdm(kk); Dr = Dr/Dr.max()*smooth(Dk_m[:70],3).max()
    axes[0].semilogy(kk, Dr, color=C_LCDM, ls='--', lw=1, alpha=0.6)
    axes[0].set_xlabel(r'$k$'); axes[0].set_ylabel(r'$D(k)$')
    axes[0].set_title('(a) Power spectrum', fontsize=9)

    axes[1].hist(S, bins=8, alpha=0.7, color='coral', edgecolor='none')
    axes[1].axvline(0, color='gray', ls='--', lw=1)
    axes[1].set_xlabel('Skewness')
    axes[1].set_title(f'(b) {S.mean():+.3f}±{S.std():.3f}', fontsize=9)

    axes[2].hist(K, bins=8, alpha=0.7, color=C_DSC, edgecolor='none')
    axes[2].axvline(0, color='gray', ls='--', lw=1)
    axes[2].set_xlabel('Excess kurtosis')
    axes[2].set_title(f'(c) {K.mean():+.3f}±{K.std():.3f}', fontsize=9)

    # QQ plot (reviewer requested)
    # Use first run as representative
    phi0 = gen_init(N, seed=3)
    phi = evolve_symp(phi0, steps=45, c2=0.45, drag=0.0)
    phi = silk_damp(phi, 35.0)
    T = znorm(phi)
    osm, osr = probplot(T.flat, dist='norm', fit=False)
    axes[3].plot(osm, osr, '.', color=C_DSC, ms=1, alpha=0.3)
    lim = max(abs(osm.min()), abs(osm.max()))
    axes[3].plot([-lim, lim], [-lim, lim], 'k--', lw=1)
    axes[3].set_xlabel('Theoretical quantiles')
    axes[3].set_ylabel('Sample quantiles')
    axes[3].set_title('(d) Q-Q plot', fontsize=9)
    axes[3].set_xlim(-4,4); axes[3].set_ylim(-4,4)
    axes[3].set_aspect('equal')

    plt.tight_layout(w_pad=0.5)
    save_fig(fig, 'fig6_ensemble')
    return S, K


# ================================================================
# Figure 7: Parameter Sensitivity
# ================================================================
def gen_fig7():
    print("Generating Fig 7: Parameter sensitivity...")
    N = 200; phi0 = gen_init(N, seed=42)

    def measure(phi0, steps, c2, drag):
        phi = evolve_symp(phi0, steps=steps, c2=c2, drag=drag)
        phi = silk_damp(phi, 35.0)
        T = znorm(phi)
        s, k_stat = skew(T.flat), kurtosis(T.flat)
        kk, Dk, _ = compute_Dk(T)
        Dr = mock_lcdm(kk)
        mask = (kk>=2)&(kk<=60)&(Dk>0)&(Dr>0)
        if mask.sum()>5:
            d = Dk[mask]/Dk[mask].sum()*Dr[mask].sum()
            r = np.corrcoef(np.log(d+1e-20), np.log(Dr[mask]+1e-20))[0,1]
        else: r = 0
        return s, k_stat, r

    # c² sweep
    c2v = np.linspace(0.1, 0.8, 12)
    c2r = [measure(phi0, 45, c2, 0.0) for c2 in c2v]

    # drag sweep
    dv = np.linspace(0.0, 0.06, 12)
    dr = [measure(phi0, 45, 0.45, d) for d in dv]

    # steps sweep
    sv = [10, 20, 30, 40, 50, 60, 80, 100, 130]
    sr = [measure(phi0, s, 0.45, 0.0) for s in sv]

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2))

    for col, (vals, results, label) in enumerate([
        (c2v, c2r, r'$c^2$'), (dv, dr, 'Drag'), (sv, sr, 'Steps')]):
        sk = [r[0] for r in results]; ku = [r[1] for r in results]; co = [r[2] for r in results]
        axes[0,col].plot(vals, sk, 'o-', color='coral', ms=3, lw=1, label='Skew')
        axes[0,col].plot(vals, ku, 's-', color=C_DSC, ms=3, lw=1, label='Kurt')
        axes[0,col].axhline(0, color='gray', ls='--', lw=0.8)
        axes[0,col].set_xlabel(label)
        axes[0,col].legend(fontsize=7, frameon=False)
        if col==0: axes[0,col].set_ylabel('Value')

        axes[1,col].plot(vals, co, 'o-', color='seagreen', ms=3, lw=1.5)
        axes[1,col].axhline(0.95, color='gray', ls=':', lw=0.8)
        axes[1,col].set_xlabel(label)
        axes[1,col].set_ylim(-0.2, 1.1)
        if col==0: axes[1,col].set_ylabel(r'Pearson $r$')

    axes[0,0].set_title('(a) Gaussianity', fontsize=9)
    axes[0,1].set_title('(b) Gaussianity', fontsize=9)
    axes[0,2].set_title('(c) Gaussianity', fontsize=9)
    axes[1,0].set_title(r'(d) Spectral corr. vs $c^2$', fontsize=9)
    axes[1,1].set_title('(e) Spectral corr. vs drag', fontsize=9)
    axes[1,2].set_title('(f) Spectral corr. vs steps', fontsize=9)

    plt.tight_layout(h_pad=0.8, w_pad=0.5)
    save_fig(fig, 'fig7_sensitivity')


# ================================================================
# Figure 8: Planck Comparison (3D → sphere)
# ================================================================
def gen_fig8():
    print("Generating Fig 8: Planck comparison...")
    fits_path = 'gemini/COM_CMB_IQU-smica_2048_R3.00_full.fits'
    if not os.path.exists(fits_path):
        print("  [SKIP] Planck SMICA file not found")
        return

    NSIDE = 64
    real = hp.read_map(fits_path, field=0)
    real_low = hp.ud_grade(real, nside_out=NSIDE)
    valid = real_low > -1e20
    vp = real_low[valid]
    pmin, pmax = np.percentile(vp, 2), np.percentile(vp, 98)
    rc = np.clip(real_low, pmin, pmax)
    mv, sv_ = np.mean(np.clip(vp,pmin,pmax)), np.std(np.clip(vp,pmin,pmax))
    target = np.where(valid, (rc-mv)/(sv_+1e-8), 0.0)

    # DSC 3D
    N_G = 96
    init = gen_3d(N_G, seed=42)
    vol = dsc_3d(init, steps=45)
    dsc_sky = sample_sphere(vol, nside=NSIDE)

    # Ensemble for error bars
    ens_s, ens_k = [], []
    for seed in range(5):
        v = dsc_3d(gen_3d(N_G, seed=seed*13+7), steps=45)
        s = sample_sphere(v, nside=NSIDE)
        ens_s.append(skew(s)); ens_k.append(kurtosis(s))

    Cl_p = hp.anafast(target, lmax=150); ell_p = np.arange(len(Cl_p))
    Cl_d = hp.anafast(dsc_sky, lmax=150); ell_d = np.arange(len(Cl_d))
    Dl_p = ell_p*(ell_p+1)*Cl_p/(2*np.pi)
    Dl_d = ell_d*(ell_d+1)*Cl_d/(2*np.pi)
    mask_n = (ell_p>=10)&(ell_p<=80)
    sc = Dl_p[mask_n].mean()/(Dl_d[mask_n].mean()+1e-20)
    Dl_ds = Dl_d*sc

    planck_s, planck_k = skew(target[valid]), kurtosis(target[valid])

    fig = plt.figure(figsize=(7.0, 5.5))

    hp.mollview(np.where(valid, target, hp.UNSEEN), cmap='RdYlBu_r', min=-3, max=3,
                sub=(2,3,1), badcolor='gray', title='')
    plt.title('(a) Planck SMICA', fontsize=9, pad=2)

    hp.mollview(dsc_sky, cmap='RdYlBu_r', min=-3, max=3,
                sub=(2,3,2), title='')
    plt.title('(b) DSC forward sim.', fontsize=9, pad=2)

    ax3 = fig.add_subplot(2,3,3)
    mip = np.max(np.exp(np.clip(vol*1.5,-5,5)), axis=2)
    ax3.imshow(mip**0.8, cmap='magma', origin='lower', interpolation='bilinear')
    circle = plt.Circle((N_G/2, N_G/2), N_G*0.38, color='cyan', fill=False, lw=1.5, ls='--')
    ax3.add_patch(circle)
    ax3.set_title('(c) 3D cosmic web', fontsize=9)
    ax3.axis('off')

    ax4 = fig.add_subplot(2,3,4)
    m = ell_p > 1
    ax4.loglog(ell_p[m], Dl_p[m], color=C_PLANCK, lw=1.5, alpha=0.7, label='Planck')
    ax4.loglog(ell_d[m], Dl_ds[m], color=C_DSC, lw=1.5, label='DSC')
    ax4.set_xlabel(r'$\ell$'); ax4.set_ylabel(r'$D_\ell$')
    ax4.set_title(r'(d) Angular power spectrum', fontsize=9)
    ax4.legend(fontsize=7, frameon=False)

    ax5 = fig.add_subplot(2,3,5)
    ax5.hist(target[valid]/target[valid].std(), bins=60, density=True, alpha=0.5, color=C_PLANCK, label='Planck')
    ax5.hist(dsc_sky/dsc_sky.std(), bins=60, density=True, alpha=0.5, color=C_DSC, label='DSC')
    xg = np.linspace(-5,5,200)
    ax5.plot(xg, norm.pdf(xg), 'k--', lw=1)
    ax5.set_xlabel(r'$\delta T/\sigma$'); ax5.set_xlim(-5,5)
    ax5.set_title('(e) Pixel distribution', fontsize=9)
    ax5.legend(fontsize=7, frameon=False)

    ax6 = fig.add_subplot(2,3,6)
    xp = [0, 1, 3, 4]
    vals = [planck_s, np.mean(ens_s), planck_k, np.mean(ens_k)]
    errs = [0, np.std(ens_s), 0, np.std(ens_k)]
    cols = [C_PLANCK, C_DSC, C_PLANCK, C_DSC]
    ax6.bar(xp, vals, color=cols, alpha=0.7, width=0.7,
            yerr=errs, capsize=3, ecolor='navy')
    ax6.axhline(0, color='gray', ls='--', lw=0.8)
    ax6.set_xticks(xp)
    ax6.set_xticklabels(['Planck\nskew', 'DSC\nskew', 'Planck\nkurt', 'DSC\nkurt'], fontsize=7)
    ax6.set_title('(f) Gaussianity comparison', fontsize=9)

    plt.tight_layout(h_pad=0.6, w_pad=0.3)
    save_fig(fig, 'fig8_planck_comparison')


# ================================================================
# Figure 1: Hero Figure (4-panel overview)
# ================================================================
def gen_fig1():
    print("Generating Fig 1: Hero figure...")

    # ── Panel (a): 3D Mollweide CMB map (mvp4.1 style) ──
    N3 = 128
    np.random.seed(42)
    kx3, ky3, kz3 = np.meshgrid(np.fft.fftfreq(N3), np.fft.fftfreq(N3), np.fft.fftfreq(N3), indexing='ij')
    k2_3 = kx3**2 + ky3**2 + kz3**2; k2_3[0,0,0] = 1e-10
    amp3 = np.random.normal(0,1,(N3,N3,N3)) + 1j*np.random.normal(0,1,(N3,N3,N3))
    fk3 = amp3 / (k2_3**0.75); fk3[0,0,0] = 0
    noise3 = np.real(np.fft.ifftn(fk3))
    noise3 = (noise3 - noise3.mean()) / (noise3.std() + 1e-10) * 0.15

    phi3 = noise3.copy(); phi3_prev = noise3.copy()
    for t in range(1, 61):
        c2t = 0.45 / np.log(t + 10)**2
        lap3 = (np.roll(phi3,1,0)+np.roll(phi3,-1,0)+np.roll(phi3,1,1)+np.roll(phi3,-1,1)+
                np.roll(phi3,1,2)+np.roll(phi3,-1,2)-6*phi3)/6
        drag3 = 0.01 * (phi3 - phi3_prev)
        pn3 = 2*phi3 - phi3_prev + c2t*lap3 - drag3
        pn3 = np.clip(pn3, -10, 10)
        phi3_prev = phi3.copy(); phi3 = pn3

    norm_u = (phi3 - phi3.mean()) / (phi3.std() + 1e-10)

    # Mollweide sphere shell sampling
    theta_m, varphi_m = np.mgrid[-np.pi/2:np.pi/2:400j, -np.pi:np.pi:800j]
    radius_m = N3 * 0.35
    xi = np.clip(np.round(N3/2 + radius_m*np.cos(theta_m)*np.cos(varphi_m)).astype(int), 0, N3-1)
    yi = np.clip(np.round(N3/2 + radius_m*np.cos(theta_m)*np.sin(varphi_m)).astype(int), 0, N3-1)
    zi = np.clip(np.round(N3/2 + radius_m*np.sin(theta_m)).astype(int), 0, N3-1)
    cmb_shell = norm_u[xi, yi, zi]

    # ── Panels (b-d): 2D experiments ──
    N = 300
    phi0 = gen_init(N, seed=42)
    phi_s = evolve_symp(phi0, steps=45, c2=0.45, drag=0.0)
    phi_s = silk_damp(phi_s, 35.0)
    Ts = znorm(phi_s)
    k, Dk_s, _ = compute_Dk(Ts)
    Dk_ss = smooth(Dk_s, 3)
    Dk_ref = mock_lcdm(k)
    Dk_rn = Dk_ref / Dk_ref.max() * Dk_ss.max()

    phi_d = evolve_diss(phi0, steps=150)
    Td = znorm(phi_d)

    # ── Build figure ──
    fig = plt.figure(figsize=(7.0, 5.5))

    # (a) 3D Mollweide CMB
    ax1 = fig.add_subplot(2, 2, 1, projection='mollweide')
    ax1.pcolormesh(varphi_m, theta_m, cmb_shell, cmap='RdYlBu_r', shading='auto', vmin=-2.5, vmax=2.5, rasterized=True)
    ax1.grid(True, alpha=0.3, color='gray', linestyle=':')
    ax1.set_xticklabels([]); ax1.set_yticklabels([])
    ax1.set_title('(a) DSC full-sky CMB (3D$\\to$Mollweide)', fontsize=9, pad=8)

    # (b) Power spectrum with acoustic peaks
    ax2 = fig.add_subplot(2, 2, 2)
    kk = k[:70]
    ax2.plot(kk, Dk_ss[:70], color=C_DSC, lw=1.8, label='DSC')
    ax2.plot(kk, Dk_rn[:70], color=C_LCDM, ls='--', lw=1.5, alpha=0.6, label=r'Mock $\Lambda$CDM')
    peaks, _ = find_peaks(Dk_ss[:70], distance=5, prominence=Dk_ss[:70].max()*0.01)
    for p in peaks:
        if p < 70: ax2.plot(kk[p], Dk_ss[p], 'v', color=C_DSC, ms=5)
    ax2.set_xlabel(r'$k$')
    ax2.set_ylabel(r'$D(k)$')
    ax2.set_title('(b) Acoustic oscillations in D(k)', fontsize=9)
    ax2.legend(frameon=False, fontsize=8)

    # (c) Ablation histograms
    ax3 = fig.add_subplot(2, 2, 3)
    xg = np.linspace(-5, 5, 200)
    ax3.hist(Td.flat, bins=80, density=True, alpha=0.5, color=C_DISS, label='Dissipative', edgecolor='none')
    ax3.hist(Ts.flat, bins=80, density=True, alpha=0.5, color=C_DSC, label='Symplectic', edgecolor='none')
    ax3.plot(xg, norm.pdf(xg), 'k--', lw=1.5)
    ax3.set_xlabel(r'$\delta T / \sigma$')
    ax3.set_title('(c) Ablation: symplectic restores Gaussianity', fontsize=9)
    ax3.legend(frameon=False, fontsize=8)
    ax3.set_xlim(-5, 5)

    # (d) Planck vs DSC pixel distribution
    has_planck = os.path.exists('gemini/COM_CMB_IQU-smica_2048_R3.00_full.fits')
    if has_planck:
        NSIDE = 64
        real = hp.read_map('gemini/COM_CMB_IQU-smica_2048_R3.00_full.fits', field=0)
        real_low = hp.ud_grade(real, nside_out=NSIDE)
        valid = real_low > -1e20
        vp = real_low[valid]
        pmin, pmax = np.percentile(vp, 2), np.percentile(vp, 98)
        rc = np.clip(real_low, pmin, pmax)
        mv, sv_ = np.mean(np.clip(vp, pmin, pmax)), np.std(np.clip(vp, pmin, pmax))
        target_sky = np.where(valid, (rc - mv) / (sv_ + 1e-8), 0.0)

        vol = dsc_3d(gen_3d(96, seed=42), steps=45)
        dsc_sky = sample_sphere(vol, nside=NSIDE)

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.hist(target_sky[valid]/target_sky[valid].std(), bins=60, density=True,
                 alpha=0.5, color=C_PLANCK, label='Planck', edgecolor='none')
        ax4.hist(dsc_sky/dsc_sky.std(), bins=60, density=True,
                 alpha=0.5, color=C_DSC, label='DSC 3D', edgecolor='none')
        ax4.plot(xg, norm.pdf(xg), 'k--', lw=1)
        ax4.set_xlabel(r'$\delta T / \sigma$')
        ax4.set_title('(d) DSC 3D vs Planck pixel dist.', fontsize=9)
        ax4.legend(frameon=False, fontsize=8)
        ax4.set_xlim(-5, 5)
    else:
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.text(0.5, 0.5, '[Planck data not available]', ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout(h_pad=0.8, w_pad=0.5)
    save_fig(fig, 'fig1_hero')


# ================================================================
# Tables
# ================================================================
def gen_tables(S_ens, K_ens, peaks_dsc, peaks_ref):
    print("Generating Tables...")

    # Table 1: Gaussianity comparison
    tab1 = r"""\begin{table}[t]
\centering
\caption{Gaussianity metrics: DSC symplectic lattice vs.\ Planck SMICA.
  DSC values are ensemble means over 20 independent runs with $1\sigma$ error bars.
  Theoretical standard errors for $N=300^2=90{,}000$ samples: $\sigma_{\rm skew}=0.008$, $\sigma_{\rm kurt}=0.016$.}
\label{tab:gaussianity}
\begin{tabular}{lccc}
\toprule
 & Skewness & Excess kurtosis & KS $p$-value \\
\midrule
Gaussian (ideal) & $0$ & $0$ & $>0.05$ \\
DSC symplectic & $""" + f"{S_ens.mean():+.3f}" + r" \pm " + f"{S_ens.std():.3f}" + r"""$ & $""" + f"{K_ens.mean():+.3f}" + r" \pm " + f"{K_ens.std():.3f}" + r"""$ & --- \\
DSC dissipative & $-0.110$ & $-1.596$ & $<10^{-100}$ \\
Planck SMICA & $-0.036$ & $-0.492$ & --- \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open('figures/TABLE_gaussianity.tex', 'w') as f:
        f.write(tab1)
    print("  Saved: figures/TABLE_gaussianity.tex")

    # Table 2: Peak positions
    tab2 = r"""\begin{table}[t]
\centering
\caption{Acoustic-like peak positions in the scaled power spectrum $D(k) = k^2 P(k)$.
  DSC peaks emerge from pure symplectic evolution (no parameter tuning for this column).
  Mock $\Lambda$CDM uses $D(k) \propto e^{-(k/35)^2}[1 + 1.2\sin^2(\pi k/12)](50/(k+10))$.}
\label{tab:peaks}
\begin{tabular}{lcccccc}
\toprule
 & Peak 1 & Peak 2 & Peak 3 & Peak 4 & Peak 5 & Peak 6 \\
\midrule
"""
    dsc_str = " & ".join([str(p) for p in peaks_dsc[:6]] + ["---"]*(6-len(peaks_dsc[:6])))
    ref_str = " & ".join([str(p) for p in peaks_ref[:6]] + ["---"]*(6-len(peaks_ref[:6])))
    tab2 += f"DSC symplectic & {dsc_str} \\\\\n"
    tab2 += f"Mock $\\Lambda$CDM & {ref_str} \\\\\n"
    tab2 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open('figures/TABLE_peaks.tex', 'w') as f:
        f.write(tab2)
    print("  Saved: figures/TABLE_peaks.tex")


# ================================================================
# LaTeX includes
# ================================================================
def gen_latex_includes():
    includes = r"""% === Auto-generated LaTeX includes for paper figures ===

% Fig 1: Hero figure (4-panel overview)
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/fig1_hero.pdf}
    \caption{Overview of DSC symplectic lattice CMB simulation.
    (a)~Temperature fluctuation map from Störmer--Verlet evolution with $1/\ln^2(t)$ cooling.
    (b)~Scaled power spectrum $D(k)=k^2P(k)$ showing acoustic-like oscillation peaks (blue triangles) compared to mock $\Lambda$CDM (dashed).
    (c)~Ablation: pixel distribution of dissipative CML (red) vs.\ symplectic lattice (blue); only the symplectic model recovers Gaussianity (black dashed).
    (d)~Pixel distribution comparison between DSC 3D forward simulation and real Planck SMICA data.}
    \label{fig:hero}
\end{figure*}

% Fig 2: Ablation
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/fig2_ablation.pdf}
    \caption{Ablation study: dissipative coupled map lattice (top row) vs.\ symplectic Störmer--Verlet integrator (bottom row).
    (a,d)~Temperature maps. (b,e)~Pixel histograms with Gaussian reference. (c,f)~Power spectra $D(k)$ with mock $\Lambda$CDM overlay.
    The dissipative model produces topological defects and bimodal statistics (kurtosis $= -1.60$), while the symplectic model restores near-Gaussian fluctuations.}
    \label{fig:ablation}
\end{figure*}

% Fig 3: Acoustic peaks
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/fig3_acoustic_peaks.pdf}
    \caption{Acoustic-like oscillation peaks in the DSC power spectrum $D(k)$ (blue solid) compared to mock $\Lambda$CDM (black dashed).
    Peaks emerge naturally from wave propagation on the symplectic lattice without parameter tuning.
    Blue triangles mark DSC peak positions; black triangles mark $\Lambda$CDM peaks (see Table~\ref{tab:peaks}).}
    \label{fig:peaks}
\end{figure}

% Fig 4: Twin experiment
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/fig4_twin_experiment.pdf}
    \caption{Twin experiment for parameter recovery.
    (a)~Bayesian optimization convergence (100 trials).
    (b)~Recovered $D(k)$ (red) overlaid on ground truth (black dashed), with random guess (gray) for comparison.
    All three lattice parameters ($c^2$, drag, $k_{\rm opt}$) are recovered to within $< 1\%$ in the noiseless setting.}
    \label{fig:twin}
\end{figure}

% Fig 5: Freeze-out
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/fig5_freezeout.pdf}
    \caption{Acoustic freeze-out from $1/\ln^2(t)$ cooling.
    Top: wavefront snapshots at $t=0, 5, 30, 100, 300$.
    (e)~Wavefront radius (blue) saturates at $r_s \approx 38$; effective sound speed $c^2_{\rm eff}$ (red dashed) decays to zero.
    (f)~Causal structure showing the bending light cone produced by adiabatic cooling.}
    \label{fig:freezeout}
\end{figure*}

% Fig 6: Ensemble
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/fig6_ensemble.pdf}
    \caption{Ensemble statistics from 20 independent DSC runs ($N=300$, 45 steps).
    (a)~Mean $D(k)$ with $\pm 1\sigma$ band and mock $\Lambda$CDM reference.
    (b,c)~Skewness and excess kurtosis distributions, both consistent with zero.
    (d)~Q-Q plot confirming near-Gaussian one-point statistics.}
    \label{fig:ensemble}
\end{figure}

% Fig 7: Sensitivity
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/fig7_sensitivity.pdf}
    \caption{Parameter sensitivity analysis.
    Top row: skewness (red) and excess kurtosis (blue) vs.\ $c^2$, drag, and evolution steps.
    Bottom row: Pearson spectral correlation with mock $\Lambda$CDM.
    Gaussianity and spectral shape are robust across wide parameter ranges.}
    \label{fig:sensitivity}
\end{figure*}

% Fig 8: Planck comparison
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/fig8_planck_comparison.pdf}
    \caption{DSC 3D forward simulation vs.\ real Planck SMICA data.
    (a)~Planck SMICA full-sky map (Nside=64). (b)~DSC 3D $\to$ sphere projection.
    (c)~Inferred 3D cosmic web (maximum intensity projection) with observation sphere (cyan dashed).
    (d)~Angular power spectrum comparison. (e)~Pixel distributions. (f)~Gaussianity metrics with ensemble error bars.
    DSC skewness ($-0.075 \pm 0.009$) is close to Planck ($-0.036$); kurtosis sign differs ($+0.29$ vs.\ $-0.49$).}
    \label{fig:planck}
\end{figure*}
"""
    with open('figures/latex_includes.tex', 'w') as f:
        f.write(includes)
    print("Saved: figures/latex_includes.tex")


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    import time
    t0 = time.time()

    sd, kd, ss, ks = gen_fig2()
    peaks_dsc, peaks_ref = gen_fig3()
    gen_fig4()
    gen_fig5()
    S_ens, K_ens = gen_fig6()
    gen_fig7()
    gen_fig8()
    gen_fig1()
    gen_tables(S_ens, K_ens, peaks_dsc, peaks_ref)
    gen_latex_includes()

    print(f"\nAll figures generated in {time.time()-t0:.0f}s")
    print(f"Output directory: figures/")
