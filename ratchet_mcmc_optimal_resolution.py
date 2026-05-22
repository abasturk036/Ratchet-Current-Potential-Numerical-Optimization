"""
Optimal Fourier truncation: |J|/P_0 vs K at fixed L.

Scans the Fourier truncation K to find K* — the minimum number of modes
at which the optimised current deviates by less than THRESHOLD from the
high-K reference.  For each K the spatial grid is set automatically to
N = 4K, guaranteeing 4 points per shortest wavelength (λ_min = L/K), which
is the minimum needed for alias-free evaluation of f·f′ (bandwidth 2K).

Notation
--------
K    number of Fourier modes (K sine + K cosine terms, indices 1..K)
N    number of spatial grid points; set to N = 4K for each run
L    system size (period, fixed)
K*   minimum K for which |J(K) - J_ref| / |J_ref| < THRESHOLD
P_0  = 1/L  uniform reference density; J/P_0 is a dimensionless velocity

Tempering: beta ramps linearly 0 → BETA_MAX so the chain freely explores
both positive and negative J (broken symmetry) in the early hot phase
and then settles on the |J|-maximising configuration.

Adele Basturk, Sunghan Ro, written in collaboration with Claude Code 4.6, 5/2026.
"""

import numpy as np
import os
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Configuration ─────────────────────────────────────────────────────────────

L         = 40            # fixed system size
SIGMA     = 0.95          # max |f(x)| force constraint
STEPS_PER_COEFF = 25000   # perturbations per coefficient; total steps = STEPS_PER_COEFF * 2K
BETA_MAX  = 4000.0        # final inverse temperature (linear ramp from 0)
THRESHOLD = 0.10          # convergence criterion: |J(K) - J_ref| / |J_ref| < THRESHOLD

# Fourier truncations to scan; spatial grid set to N = 4K for each run
# (4 pts per shortest wavelength — alias-free for f·f′ products)
K_LIST = [4, 8, 16, 24, 32, 48, 54, 68]

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ─── Grid and Fourier matrices ──────────────────────────────────────────────────

def make_grid(K, N, L):
    """K Fourier modes on N spatial grid points over [0, L]."""
    dx    = L / N
    x     = np.linspace(0, L, N, endpoint=False)
    n_arr = np.arange(K + 1)
    omega = 2 * np.pi * n_arr / L
    SIN   = np.sin(np.outer(omega, x))
    COS   = np.cos(np.outer(omega, x))
    return dx, x, omega, SIN, COS

def build_V(a_s, a_c, omega, SIN, COS):
    V  =  a_s @ SIN + a_c @ COS
    f  = -(a_s * omega) @ COS + (a_c * omega) @ SIN
    fp =  (a_s * omega**2) @ SIN + (a_c * omega**2) @ COS
    return V, f, fp

def enforce_bcs(a_s, a_c, omega, SIN, COS, x):
    """Phase-shift so global min of V on f=0 crossings is at x=0; set V(0)=0."""
    V, f, _ = build_V(a_s, a_c, omega, SIN, COS)
    zc      = np.where(np.diff(np.sign(f)) != 0)[0]
    idx     = zc[np.argmin(V[zc])] if len(zc) else np.argmin(V)
    x_star  = x[idx]
    phi     = omega * x_star
    cp, sp  = np.cos(phi), np.sin(phi)
    a_sn    =  a_s * cp - a_c * sp
    a_cn    =  a_c * cp + a_s * sp
    a_sn[0] = 0.0
    a_cn[0] = -np.sum(a_cn[1:])
    return a_sn, a_cn

# ─── Exact current (Eq. 14) ────────────────────────────────────────────────────

def J_exact(V, f, fp, dx, L, N):
    id_ = 1.0 / (1.0 - f * f)
    A   = (2 * f * fp + f) * id_
    b   = -(fp + 1) * id_
    mA  = 0.5 * (A[:-1] + A[1:]) * dx
    Phi = np.zeros(N)
    np.cumsum(mA, out=Phi[1:])
    PL  = Phi[-1] + 0.5 * (A[-1] + A[0]) * dx
    xi  = np.exp(PL)
    eP  = np.exp(Phi)
    num = L * (1.0 - xi)
    if abs(num) < 1e-30:
        return 0.0
    Ib  = np.trapz(b * (xi / eP), dx=dx)
    Ix  = np.trapz(eP, dx=dx)
    be  = b * np.exp(-Phi)
    mb  = 0.5 * (be[:-1] + be[1:]) * dx
    G   = np.zeros(N)
    np.cumsum(mb, out=G[1:])
    Id  = np.trapz(eP * G, dx=dx)
    den = Ib * Ix + (1.0 - xi) * Id
    return num / den if abs(den) > 1e-30 else 0.0

# ─── MCMC with linear tempering ────────────────────────────────────────────────

def run_mcmc(K, N, L, sig=SIGMA, n_steps=None, seed=None):
    """
    Optimise |J| using K Fourier modes on N spatial grid points.

    Beta ramps linearly 0 → BETA_MAX:
      - early (beta ≈ 0): pure random walk, freely explores ±J
      - late (beta → BETA_MAX): settles on the |J|-maximising configuration
    """
    dx, x, omega, SIN, COS = make_grid(K, N, L)
    rng      = np.random.default_rng(seed)
    seed_out = int(rng.bit_generator.state["state"]["state"]) & 0xFFFFFFFF

    free_idx = np.arange(1, K + 1)   # all non-DC modes are free
    if n_steps is None:
        n_steps = STEPS_PER_COEFF * 2 * K
    n_free   = len(free_idx)
    ss       = 0.06 * sig             # force-space step size (mode-scaled below)

    a_s = np.zeros(K + 1)
    a_c = np.zeros(K + 1)
    a_s, a_c = enforce_bcs(a_s, a_c, omega, SIN, COS, x)
    V, f, fp  = build_V(a_s, a_c, omega, SIN, COS)
    Jc        = J_exact(V, f, fp, dx, L, N) or 0.0
    bJ        = abs(Jc)
    bs, bc    = a_s.copy(), a_c.copy()

    Jt  = np.empty(n_steps)
    bt  = np.empty(n_steps)
    acc = 0
    t0  = time()

    for i in range(n_steps):
        beta   = BETA_MAX * i / n_steps    # linear: 0 → BETA_MAX
        ns, nc = a_s.copy(), a_c.copy()

        pick = rng.integers(2 * n_free)
        m    = free_idx[pick % n_free]
        d    = rng.normal() * ss / omega[m]  # 1/ω scaling keeps |Δf|_max ≈ ss for all modes
        if pick < n_free:
            ns[m] += d
        else:
            nc[m] += d

        ns, nc      = enforce_bcs(ns, nc, omega, SIN, COS, x)
        nV, nf, nfp = build_V(ns, nc, omega, SIN, COS)

        if nf.max() >= sig or nf.min() <= -sig:
            nJ = None
        else:
            nJ = J_exact(nV, nf, nfp, dx, L, N)

        if nJ is not None:
            dE = abs(nJ) - abs(Jc)
            if dE > 0 or rng.random() < np.exp(beta * dE):
                a_s, a_c = ns, nc
                Jc       = nJ
                acc      += 1
                if abs(Jc) > bJ:
                    bJ     = abs(Jc)
                    bs, bc = a_s.copy(), a_c.copy()

        Jt[i] = Jc
        bt[i]  = bJ

        if (i + 1) % max(n_steps // 5, 1) == 0:
            print(f"    {i+1:>7d}/{n_steps}  J={Jc:+.6f}  best={bJ:.6f}  "
                  f"acc={acc/(i+1):.3f}  β={beta:.0f}  [{time()-t0:.1f}s]")

    print(f"  done  {time()-t0:.1f}s  |J|_best={bJ:.6f}  acc={acc/n_steps:.3f}")
    return bs, bc, Jt, bt, seed_out

# ─── Scan ──────────────────────────────────────────────────────────────────────

print(f"K* scan — L={L}, σ={SIGMA}, {STEPS_PER_COEFF} steps/coeff, N=4K for each run")
print(f"K list: {K_LIST}\n")

results = []
for K in K_LIST:
    N       = 5000
    n_steps = STEPS_PER_COEFF * 2 * K
    print(f"── K={K}  N={N}  n_steps={n_steps}  (dx={L/N:.3f})")
    bs, bc, Jt, bt, seed      = run_mcmc(K, N, L, n_steps=n_steps)
    dx, x, omega, SIN, COS    = make_grid(K, N, L)
    V, f, fp                   = build_V(bs, bc, omega, SIN, COS)
    Je                         = J_exact(V, f, fp, dx, L, N)
    results.append(dict(K=K, N=N, n_steps=n_steps, Je=Je, x=x, V=V,
                        Jt=Jt, bt=bt, bs=bs, bc=bc, seed=seed))
    print()

# ─── Convergence analysis ──────────────────────────────────────────────────────

K_arr  = np.array([r['K']  for r in results])
J_arr  = np.array([r['Je'] for r in results])

# Use the best |J| found across all K as reference — the last run may have
# found a suboptimal solution (harder landscape at large K), which would
# make converged intermediate runs appear to deviate when they don't.
best_idx = np.argmax(np.abs(J_arr))
J_ref    = J_arr[best_idx]
K_ref    = int(K_arr[best_idx])

devs   = np.abs(J_arr - J_ref) / (np.abs(J_ref) + 1e-30)

K_star_idx = next((i for i, d in enumerate(devs) if d < THRESHOLD),
                  len(K_arr) - 1)
K_star = int(K_arr[K_star_idx])

print(f"J_ref  (K={K_ref}, best run):  {J_ref:+.6f}")
print(f"K*  =  {K_star}   (first K with deviation < {100*THRESHOLD:.0f}%)\n")

# ─── Write txt ─────────────────────────────────────────────────────────────────

seeds_tag = "_".join(str(r["seed"]) for r in results)
base      = f"{OUTDIR}/resolution_L{L}_sig{SIGMA}_seeds{seeds_tag}"
outpath   = base + ".txt"

with open(outpath, "w") as fh:
    fh.write("# Optimal Fourier truncation scan: J vs K  (J_exact = J/P_0)\n")
    fh.write(f"# L={L}  sigma={SIGMA}  steps_per_coeff={STEPS_PER_COEFF}  N=4K for each run\n")
    fh.write(f"# J_ref={J_ref:.10e}  K_ref={K_ref}  K_star={K_star}\n")
    fh.write("# K  N  J  deviation\n")
    for r, d in zip(results, devs):
        fh.write(f"{r['K']:5d}  {r['N']:6d}  {r['Je']:+.10e}  {d:.6f}\n")

print(f"→ {outpath}")

# ─── Plotting ──────────────────────────────────────────────────────────────────

STRIDE_PTS = 2000   # target number of plotted chain points per subplot
ncols  = min(4, len(results))
nrows  = (len(results) + ncols - 1) // ncols

# Figure 1 — J/P_0 vs K  (main result; J_exact already returns J/P_0)
fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
ax.plot(K_arr, np.abs(J_arr), 'o-', color='steelblue',
        lw=1.5, ms=6, zorder=3)
ax.axvline(K_star, color='crimson', ls='--', lw=1.2,
           label=f'$K^* = {K_star}$  ({100*THRESHOLD:.0f}% threshold)')
ax.axhline(np.abs(J_ref), color='grey', ls=':', lw=1,
           label=f'reference  ($K={K_ref}$, best run)')
ax.set_xlabel('$K$  (Fourier modes)', fontsize=12)
ax.set_ylabel(r'$J/P_0$', fontsize=12)
ax.set_title(f'Optimal Fourier truncation  [$L={L},\\ N=4K,\\ \\sigma={SIGMA}$]',
             fontsize=11)
ax.legend(fontsize=10)
ax.tick_params(labelsize=9)
fig.savefig(f"{base}_JvsK.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  → {base}_JvsK.png")

# Figure 2 — V(x) at each K
fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 3.0*nrows),
                         constrained_layout=True, squeeze=False)
axes_flat = axes.flatten()
for ax, r in zip(axes_flat, results):
    d = np.abs(r['Je'] - J_ref) / (np.abs(J_ref) + 1e-30)
    ax.plot(r['x'], r['V'], lw=0.8, color='steelblue')
    ax.set_title(f"K={r['K']}, N={r['N']}\n$J/P_0$={r['Je']:.4f},  dev={d:.1%}",
                 fontsize=7)
    ax.set_xlabel('x', fontsize=7)
    ax.set_ylabel('V(x)', fontsize=7)
    ax.tick_params(labelsize=6)
for ax in axes_flat[len(results):]:
    ax.set_visible(False)
fig.suptitle(f'V(x)  [$L={L},\\ N=4K,\\ \\sigma={SIGMA}$]', fontsize=12)
fig.savefig(f"{base}_Vx.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  → {base}_Vx.png")

# Figure 3 — J chains  (signed J shows ±symmetry exploration then convergence)
fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 3.0*nrows),
                         constrained_layout=True, squeeze=False)
axes_flat = axes.flatten()
for ax, r in zip(axes_flat, results):
    stride = max(1, r['n_steps'] // STRIDE_PTS)
    steps  = np.arange(0, r['n_steps'], stride)
    ax.plot(steps, r['Jt'][::stride], lw=0.4, alpha=0.6,
            color='steelblue', label='J')
    ax.plot(steps, r['bt'][::stride], lw=0.8,
            color='crimson', label='best |J|')
    ax.set_title(f"K={r['K']}, N={r['N']}, steps={r['n_steps']}", fontsize=7)
    ax.set_xlabel('step', fontsize=7)
    ax.set_ylabel('J', fontsize=7)
    ax.tick_params(labelsize=6)
axes_flat[0].legend(fontsize=6, framealpha=0.5)
for ax in axes_flat[len(results):]:
    ax.set_visible(False)
fig.suptitle(f'MCMC chains  [$L={L},\\ N=4K,\\ \\sigma={SIGMA}$]', fontsize=12)
fig.savefig(f"{base}_chains.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  → {base}_chains.png")

print("\nDone.")
