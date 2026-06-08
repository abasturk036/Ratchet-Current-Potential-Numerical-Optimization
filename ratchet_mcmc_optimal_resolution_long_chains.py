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

Output: one .txt file per K (written immediately on completion) plus a
summary .txt after all runs finish.  No plotting.

Adele Basturk, Sunghan Ro, written in collaboration with Claude Code 4.6, 5/2026.
"""

import numpy as np
import os
import collections
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

'''
TINY = np.finfo(float).tiny   # smallest normal float64 (~2.2e-308)

# Module-level FP flag counter — reset before each K run, written to file after.
__fp_flags = collections.Counter()

def _flag(name, val):
    """Record overflow / invalid / underflow flags for val into __fp_flags."""
    if np.isscalar(val):
        if np.isinf(val):                          __fp_flags[f"{name}:overflow"]  += 1
        elif np.isnan(val):                        __fp_flags[f"{name}:invalid"]   += 1
        elif val != 0.0 and abs(val) < TINY:       __fp_flags[f"{name}:underflow"] += 1
    else:
        if np.any(np.isinf(val)):                  __fp_flags[f"{name}:overflow"]  += 1
        if np.any(np.isnan(val)):                  __fp_flags[f"{name}:invalid"]   += 1
        if np.any((val != 0.0) & (np.abs(val) < TINY)):
                                                   __fp_flags[f"{name}:underflow"] += 1

def _flag_cancel(name, a, b, result):
    """Record cancellation when |result| loses 8+ digits vs max(|a|,|b|)."""
    scale = max(abs(a), abs(b), 1e-300)
    if abs(result) < 1e-8 * scale:
        __fp_flags[f"{name}:cancellation"] += 1

'''

# ─── Configuration ─────────────────────────────────────────────────────────────

L         = 100            # fixed system size
SIGMA     = 0.95          # max |f(x)| force constraint
STEPS_PER_COEFF = 2500   # perturbations per coefficient; total steps = STEPS_PER_COEFF * 2K
BETA_MAX  = 4000.0        # final inverse temperature (linear ramp from 0)
THRESHOLD = 0.10          # convergence criterion: |J(K) - J_ref| / |J_ref| < THRESHOLD
CHAIN_STRIDE = 500       # write every CHAIN_STRIDE-th step to file

# Fourier truncations to scan; spatial grid set to N = 5000 for each run
# (4 pts per shortest wavelength — alias-free for f·f′ products)
K_LIST = [16, 32, 48, 64]

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

def _slog_add(s1, l1, s2, l2):
    """Add two signed-log numbers (s, ln|.|); cancellation-safe via log1p/expm1."""
    if s1 == 0: return s2, l2
    if s2 == 0: return s1, l1
    if l1 < l2: s1, l1, s2, l2 = s2, l2, s1, l1
    d = l2 - l1
    if s1 == s2: return s1, l1 + np.log1p(np.exp(d))
    v = -np.expm1(d)
    return (0, -np.inf) if v <= 0 else (s1, l1 + np.log(v))
 
 
def J_exact(V, f, fp, dx, L, N):
    """
    Stable log-space evaluation of the ratchet current J/P0 (Eq. 15), in the
    factored form on the page:
 
        ln|J/P0| = - ln[  xi_0L * I1  +  (1 - xi_0L) * I2  ]   # bracket, both terms ~ O(1)
                   - (Phi_max - Phi_min)                       # the pulled-out offset R
                   + ln[ L (1 - xi_0L) ]                       # numerator (ADDED, see note 1)
 
      xi_0L = e^{PL},   m = PL - Phi_min,
      I1 = ( int_0^L e^{Phi-Phi_max} dx ) * ( int_0^L b e^{(PL-Phi)-m} dx' )
      I2 = int_0^L e^{Phi-Phi_max} dx * [ int_0^x b e^{-(Phi-Phi_min)} dx' ]   

    """
    if np.abs(f).max() >= 1.0:
        raise ValueError(f"max|f| = {np.abs(f).max():.5f} >= 1: out of model domain")
 
    inv = 1.0 / (1.0 - f * f)
    A = (2.0 * f * fp + f) * inv
    b = -(fp + 1.0) * inv
 
    Phi = np.zeros(N)
    np.cumsum(0.5 * (A[:-1] + A[1:]) * dx, out=Phi[1:])     # Phi(x) = cumulative int A
    PL = Phi[-1] + 0.5 * (A[-1] + A[0]) * dx                # full-period exponent
    if PL == 0.0:
        return 0.0
    Pmax, Pmin = Phi.max(), Phi.min()
    R = Pmax - Pmin                                         # the artificial offset
    m = PL - Pmin
 
    # rescaled integrals -- every exp() argument is <= 0, so nothing overflows
    tIx = np.trapezoid(np.exp(Phi - Pmax), dx=dx)                       # > 0
    vb  = np.trapezoid(b * np.exp((PL - Phi) - m), dx=dx)               # signed
    Kc  = np.zeros(N); be = b * np.exp(-(Phi - Pmin))
    np.cumsum(0.5 * (be[:-1] + be[1:]) * dx, out=Kc[1:])
    tId = np.trapezoid(np.exp(Phi - Pmax) * Kc, dx=dx)                  # signed
 
    sE, lE = _slog_add(1, 0.0, -1, PL)                     # (1 - xi_0L) in signed-log
 
    # bracket = xi_0L * tIx * vb  +  (1 - xi_0L) * tId
    lt1 = PL + np.log(tIx) + np.log(abs(vb)); s1 = np.sign(vb)
    lt2 = lE + np.log(abs(tId));              s2 = sE * np.sign(tId)
    sB, lB = _slog_add(s1, lt1, s2, lt2)
 
    # ln|J/P0| = ln[L(1-xi_0L)] - R - ln|bracket| ;  sign = sign(num)*sign(den)
    lnJ = (np.log(L) + lE) - R - lB
    return sE * sB * np.exp(lnJ)

'''
# ─── Exact current (Eq. 14) ────────────────────────────────────────────────────
# --> Not in log-space, 
def J_exact(V, f, fp, dx, L, N):
    id_ = 1.0 / (1.0 - f * f);  _flag("id_", id_)
    A   = (2 * f * fp + f) * id_;  _flag("A", A)
    b   = -(fp + 1) * id_;  _flag("b", b)
    mA  = 0.5 * (A[:-1] + A[1:]) * dx;  _flag("mA", mA)
    Phi = np.zeros(N)
    np.cumsum(mA, out=Phi[1:]);  _flag("Phi", Phi)
    PL  = Phi[-1] + 0.5 * (A[-1] + A[0]) * dx;  _flag("PL", PL)
    xi  = np.exp(PL);  _flag("xi", xi)
    eP  = np.exp(Phi);  _flag("eP", eP)
    num = L * (1.0 - xi);  _flag("num", num);  _flag_cancel("num", L, L * xi, num)
    if abs(num) < 1e-30:
        return 0.0
    Ib  = np.trapz(b * (xi / eP), dx=dx);  _flag("Ib", Ib)
    Ix  = np.trapz(eP, dx=dx);  _flag("Ix", Ix)
    be  = b * np.exp(-Phi);  _flag("be", be)
    mb  = 0.5 * (be[:-1] + be[1:]) * dx
    G   = np.zeros(N)
    np.cumsum(mb, out=G[1:]);  _flag("G", G)
    Id  = np.trapz(eP * G, dx=dx);  _flag("Id", Id)
    term1 = Ib * Ix
    term2 = (1.0 - xi) * Id
    den   = term1 + term2;  _flag("den", den);  _flag_cancel("den", term1, term2, den)
    return num / den if abs(den) > 1e-30 else 0.0
'''

# ─── MCMC with linear tempering ────────────────────────────────────────────────

def run_mcmc(K, N, L, sig=SIGMA, n_steps=None, seed=None):
    """
    Optimise |J| using K Fourier modes on N spatial grid points.

    Beta ramps linearly 0 → BETA_MAX:
      - early (beta ≈ 0): close pure random walk, freely explores ±J
      - late (beta → BETA_MAX): settles on the |J|-maximising configuration

    Returns (best_a_s, best_a_c, chain_samples, acc_mode, seed_out).
    chain_samples is a list of (step, J, best_J) tuples at every CHAIN_STRIDE steps.
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

    chain_samples = []   # (step, J, best_J) at every CHAIN_STRIDE steps
    acc           = 0
    t0            = time()

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

        if i % CHAIN_STRIDE == 0:
            chain_samples.append((i, Jc, bJ))

        if (i + 1) % max(n_steps // 5, 1) == 0:
            print(f"    {i+1:>7d}/{n_steps}  J={Jc:+.6f}  best={bJ:.6f}  "
                  f"acc={acc/(i+1):.3f}  β={beta:.0f}  [{time()-t0:.1f}s]")

    print(f"  done  {time()-t0:.1f}s  |J|_best={bJ:.6f}  acc={acc/n_steps:.3f}")
    return bs, bc, chain_samples, seed_out

# ─── Scan ──────────────────────────────────────────────────────────────────────

print(f"K* scan — L={L}, σ={SIGMA}, {STEPS_PER_COEFF} steps/coeff, N=4K for each run")
print(f"K list: {K_LIST}\n")

summary = []   # (K, N, Je, seed) for the final summary file

for K in K_LIST:
    N       = 5000
    n_steps = STEPS_PER_COEFF * 2 * K
    print(f"── K={K}  N={N}  n_steps={n_steps}  (dx={L/N:.3f})")

    #__fp_flags.clear()
    bs, bc, chain_samples, seed = run_mcmc(K, N, L, n_steps=n_steps)

    dx, x, omega, SIN, COS = make_grid(K, N, L)
    V, f, fp               = build_V(bs, bc, omega, SIN, COS)
    Je                     = J_exact(V, f, fp, dx, L, N)
    print(f"  K={K}  J={Je:+.10f}  max|f|={np.max(np.abs(f)):.4f}\n")

    # ── Write per-K output file ────────────────────────────────────────────────
    outpath = f"{OUTDIR}/results_K{K}_L{L}_sig{SIGMA}_seed{seed}.txt"
    with open(outpath, "w") as fh:
        fh.write(f"# Ratchet MCMC: K={K}  L={L}  N={N}  sigma={SIGMA}\n")
        fh.write(f"# steps_per_coeff={STEPS_PER_COEFF}  n_steps={n_steps}  seed={seed}\n")
        fh.write(f"# J_exact={Je:.10e}  max_abs_f={np.max(np.abs(f)):.6f}\n")

        # MCMC chain (every CHAIN_STRIDE-th step)
        fh.write(f"\n# === MCMC CHAIN (every {CHAIN_STRIDE} steps, "
                 f"{len(chain_samples)} rows): step  J  best_J ===\n")
        fh.write("# step  J  best_J\n")
        for step, Jval, bJval in chain_samples:
            fh.write(f"{step}  {Jval:.10e}  {bJval:.10e}\n")
    '''
        # FP flags
        fh.write(f"\n# === FLOATING-POINT FLAGS ({len(__fp_flags)} unique) ===\n")
        fh.write("# count  message\n")
        if __fp_flags:
            for msg, count in __fp_flags.most_common():
                fh.write(f"{count}  {msg}\n")
        else:
            fh.write("# none\n")
    '''
    print(f"  → {outpath}")

    # ── Per-K plot: V(x), f(x), J chain ──────────────────────────────────────
    ch  = np.array(chain_samples)   # shape (n_samples, 3): step, J, best_J
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    fig.suptitle(f'K={K}  L={L}  N={N}  σ={SIGMA}  seed={seed}  '
                 f'best $|J|$={Je:.6f}', fontsize=9)

    axes[0].plot(x, V, lw=0.9, color='steelblue')
    axes[0].axhline(0, color='grey', lw=0.4, ls='--')
    axes[0].set_xlabel('$x$', fontsize=10);  axes[0].set_ylabel('$V(x)$', fontsize=10)
    axes[0].set_title('$V(x)$', fontsize=10);  axes[0].grid(True, alpha=0.2)

    axes[1].plot(x, f, lw=0.9, color='steelblue')
    axes[1].axhline( SIGMA, color='k', lw=0.7, ls=':', label=f'±{SIGMA}')
    axes[1].axhline(-SIGMA, color='k', lw=0.7, ls=':')
    axes[1].axhline(0, color='grey', lw=0.4, ls='--')
    axes[1].set_xlabel('$x$', fontsize=10);  axes[1].set_ylabel('$f(x)$', fontsize=10)
    axes[1].set_title('$f(x) = -V\'(x)$', fontsize=10)
    axes[1].legend(fontsize=8);  axes[1].grid(True, alpha=0.2)

    axes[2].plot(ch[:, 0], ch[:, 1],         lw=0.4, alpha=0.6, color='steelblue', label='$J$')
    axes[2].plot(ch[:, 0], np.abs(ch[:, 2]), lw=0.8,             color='crimson',   label='best $|J|$')
    axes[2].set_xlabel('step', fontsize=10);  axes[2].set_ylabel('$J/P_0$', fontsize=10)
    axes[2].set_title('MCMC chain', fontsize=10)
    axes[2].legend(fontsize=8);  axes[2].grid(True, alpha=0.2)

    chain_png = f"{OUTDIR}/chain_K{K}_L{L}_sig{SIGMA}_seed{seed}.png"
    fig.savefig(chain_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {chain_png}")

    summary.append(dict(K=K, N=N, Je=Je, seed=seed))

# ─── Summary file ──────────────────────────────────────────────────────────────

K_arr = np.array([r['K']  for r in summary])
J_arr = np.array([r['Je'] for r in summary])

best_idx = np.argmax(np.abs(J_arr))
J_ref    = J_arr[best_idx]
K_ref    = int(K_arr[best_idx])

devs       = np.abs(J_arr - J_ref) / (np.abs(J_ref) + 1e-30)
K_star_idx = next((i for i, d in enumerate(devs) if d < THRESHOLD), len(K_arr) - 1)
K_star     = int(K_arr[K_star_idx])

print(f"J_ref  (K={K_ref}, best run):  {J_ref:+.6f}")
print(f"K*  =  {K_star}   (first K with deviation < {100*THRESHOLD:.0f}%)\n")

seeds_tag   = "_".join(str(r["seed"]) for r in summary)
summary_path = f"{OUTDIR}/summary_L{L}_sig{SIGMA}_seeds{seeds_tag}.txt"

with open(summary_path, "w") as fh:
    fh.write("# Optimal Fourier truncation scan: J vs K  (J_exact = J/P_0)\n")
    fh.write(f"# L={L}  sigma={SIGMA}  steps_per_coeff={STEPS_PER_COEFF}  N=4K for each run\n")
    fh.write(f"# J_ref={J_ref:.10e}  K_ref={K_ref}  K_star={K_star}\n")
    fh.write("# K  N  J  deviation  seed\n")
    for r, d in zip(summary, devs):
        fh.write(f"{r['K']:5d}  {r['N']:6d}  {r['Je']:+.10e}  {d:.6f}  {r['seed']}\n")

print(f"→ {summary_path}")

# ─── J vs K summary plot ───────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
ax.plot(K_arr, np.abs(J_arr), 'o-', color='steelblue', lw=1.5, ms=6, zorder=3)
ax.axvline(K_star, color='crimson', ls='--', lw=1.2,
           label=f'$K^* = {K_star}$  ({100*THRESHOLD:.0f}% threshold)')
ax.axhline(np.abs(J_ref), color='grey', ls=':', lw=1,
           label=f'reference  ($K={K_ref}$, best run)')
ax.set_xlabel('$K$  (Fourier modes)', fontsize=12)
ax.set_ylabel('$|J/P_0|$', fontsize=12)
ax.set_title(f'Optimal Fourier truncation  [$L={L},\\ N=4K,\\ \\sigma={SIGMA}$]', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
jvsk_png = f"{OUTDIR}/summary_L{L}_sig{SIGMA}_seeds{seeds_tag}_JvsK.png"
fig.savefig(jvsk_png, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"→ {jvsk_png}")

print("\nDone.")
