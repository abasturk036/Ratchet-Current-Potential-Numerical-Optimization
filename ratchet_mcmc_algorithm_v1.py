"""
MCMC optimization of ratchet current J/P0 — exact formula (Eq. 14).
Run-and-tumble particle in periodic potential V(x), alpha = v = 1.

Fourier-parameterized V(x), fixed max force sigma = max|f|, with f=-V'.

Adele Basturk, Sunghan Ro, written in collaboration with Claude Code Opus 4.6 4/20/2026.

4/26/2025 --> code modified to output text files and optimized to run on FASRC with L, N, K as inputs.

New algorithm implemented to enforce BCs by phase-shifting to the global minimum of V on the 
f=0 curve, placing the minimum at x=0 and shifting the DC mode to set V(0)=0. 
"""

import sys
import os
import argparse
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from time import time

parser = argparse.ArgumentParser(description="MCMC ratchet current optimisation")
parser.add_argument("--L",      type=float, default=40,   help="Domain length in units of pi (L = L_pi * pi)")
parser.add_argument("--N",      type=int,   default=4096, help="Number of grid points")
parser.add_argument("--K",      type=int,   default=32,   help="Number of Fourier modes")
parser.add_argument("--outdir", type=str,   default="/n/home13/abasturk/jobs/output",
                    help="Directory for output files")
args = parser.parse_args()

L = args.L * 2*np.pi
N = args.N
K = args.K
OUTDIR = args.outdir.rstrip("/")
os.makedirs(OUTDIR, exist_ok=True)

dx = L/N; x = np.linspace(0, L, N, endpoint=False)
_Lpi = f"{args.L:g}pi"
_tag = f"_L{_Lpi}_K{K}_N{N}"   # appended to every output filename; updated with seeds after MCMC
k_idx = np.arange(0, K+1); omega = 2*np.pi*k_idx/L
SIN = np.sin(np.outer(omega, x)); COS = np.cos(np.outer(omega, x))

#Build's V(x) profile given Fourier coefficients a_s, a_c, and also computes f=-V' and fp=-V'' for convenience.
def build(a_s, a_c):
    V  = a_s@SIN + a_c@COS
    f  = -(a_s*omega)@COS + (a_c*omega)@SIN
    fp = (a_s*omega**2)@SIN + (a_c*omega**2)@COS
    return V, f, fp

#a_s and A_c are the Fourier coefficients of V(k) = a_s*sin(2πkx/L) + a_c*cos(2πkx/L).

def enforce_bcs(a_s, a_c):
    """
    Enforce BCs by phase-shifting to the global minimum of V on the f=0:
      1. Find x* = argmin_{x : f(x)=0} V(x)  (smallest V among zeros of f).
      2. Phase-shift coefficients so V(x) -> V(x + x*), placing V_min at x=0:
             a_s[n] -> a_s[n]*cos(n*k*x*) - a_c[n]*sin(n*k*x*)
             a_c[n] -> a_c[n]*cos(n*k*x*) + a_s[n]*sin(n*k*x*)
         The n=0 mode is unaffected (omega[0]=0 => phi[0]=0).
      3. DC shift so V_min = V(0) = 0:
             a_c[0] = -sum_{n=1}^{K} a_c_new[n]
         (V(0) = sum(a_c_new) since COS[n,0]=1; this is a_0 in the user's notation
          where cosine coefficients are called a_s.)
    """
    V, f, _ = build(a_s, a_c)

    # Step 1: find x* = min V at a zero of f
    zc = np.where(np.diff(np.sign(f)) != 0)[0]
    x_star_idx = zc[np.argmin(V[zc])] if len(zc) else np.argmin(V)
    x_star = x[x_star_idx]

    # Step 2: phase shift  V(x) -> V(x + x*)
    phi     = omega * x_star          # phi[n] = n*k*x*;  phi[0] = 0
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    a_s_new = a_s * cos_phi - a_c * sin_phi
    a_c_new = a_c * cos_phi + a_s * sin_phi

    # Step 3: set DC so V(0) = 0  [V(0) = sum(a_c_new) after shift]
    a_s_new[0] = 0.0                          # n=0 sine mode is always zero
    a_c_new[0] = -np.sum(a_c_new[1:])         # DC offset = -sum of AC cosine modes

    return a_s_new, a_c_new


#compute J from equation 14 in Systematic Ratchet Current - eq 14
def J_exact(V, f, fp, f_max=0.995):
    """Exact current from Eq. (14). 
    "Toward a Systematic Understanding of Ratchet Current - Analytical Results" """
    #if np.max(np.abs(f)) >= f_max: return None
    id_ = 1.0/(1-f*f); A = (2*f*fp+f)*id_; b = -(fp+1)*id_
    mA = 0.5*(A[:-1]+A[1:])*dx
    Phi = np.zeros(N); np.cumsum(mA, out=Phi[1:])
    PL = Phi[-1]+0.5*(A[-1]+A[0])*dx #integral from 0 to L, unexponentiated
    xi = np.exp(PL) #\xi_{0,L}
    eP = np.exp(Phi) #\xi_{0, x}
    xi_xL = xi / eP  #\xi_{x, L} = e^{Phi(L) - Phi(x)}
    num = L*(1-xi) #numerator
    if abs(num)<1e-30: return 0.0
    Ib = np.trapz(b*xi_xL, dx=dx) #\int_{0,L} dx' b(x') \xi_{x, L}
    Ix = np.trapz(eP, dx=dx)
    be = b*np.exp(-Phi); mb = 0.5*(be[:-1]+be[1:])*dx
    G = np.zeros(N); np.cumsum(mb, out=G[1:])
    Id = np.trapz(eP*G, dx=dx)
    den = Ib*Ix + (1-xi)*Id
    return num/den if abs(den)> 1e-30 else 0.0

# ─── MCMC ────────────────────────────────
def mcmc(sig, n=int(2e5), seed=42):
    rng = np.random.default_rng(seed)
    _seed_used = rng.bit_generator.state["state"]["state"]  # actual 128-bit state; use low 32 bits for filename
    _seed_used = int(_seed_used) & 0xFFFFFFFF
    print(f"{'='*55}  sigma={sig}  seed={_seed_used}")
    # Coefficients length K+1; index 0 is the DC mode (set by enforce_bcs, never perturbed).
    a_s = np.zeros(K+1); a_c = np.zeros(K+1)
    # 
    a_s, a_c = enforce_bcs(a_s, a_c)
    V,f,fp = build(a_s, a_c)
    Jc = J_exact(V,f,fp,sig) or 0.0
    bJ, bs, bc = abs(Jc), a_s.copy(), a_c.copy()
    Jt = np.empty(n); bt = np.empty(n); acc=0
    #sig_ref, L_ref = 0.10, 2*np.pi
    scale = 1 #(sig_ref/sig)**4 * (L/L_ref) # Likely needs editing
    b0 = 1000 * scale; b1 = 500000 * scale
    ss = 0.06*sig
    t0 = time()
    for i in range(n):
        kappa = np.log(b1/b0) / n; beta = b0*np.exp(kappa*i)
        ns,nc = a_s.copy(),a_c.copy()
        j=rng.integers(2*K); m=(j%K)+1; d=rng.normal()*ss  # m in 1..K only
        if j<K: ns[m]+=d
        else: nc[m]+=d
        ns,nc = enforce_bcs(ns,nc)
        nV,nf,nfp = build(ns,nc)
        nJ = J_exact(nV,nf,nfp,sig)
        if nf.max() >= sig or nf.min() <= -sig: nJ = None  # enforce max|f| < sig
        if nJ is not None:
            dE=abs(nJ)-abs(Jc)
            if dE>0 or rng.random()<np.exp(beta*dE):
                a_s,a_c=ns,nc; Jc=nJ; acc+=1
                if abs(Jc)>bJ: bJ=abs(Jc); bs,bc=a_s.copy(),a_c.copy()
        Jt[i]=Jc; bt[i]=bJ
        if (i+1)%50000==0:
            print(f"  {i+1:>7d}/{n}  J={Jc:+.8f}  best={bJ:.8f}  sigma={sig}  "
                  f"acc={acc/(i+1):.3f}  [{time()-t0:.1f}s]")
    print(f"  done {time()-t0:.1f}s  acc={acc/n:.3f}")
    return bs, bc, Jt, bt, _seed_used

# ─── Run regimes ─────────────────────────
configs = [
    (0.95, int(2e6)),
]

results = {}; _seeds = []
for sig, ns in configs:
    bs,bc,Jt,bt,seed_used = mcmc(sig, ns, seed=None)
    _seeds.append(seed_used)
    V,f,fp = build(bs,bc)
    Je = J_exact(V,f,fp,sig)
    mf = np.max(np.abs(f))
    Je_str = f"{Je:+.10f}" if Je is not None else "None (|f|>=sig)"
    print(f"\n  sigma={sig}:  J_exact={Je_str}  max|f|={mf:.4f}\n")
    results[sig] = dict(bs=bs, bc=bc, Jt=Jt, bt=bt, V=V, f=f, fp=fp, Je=Je)

_tag = f"_L{_L_2pi}_K{K}_N{N}_seeds{'_'.join(str(s) for s in _seeds)}"

# ─── Save results to text file ────────────────────────────────────────────────
for sig in results:
    r = results[sig]
    outpath = f"{OUTDIR}/results_sig{sig}{_tag}.txt"
    with open(outpath, "w") as fh:
        fh.write(f"# Ratchet MCMC results\n")
        fh.write(f"# sigma={sig}  L={L/np.pi:.6g}pi  N={N}  K={K}\n")
        fh.write(f"# J_exact={r['Je']:+.10f}\n" if r['Je'] is not None else "# J_exact=None\n")
        fh.write(f"#\n")

        fh.write(f"# === PROFILE (N={N} rows): x  V(x)  f(x) ===\n")
        fh.write("# x  V  f\n")
        for xi, Vi, fi in zip(x, r["V"], r["f"]):
            fh.write(f"{xi:.10e}  {Vi:.10e}  {fi:.10e}\n")

        fh.write(f"#\n# === FOURIER COEFFICIENTS (K+1={K+1} rows): mode  a_s  a_c ===\n")
        fh.write("# mode  a_s  a_c\n")
        for m, (asi, aci) in enumerate(zip(r["bs"], r["bc"])):
            fh.write(f"{m}  {asi:.10e}  {aci:.10e}\n")

        fh.write(f"#\n# === MCMC J CHAIN ({len(r['Jt'])} rows): step  J  best_J ===\n")
        fh.write("# step  J  best_J\n")
        for i, (Ji, bJi) in enumerate(zip(r["Jt"], r["bt"])):
            fh.write(f"{i}  {Ji:.10e}  {bJi:.10e}\n")
    print(f"-> {outpath}")

'''
# ─── Figure 1: trajectories + final profiles (3x4 grid) ──────────────────────
cols = {0.95:"#dc2626"}
slbl = {s: f"$\\sigma={s}$" for s in cols}

fig = plt.figure(figsize=(6, 12))
fig.suptitle("Ratchet Current Optimization via MCMC\n"
             f"Exact steady-state formula  |  Run-and-tumble, $\\alpha=v=1$, period $L={L/np.pi:.4g}\\pi$",
             fontsize=15, fontweight="bold", y=0.99)
gs = GridSpec(3, 1, figure=fig, hspace=0.35, wspace=0.3)

for col, sig in enumerate([0.95]):
    r=results[sig]; c=cols[sig]

    # Row 0: V(x) final profile
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, r["V"], color=c, lw=1.8)
    Je_lbl = f"{r['Je']:+.6f}" if r['Je'] is not None else "N/A"
    ax.set_title(f"$V(x)$ — {slbl[sig]}\n$J/P_0={Je_lbl}$", fontsize=10)
    ax.set_xlabel("$x$", fontsize=9); ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.tick_params(labelsize=8); ax.grid(True, alpha=0.2)

    # Row 1: f(x) final profile
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, r["f"], color=c, lw=1.8)
    ax.axhline(sig, color="k", lw=0.6, ls=":"); ax.axhline(-sig, color="k", lw=0.6, ls=":")
    ax.set_title(f"$f(x)=-V'$ — {slbl[sig]}", fontsize=10)
    ax.set_xlabel("$x$", fontsize=9); ax.tick_params(labelsize=8); ax.grid(True, alpha=0.2)

    # Row 2: convergence trajectory
    ax = fig.add_subplot(gs[2, 0])
    thin = max(1, len(r["Jt"])//1500)
    ax.plot(r["Jt"][::thin], color=c, lw=0.3, alpha=0.4, label="$J/P_0$")
    ax.plot(r["bt"][::thin], color="k", lw=1.0, label="best $|J|$")
    ax.set_title(f"Convergence — {slbl[sig]}", fontsize=10)
    ax.set_xlabel("step", fontsize=9); ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="lower right"); ax.grid(True, alpha=0.2)

fig.savefig(f"{OUTDIR}/ratchet_trajectories_HMCMC April{_tag}.png",
            dpi=150, bbox_inches="tight")
#print("-> ratchet_trajectories.png")
'''