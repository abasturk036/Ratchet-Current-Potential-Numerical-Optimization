"""
MCMC optimization of ratchet current J/P0 — exact formula (Eq. 14).
Run-and-tumble particle in periodic potential V(x), alpha = v = 1.

Fourier-parameterized V(x), fixed max force sigma = max|f|, with f=-V'.
Four regimes compared.

Perturbative formula verified against direct ODE solution.

Adele Basturk, written in collaboration with Claude Code Opus 4.6 4/6/2026.
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from time import time

L, N, K = 40*np.pi, 512, 32
dx = L/N; x = np.linspace(0, L, N, endpoint=False)
_Lpi = f"{round(L/np.pi)}pi"
_tag = f"_L{_Lpi}_K{K}_N{N}"   # appended to every output filename; updated with seeds after MCMC
k_idx = np.arange(1, K+1); omega = 2*np.pi*k_idx/L
SIN = np.sin(np.outer(omega, x)); COS = np.cos(np.outer(omega, x))

#Build's V(x) profile given Fourier coefficients a_s, a_c, and also computes f=-V' and fp=-V'' for convenience.
def build(a_s, a_c):
    V  = a_s@SIN + a_c@COS
    f  = -(a_s*omega)@COS + (a_c*omega)@SIN
    fp = (a_s*omega**2)@SIN + (a_c*omega**2)@COS
    return V, f, fp

#returns the peak force amplitude for given Fourier coefficients.
def rough(a_s, a_c):
    _, f, _ = build(a_s, a_c)
    return np.max(np.abs(f))

#a_s and A_c are the Fourier coefficients of V(k) = a_s*sin(2πkx/L) + a_c*cos(2πkx/L).

def enforce_bcs(a_s, a_c):
    """
    Enforce V(0)=0 and f(0)=0.
      V(0) = sum(a_c) = 0  =>  subtract uniform component of a_c
      f(0) = -dot(a_s, omega) = 0  =>  subtract component of a_s along omega
    """
    a_c = a_c - np.sum(a_c) / K
    a_s = a_s - (np.dot(a_s, omega) / np.dot(omega, omega)) * omega
    return a_s, a_c

#rescale Fourier coefficients so that max|f| = sig.
def proj(a_s, a_c, sig):
    a_s, a_c = enforce_bcs(a_s, a_c)
    r = rough(a_s, a_c)
    if r < 1e-30: return a_s, a_c
    s = sig/r; return a_s*s, a_c*s

#compute J from equation 14 in Systematic Ratchet Current - eq 14
def J_exact(V, f, fp, f_max=0.995):
    """Exact current from Eq. (14)."""
    if np.max(np.abs(f)) >= f_max: return None
    id_ = 1.0/(1-f*f); A = (2*f*fp+f)*id_; b = (fp-1)*id_
    mA = 0.5*(A[:-1]+A[1:])*dx
    Phi = np.zeros(N); np.cumsum(mA, out=Phi[1:])
    PL = Phi[-1]+0.5*(A[-1]+A[0])*dx
    xi = np.exp(PL); eP = np.exp(Phi)
    num = L*(1-xi)
    if abs(num)<1e-30: return 0.0
    Ib = np.trapezoid(b*eP, dx=dx)
    Ix = np.trapezoid(eP, dx=dx)
    be = b*np.exp(-Phi); mb = 0.5*(be[:-1]+be[1:])*dx
    G = np.zeros(N); np.cumsum(mb, out=G[1:])
    Id = np.trapezoid(eP*G, dx=dx)
    den = Ib*Ix + (1-xi)*Id
    return num/den if abs(den)>1e-30 else 0.0

def ode_coeffs(V, f):
    """
    Compute ODE coefficients a,b,c,d,e from the optimal V(x), f(x)=-V'(x)
    via Eqs. 37-42.  p(x) = -f(x) = V'(x).
      I2  = integral of p^2,  I3 = integral of p^3
      Ib1 = integral of V,    Ib2 = integral of V^2
      a = (1/L)*I3*(2 + 6*Ib1^2/L)     [Eq. 37]
      b = (2/L)*I3                       [Eq. 38]
      c = 2*b  (= (4/L)*I3)             [Eq. 39]
      d = 1 - 2*Ib1/L - 2*I2/L
            + 3*Ib1^2/L^2 - Ib2/L      [Eq. 40]
      e = 5                              [Eq. 42, fixed]
    """
    p   = -f
    I2  = np.trapezoid(p**2, dx=dx)
    I3  = np.trapezoid(p**3, dx=dx)
    Ib1 = np.trapezoid(V,    dx=dx)
    Ib2 = np.trapezoid(V**2, dx=dx)
    a = (1/L)*I3*(2 + 6*Ib1**2/L)
    b = (2/L)*I3
    c = 2*b
    d = 1 - 2*Ib1/L - 2*I2/L + 3*Ib1**2/L**2 - Ib2/L
    e = 5.0
    return a, b, c, d, e

def ode_validity(V, f):
    """
    Plug the MCMC-optimal V into the emergent nonlinear ODE (Eq. 36):
        dV/dx = c*p + 2d*p^2 + (16e/5)*p^3
        dp/dx = -(a + b*V)
    with coefficients from Eqs. 37-42.

    Integrates the ODE over [0, L] starting from V(0), p(0) and compares
    the orbit to the original V to assess self-consistency in the small-
    forcing limit.

    Returns dict with coefficients, ODE solution arrays, and residual norms.
    """
    from scipy.integrate import solve_ivp

    a, b, c, d, e = ode_coeffs(V, f)
    p0 = -f  # p = V'

    def rhs(_, y):
        Vy, py = y
        dV = c*py + 2*d*py**2 + (16/5)*e*py**3
        dp = -(a + b*Vy)
        return [dV, dp]

    sol = solve_ivp(rhs, [0, L], [V[0], p0[0]],
                    t_eval=x, method="RK45",
                    rtol=1e-9, atol=1e-12)

    V_ode = sol.y[0] if sol.success else np.full(N, np.nan)
    p_ode = sol.y[1] if sol.success else np.full(N, np.nan)

    # residuals: how well does the ODE orbit reproduce the input V?
    res_V = np.sqrt(np.mean((V_ode - V)**2)) / (np.std(V) + 1e-30)
    res_p = np.sqrt(np.mean((p_ode - p0)**2)) / (np.std(p0) + 1e-30)

    # periodicity of the ODE orbit
    per_V = abs(V_ode[-1] - V_ode[0]) / (np.std(V_ode) + 1e-30)
    per_p = abs(p_ode[-1] - p_ode[0]) / (np.std(p_ode) + 1e-30)

    return dict(a=a, b=b, c=c, d=d, e=e,
                V_ode=V_ode, p_ode=p_ode,
                res_V=res_V, res_p=res_p,
                per_V=per_V, per_p=per_p,
                ode_success=sol.success)

def J_p3(V, f):
    """O(f^3): sign verified against direct ODE solver."""
    return (1.0/L)*np.trapezoid(f**3, dx=dx)

def J_p5(V, f):
    """O(f^5): corrected sign."""
    I3 = np.trapezoid(f**3,dx=dx); I5 = np.trapezoid(f**5,dx=dx)
    IV = np.trapezoid(V,dx=dx); If2= np.trapezoid(f**2,dx=dx)
    IV2= np.trapezoid(V**2,dx=dx)
    return (1.0/L)*(
        I3 + I5
        - (2.0/L)*I3*IV
        - (2.0/L)*I3*If2
        + (3.0/L**2)*I3*IV**2
        - (1.0/L)*I3*IV2
    )

# ─── Quick verification ─────────────────
print("="*65)
print("  SIGN VERIFICATION (direct ODE vs exact vs perturbative)")
print("="*65)
for eps in [0.05, 0.10, 0.20]:
    V = eps*(np.sin(x)+0.3*np.sin(2*x))
    f = -eps*(np.cos(x)+0.6*np.cos(2*x))
    fp = eps*(np.sin(x)+1.2*np.sin(2*x))
    Je = J_exact(V,f,fp)
    J3 = J_p3(V,f)
    J5 = J_p5(V,f)
    # Direct solve
    id_=1/(1-f*f); A_=(2*f*fp+f)*id_; b_=(fp-1)*id_
    M = np.zeros((N+1,N+1)); rhs = np.zeros(N+1)
    for i in range(N):
        ip=(i+1)%N; im=(i-1)%N
        M[i,ip]+=1/(2*dx); M[i,im]-=1/(2*dx); M[i,i]-=A_[i]; M[i,N]=-b_[i]
    M[N,:N]=dx/L; rhs[N]=1.0
    Jd = np.linalg.solve(M,rhs)[N]
    print(f"  eps={eps:.2f}  J_direct={Jd:+.8f}  J_exact={Je:+.8f}  "
          f"J_O3={J3:+.8f}  J_O5={J5:+.8f}")
print()

# ─── MCMC ────────────────────────────────
def mcmc(sig, n=int(2e5), seed=42):
    rng = np.random.default_rng(seed)
    _seed_used = rng.bit_generator.state["state"]["state"]  # actual 128-bit state; use low 32 bits for filename
    _seed_used = int(_seed_used) & 0xFFFFFFFF
    print(f"{'='*55}  sigma={sig}  seed={_seed_used}")
    a_s = rng.normal(0,1,K); a_c = rng.normal(0,1,K)
    #a_s, a_c = proj(a_s, a_c, sig)
    V,f,fp = build(a_s, a_c)
    Jc = J_exact(V,f,fp,sig) or 0.0
    if Jc > 1: print(Jc)
    bJ, bs, bc = abs(Jc), a_s.copy(), a_c.copy()
    Jt = np.empty(n); bt = np.empty(n); acc=0
    ms = 1.0/np.sqrt(k_idx.astype(float))
    # Calibrated: b0=200, b1=50000 worked for sig=0.1, L=2*pi.
    # Scale by (sig_ref/sig)^3 * (L/L_ref) to keep beta*J_typ constant across regimes.
    sig_ref, L_ref = 0.10, 2*np.pi
    scale = (sig_ref/sig)**4 * (L/L_ref)
    b0 = 200 * scale; b1 = 500000 * scale
    ss = 0.06*sig
    t0 = time()
    for i in range(n):
        kappa = np.log(b1/b0) / n; beta = b0*np.exp(kappa*i); sz = ss

        ns,nc = a_s.copy(),a_c.copy()
        j=rng.integers(2*K); m=j%K; d=rng.normal()*sz*ms[m]
        if j<K: ns[m]+=d
        else: nc[m]+=d
        ns,nc = proj(ns,nc,sig)
        nV,nf,nfp = build(ns,nc)
        nJ = J_exact(nV,nf,nfp,sig)
        if nJ is not None:
            dE=abs(nJ)-abs(Jc)
            if dE>0 or rng.random()<np.exp(beta*dE):
                a_s,a_c=ns,nc; Jc=nJ; acc+=1
                if abs(Jc)>bJ: bJ=abs(Jc); bs,bc=a_s.copy(),a_c.copy()
        Jt[i]=Jc; bt[i]=bJ
        if (i+1)%50000==0:
            print(f"  {i+1:>7d}/{n}  J={Jc:+.8f}  best={bJ:.8f}  "
                  f"acc={acc/(i+1):.3f}  [{time()-t0:.1f}s]")
    print(f"  done {time()-t0:.1f}s  acc={acc/n:.3f}")
    return bs, bc, Jt, bt, _seed_used

# ─── Run regimes ─────────────────────────
configs = [
    (0.10, int(2e5)),
    (0.40, int(2e5)),
    (0.60, int(2e5)),
    (0.95, int(2e5)),
]

results = {}; _seeds = []
for sig, ns in configs:
    bs,bc,Jt,bt,seed_used = mcmc(sig, ns, seed=None)
    _seeds.append(seed_used)
    V,f,fp = build(bs,bc)
    Je = J_exact(V,f,fp,sig)
    J3 = J_p3(V,f); J5 = J_p5(V,f)
    ode = ode_validity(V, f)
    results[sig] = dict(bs=bs,bc=bc,Jt=Jt,bt=bt,V=V,f=f,fp=fp,
                        Je=Je,J3=J3,J5=J5,ode=ode)
    mf = np.max(np.abs(f))
    re3 = abs(J3-Je)/abs(Je)*100 if (Je is not None and abs(Je)>1e-15) else float('nan')
    re5 = abs(J5-Je)/abs(Je)*100 if (Je is not None and abs(Je)>1e-15) else float('nan')
    Je_str = f"{Je:+.10f}" if Je is not None else "None (|f|>=sig)"
    print(f"\n  sigma={sig}:  J_exact={Je_str}  J_O3={J3:+.10f}  J_O5={J5:+.10f}")
    print(f"    max|f|={mf:.4f}  rel_err_O3={re3:.1f}%  rel_err_O5={re5:.1f}%")
    print(f"  ODE (Eq.36) validity  [small-forcing self-consistency check]")
    print(f"    a={ode['a']:+.6f}  b={ode['b']:+.6f}  c={ode['c']:+.6f}  "
          f"d={ode['d']:+.6f}  e={ode['e']:.1f}")
    print(f"    residual V: {ode['res_V']:.4f}  residual p: {ode['res_p']:.4f}  "
          f"(normalised RMS; <0.05 = self-consistent)")
    print(f"    periodicity ΔV/std: {ode['per_V']:.4f}  Δp/std: {ode['per_p']:.4f}  "
          f"(< 0.05 = periodic orbit)")
    if not ode['ode_success']:
        print(f"    WARNING: ODE integration failed")
    print()
_tag = f"_L{_Lpi}_K{K}_N{N}_seeds{'_'.join(str(s) for s in _seeds)}"

# ─── Figure 1: main results (2x4 grid) ──
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Ratchet Current Optimization via MCMC\n"
             f"Exact steady-state formula  |  Run-and-tumble, $\\alpha=v=1$, period $L={L/np.pi:.4g}\\pi$",
             fontsize=15, fontweight="bold", y=0.99)
gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

cols = {0.10:"#16a34a", 0.40:"#2563eb", 0.60:"#f59e0b", 0.95:"#dc2626"}
slbl = {s: f"$\\sigma={s}$" for s in cols}

for col, sig in enumerate([0.10, 0.40, 0.60, 0.95]):
    r=results[sig]; c=cols[sig]
    
    ax = fig.add_subplot(gs[0, col])
    ax.plot(x, r["V"], color=c, lw=1.8)
    Je_lbl = f"{r['Je']:+.6f}" if r['Je'] is not None else "N/A"
    ax.set_title(f"$V(x)$ — {slbl[sig]}\n$J/P_0={Je_lbl}$", fontsize=10)
    ax.set_xlabel("$x$", fontsize=9); ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.tick_params(labelsize=8); ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[1, col])
    ax.plot(x, r["f"], color=c, lw=1.8)
    ax.axhline(sig, color="k", lw=0.6, ls=":"); ax.axhline(-sig, color="k", lw=0.6, ls=":")
    ax.set_title(f"$f(x)=-V'$ — {slbl[sig]}", fontsize=10)
    ax.set_xlabel("$x$", fontsize=9); ax.tick_params(labelsize=8); ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[2, col])
    thin = max(1, len(r["Jt"])//1500)
    ax.plot(r["Jt"][::thin], color=c, lw=0.3, alpha=0.4, label="$J/P_0$")
    ax.plot(r["bt"][::thin], color="k", lw=1.0, label="best $|J|$")
    ax.set_title(f"Convergence — {slbl[sig]}", fontsize=10)
    ax.set_xlabel("step", fontsize=9); ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="lower right"); ax.grid(True, alpha=0.2)

plt.savefig(f"/Users/Adele/Downloads/Systematic Ratchet Current/ratchet_trajectories_HMCMC April{_tag}.png", dpi=150, bbox_inches="tight")
print("-> ratchet_main.png")

# ─── Figure 2: comparison table ──────────
fig2, ax2 = plt.subplots(figsize=(12, 4)); ax2.axis("off")
rows = []
for sig in [0.10, 0.40, 0.60, 0.95]:
    r=results[sig]; mf=np.max(np.abs(r['f'])); Je=r['Je']
    re3 = abs(r['J3']-Je)/abs(Je)*100 if (Je is not None and abs(Je)>1e-15) else float('nan')
    re5 = abs(r['J5']-Je)/abs(Je)*100 if (Je is not None and abs(Je)>1e-15) else float('nan')
    Je_str = f"{Je:+.8f}" if Je is not None else "N/A"
    rows.append([f"{sig}", Je_str, f"{r['J3']:+.8f}",
                 f"{r['J5']:+.8f}", f"{re3:.1f}%", f"{re5:.1f}%",
                 f"{mf:.3f}"])
tbl = ax2.table(cellText=rows,
    colLabels=["$\\sigma$", "$J/P_0$ exact", "$J/P_0$ O$(f^3)$",
               "$J/P_0$ O$(f^5)$", "Err O(3)", "Err O(5)", "max$|f|$"],
    loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.0, 1.8)
for (r_,c_), cell in tbl.get_celld().items():
    if r_==0: cell.set_facecolor("#dbeafe"); cell.set_text_props(fontweight="bold")
fig2.suptitle("Exact vs. Perturbative Current at MCMC-Optimal Shape\n"
              "(Sign-verified against direct ODE solution)",
              fontsize=13, fontweight="bold")
fig2.tight_layout()
fig2.savefig(f"/Users/Adele/Downloads/Systematic Ratchet Current/ratchet_exact_v_perturbative_HMCMC April{_tag}.png", dpi=150, bbox_inches="tight")
print("-> ratchet_table.png")

'''
# ─── Figure 3: density for two regimes ───
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

for ax, sig, c in [(ax3a, 0.10, "#2563eb"), (ax3b, 0.30, "#dc2626")]:
    r=results[sig]; V,f,fp = r["V"],r["f"],r["fp"]
    id_=1/(1-f*f); Aa=(2*f*fp+f)*id_; bb=(fp-1)*id_
    mA=0.5*(Aa[:-1]+Aa[1:])*dx
    Ph=np.zeros(N); np.cumsum(mA,out=Ph[1:])
    PL=Ph[-1]+0.5*(Aa[-1]+Aa[0])*dx
    eP=np.exp(Ph); xi=np.exp(PL)
    Ib=np.trapezoid(bb*eP,dx=dx); Cv=(1-xi)/Ib if abs(Ib)>1e-30 else 0
    be_=bb*np.exp(-Ph); mb_=0.5*(be_[:-1]+be_[1:])*dx
    G_=np.zeros(N); G_[0]=0; np.cumsum(mb_,out=G_[1:])
    Pr=eP+Cv*eP*G_; Pr/=np.mean(Pr)
    ax.fill_between(x,0,Pr,color=c,alpha=0.12)
    ax.plot(x,Pr,color=c,lw=2,label="$P(x)$ steady-state")
    Vs=V/np.max(np.abs(V))*np.max(Pr)*0.3
    ax.plot(x,Vs,color="grey",lw=1.5,ls="--",alpha=0.7,label="$V(x)$ (scaled)")
    ax.set_xlabel("$x$",fontsize=12); ax.set_ylabel("density",fontsize=12)
    Je_lbl = f"{r['Je']:+.4f}" if r['Je'] is not None else "N/A"
    ax.set_title(f"Steady-state density ($\\sigma={sig}$, $J/P_0={Je_lbl}$)",fontsize=12)
    ax.legend(fontsize=10); ax.grid(True,alpha=0.25)

fig3.tight_layout()
fig3.savefig(f"/Users/Adele/Downloads/Systematic Ratchet Current/ratchet_density_HMCMC April{_tag}.png",dpi=150,bbox_inches="tight")
print("-> ratchet_density.png")
'''

# ─── Figure 4: Eq. 36 residual table ─────────────────────────────────────────
# Eq. 36: a + bV + c*V'' + d*∂x(V')² + e*∂x(V')⁴ = 0
# V' = -f,  V'' = -fp,  ∂x(V')² = ∂x(f²) = 2f·fp,  ∂x(V')⁴ = 4f³·fp
fig4, ax4 = plt.subplots(figsize=(13, 3)); ax4.axis("off")
ode_rows = []
for sig in [0.10, 0.40, 0.60, 0.95]:
    o  = results[sig]["ode"]
    V_ = results[sig]["V"]
    f_ = results[sig]["f"]
    fp_= results[sig]["fp"]
    a_, b_, c_, d_, e_ = o['a'], o['b'], o['c'], o['d'], o['e']
    R  = a_ + b_*V_ + c_*(-fp_) + d_*(2*f_*fp_) + e_*(4*f_**3*fp_)
    ode_rows.append([
        f"{sig}",
        f"{a_:+.5f}", f"{b_:+.5f}", f"{c_:+.5f}",
        f"{d_:+.5f}", f"{e_:.1f}",
        f"{np.sqrt(np.mean(R**2)):.3e}",
    ])
col_labels = [
    r"$\sigma$",
    r"$a$", r"$b$", r"$c$", r"$d$", r"$e$",
    r"$\|R\|_\mathrm{rms}$",
]
tbl4 = ax4.table(cellText=ode_rows, colLabels=col_labels,
                 loc="center", cellLoc="center")
tbl4.auto_set_font_size(False); tbl4.set_fontsize(11); tbl4.scale(1.0, 2.2)
for (r_, c_), cell in tbl4.get_celld().items():
    if r_ == 0:
        cell.set_facecolor("#dbeafe"); cell.set_text_props(fontweight="bold")

fig4.suptitle(
    r"Eq. 36: $a + bV + c\,\partial_x^2 V + d\,\partial_x(\partial_x V)^2 + e\,\partial_x(\partial_x V)^4 = 0$"
    "\n" r"Coefficients from Eqs. 37–42, evaluated at MCMC-optimal $V(x)$",
    fontsize=12, fontweight="bold",
)
fig4.tight_layout()
fig4.savefig(f"/Users/Adele/Downloads/Systematic Ratchet Current/ratchet_ode_residuals_HMCMC April{_tag}.png",
             dpi=150, bbox_inches="tight")
print("-> ratchet_ode_residuals.png")

# ─── Figure 5: Fourier mode structure ────────────────────────────────────────
# Power in each mode k: S(k) = (a_s[k]^2 + a_c[k]^2) / 2, normalised by L
# so that sum_k S(k) = sigma^2 (roughness per unit length)
fig5, axes5 = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
fig5.suptitle(
    "Fourier Mode Structure of MCMC-Optimal $V(x)$\n"
    r"Power per mode $S_k = (a_k^2 + b_k^2)/2$",
    fontsize=13, fontweight="bold"
)

for ax, sig in zip(axes5.flat, [0.10, 0.40, 0.60, 0.95]):
    r  = results[sig]
    bs = r["bs"]; bc = r["bc"]
    Sk = (bs**2 + bc**2) / 2                 # power per mode
    total = np.sum(Sk)

    bars = ax.bar(k_idx, Sk, color=cols[sig], alpha=0.75, edgecolor="k", lw=0.5)
    ax.set_title(f"$\\sigma={sig}$,  $\\sum S_k = {total:.4f}$", fontsize=11)
    ax.set_xlabel("mode $k$", fontsize=10)
    ax.set_ylabel("$S_k$", fontsize=10)
    ax.set_xticks(k_idx)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    # annotate dominant mode
    kdom = k_idx[np.argmax(Sk)]
    ax.annotate(f"$k={kdom}$", xy=(kdom, Sk[kdom-1]),
                xytext=(kdom + 0.3, Sk[kdom-1]*1.05),
                fontsize=9, color="k",
                arrowprops=dict(arrowstyle="-", lw=0.8))

fig5.tight_layout()
fig5.savefig(f"/Users/Adele/Downloads/Systematic Ratchet Current/ratchet_fourier_modes_HMCMC April{_tag}.png",
             dpi=150, bbox_inches="tight")
print("-> ratchet_fourier_modes.png")
