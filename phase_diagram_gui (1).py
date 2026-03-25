"""
Phase Diagram GUI for the Ratchet Current ODE
==============================================
Integrated form of the ODE (Eq. 53):

    F(V,p) = aV + (b/2)V^2 + (c/2)p^2 + (2d/3)p^3 + (4e/5)p^5 = 0

Autonomous system:
    dV/dx = p
    dp/dx = -(a + b*V + d*p^2 + e*p^4) / c

Sliders: a, b, c, d, e  (ODE coefficients)
         V0, p0          (initial condition)
         transient %     (fraction of trajectory to skip before plotting)

Run:
    pip install numpy matplotlib scipy
    /opt/homebrew/bin/python3 phase_diagram_gui.py

Written in collaboration w/ Claude Sonnet 4.6, Adele Basturk
"""

import sys
import traceback
import numpy as np

import matplotlib
for _b in ["MacOSX", "QtAgg", "Qt5Agg", "TkAgg"]:
    try:
        matplotlib.use(_b, force=True)
        import matplotlib.pyplot as _t; _t.figure(); _t.close("all")
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from scipy.integrate import odeint

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULTS = dict(a=0.1, b=0.5, c=1.0, d=-1.5, e=0.8, V0=0.3, p0=0.1, trans=0.5)

PARAM_RANGES = dict(
    a  = (-2.0,  2.0),
    b  = (-3.0,  3.0),
    c  = (-3.0,  3.0),
    d  = (-3.0,  3.0),
    e  = (-3.0,  3.0),
    V0 = (-2.4,  2.4),
    p0 = (-2.4,  2.4),
    trans = (0.0, 0.95),   # fraction of trajectory treated as transient
)

V_RANGE = (-2.5,  2.5)
P_RANGE = (-2.5,  2.5)
X_END   = 120.0            # longer integration so transient has time to die
N_STEPS = 2400
N_SEEDS = 8                # background grid seeds per axis
QUIV_N  = 18

COLORS = dict(
    bg        = "#0e0e18",
    panel     = "#12121f",
    spine     = "#2a2a44",
    tick      = "#6677aa",
    title     = "#c8d0f0",
    label     = "#8899bb",
    nullcl    = "#00e5ff",
    traj_bg   = "#6655aa",
    traj_main = "#f0c040",   # highlighted single trajectory in phase plane
    ic_dot    = "#ff4466",
    sample_V  = "#7ec8e3",
    sample_p  = "#98e09a",
    transient = "#444466",   # greyed-out transient portion
)

SL_COLORS = ["#5e81ac","#81a1c1","#88c0d0","#8fbcbb","#a3be8c",
             "#ebcb8b","#d08770","#bf616a"]


# ── ODE ──────────────────────────────────────────────────────────────────────

def rhs(y, x, a, b, c, d, e):
    V, p = float(y[0]), float(y[1])
    dpdx = 0.0 if abs(c) < 1e-12 else -(a + b*V + d*p*p + e*p**4) / c
    return [p, dpdx]


def integrate(V0, p0, a, b, c, d, e, x_end=X_END, n=N_STEPS):
    xs = np.linspace(0.0, x_end, n)
    try:
        sol = odeint(rhs, [V0, p0], xs, args=(a, b, c, d, e),
                     rtol=1e-6, atol=1e-8, mxstep=4000)
        return xs, sol[:, 0], sol[:, 1]
    except Exception:
        return None


def nullcline(a, b, c, d, e, npts=400):
    ps = np.linspace(P_RANGE[0], P_RANGE[1], npts)
    Vout, Pout = [], []
    for p in ps:
        tail = (c/2)*p**2 + (2*d/3)*p**3 + (4*e/5)*p**5
        A, B, C = b/2, a, tail
        if abs(A) < 1e-12:
            sols = [-C/B] if abs(B) > 1e-12 else []
        else:
            disc = B*B - 4*A*C
            if disc < 0: continue
            s = np.sqrt(disc)
            sols = [(-B+s)/(2*A), (-B-s)/(2*A)]
        for V in sols:
            if V_RANGE[0] <= V <= V_RANGE[1]:
                Vout.append(V); Pout.append(p)
    return np.array(Vout), np.array(Pout)


def quiver_field(a, b, c, d, e, n=QUIV_N):
    Vg = np.linspace(*V_RANGE, n)
    pg = np.linspace(*P_RANGE, n)
    VV, PP = np.meshgrid(Vg, pg)
    U = PP.copy()
    W = (-(a + b*VV + d*PP**2 + e*PP**4)/c
         if abs(c) > 1e-12 else np.zeros_like(U))
    mag = np.hypot(U, W) + 1e-12
    return Vg, pg, U/mag, W/mag, np.log1p(mag)


# ── App ───────────────────────────────────────────────────────────────────────

class PhaseApp:

    def __init__(self):
        self.p = dict(DEFAULTS)
        self._build()
        self._draw()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self):
        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor(COLORS["bg"])
        fig.canvas.manager.set_window_title("Ratchet ODE – Phase Portrait")
        self.fig = fig

        gs = gridspec.GridSpec(3, 2,
                               left=0.06, right=0.62,
                               top=0.93, bottom=0.07,
                               hspace=0.50, wspace=0.34)

        self.ax_ph = fig.add_subplot(gs[:, 0])   # phase plane (full height)
        self.ax_V  = fig.add_subplot(gs[0, 1])   # V(x)
        self.ax_p  = fig.add_subplot(gs[1, 1])   # p(x)
        self.ax_pp = fig.add_subplot(gs[2, 1])   # p vs V  (single trajectory, zoomed)

        for ax in (self.ax_ph, self.ax_V, self.ax_p, self.ax_pp):
            self._style(ax)

        self.ax_ph.set_title("Phase Plane", fontsize=11, fontweight="bold")
        self.ax_ph.set_xlabel("V"); self.ax_ph.set_ylabel("p = V′")

        self.ax_V.set_ylabel("V(x)")
        self.ax_p.set_ylabel("p(x)")
        self.ax_p.set_xlabel("x")
        self.ax_pp.set_xlabel("V"); self.ax_pp.set_ylabel("p")
        self.ax_pp.set_title("Single orbit (steady state)", fontsize=8)

        # ── sliders: two columns ──────────────────────────────────────────────
        # left column: a b c d e
        # right column: V0 p0 trans
        sl_specs = [
            # (name, col, row)
            ("a",     0, 0), ("b",     0, 1), ("c",     0, 2),
            ("d",     0, 3), ("e",     0, 4),
            ("V0",    1, 0), ("p0",    1, 1), ("trans", 1, 2),
        ]
        x0 = [0.66, 0.86]   # left edge of each column
        w, h = 0.11, 0.026
        y0, dy = 0.87, 0.095

        self.sliders = {}
        for i, (name, col, row) in enumerate(sl_specs):
            lo, hi = PARAM_RANGES[name]
            ax_sl = fig.add_axes([x0[col], y0 - row*dy, w, h],
                                 facecolor="#161626")
            label = "skip%" if name == "trans" else name
            sl = Slider(ax_sl, label, lo, hi,
                        valinit=self.p[name], color=SL_COLORS[i])
            sl.label.set_color(COLORS["title"])
            sl.valtext.set_color(COLORS["title"])
            sl.on_changed(self._changed)
            self.sliders[name] = sl

        # column headers
        for col, txt in zip(x0, ["ODE coeffs", "Initial cond / transient"]):
            fig.text(col, y0 + 0.025, txt, fontsize=7.5,
                     color=COLORS["label"], family="monospace")

        # reset button
        ax_btn = fig.add_axes([x0[0]+0.02, y0 - 5*dy - 0.01, 0.09, 0.036])
        self._btn = Button(ax_btn, "Reset", color="#1a1a2e", hovercolor="#2d2d50")
        self._btn.label.set_color(COLORS["title"])
        self._btn.on_clicked(self._reset)

        # legend text
        fig.text(0.66, 0.22,
                 "Cyan  = zero-energy curve\n"
                 "Gold  = chosen (V₀,p₀) orbit\n"
                 "──── = steady state\n"
                 "···· = transient (skip%)\n"
                 "skip% = fraction discarded\n"
                 "         before plotting",
                 fontsize=7.5, color="#555575",
                 family="monospace", va="top")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _style(self, ax):
        ax.set_facecolor(COLORS["panel"])
        for sp in ax.spines.values(): sp.set_color(COLORS["spine"])
        ax.tick_params(colors=COLORS["tick"], labelsize=8)
        ax.xaxis.label.set_color(COLORS["label"])
        ax.yaxis.label.set_color(COLORS["label"])
        ax.title.set_color(COLORS["title"])

    def _clear(self, ax):
        ax.cla(); self._style(ax)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _changed(self, _):
        for k in self.p: self.p[k] = float(self.sliders[k].val)
        self._draw()

    def _reset(self, _):
        for k, sl in self.sliders.items(): sl.set_val(DEFAULTS[k])

    # ── draw ──────────────────────────────────────────────────────────────────

    def _draw(self):
        a, b, c, d, e = (self.p[k] for k in "abcde")
        V0, p0, trans = self.p["V0"], self.p["p0"], self.p["trans"]

        # integrate chosen trajectory
        res = integrate(V0, p0, a, b, c, d, e)
        split = int(trans * N_STEPS)   # index where steady state begins

        # ── phase plane ───────────────────────────────────────────────────────
        ax = self.ax_ph
        self._clear(ax)
        ax.set_xlim(*V_RANGE); ax.set_ylim(*P_RANGE)
        ax.set_xlabel("V"); ax.set_ylabel("p = V′")
        ax.set_title("Phase Plane", fontsize=11, fontweight="bold")
        ax.axhline(0, color=COLORS["spine"], lw=0.5, ls="--")
        ax.axvline(0, color=COLORS["spine"], lw=0.5, ls="--")

        # vector field
        if abs(c) > 1e-12:
            try:
                Vg, pg, U, W, lmag = quiver_field(a, b, c, d, e)
                ax.quiver(Vg, pg, U, W, lmag, cmap="Blues",
                          alpha=0.45, scale=22, width=0.003, headwidth=4)
            except Exception: pass

        # background grid trajectories
        seeds = np.linspace(-2.0, 2.0, N_SEEDS)
        for _V0 in seeds:
            for _p0 in seeds:
                r = integrate(_V0, _p0, a, b, c, d, e)
                if r is None: continue
                _, Vt, pt = r
                ok = (np.abs(Vt) < 3.5) & (np.abs(pt) < 3.5)
                ax.plot(Vt[ok], pt[ok], lw=0.35, alpha=0.22,
                        color=COLORS["traj_bg"])

        # zero-energy curve
        try:
            Vc, Pc = nullcline(a, b, c, d, e)
            if len(Vc):
                ax.scatter(Vc, Pc, s=1.0, c=COLORS["nullcl"],
                           alpha=0.75, linewidths=0)
        except Exception: pass

        # chosen trajectory: transient (dotted) + steady (solid gold)
        if res is not None:
            xs, Vt, pt = res
            ok = (np.abs(Vt) < 3.5) & (np.abs(pt) < 3.5)
            # transient
            tr_ok = ok.copy(); tr_ok[split:] = False
            ax.plot(Vt[tr_ok], pt[tr_ok], lw=0.7, ls=":",
                    alpha=0.5, color=COLORS["traj_main"])
            # steady state
            ss_ok = ok.copy(); ss_ok[:split] = False
            ax.plot(Vt[ss_ok], pt[ss_ok], lw=1.2,
                    alpha=0.9, color=COLORS["traj_main"])

        # initial condition dot
        ax.scatter([V0], [p0], s=40, c=COLORS["ic_dot"],
                   zorder=5, linewidths=0)

        # ── V(x) and p(x) ────────────────────────────────────────────────────
        for ax2 in (self.ax_V, self.ax_p, self.ax_pp):
            self._clear(ax2)

        if res is not None:
            xs, Vt, pt = res

            # transient shading boundary
            x_split = xs[split] if split < len(xs) else xs[-1]

            for ax2, yt, col, lbl in [
                (self.ax_V, Vt, COLORS["sample_V"], "V(x)"),
                (self.ax_p, pt, COLORS["sample_p"], "p(x)"),
            ]:
                # transient portion
                ax2.plot(xs[:split], yt[:split], lw=0.8, ls=":",
                         alpha=0.45, color=COLORS["transient"])
                # steady portion
                ax2.plot(xs[split:], yt[split:], lw=1.0, color=col)
                ax2.axvline(x_split, color="#ff4466", lw=0.7,
                            ls="--", alpha=0.6)
                ax2.axhline(0, color=COLORS["spine"], lw=0.4, ls="--")
                ax2.set_xlim(0, X_END)
                ax2.set_ylabel(lbl, color=COLORS["label"])

            self.ax_V.set_title(
                f"Trajectory  V₀={V0:.2f}, p₀={p0:.2f}  "
                f"[skip first {trans*100:.0f}%]",
                fontsize=8)
            self.ax_p.set_xlabel("x", color=COLORS["label"])

            # steady-state orbit in its own phase panel
            ax_pp = self.ax_pp
            Vss, pss = Vt[split:], pt[split:]
            ok = (np.abs(Vss) < 3.5) & (np.abs(pss) < 3.5)
            ax_pp.plot(Vss[ok], pss[ok], lw=1.0,
                       color=COLORS["traj_main"], alpha=0.85)
            ax_pp.set_xlabel("V", color=COLORS["label"])
            ax_pp.set_ylabel("p", color=COLORS["label"])
            ax_pp.set_title("Single orbit (steady state)", fontsize=8)
            ax_pp.axhline(0, color=COLORS["spine"], lw=0.4, ls="--")
            ax_pp.axvline(0, color=COLORS["spine"], lw=0.4, ls="--")

        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        app = PhaseApp()
        app.run()
    except Exception:
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  /opt/homebrew/bin/pip3 install numpy matplotlib scipy")
        sys.exit(1)
