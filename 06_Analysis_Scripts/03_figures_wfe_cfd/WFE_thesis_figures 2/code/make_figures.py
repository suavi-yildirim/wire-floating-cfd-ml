"""
Thesis figure set - Wire Floating Experiment (WFE), Tube Flow Analysis (TFA), CFD.
Run:  python3 make_figures.py [fig01 fig02 ...]      (no args = all)
      WFE_DATA=/path/to/Thesis_Data python3 make_figures.py   (if the data is not in ../data)
Outputs go to style.OUT as vector PDF (for LaTeX) + 400-dpi PNG (for slides/preview).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

import style as S
from loaders import (DA_C10_FULL, TFA_CONFIGS, WFE_DIR, parse_wfe, tfa_points, uid_to_geom,
                     wfe_run_table, cfd_cases, cfd_history)
from physics import cfd_table

S.apply()
CACHE = Path(__file__).resolve().parent / ".cache"
CACHE.mkdir(exist_ok=True)

# ======================================================================================
# shared data
# ======================================================================================
_RUNS = None


def runs() -> pd.DataFrame:
    """All stable WFE runs with a known tube channel (793 of 794 raw-stable runs)."""
    global _RUNS
    if _RUNS is None:
        p = CACHE / "cache_runs.pkl"
        if p.exists():
            df = pd.read_pickle(p)
        else:
            df = wfe_run_table()          # parses all raw WFE logs (~10 s)
            df.to_pickle(p)
        df = df[df["stable"] & df["p_mbar"].notna() & df["D"].isin([8.0, 10.0])].copy()
        df["tube"] = np.where(df["D"] == 8.0, "WC08", "WC10")
        df["section"] = df["uid"].map(lambda u: uid_to_geom(int(u))[0])
        df["length"] = df["uid"].map(lambda u: uid_to_geom(int(u))[1])
        _RUNS = df
    return _RUNS


def _ts_time(s: str) -> pd.Timestamp:
    s = str(s).strip().replace(",", ".")
    return pd.to_datetime(s, format="%d.%m.%Y %H:%M:%S.%f", errors="coerce")


def quad0(q, a, b):
    """Porous-media / system characteristic  dp = a Q^2 + b Q  (through the origin)."""
    return a * q ** 2 + b * q


# ======================================================================================
# Fig 01 - anatomy of one levitation experiment (raw 50-ms log)
# ======================================================================================
def fig01_run_anatomy(file="WFE_19_2775_20260326_1135.csv"):
    r = parse_wfe(WFE_DIR / file)
    ts, m = r.ts, r.meta
    t0 = _ts_time(m["Experiment_Start_Time"])
    st = r.stable.copy()
    st["t_start"] = (st["Start_Time"].map(_ts_time) - t0).dt.total_seconds()
    st["t_end"] = st["t_start"] + st["Duration"] / 1000
    w = st["Duration"]
    p_bar = (st["Pressure_Stable_Average"] * w).sum() / w.sum()
    q_bar = (st["Flow_Stable_Average"] * w).sum() / w.sum() / 100
    t_lift = (_ts_time(r.increases["Increase_Time"].iloc[0]) - t0).total_seconds()
    p_lift = float(r.increases["Increase_Pressure"].iloc[0])
    sensors = [0.027, 0.055, 0.095, 0.135, 0.175]

    fig, axs = plt.subplots(3, 1, figsize=(S.TEXTWIDTH, 10.2 * S.CM), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.35, 1.0, 1.0], hspace=0.42))
    STAB = "#E6EED8"
    for ax in axs:
        for _, ph in st.iterrows():
            ax.axvspan(ph.t_start, ph.t_end, color=STAB, lw=0, zorder=0)
        ax.axvline(t_lift, color=S.INK3, lw=0.6, zorder=1)

    # (a) pressure
    a = axs[0]
    a.plot(ts.t_s, ts.p_set_mbar, color="#9AA5B1", lw=0.9, label="set point $p_\\mathrm{set}$")
    a.plot(ts.t_s, ts.p_mbar, color=S.INK, lw=1.1, label="measured $p$")
    for _, ph in st.iterrows():
        a.hlines(ph.Pressure_Stable_Average, ph.t_start, ph.t_end, color=S.FAPS_GREEN, lw=1.8, zorder=4)
    a.text(0.60, 0.86, f"$\\bar p_\\mathrm{{stable}}$ = {p_bar:.0f} mbar (duration-weighted)",
           transform=a.transAxes, fontsize=8, color="#4A6B12", ha="left")
    a.plot([t_lift], [p_lift], "o", ms=3.5, color=S.INK, mec="white", mew=0.6, zorder=5)
    a.annotate(f"lift-off detected: {p_lift:.0f} mbar", (t_lift, p_lift), xytext=(3.2, 4300),
               textcoords="data", fontsize=7.5, color=S.INK2, va="center",
               arrowprops=dict(arrowstyle="-", color=S.INK3, lw=0.6, shrinkA=1, shrinkB=2))
    a.set_ylabel("Inlet pressure\n$p$ [mbar]")
    a.set_ylim(0, 4600)
    a.yaxis.set_major_locator(mticker.MultipleLocator(1000))
    a.legend(loc="lower right", ncol=2, bbox_to_anchor=(1.0, 0.0))
    S.panel_title(a, "Pressure-controlled levitation, WC10 · UID 19 (4.0 mm$^2$, 2.0 m)", "a")

    # (b) flow
    b = axs[1]
    b.plot(ts.t_s, ts.Q_Nm3h, color=S.INK, lw=1.1)
    for _, ph in st.iterrows():
        b.hlines(ph.Flow_Stable_Average / 100, ph.t_start, ph.t_end, color=S.FAPS_GREEN, lw=1.8, zorder=4)
    b.text(0.60, 0.86, f"$\\bar{{\\dot V}}_\\mathrm{{N}}$ = {q_bar:.1f} Nm$^3$/h", transform=b.transAxes,
           fontsize=8, color="#4A6B12")
    b.set_ylabel("Flow rate\n$\\dot V_\\mathrm{N}$ [Nm$^3$/h]")
    b.set_ylim(0, 40)
    S.panel_title(b, "Volumetric flow (standard conditions)", "b")

    # (c) wire-tip position from the light-barrier array
    c = axs[2]
    tmin, tmax = float(m["Target_Min"]) * 1000, float(m["Target_Max"]) * 1000
    c.axhspan(tmin, tmax, color="#DDE6F0", lw=0, zorder=0.5)
    for i, y in enumerate(sensors, 1):
        c.axhline(y * 1000, color=S.INK3, lw=0.5, zorder=1)
        c.text(1.006, y * 1000, f"S{i}", transform=c.get_yaxis_transform(), fontsize=7,
               color=S.INK3, va="center")
    c.step(ts.t_s, ts.pos_m * 1000, where="post", color=S.INK, lw=1.2, zorder=3)
    c.set_ylabel("Wire-tip position\n$z$ [mm]")
    c.set_ylim(-8, 250)
    c.text(8.0, (tmin + tmax) / 2, "target window", fontsize=7.5, color="#3A5A80", va="center", ha="left",
           bbox=dict(boxstyle="square,pad=0.15", fc="#DDE6F0", ec="none"), zorder=2)
    S.panel_title(c, "Wire position quantised by five light barriers", "c")
    c.set_xlabel("Time $t$ [s]")
    c.set_xlim(0, ts.t_s.iloc[-1])
    c.xaxis.set_major_locator(mticker.MultipleLocator(10))

    # phase legend (shared)
    hs = [Patch(color=STAB, label="stable phase ($\\geq$1 s, wire inside target window)"),
          Line2D([], [], color=S.FAPS_GREEN, lw=1.8, label="phase average"),
          Line2D([], [], color=S.INK3, lw=0.6, label="first barrier activation")]
    fig.legend(handles=hs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.045), frameon=False)
    for ax in axs:
        ax.grid(axis="x", visible=False)
    fig.align_ylabels(axs)
    return S.save(fig, "fig01_wfe_run_anatomy")


# --------------------------------------------------------------------------------------
# TFA helpers
# --------------------------------------------------------------------------------------
from scipy.interpolate import PchipInterpolator  # noqa: E402

_TFA = {}


def tfa(key: str):
    """Valid quasi-static TFA points of one configuration and a monotone PCHIP curve p(Q).
    Both ramp files (100 mbar and 20 mbar steps) are pooled; points above flow-meter
    full scale are dropped (see loaders.Q_FULLSCALE)."""
    if isinstance(key, (list, tuple)):
        files, key = list(key), "|".join(key)
    else:
        files = TFA_CONFIGS[key]
    if key not in _TFA:
        d = tfa_points(files)
        d = d[d.valid].copy()
        b = d.groupby(d.Q_Nm3h.round(2)).p_mbar.mean().reset_index().sort_values("Q_Nm3h")
        b = b[b.p_mbar.cummax() == b.p_mbar]
        x = np.r_[0.0, b.Q_Nm3h.values]
        y = np.r_[0.0, b.p_mbar.values]
        x, i = np.unique(x, return_index=True)
        _TFA[key] = (d, PchipInterpolator(x, y[i], extrapolate=False), float(x.max()))
    return _TFA[key]


def operating_window(tube: str):
    g = runs()[runs().tube == tube].Q_Nm3h
    return float(g.min()), float(g.max())


def group_axis(ax, n_per=4, groups=S.SECTIONS, gap=1.0):
    """x positions for 5 section groups x 4 lengths, with a gap between groups."""
    xs, centers = [], []
    for gi in range(len(groups)):
        base = gi * (n_per + gap)
        xs.extend(base + np.arange(n_per))
        centers.append(base + (n_per - 1) / 2)
    return np.array(xs), np.array(centers)


def label_groups(ax, centers, xs, y=-0.13, lengths=S.LENGTHS):
    ax.set_xlim(xs.min() - 0.8, xs.max() + 0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{L:g}" for L in lengths] * (len(xs) // len(lengths)), fontsize=7)
    for c, sec in zip(centers, S.SECTIONS):
        ax.text(c, y, f"{sec:.1f} mm$^2$", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=8, color=S.INK)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.grid(axis="x", visible=False)


# ======================================================================================
# Fig 02 - cumulative system resistance from the raw TFA ramps
# ======================================================================================
def fig02_tfa_buildup():
    stages = [("VF", "valve + flow meter (VF)"), ("+FT", "+ feed tube 2.2 m"),
              ("+LS", "+ launching station"), ("+T", "+ conveying tube 4.3 m"),
              ("+Da", "+ silencer = WFE system")]
    fig, axs = plt.subplots(1, 2, figsize=(S.TEXTWIDTH, 7.6 * S.CM), sharey=True,
                            gridspec_kw=dict(wspace=0.08))
    for ax, tube, letter in zip(axs, ["WC08", "WC10"], "ab"):
        ramp = S.TUBE_RAMP[tube]
        lo, hi = operating_window(tube)
        ax.axvspan(lo, hi, color=S.BAND, lw=0, zorder=0)
        ax.text((lo + hi) / 2, 5750, "WFE\noperating\nrange", ha="center", va="top", fontsize=7,
                color=S.INK2, linespacing=0.95)
        for (k, lab), col in zip(stages, ramp):
            key = "VF" if k == "VF" else f"{tube}:{k}"
            d, f, qmax = tfa(key)
            ax.scatter(d.Q_Nm3h, d.p_mbar, s=3.5, color=col, alpha=0.45, lw=0, zorder=2)
            q = np.linspace(0, qmax, 300)
            ax.plot(q, f(q), color=col, lw=1.4, zorder=3, label=lab)
        ax.set_xlim(0, 66)
        ax.set_ylim(0, 6000)
        ax.set_xlabel("Standard volumetric flow rate $\\dot V_\\mathrm{N}$ [Nm$^3$/h]")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1000))
        ls = "LS-08-A" if tube == "WC08" else "LS-10-B"
        S.panel_title(ax, f"{S.TUBE_LABEL[tube]} · {ls}", letter)
    axs[0].set_ylabel("System pressure $p$ [mbar]")
    from matplotlib.legend_handler import HandlerTuple
    hs = [(Line2D([], [], color=b, lw=1.6), Line2D([], [], color=g, lw=1.6))
          for b, g in zip(S.BLUE_RAMP, S.GREEN_RAMP)]
    fig.legend(hs, [lab for _, lab in stages], handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10), handlelength=2.2, fontsize=7.2,
               columnspacing=1.4)
    return S.save(fig, "fig02_tfa_system_buildup")


# ======================================================================================
# Fig 03 - silencer throttle characteristic (Da-C10 on the full WC08 system)
# ======================================================================================
def fig03_silencer_throttle(operating=7):
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    settings = sorted(DA_C10_FULL)
    cmap = LinearSegmentedColormap.from_list("faps", ["#C9DCB0", "#7FA43F", "#3E5C12", "#1C2B06"])
    norm = plt.Normalize(min(settings), max(settings))
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.TEXTWIDTH, 7.0 * S.CM),
                               gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.30))
    q_at = []
    for k in settings:
        d, f, qmax = tfa(DA_C10_FULL[k])
        col = S.ACCENT if k == operating else cmap(norm(k))
        lw, z = (2.0, 5) if k == operating else (1.1, 3)
        q = np.linspace(0, qmax, 300)
        a.plot(q, f(q), color=col, lw=lw, zorder=z)
        a.scatter(d.Q_Nm3h, d.p_mbar, s=2.5, color=col, alpha=0.35, lw=0, zorder=2)
        # flow capacity at 1000 mbar (inverse of the monotone curve)
        qq = np.linspace(0, qmax, 4000)
        pp = f(qq)
        q_at.append((k, np.interp(1000, pp[np.isfinite(pp)], qq[np.isfinite(pp)])))
    lo, hi = operating_window("WC08")
    a.axvspan(lo, hi, color=S.BAND, lw=0, zorder=0)
    a.annotate("WC08 WFE range", xy=(hi, 230), xytext=(hi + 2.2, 230), fontsize=7, color=S.INK2, va="center",
               arrowprops=dict(arrowstyle="-|>", color=S.INK3, lw=0.6, mutation_scale=6, shrinkA=0, shrinkB=0))
    a.set_xlim(0, 62); a.set_ylim(0, 6100)
    a.set_xlabel("Standard volumetric flow rate $\\dot V_\\mathrm{N}$ [Nm$^3$/h]")
    a.set_ylabel("System pressure $p$ [mbar]")
    a.xaxis.set_major_locator(mticker.MultipleLocator(10))
    a.yaxis.set_major_locator(mticker.MultipleLocator(1000))
    S.panel_title(a, "Throttle sweep of silencer Da-C10 on the WC08 system", "a")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = a.inset_axes([0.56, 0.13, 0.36, 0.035])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Throttle [turns]: open $\\rightarrow$ closed", fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2, pad=1.5)
    cb.outline.set_visible(False)
    cb.set_ticks([0, 5, 10, 15])
    cax.xaxis.set_label_position("top")
    a.annotate(f"Da-C10-{operating}\n(WFE setting)", xy=(40, tfa(DA_C10_FULL[operating])[1](40)),
               xytext=(47, 2350), fontsize=7.5, color=S.ACCENT, ha="left",
               arrowprops=dict(arrowstyle="-", color=S.ACCENT, lw=0.7))
    k, qv = np.array(q_at).T
    b.plot(k, qv, color=S.INK2, lw=1.0, zorder=2)
    b.scatter(k, qv, c=[S.ACCENT if kk == operating else cmap(norm(kk)) for kk in k], s=26,
              edgecolor="white", linewidth=0.7, zorder=3)
    b.set_xlabel("Throttle position [turns]")
    b.set_ylabel("Flow at $p$ = 1000 mbar\n$\\dot V_\\mathrm{N}$ [Nm$^3$/h]")
    b.set_xlim(-0.8, 15.8); b.set_ylim(0, None)
    b.xaxis.set_major_locator(mticker.MultipleLocator(5))
    S.panel_title(b, "Flow capacity vs. throttle", "b")
    i_op = list(k).index(operating)
    b.annotate(f"{qv[i_op]:.1f} Nm$^3$/h", (k[i_op], qv[i_op]), xytext=(6, 6), textcoords="offset points",
               fontsize=7.5, color=S.ACCENT)
    return S.save(fig, "fig03_silencer_throttle_map")


# ======================================================================================
# Fig 05 - operating map on the empty-system characteristic + experimental force balance
# ======================================================================================
def excess_pressure_table():
    d = runs().copy()
    out = []
    for tube, g in d.groupby("tube"):
        _, f, _ = tfa(f"{tube}:+Da")
        g = g.copy()
        g["p_empty"] = f(g.Q_Nm3h)
        g["dp_ex"] = g.p_mbar - g.p_empty
        out.append(g)
    d = pd.concat(out)
    c = cfd_table()[["tube", "uid", "W", "D_mm"]]
    agg = d.groupby(["tube", "uid"]).agg(section=("section", "first"), length=("length", "first"),
                                          Q=("Q_Nm3h", "mean"), p=("p_mbar", "mean"),
                                          dp_ex=("dp_ex", "mean"), dp_sd=("dp_ex", "std"),
                                          n=("dp_ex", "size")).reset_index()
    agg = agg.merge(c, on=["tube", "uid"])
    A = np.pi / 4 * (agg.D_mm / 1e3) ** 2
    agg["F_ex"] = agg.dp_ex * 100 * A
    agg["F_ex_sd"] = agg.dp_sd * 100 * A
    return d, agg


def fig05_operating_map():
    d, agg = excess_pressure_table()
    fig, axs = plt.subplots(1, 3, figsize=(S.TEXTWIDTH, 6.9 * S.CM),
                            gridspec_kw=dict(width_ratios=[1, 1, 1.12], wspace=0.36))
    lims = {"WC08": ((2, 13), (0, 1600)), "WC10": ((12, 34), (800, 3800))}
    for ax, tube, letter in zip(axs[:2], ["WC08", "WC10"], "ab"):
        (x0, x1), (y0, y1) = lims[tube]
        _, f, _ = tfa(f"{tube}:+Da")
        q = np.linspace(0, x1, 400)
        ax.plot(q, f(q), color=S.INK, lw=1.2, zorder=5)
        filled = tube == "WC08"
        for sec in S.SECTIONS:
            g = d[(d.section == sec) & (d.tube == tube)]
            ax.scatter(g.Q_Nm3h, g.p_mbar, s=10, marker=S.SEC_MARKER[sec],
                       facecolor=S.SEC_COLOR[sec] if filled else "white",
                       edgecolor="white" if filled else S.SEC_COLOR[sec],
                       linewidth=0.3 if filled else 0.6, alpha=0.9, zorder=3)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xlabel("$\\dot V_\\mathrm{N}$ [Nm$^3$/h]")
        S.panel_title(ax, tube, letter)
        # label the empty-system curve
        xl = x0 + 0.50 * (x1 - x0)
        ax.annotate("empty system\n(TFA)", xy=(xl, f(xl)), xytext=(x0 + 0.56 * (x1 - x0), y0 + 0.10 * (y1 - y0)),
                    fontsize=7, color=S.INK, ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=S.INK, lw=0.6, shrinkA=1, shrinkB=1))
        # excess-pressure illustration on the heaviest long wire
        g = agg[(agg.tube == tube) & (agg.uid == 24)].iloc[0]
        pe = g.p - g.dp_ex
        ax.annotate("", xy=(g.Q, g.p), xytext=(g.Q, pe),
                    arrowprops=dict(arrowstyle="<|-|>", color=S.INK, lw=0.7, mutation_scale=6,
                                    shrinkA=0, shrinkB=0), zorder=6)
        ax.text(g.Q - 0.012 * (x1 - x0) * 3, (g.p + pe) / 2, "$\\Delta p_\\mathrm{ex}$", fontsize=8,
                ha="right", va="center", zorder=6)
    axs[0].set_ylabel("Stable inlet pressure $p$ [mbar]")
    axs[0].xaxis.set_major_locator(mticker.MultipleLocator(2))
    axs[1].xaxis.set_major_locator(mticker.MultipleLocator(5))
    axs[0].yaxis.set_major_locator(mticker.MultipleLocator(250))
    axs[1].yaxis.set_major_locator(mticker.MultipleLocator(500))
    # (c) experimental force balance
    b = axs[2]
    xx = np.array([0.02, 5])
    b.fill_between(xx, xx * 0.8, xx * 1.25, color=S.BAND, lw=0, zorder=0)
    b.plot(xx, xx, color=S.INK2, lw=0.8, zorder=1)
    for sec in S.SECTIONS:
        for tube in ["WC08", "WC10"]:
            g = agg[(agg.section == sec) & (agg.tube == tube)]
            filled = tube == "WC08"
            ci = 1.96 * g.F_ex_sd / np.sqrt(g.n)
            b.errorbar(g.W, g.F_ex, yerr=ci, fmt="none", ecolor=S.SEC_COLOR[sec], elinewidth=0.7,
                       alpha=0.8, zorder=2)
            b.scatter(g.W, g.F_ex, s=24, marker=S.SEC_MARKER[sec],
                      facecolor=S.SEC_COLOR[sec] if filled else "white",
                      edgecolor="white" if filled else S.SEC_COLOR[sec],
                      linewidth=0.7 if filled else 1.0, zorder=3)
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlim(0.03, 4); b.set_ylim(0.03, 8)
    b.text(1.6, 0.95, "1 : 1", fontsize=7, color=S.INK2, ha="left", va="top")
    b.set_xlabel("Wire weight $W$ [N]")
    b.set_ylabel("$\\Delta p_\\mathrm{ex}\\,A_\\mathrm{tube}$ [N]")
    S.panel_title(b, "Experimental force balance", "c")
    meds = {t: (g.F_ex / g.W)[g.W > 0.4].median() for t, g in agg.groupby("tube")}
    b.text(0.97, 0.04, f"median ratio, $W$ > 0.4 N:\nWC08 {meds['WC08']:.2f}   WC10 {meds['WC10']:.2f}",
           transform=b.transAxes, fontsize=6.8, color=S.INK2, ha="right", va="bottom")
    hs = S.section_handles(lw=0)
    hs += [Line2D([], [], ls="", marker="o", ms=4.8, mfc=S.INK2, mec="white", label="WC08 (filled)"),
           Line2D([], [], ls="", marker="o", ms=4.8, mfc="white", mec=S.INK2, label="WC10 (open)")]
    fig.legend(handles=hs, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.085), handletextpad=0.2,
               columnspacing=0.9)
    return S.save(fig, "fig05_operating_map_force_balance")


# ======================================================================================
# Fig 06 - repeatability: CV per configuration + variance attributable to wire instance
# ======================================================================================
def repeatability_table():
    r = runs()
    rows = []
    for (tube, u), g in r.groupby(["tube", "uid"]):
        gm = g.p_mbar.mean()
        grp = g.groupby("instance").p_mbar
        n_i, m_i = grp.size(), grp.mean()
        k = len(n_i)
        ssw = ((g.p_mbar - g.instance.map(m_i)) ** 2).sum()
        dfw = len(g) - k
        msw = ssw / dfw
        if k > 1:
            msb = (n_i * (m_i - gm) ** 2).sum() / (k - 1)
            n0 = (len(g) - (n_i ** 2).sum() / len(g)) / (k - 1)
            s2b = max((msb - msw) / n0, 0.0)
            icc = s2b / (s2b + msw)
        else:
            icc = np.nan
        sec, L = uid_to_geom(int(u))
        rows.append(dict(tube=tube, uid=u, section=sec, length=L, n=len(g), k=k,
                         cv=100 * g.p_mbar.std() / gm, icc=icc))
    return pd.DataFrame(rows)


def fig06_repeatability():
    t = repeatability_table()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("cv", ["#F4F6F8", "#B9C7D6", "#6F8BA8", "#34506E"])
    vmax = 20
    fig = plt.figure(figsize=(S.TEXTWIDTH, 6.6 * S.CM))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.055, 0.42, 1.0], wspace=0.12)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    c = fig.add_subplot(gs[0, 4])
    for ax, tube, letter in zip(axs, ["WC08", "WC10"], "ab"):
        g = t[t.tube == tube].pivot(index="section", columns="length", values="cv").loc[S.SECTIONS[::-1]]
        n = t[t.tube == tube].pivot(index="section", columns="length", values="n").loc[S.SECTIONS[::-1]]
        im = ax.imshow(g.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.values[i, j]
                col = "white" if v > 11 else S.INK
                ax.text(j, i - 0.08, f"{v:.1f}", ha="center", va="center", fontsize=7.5, color=col)
                ax.text(j, i + 0.28, f"$n$={int(n.values[i, j])}", ha="center", va="center", fontsize=5.6,
                        color=col, alpha=0.85)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{L:g}" for L in S.LENGTHS])
        ax.set_yticks(range(5)); ax.set_yticklabels([f"{s:.1f}" for s in S.SECTIONS[::-1]])
        ax.set_xlabel("Wire length $L$ [m]")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True); ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="minor", length=0)
        S.panel_title(ax, tube, letter)
    axs[0].set_ylabel("Cross-section [mm$^2$]")
    axs[1].set_yticklabels([])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Coefficient of variation of $\\bar p_\\mathrm{stable}$ [%]", fontsize=7.5)
    cb.ax.tick_params(labelsize=7, length=2); cb.outline.set_visible(False)
    cb.set_ticks([0, 5, 10, 15, 20])
    # (c) share of variance between instances
    rng = np.random.default_rng(3)
    for yi, tube in enumerate(["WC08", "WC10"]):
        g = t[(t.tube == tube) & t.icc.notna()]
        y = yi + rng.uniform(-0.16, 0.16, len(g))
        c.scatter(100 * g.icc, y, s=20, color=S.TUBE_COLOR[tube], alpha=0.85,
                  marker="o", edgecolor="white", linewidth=0.5, zorder=3)
        med = 100 * g.icc.median()
        c.vlines(med, yi - 0.3, yi + 0.3, color=S.INK, lw=1.4, zorder=4)
        c.text(med, yi + 0.33, f"median {med:.0f} %", ha="center", va="bottom", fontsize=7, color=S.INK)
    c.set_yticks([0, 1]); c.set_yticklabels(["WC08", "WC10"])
    c.set_ylim(-0.55, 1.75); c.set_xlim(-3, 103)
    c.set_xlabel("Variance between wire instances [%]")
    c.grid(axis="y", visible=False)
    c.xaxis.set_major_locator(mticker.MultipleLocator(25))
    S.panel_title(c, "Instance-to-instance share (ICC)", "c")
    c.text(0.99, 0.02, "one dot per wire type", transform=c.transAxes,
           ha="right", va="bottom", fontsize=6.5, color=S.INK3)
    return S.save(fig, "fig06_repeatability")


# ======================================================================================
# Fig 04 - stable levitation pressure across the design space (all runs)
# ======================================================================================
def fig04_pressure_vs_length():
    d = runs()
    fig, axs = plt.subplots(1, 2, figsize=(S.TEXTWIDTH, 7.4 * S.CM),
                            gridspec_kw=dict(wspace=0.28))
    rng = np.random.default_rng(7)
    dodge = dict(zip(S.SECTIONS, np.linspace(-0.12, 0.12, 5)))
    for ax, tube, letter in zip(axs, ["WC08", "WC10"], "ab"):
        g = d[d.tube == tube]
        for sec in S.SECTIONS:
            gs = g[g.section == sec]
            x = gs.length + dodge[sec] + rng.uniform(-0.035, 0.035, len(gs))
            ax.scatter(x, gs.p_mbar, s=5, color=S.SEC_COLOR[sec], alpha=0.28, lw=0, zorder=2)
            agg = gs.groupby("length").p_mbar.agg(["mean", "std", "size"]).reset_index()
            ax.plot(agg.length + dodge[sec], agg["mean"], color=S.SEC_COLOR[sec], lw=1.4,
                    marker=S.SEC_MARKER[sec], ms=5, mec="white", mew=0.7, zorder=4)
        ax.set_xticks(S.LENGTHS)
        ax.set_xticklabels([f"{L:g}" for L in S.LENGTHS])
        ax.set_xlim(0.2, 3.8)
        ax.set_xlabel("Wire length $L$ [m]")
        ax.grid(axis="x", visible=False)
        n = len(g)
        S.panel_title(ax, f"{S.TUBE_LABEL[tube]},  $n$ = {n} runs", letter)
    axs[0].set_ylim(0, 1500)
    axs[1].set_ylim(0, 4000)
    axs[0].yaxis.set_major_locator(mticker.MultipleLocator(250))
    axs[1].yaxis.set_major_locator(mticker.MultipleLocator(500))
    axs[0].set_ylabel("Stable inlet pressure $\\bar p_\\mathrm{stable}$ [mbar]")
    hs = S.section_handles()
    hs.append(Line2D([], [], ls="", marker="o", ms=3, color=S.INK3, alpha=0.5, label="single run"))
    fig.legend(handles=hs, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.065),
               title=None, frameon=False)
    return S.save(fig, "fig04_stable_pressure_vs_length")


# ======================================================================================
# CFD helpers
# ======================================================================================
import pickle  # noqa: E402

_HIST = None


def histories():
    global _HIST
    if _HIST is None:
        pth = CACHE / "cache_cfd_hist.pkl"
        if pth.exists():
            _HIST = pickle.load(open(pth, "rb"))
        else:
            _HIST = {c: cfd_history(c) for c in cfd_cases()}
            pickle.dump(_HIST, open(pth, "wb"))
    return _HIST


def tube_scatter(ax, d, x, y, s=26, by_tube_color=False, **kw):
    """Cross-section = colour + marker; tube = filled (WC08) / open (WC10)."""
    for sec in S.SECTIONS:
        for tube in ["WC08", "WC10"]:
            g = d[(d.section == sec) & (d.tube == tube)]
            col = S.TUBE_COLOR[tube] if by_tube_color else S.SEC_COLOR[sec]
            filled = tube == "WC08"
            ax.scatter(g[x], g[y], s=s, marker=S.SEC_MARKER[sec],
                       facecolor=col if filled else "white", edgecolor="white" if filled else col,
                       linewidth=0.7 if filled else 1.0, zorder=3, **kw)


def encoding_legend(fig, y=1.08, tube_color=False, ncol=7):
    if tube_color:
        hs = [Line2D([], [], ls="", marker=S.SEC_MARKER[s], ms=4.8, mfc=S.INK2, mec="white", label=f"{s:.1f} mm$^2$")
              for s in S.SECTIONS]
        hs += [Line2D([], [], ls="", marker="o", ms=5, mfc=S.TUBE_COLOR["WC08"], mec="white", label="WC08"),
               Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=S.TUBE_COLOR["WC10"], mew=1.0, label="WC10")]
    else:
        hs = S.section_handles(lw=0)
        hs += [Line2D([], [], ls="", marker="o", ms=4.8, mfc=S.INK2, mec="white", label="WC08 (filled)"),
               Line2D([], [], ls="", marker="o", ms=4.8, mfc="white", mec=S.INK2, label="WC10 (open)")]
    fig.legend(handles=hs, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y), handletextpad=0.2,
               columnspacing=0.9)


# ======================================================================================
# Fig 07 - iterative convergence of all 40 straight-wire runs
# ======================================================================================
def iterative_change(n_last: int = 20) -> pd.DataFrame:
    """Signed change of the two reported quantities over the final n_last iterations [%]
    and a convergence tier: A < 1 %, B 1-3 %, C > 3 % (largest of the two)."""
    rows = []
    for case, h in histories().items():
        m = h["mass-flow-inlet"].values
        F = h[[k for k in h.columns if k.startswith("wire-drag-force(")]].sum(axis=1).values
        k = min(n_last, len(m) - 1)
        rows.append(dict(Simulation_ID=case, tube=case[:4], iters=int(h["iter"].iloc[-1]),
                         dm=100 * (m[-1] - m[-1 - k]) / m[-1], dF=100 * (F[-1] - F[-1 - k]) / F[-1]))
    r = pd.DataFrame(rows)
    r["drift"] = r[["dm", "dF"]].abs().max(axis=1)
    r["tier"] = np.select([r.drift < 1, r.drift < 3], ["A", "B"], "C")
    return r


TIER_FILL = {"A": "#E6EED8", "B": "#F4F1DE", "C": "#F6E3E3"}
TIER_TEXT = {"A": "#4A6B12", "B": "#7A6A12", "C": "#9A2A2A"}


def fig07_cfd_convergence():
    H = histories()
    d = cfd_table().set_index("Simulation_ID")
    ic = iterative_change().set_index("Simulation_ID")
    fig, axs = plt.subplots(2, 2, figsize=(S.TEXTWIDTH, 10.6 * S.CM),
                            gridspec_kw=dict(hspace=0.55, wspace=0.62, top=0.90))
    a, b, c, e = axs.ravel()
    for case, h in H.items():
        col = S.TUBE_COLOR[case[:4]]
        it = h["iter"].values
        m = h["mass-flow-inlet"].values
        F = h[[k for k in h.columns if k.startswith("wire-drag-force(")]].sum(axis=1).values
        a.plot(it, m / d.loc[case, "mdot_exp"], color=col, lw=0.6, alpha=0.55)
        b.plot(it, F / d.loc[case, "W"], color=col, lw=0.6, alpha=0.55)
        for ax, y in ((a, m[-1] / d.loc[case, "mdot_exp"]), (b, F[-1] / d.loc[case, "W"])):
            ax.plot(it[-1], y, "o", ms=2.6, color=col, mec="white", mew=0.3, zorder=4)
    for ax, ylim, lab in ((a, (0, 1.6), "$\\dot m_\\mathrm{CFD}\\,/\\,\\dot m_\\mathrm{exp}$ [–]"),
                          (b, (0, 2.0), "$F_\\mathrm{D}\\,/\\,W$ [–]")):
        ax.axhspan(0.9, 1.1, color=S.BAND, lw=0, zorder=0)
        ax.axhline(1, color=S.INK2, lw=0.6)
        ax.set_ylim(*ylim); ax.set_xlim(0, 250)
        ax.set_ylabel(lab); ax.set_xlabel("Iteration [–]")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    a.text(246, 1.12, "±10 %", fontsize=7, color=S.INK3, ha="right", va="bottom")
    S.panel_title(a, "Inlet mass flow vs. experimental target", "a")
    S.panel_title(b, "Aerodynamic force vs. wire weight", "b")
    a.text(0.98, 0.04, "dots: value at the last iteration", transform=a.transAxes, ha="right", va="bottom",
           fontsize=6.6, color=S.INK3)

    # (c) change over the final 20 iterations
    lo, hi = 0.02, 40
    for t, (x0, x1), (y0, y1) in [("A", (lo, 1), (lo, 1)), ("B", (lo, 3), (lo, 3)), ("C", (lo, hi), (lo, hi))][::-1]:
        c.fill_between([x0, x1], y0, y1, color=TIER_FILL[t], lw=0, zorder=0)
    for tube in ["WC08", "WC10"]:
        g = ic[ic.tube == tube]
        filled = tube == "WC08"
        c.scatter(g.dm.abs().clip(lower=lo * 1.2), g.dF.abs().clip(lower=lo * 1.2), s=22,
                  facecolor=S.TUBE_COLOR[tube] if filled else "white",
                  edgecolor="white" if filled else S.TUBE_COLOR[tube], linewidth=0.7 if filled else 1.0, zorder=3)
    c.set_xscale("log"); c.set_yscale("log")
    c.set_xlim(lo, hi); c.set_ylim(lo, hi)
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:g}")
    c.xaxis.set_major_formatter(fmt); c.yaxis.set_major_formatter(fmt)
    c.set_xlabel("$|\\Delta\\dot m_\\mathrm{in}|$ over last 20 it. [%]")
    c.set_ylabel("$|\\Delta F_\\mathrm{D}|$ over last 20 it. [%]")
    counts = ic.tier.value_counts()
    c.text(0.025, 0.06, f"< 1 %\n{counts.get('A', 0)} runs", fontsize=6.8, color=TIER_TEXT["A"], va="bottom")
    c.text(1.15, 0.06, f"1–3 %\n{counts.get('B', 0)}", fontsize=6.8, color=TIER_TEXT["B"], va="bottom")
    c.text(4.0, 0.06, f"> 3 %\n{counts.get('C', 0)} runs", fontsize=6.8, color=TIER_TEXT["C"], va="bottom")
    S.panel_title(c, "Monitor change at the end of each run", "c")

    # (d) robustness of the validation metrics
    full = d.join(ic[["tier"]])
    sub = full[full.tier != "C"]
    metrics = []
    for tube in ["WC08", "WC10"]:
        for lab, fn in [("$\\dot m$ MAPE", lambda g: 100 * g.mdot_err.abs().mean()),
                        ("$\\dot m$ bias", lambda g: 100 * g.mdot_err.mean()),
                        ("med. $F_\\mathrm{D}/W - 1$", lambda g: 100 * (g.FD_over_W.median() - 1))]:
            metrics.append((f"{tube}  {lab}", tube, fn(full[full.tube == tube]), fn(sub[sub.tube == tube])))
    y = np.arange(len(metrics))[::-1]
    for yi, (lab, tube, v_all, v_sub) in zip(y, metrics):
        e.plot([v_all, v_sub], [yi, yi], color=S.INK3, lw=0.8, zorder=1)
        e.scatter(v_all, yi, s=26, color=S.TUBE_COLOR[tube], edgecolor="white", linewidth=0.6, zorder=3)
        e.scatter(v_sub, yi, s=26, facecolor="white", edgecolor=S.TUBE_COLOR[tube], linewidth=1.0, zorder=3)
    e.axvline(0, color=S.INK3, lw=0.6)
    e.set_yticks(y)
    e.set_yticklabels([m_[0] for m_ in metrics], fontsize=6.8)
    e.set_xlim(-25, 25)
    e.xaxis.set_major_locator(mticker.MultipleLocator(10))
    e.grid(axis="y", visible=False)
    e.set_xlabel("Validation metric [%]")
    S.panel_title(e, "Robustness: all runs vs. flat runs", "d")
    hs_d = [Line2D([], [], ls="", marker="o", ms=5, mfc=S.INK2, mec="white", label=f"all 40 runs"),
            Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=S.INK2, mew=1.0,
                   label=f"flat runs only (< 3 %, $n$ = {len(sub)})")]
    e.legend(handles=hs_d, loc="upper left", fontsize=6.6, handletextpad=0.3, borderaxespad=0.2)
    e.set_ylim(-0.7, len(metrics) + 1.0)

    hs = [Line2D([], [], color=S.TUBE_COLOR["WC08"], lw=1.4, label="WC08 (20 runs)"),
          Line2D([], [], color=S.TUBE_COLOR["WC10"], lw=1.4, label="WC10 (20 runs)")]
    fig.legend(handles=hs, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))
    return S.save(fig, "fig07_cfd_convergence")


# ======================================================================================
# Fig 08 - validation: mass flow parity and force balance
# ======================================================================================
def fig08_cfd_validation():
    d = cfd_table().copy()
    d["mdot_exp_g"] = d.mdot_exp * 1e3
    d["mdot_cfd_g"] = d.mdot_cfd * 1e3
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.TEXTWIDTH, 7.4 * S.CM), gridspec_kw=dict(wspace=0.32))
    xx = np.array([0, 14])
    a.fill_between(xx, xx * 0.9, xx * 1.1, color=S.BAND, lw=0, zorder=0)
    a.plot(xx, xx, color=S.INK2, lw=0.8, zorder=1)
    tube_scatter(a, d, "mdot_exp_g", "mdot_cfd_g")
    a.set_xlim(0, 13); a.set_ylim(0, 13); a.set_aspect("equal")
    a.set_xlabel("Experimental mass flow $\\dot m_\\mathrm{exp}$ [g/s]")
    a.set_ylabel("CFD mass flow $\\dot m_\\mathrm{CFD}$ [g/s]")
    a.xaxis.set_major_locator(mticker.MultipleLocator(2)); a.yaxis.set_major_locator(mticker.MultipleLocator(2))
    lines = ["shaded band: ±10 %"]
    for tube in ["WC08", "WC10"]:
        g = d[d.tube == tube]
        lines.append(f"{tube}: MAPE {100 * g.mdot_err.abs().mean():.1f} %, bias {100 * g.mdot_err.mean():+.1f} %")
    a.text(0.03, 0.97, "\n".join(lines), transform=a.transAxes, fontsize=6.8, color=S.INK2, va="top")
    S.panel_title(a, "Mass-flow validation (pressure prescribed)", "a")
    xx = np.array([0.02, 6])
    b.fill_between(xx, xx * 0.9, xx * 1.1, color=S.BAND, lw=0, zorder=0)
    b.plot(xx, xx, color=S.INK2, lw=0.8, zorder=1)
    tube_scatter(b, d, "W", "F_D")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlim(0.03, 4); b.set_ylim(0.03, 4); b.set_aspect("equal")
    b.set_xlabel("Wire weight $W$ [N]")
    b.set_ylabel("CFD aerodynamic force $F_\\mathrm{D}$ [N]")
    b.text(2.6, 1.55, "1 : 1", fontsize=7, color=S.INK2, ha="left", va="top")
    lines = ["shaded band: ±10 %"] + [f"{t}: median $F_\\mathrm{{D}}/W$ = {g.FD_over_W.median():.2f}"
                                      for t, g in d.groupby("tube")]
    b.text(0.03, 0.97, "\n".join(lines), transform=b.transAxes, fontsize=6.8, color=S.INK2, va="top")
    S.panel_title(b, "Vertical force balance", "b")
    encoding_legend(fig, y=1.085)
    return S.save(fig, "fig08_cfd_validation_parity")


# ======================================================================================
# Fig 09 - error attribution: force imbalance <- mass-flow error <- porous-fit range
# ======================================================================================
def porous_misfit():
    """Relative misfit of the full-range Darcy-Forchheimer fit (dp = aQ^2 + bQ) to the
    TFA-derived component pressure drop, per component and tube (author's derivation:
    silencer = (+Da) - (+T), feed tube = (+FT) - VF)."""
    from scipy.interpolate import interp1d
    out = {}
    for comp, (tot, ref) in {"silencer": ("+Da", "+T"), "feed tube": ("+FT", None)}.items():
        for tube in ["WC08", "WC10"]:
            dt = tfa(f"{tube}:{tot}")[0].groupby("Q_Nm3h", as_index=False).p_mbar.mean()
            dr = (tfa(f"{tube}:{ref}") if ref else tfa("VF"))[0].groupby("Q_Nm3h", as_index=False).p_mbar.mean()
            lo, hi = max(dt.Q_Nm3h.min(), dr.Q_Nm3h.min()), min(dt.Q_Nm3h.max(), dr.Q_Nm3h.max())
            dt = dt[(dt.Q_Nm3h >= lo) & (dt.Q_Nm3h <= hi)].sort_values("Q_Nm3h")
            dp = dt.p_mbar.values - interp1d(dr.Q_Nm3h, dr.p_mbar)(dt.Q_Nm3h.values)
            q = dt.Q_Nm3h.values
            m = dp > 5
            popt, _ = curve_fit(quad0, q[m], dp[m])
            # smooth the measured component curve for display (running median on Q bins)
            dfm = pd.DataFrame(dict(q=q[m], dp=dp[m]))
            dfm["bin"] = (dfm.q / 1.0).round()
            sm = dfm.groupby("bin").agg(q=("q", "mean"), dp=("dp", "median")).reset_index()
            sm["rel"] = (quad0(sm.q, *popt) - sm.dp) / sm.dp
            out[(comp, tube)] = sm
    return out


def fig09_cfd_error_attribution():
    d = cfd_table().copy()
    d["Q_exp"] = d["Stable_Average_Flow[Nm^3/h]"]
    fig, axs = plt.subplots(1, 3, figsize=(S.TEXTWIDTH, 6.6 * S.CM),
                            gridspec_kw=dict(wspace=0.46, width_ratios=[1, 1, 1]))
    a, b, c = axs
    # (a)
    e = np.linspace(-0.22, 0.24, 100)
    x = np.log1p(d.mdot_err); y = np.log(d.FD_over_W)
    n, c0 = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    a.plot(100 * e, (1 + e) ** n * np.exp(c0), color=S.INK, lw=1.0, zorder=2)
    a.axhline(1, color=S.INK3, lw=0.6); a.axvline(0, color=S.INK3, lw=0.6)
    tube_scatter(a, d.assign(err=100 * d.mdot_err), "err", "FD_over_W", s=22, by_tube_color=True)
    a.set_xlabel("Mass-flow error $\\varepsilon_{\\dot m}$ [%]")
    a.set_ylabel("$F_\\mathrm{D}\\,/\\,W$ [–]")
    a.set_xlim(-22, 24); a.set_ylim(0.35, 1.65)
    a.text(0.03, 0.97, f"$r$ = {r:.2f}\n$F_\\mathrm{{D}}/W \\propto (1+\\varepsilon)^{{{n:.1f}}}$\n"
           f"at $\\varepsilon$ = 0: {np.exp(c0):.2f}", transform=a.transAxes, fontsize=6.8, va="top", color=S.INK2)
    S.panel_title(a, "Force vs. flow error", "a")
    # (b)
    txt = []
    for tube in ["WC08", "WC10"]:
        g = d[d.tube == tube]
        k, c1 = np.polyfit(g.Q_exp, 100 * g.mdot_err, 1)
        qq = np.linspace(g.Q_exp.min(), g.Q_exp.max(), 10)
        b.plot(qq, k * qq + c1, color=S.TUBE_COLOR[tube], lw=1.0, zorder=2)
        rr = np.corrcoef(g.Q_exp, g.mdot_err)[0, 1]
        txt.append((tube, f"{k:+.1f} %/(Nm$^3$/h), $r$ = {rr:.2f}"))
    for i, (tube, t_) in enumerate(txt):
        b.text(0.97, 0.04 + 0.085 * (1 - i), t_, transform=b.transAxes, ha="right", va="bottom", fontsize=6.6,
               color=S.TUBE_COLOR[tube], bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"), zorder=6)
    b.axhline(0, color=S.INK3, lw=0.6)
    tube_scatter(b, d.assign(err=100 * d.mdot_err), "Q_exp", "err", s=22, by_tube_color=True)
    b.set_xlim(0, 42); b.set_ylim(-30, 24)
    b.set_xlabel("Operating flow $\\dot V_\\mathrm{N}$ [Nm$^3$/h]")
    b.set_ylabel("Mass-flow error $\\varepsilon_{\\dot m}$ [%]")
    S.panel_title(b, "Flow error vs. flow rate", "b")
    # (c)
    mis = porous_misfit()
    for tube in ["WC08", "WC10"]:
        lo, hi = operating_window(tube)
        c.axvspan(lo, hi, color=S.TUBE_COLOR[tube], alpha=0.10, lw=0, zorder=0)
        c.text((lo + hi) / 2, 0.03, f"{tube}\nrange", transform=c.get_xaxis_transform(), ha="center", va="bottom",
               fontsize=6.6, color=S.TUBE_COLOR[tube])
        for comp, ls in [("silencer", "-"), ("feed tube", (0, (3, 1.6)))]:
            sm = mis[(comp, tube)]
            sm = sm[sm.q >= 2.5]
            c.plot(sm.q, 100 * sm.rel, color=S.TUBE_COLOR[tube], lw=1.2, ls=ls, zorder=3)
    c.axhline(0, color=S.INK3, lw=0.6)
    c.set_xlim(0, 56); c.set_ylim(-40, 100)
    c.set_xlabel("$\\dot V_\\mathrm{N}$ [Nm$^3$/h]")
    c.set_ylabel("$\\Delta p_\\mathrm{fit}/\\Delta p_\\mathrm{TFA} - 1$ [%]")
    c.yaxis.set_major_locator(mticker.MultipleLocator(20))
    S.panel_title(c, "Porous-zone fit misfit", "c")
    hs = [Line2D([], [], color=S.INK2, lw=1.2, ls="-", label="silencer"),
          Line2D([], [], color=S.INK2, lw=1.2, ls=(0, (3, 1.6)), label="feed tube")]
    c.legend(handles=hs, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    encoding_legend(fig, y=1.10, tube_color=True)
    return S.save(fig, "fig09_cfd_error_attribution")


# ======================================================================================
# Fig 10 - where the levitation force comes from (force decomposition, normalised by weight)
# ======================================================================================
def fig10_drag_decomposition():
    d = cfd_table().sort_values(["tube", "section", "L"])
    fig, axs = plt.subplots(2, 1, figsize=(S.TEXTWIDTH, 12.4 * S.CM), sharex=True,
                            gridspec_kw=dict(hspace=0.50, top=0.88, height_ratios=[1, 1.45]))
    for ax, tube, letter in zip(axs, ["WC08", "WC10"], "ab"):
        g = d[d.tube == tube]
        xs, centers = group_axis(ax)
        bw = 0.74
        bot = g.F_bot.values / g.W.values
        wall = g.F_wall.values / g.W.values
        top = g.F_top.values / g.W.values
        net = g.F_D.values / g.W.values
        ax.bar(xs, bot, bw, color=S.COMP["bottom"], lw=0, zorder=2)
        ax.bar(xs, wall, bw, bottom=bot, color=S.COMP["wall"], lw=0, zorder=2)
        ax.bar(xs, top, bw, color=S.COMP["top"], lw=0, zorder=2)
        # 2-px surface gap between stacked segments
        ax.hlines(bot, xs - bw / 2, xs + bw / 2, color="white", lw=0.8, zorder=3)
        ax.axhline(0, color=S.INK2, lw=0.6, zorder=3)
        ax.axhline(1, color=S.INK, lw=0.7, ls=(0, (4, 2)), zorder=1)
        ax.scatter(xs, net, marker="D", s=16, color=S.INK, edgecolor="white", linewidth=0.6, zorder=5)
        ax.set_ylim(*((-2.6, 3.8) if tube == "WC08" else (-5.2, 6.3)))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1 if tube == "WC08" else 2))
        ax.set_ylabel("Force / wire weight [–]")
        S.panel_title(ax, S.TUBE_LABEL[tube], letter)
        label_groups(ax, centers, xs, y=-0.13)
        ax.tick_params(axis="x", labelbottom=True)
        for c in centers[1:]:
            ax.axvline(c - 2.5, color=GRID_SEP, lw=0.6, zorder=0)
    axs[1].text(-0.035, -0.04, "$L$ [m]", transform=axs[1].transAxes, ha="right", va="top", fontsize=7, color=S.INK2)
    axs[0].text(-0.035, -0.04, "$L$ [m]", transform=axs[0].transAxes, ha="right", va="top", fontsize=7, color=S.INK2)
    hs = [Patch(color=S.COMP["bottom"], label="pressure on bottom ferrule (+)"),
          Patch(color=S.COMP["wall"], label="shear on wire surface (+)"),
          Patch(color=S.COMP["top"], label="pressure on top ferrule (−)"),
          Line2D([], [], ls="", marker="D", ms=4.5, color=S.INK, mec="white", label="net $F_\\mathrm{D}/W$"),
          Line2D([], [], color=S.INK, lw=0.7, ls=(0, (4, 2)), label="equilibrium")]
    fig.legend(handles=hs, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995), handlelength=1.4)
    return S.save(fig, "fig10_force_decomposition")


GRID_SEP = "#D5D9DE"


# ======================================================================================
# Fig 11 - pressure budget of the conveying system (CFD report expressions)
# ======================================================================================
def fig11_pressure_budget():
    d = cfd_table().sort_values(["tube", "section", "L"])
    fig, axs = plt.subplots(2, 1, figsize=(S.TEXTWIDTH, 11.9 * S.CM), sharex=True,
                            gridspec_kw=dict(hspace=0.50, top=0.855))
    for ax, tube, letter in zip(axs, ["WC08", "WC10"], "ab"):
        g = d[d.tube == tube]
        xs, centers = group_axis(ax)
        bw = 0.74
        parts = [("dp_feed", "feed", "feed tube (porous zone)"),
                 ("dp_ls_tube", "wire", "launching station + conveying tube with wire"),
                 ("dp_sil", "sil", "silencer (porous zone)")]
        base = np.zeros(len(g))
        for col, key, lab in parts:
            v = g[col].values / 1000
            ax.bar(xs, v, bw, bottom=base, color=S.BUDGET[key], lw=0, zorder=2, label=lab)
            base = base + v
            ax.hlines(base, xs - bw / 2, xs + bw / 2, color="white", lw=0.8, zorder=3)
        ax.scatter(xs, g.p_in.values / 1000, marker="_", s=60, color=S.INK, linewidth=1.2, zorder=5)
        share = 100 * g.dp_ls_tube.values / g.p_in.values
        for x_, b_, sh in zip(xs, g.p_in.values / 1000, share):
            ax.text(x_, b_ + (0.02 if tube == "WC08" else 0.02) * ax.get_ylim()[1] + 2, f"{sh:.0f}",
                    ha="center", va="bottom", fontsize=5.8, color="#4A6B12")
        ax.set_ylabel("Pressure drop [kPa]")
        S.panel_title(ax, S.TUBE_LABEL[tube] + "   (numbers: wire-region share of $p_\\mathrm{in}$ in %)", letter)
        label_groups(ax, centers, xs, y=-0.13)
        ax.tick_params(axis="x", labelbottom=True)
        for c in centers[1:]:
            ax.axvline(c - 2.5, color=GRID_SEP, lw=0.6, zorder=0)
    axs[0].set_ylim(0, 150); axs[1].set_ylim(0, 370)
    axs[0].yaxis.set_major_locator(mticker.MultipleLocator(25))
    axs[1].yaxis.set_major_locator(mticker.MultipleLocator(50))
    for ax in axs:
        ax.text(-0.035, -0.04, "$L$ [m]", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=S.INK2)
    hs, ls_ = axs[0].get_legend_handles_labels()
    hs.append(Line2D([], [], ls="", marker="_", ms=9, mew=1.2, color=S.INK))
    ls_.append("inlet pressure $p_\\mathrm{in}$ (measured, BC)")
    fig.legend(hs, ls_, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995), handlelength=1.4)
    return S.save(fig, "fig11_cfd_pressure_budget")


# ======================================================================================
# Fig 12 - wall-shear scaling on the wire and the skin-friction share
# ======================================================================================
def fig12_wall_friction():
    d = cfd_table().copy()
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.TEXTWIDTH, 7.0 * S.CM), gridspec_kw=dict(wspace=0.30))
    re = np.logspace(np.log10(4e3), np.log10(8e4), 100)
    blas = 0.079 * re ** -0.25
    ratio = (d.Cf / (0.079 * d.Re_h ** -0.25)).median()
    a.plot(re, blas, color=S.INK2, lw=0.9, ls=(0, (4, 2)), zorder=1,
           label="Blasius (smooth pipe), $0.079\\,Re^{-1/4}$")
    a.plot(re, ratio * blas, color=S.INK, lw=1.1, zorder=1, label=f"{ratio:.2f} × Blasius (median of 40 runs)")
    tube_scatter(a, d, "Re_h", "Cf", s=24)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(5e3, 7e4); a.set_ylim(4.5e-3, 1.2e-2)
    a.xaxis.set_major_locator(mticker.FixedLocator([5e3, 1e4, 2e4, 5e4]))
    a.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1e3:g}"))
    a.xaxis.set_minor_formatter(mticker.NullFormatter())
    a.yaxis.set_major_locator(mticker.FixedLocator([5e-3, 6e-3, 8e-3, 1e-2]))
    a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v * 1e3:g}"))
    a.yaxis.set_minor_formatter(mticker.NullFormatter())
    a.set_xlabel("Annular Reynolds number $Re_{D_h}\\,\\times 10^{-3}$ [–]")
    a.set_ylabel("Wire skin-friction coeff. $C_f\\,\\times 10^{3}$ [–]")
    a.legend(loc="lower left", fontsize=6.8, handlelength=2.0)
    S.panel_title(a, "Wall shear on the wire surface", "a")
    for sec in S.SECTIONS:
        for tube, ls in [("WC08", "-"), ("WC10", (0, (3, 1.6)))]:
            g = d[(d.section == sec) & (d.tube == tube)].sort_values("L")
            filled = tube == "WC08"
            b.plot(g.L, 100 * g.skin_share, color=S.SEC_COLOR[sec], lw=1.1, ls=ls, zorder=2)
            b.scatter(g.L, 100 * g.skin_share, s=20, marker=S.SEC_MARKER[sec],
                      facecolor=S.SEC_COLOR[sec] if filled else "white",
                      edgecolor="white" if filled else S.SEC_COLOR[sec], linewidth=0.7 if filled else 1.0, zorder=3)
    b.set_xticks(S.LENGTHS); b.set_xlim(0.3, 3.7); b.set_ylim(0, 100)
    b.set_xlabel("Wire length $L$ [m]")
    b.set_ylabel("Skin-friction share $F_\\mathrm{wall}/F_\\mathrm{D}$ [%]")
    b.grid(axis="x", visible=False)
    S.panel_title(b, "Shear vs. ferrule pressure forces", "b")
    hs = S.section_handles(lw=0)
    hs += [Line2D([], [], color=S.INK2, lw=1.1, ls="-", marker="o", ms=4, mfc=S.INK2, mec="white", label="WC08"),
           Line2D([], [], color=S.INK2, lw=1.1, ls=(0, (3, 1.6)), marker="o", ms=4, mfc="white", mec=S.INK2, label="WC10")]
    fig.legend(handles=hs, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.085), handletextpad=0.25,
               columnspacing=0.9)
    return S.save(fig, "fig12_wall_friction_scaling")


# ======================================================================================
FIGS = {k: v for k, v in globals().items() if k.startswith("fig") and callable(v)}

if __name__ == "__main__":
    want = sys.argv[1:] or list(FIGS)
    for k in FIGS:
        if any(k.startswith(w) for w in want):
            print(k, "->", FIGS[k]())
