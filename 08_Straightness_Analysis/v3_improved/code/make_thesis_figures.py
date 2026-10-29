"""Thesis figures for the straightness negative result (Section 7.1 and appendix).

Usage: python make_thesis_figures.py <v3_improved dir>
Reads  more/sheets_clean, more/analysis (wire_metrics, *_curve.csv, explore_tests.csv)
Writes thesis_figures/: fig_annotated_fitted_all, fig_annotated_fitted_2m, fig_correlation_heatmap
(fig_true_scale_profiles is made by plot_true_scale_compact.py; fig_dev_vs_pressure is copied from
more/analysis/figures/fig_marked_dev_vs_pressure, made by plot_marked.py.)
"""
import os, sys, shutil
import cv2, numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

BASE = sys.argv[1] if len(sys.argv) > 1 else "."
os.chdir(BASE); os.makedirs("thesis_figures", exist_ok=True)
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "pdf.fonttype": 42})


def panel(sheet, pad_y, pad_x=40):
    """Clean sheet with the fitted spline (orange), cropped around the magenta dots."""
    im = cv2.imread(f"more/sheets_clean/{sheet}.jpg"); c = pd.read_csv(f"more/analysis/{sheet}_curve.csv")
    for x, y in zip(c.x.astype(int), c.y.astype(int)):
        cv2.circle(im, (x, y), 1, (0, 140, 255), -1)
    b, g, r = [im[..., i].astype(int) for i in range(3)]
    ys, xs = np.where((r > 200) & (b > 200) & (g < 120))
    y0, y1 = max(0, ys.min() - pad_y), min(im.shape[0], ys.max() + pad_y)
    x0, x1 = max(0, xs.min() - pad_x), min(im.shape[1], xs.max() + pad_x)
    return cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)


def label(ax, text, fs):
    ax.text(0.005, 0.93, text, transform=ax.transAxes, fontsize=fs, va="top",
            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))


def fig_all():
    W = pd.read_csv("more/analysis/wire_metrics.csv"); W = W[W.status == "ok"]
    pairs = []
    for (u, i), g in W.groupby(["uid", "instance"]):
        d = dict(zip(g.plane, g.sheet))
        if {"X", "Y"} <= set(d):
            pairs.append((int(u), int(i), d["X"], d["Y"]))
    panels = [(panel(sx, 45), panel(sy, 45)) for _, _, sx, sy in pairs]
    asp = np.mean([p.shape[0] / p.shape[1] for pr in panels for p in pr])
    row_h = (7.16 / 2 - 0.02) * asp; n = len(pairs)
    fig, axs = plt.subplots(n, 2, figsize=(7.16, row_h * n + 0.3), squeeze=False)
    for r_, ((u, i, _, _), pr) in enumerate(zip(pairs, panels)):
        for c_, (p, pl) in enumerate(zip(pr, "XY")):
            ax = axs[r_, c_]; ax.imshow(p, aspect="auto"); ax.axis("off")
            label(ax, f"UID {u}, inst. {i}, plane {pl}", 6)
    fig.text(0.5, 0.003, "Magenta: hand-placed wire points; orange: fitted spline; cyan: rule marks 300 mm apart (scale reference).",
             ha="center", fontsize=6.5)
    fig.subplots_adjust(left=0.003, right=0.997, top=0.998, bottom=0.02, hspace=0.03, wspace=0.012)
    fig.savefig("thesis_figures/fig_annotated_fitted_all.pdf"); fig.savefig("thesis_figures/fig_annotated_fitted_all.png", dpi=220)


def fig_2m():
    sheets = [("IMG_6095_19_1_X", "UID 19, inst. 1"), ("IMG_6096_19_2_X", "UID 19, inst. 2"), ("IMG_6097_19_3_X", "UID 19, inst. 3"),
              ("IMG_6087_23_1_X", "UID 23, inst. 1"), ("IMG_6088_23_2_X", "UID 23, inst. 2"), ("IMG_6089_23_3_X", "UID 23, inst. 3")]
    P = [panel(s, 60) for s, _ in sheets]
    row_h = 7.0 * np.mean([p.shape[0] / p.shape[1] for p in P])
    fig, axs = plt.subplots(len(P), 1, figsize=(7.16, row_h * len(P) + 0.3), squeeze=False)
    for ax, p, (s, t) in zip(axs.ravel(), P, sheets):
        ax.imshow(p, aspect="auto"); ax.axis("off"); label(ax, f"{t}, plane X (single view)", 6.5)
    fig.text(0.5, 0.003, "2.0 m wires. Magenta: hand-placed wire points; orange: fitted spline; cyan: rule marks (scale). "
             "The wire extends beyond the left edge of the frame.", ha="center", fontsize=6)
    fig.subplots_adjust(left=0.003, right=0.997, top=0.998, bottom=0.025, hspace=0.04)
    fig.savefig("thesis_figures/fig_annotated_fitted_2m.pdf"); fig.savefig("thesis_figures/fig_annotated_fitted_2m.png", dpi=220)


def fig_heatmap():
    T = pd.read_csv("more/analysis/explore_tests.csv")
    shape = ["dev_max", "dev_rms", "bend", "kappa_rms", "kappa_max", "kappa_mean", "bend_energy", "arc_excess_pct", "inflections"]
    resp = ["p_stable", "q_stable", "p_lift", "t_lift", "t_stable", "t_lift_to_stable", "n_incr", "p_overshoot", "p_sd_settled", "lift_ratio"]
    sl = {"dev_max": "Max. deviation", "dev_rms": "RMS deviation", "bend": "Max. bend (20 mm)", "kappa_rms": "Curvature RMS",
          "kappa_max": "Curvature max.", "kappa_mean": "Curvature mean", "bend_energy": "Bending energy",
          "arc_excess_pct": "Arc excess", "inflections": "Inflections"}
    rl = {"p_stable": "Stable pressure", "q_stable": "Stable flow", "p_lift": "Lift-off pressure", "t_lift": "Lift-off time",
          "t_stable": "Time to stable", "t_lift_to_stable": "Lift-off to stable", "n_incr": "No. of increases",
          "p_overshoot": "Pressure overshoot", "p_sd_settled": "Settling scatter", "lift_ratio": "Lift/stable ratio"}
    R = T.pivot(index="shape", columns="response", values="rho").reindex(index=shape, columns=resp)
    P = T.pivot(index="shape", columns="response", values="p_perm").reindex(index=shape, columns=resp)
    fig, ax = plt.subplots(figsize=(7.16, 3.9))
    im = ax.imshow(R.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(shape)):
        for j in range(len(resp)):
            v = R.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5, color="black" if abs(v) < 0.6 else "white")
                if P.values[i, j] < 0.10:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.0))
    ax.set_xticks(range(len(resp))); ax.set_xticklabels([rl[r] for r in resp], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(shape))); ax.set_yticklabels([sl[s] for s in shape], fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label(r"Spearman $\rho$ (within wire type)", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.text(0.01, 0.005, f"n = {int(T.n.max())} instances, 6 wire types; boxed: uncorrected p < 0.10; no cell significant after "
             f"Benjamini–Hochberg correction (min. q = {T.q_BH.min():.2f}).", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("thesis_figures/fig_correlation_heatmap.pdf"); fig.savefig("thesis_figures/fig_correlation_heatmap.png", dpi=250)


def copy_dev_vs_pressure():
    for ext in ("pdf", "png"):
        shutil.copy(f"more/analysis/figures/fig_marked_dev_vs_pressure.{ext}", f"thesis_figures/fig_dev_vs_pressure.{ext}")


if __name__ == "__main__":
    fig_all(); fig_2m(); fig_heatmap(); copy_dev_vs_pressure()
