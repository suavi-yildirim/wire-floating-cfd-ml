# Script from the 2026-10-19 thesis session (transcript command s067).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D, OUT = sys.argv[1], sys.argv[2]
def load(name):
    rows = []
    with open(f"{D}/{name}-rfile.out") as f:
        for line in f:
            p = line.split()
            if len(p) == 2 and p[0].isdigit():
                rows.append((int(p[0]), float(p[1])))
    a = np.array(rows); return a[:, 0], a[:, 1]
series = [
    ("mass-flow-inlet", r"Inlet mass flow rate (kg/s)", 1.0),
    ("silencer-pressure-drop", r"Silencer pressure drop (kPa)", 1e-3),
    ("launching-station-pressure-drop", r"Launching-station pressure drop (kPa)", 1e-3),
]
plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm", "axes.linewidth": 0.6})
fig, axes = plt.subplots(3, 1, figsize=(4.8, 5.6), sharex=True)
for ax, (name, label, sc) in zip(axes, series):
    it, v = load(name)
    ax.plot(it, v * sc, color="#1f4e79", lw=1.0)
    ax.set_ylabel(label, fontsize=8)
    ax.grid(True, ls=":", lw=0.4)
    last = it[-1]; m = it >= last - 50
    ch = (v[m].max() - v[m].min()) / abs(v[-1]) * 100
    ax.text(0.98, 0.9, rf"change over last 50 it.: {ch:.1f}\,\%", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    print(name, "last it", last, "value", v[-1], "range last50 %", round(ch, 2), "value at it", it[np.argmin(abs(it-(last-50)))])
axes[-1].set_xlabel("Iteration")
fig.tight_layout(); fig.savefig(OUT); print("saved", OUT)