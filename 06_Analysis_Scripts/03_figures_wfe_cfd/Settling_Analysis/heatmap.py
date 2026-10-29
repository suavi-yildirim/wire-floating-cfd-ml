import csv, glob, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import settling as s

FAPS = LinearSegmentedColormap.from_list("faps_seq", ["#E8F3D1", "#97C139", "#34677D", "#004359"])
SECTIONS = ["0,5", "1", "1,5", "4", "6"]
LENGTHS = ["0,5", "1", "2", "3,5"]

def header(path, key):
    return s.header_value(path, key)

records = []
for p in sorted(glob.glob(os.path.join(s.DATA_DIR, "WFE_*.csv"))):
    text = open(p, encoding="utf-8", errors="replace").read().replace("\r", "")
    if not any(l.strip() == "stable: True" for l in text.split("\n")):
        continue  # controller stability flag (794 runs)
    diameter = header(p, "TUBE_diameter") or ""
    tube = {"8": "WC08", "10": "WC10"}.get(diameter)
    if tube is None:
        continue
    sec = header(p, "WIRE_section")
    length = header(p, "WIRE_lenght")
    if sec is None or length is None:
        continue
    sec = sec.replace(".", ",")
    length = length.replace(".", ",")
    t = s.settling_time(p, s.load_response(p))
    records.append((os.path.basename(p), tube, sec, length, t))

print("controller-stable runs:", len(records))
with open("settling_summary.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "tube", "section_mm2", "length_m", "settling_s"])
    for r in records:
        w.writerow([r[0], r[1], r[2], r[3], "" if r[4] is None else round(r[4], 2)])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, tube in zip(axes, ["WC08", "WC10"]):
    grid = np.full((len(SECTIONS), len(LENGTHS)), np.nan)
    labels = [["" for _ in LENGTHS] for _ in SECTIONS]
    for i, sec in enumerate(SECTIONS):
        for j, length in enumerate(LENGTHS):
            cell = [r for r in records if r[1] == tube and r[2] == sec and r[3] == length]
            done = [r[4] for r in cell if r[4] is not None]
            if done:
                grid[i, j] = statistics.median(done)
            labels[i][j] = f"{'-' if not done else f'{grid[i, j]:.1f}'}\n{len(done)}/{len(cell)}"
    im = ax.imshow(grid, cmap=FAPS, aspect="auto", vmin=0, vmax=np.nanmax(grid))
    norm = Normalize(vmin=0, vmax=np.nanmax(grid))
    for i in range(len(SECTIONS)):
        for j in range(len(LENGTHS)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8, color=("white" if np.isfinite(grid[i, j]) and sum(FAPS(norm(grid[i, j]))[:3]) / 3 < 0.55 else "#1d1d1b"))
    ax.set_xticks(range(len(LENGTHS)), [f"{l} m" for l in LENGTHS])
    ax.set_yticks(range(len(SECTIONS)), [f"{s_} mm²" for s_ in SECTIONS])
    ax.set_xlabel("Wire length")
    ax.set_title(tube)
axes[0].set_ylabel("Cross-section")
fig.colorbar(im, ax=axes, label="Median settling time (s)", shrink=0.8)
left, right = axes[0].get_position(), axes[1].get_position()
fig.text((left.x0 + right.x1) / 2, right.y1 + 0.10, "Median settling time per configuration", fontsize=12, ha="center", va="bottom")
fig.savefig("settling_heatmap.pdf", bbox_inches="tight")
fig.savefig("settling_heatmap.png", bbox_inches="tight", dpi=200)
print("saved settling_heatmap.pdf")
