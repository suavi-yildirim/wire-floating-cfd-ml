"""Single figure style for the WFE thesis (FAPS template: A4, \\textwidth = 16 cm, Latin Modern)."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CM = 1 / 2.54
TEXTWIDTH = 16.0 * CM          # inches
HALF = 7.8 * CM

# ---- colour system ---------------------------------------------------------------------
INK = "#1A1A1A"        # primary text
INK2 = "#4D4D4D"       # secondary text / axes
INK3 = "#8C8C8C"       # muted annotations
GRID = "#E3E3E3"
BAND = "#EEF1F4"       # tolerance bands / shaded regions

# Cross-section = the physical insulation colour of each wire, re-stepped so that all
# 10 pairs clear CVD dE>=8 and normal-vision dE>=15 (validated, --pairs all).
SECTIONS = [0.5, 1.0, 1.5, 4.0, 6.0]
SEC_COLOR = {0.5: "#E86A17", 1.0: "#2F7DD9", 1.5: "#DDAA00", 4.0: "#7A3CB8", 6.0: "#A8102F"}
SEC_MARKER = {0.5: "o", 1.0: "s", 1.5: "^", 4.0: "D", 6.0: "v"}
SEC_NAME = {0.5: "orange", 1.0: "blue", 1.5: "yellow", 4.0: "violet", 6.0: "red"}

# Tube channel (used only in figures where cross-section is not colour-coded)
TUBE_COLOR = {"WC08": "#2A5C9E", "WC10": "#6A8A22"}     # FAU-blue-like / FAPS green (validated pair)
TUBE_LABEL = {"WC08": r"WC08 ($D$ = 8 mm)", "WC10": r"WC10 ($D$ = 10 mm)"}

LENGTHS = [0.5, 1.0, 2.0, 3.5]
LEN_SIZE = {0.5: 18, 1.0: 30, 2.0: 46, 3.5: 68}           # scatter marker area by wire length

FAPS_GREEN = "#668E23"
# ordinal ramps (validated --ordinal): light -> dark
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
GREEN_RAMP = ["#9DBE6C", "#7FA43F", "#668E23", "#4A6A16", "#2E440B"]
TUBE_RAMP = {"WC08": BLUE_RAMP, "WC10": GREEN_RAMP}
# force components on the wire (validated, all pairs)
COMP = {"bottom": "#2F6DB5", "wall": "#668E23", "top": "#C2477A"}
# pressure budget - emphasis palette: wire region in green, porous zones in slate
BUDGET = {"feed": "#56677A", "wire": "#668E23", "sil": "#A7B3C0"}
ACCENT = "#E86A17"


def _pin_lm10():
    """Latin Modern ships several optical sizes under one family name; keep the 10-pt design."""
    from matplotlib import font_manager as fm
    fm.fontManager.ttflist = [f for f in fm.fontManager.ttflist
                              if not (f.name == "Latin Modern Roman" and "lmroman10" not in f.fname)]


def apply():
    _pin_lm10()
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": True,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.title_fontsize": 7.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.45,
        "grid.linestyle": "-",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.3,
        "lines.markersize": 4.5,
        "lines.markeredgewidth": 0.6,
        "patch.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.35,
        "legend.columnspacing": 1.0,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlepad": 4,
        "axes.labelpad": 3,
        "figure.constrained_layout.use": False,
    })


def fit_width(fig, target_w: float = None, tol: float = 0.004):
    """Grow/shrink the canvas until the *cropped* figure is exactly target_w inches wide,
    so every figure can be included at width=\\textwidth without rescaling its fonts."""
    target_w = target_w or TEXTWIDTH
    for _ in range(6):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        dw = target_w - bb.width
        if abs(dw) < tol:
            break
        fw, fh = fig.get_size_inches()
        fig.set_size_inches(fw + dw, fh)


def save(fig, name: str, width: float = None):
    fit_width(fig, width)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return OUT / f"{name}.png"


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.02, ha="right"):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="bottom", ha=ha, color=INK)


def panel_title(ax, text: str, letter: str | None = None):
    """Left-aligned panel header: '(a)  WC08 (D = 8 mm)'."""
    s = f"$\\bf{{({letter})}}$  {text}" if letter else text
    ax.set_title(s, loc="left", fontsize=9, color=INK, pad=5)


def section_handles(with_marker=True, lw=1.3, ms=4.8):
    hs = []
    for s in SECTIONS:
        hs.append(Line2D([], [], color=SEC_COLOR[s], lw=lw if lw else 0,
                         marker=SEC_MARKER[s] if with_marker else None, ms=ms,
                         mec="white", mew=0.6, label=f"{s:.1f} mm$^2$"))
    return hs


def length_handles():
    return [Line2D([], [], ls="", marker="o", ms=(LEN_SIZE[L]) ** 0.5, mfc="none", mec=INK2, mew=0.7,
                   label=f"{L:g} m") for L in LENGTHS]


def soften_spines(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
