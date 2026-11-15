"""Shared figure style (reference palette of the dataviz method; light surface, static PNG).

Configs keep one color each everywhere (color follows the entity): A = slot 1 blue, B = slot 2 orange,
C = slot 3 aqua. Slots 1–3 were validated all-pairs on the light surface (validate_palette.js); aqua is
below 3:1 contrast, so every chart that uses it also labels the marks directly.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b1"
# Color encodes the window type (long / short symmetric / dual), so the 5.8 and 2.9 ms configs of a type share it.
CONFIG_COLOR = {"A_sym1024": BLUE, "B_sym256": ORANGE, "C_dual": AQUA, "D_dual_2p9ms": AQUA, "E_sym128": ORANGE}
CONFIG_LABEL = {"A_sym1024": "A  long window (23.2 ms)", "B_sym256": "B  short window (5.8 ms)",
                "C_dual": "C  dual window (5.8 ms)", "D_dual_2p9ms": "D  dual window (2.9 ms)",
                "E_sym128": "E  short window (2.9 ms)"}
CONFIG_SHORT = {"A_sym1024": "A", "B_sym256": "B", "C_dual": "C", "D_dual_2p9ms": "D", "E_sym128": "E"}
WINDOW_TYPE = {"A_sym1024": "long", "B_sym256": "short", "C_dual": "dual", "D_dual_2p9ms": "dual", "E_sym128": "short"}


def _register_fonts():
    import glob
    import os

    from matplotlib import font_manager
    for f in glob.glob(os.path.expanduser("~/.local/share/fonts/**/Carlito-*.ttf"), recursive=True):
        font_manager.fontManager.addfont(f)


def apply_style():
    _register_fonts()
    plt.rcParams.update({
        "font.family": ["Carlito", "DejaVu Sans"], "font.size": 10.5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.labelsize": 10.5,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.major.pad": 5, "ytick.major.pad": 5,
        "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
        "legend.handletextpad": 0.4, "legend.columnspacing": 1.8, "legend.handlelength": 1.6,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.2,
    })


def header(fig, x, top, title, subtitle=""):
    """Title + subtitle above the plot area (figure coordinates; ``top`` = top of the axes)."""
    line = 1.4 * 10 / 72 / fig.get_size_inches()[1]
    n = subtitle.count("\n") + 1 if subtitle else 0
    y = top + 0.03
    if subtitle:
        fig.text(x, y, subtitle, fontsize=10, color=INK2, ha="left", va="bottom", linespacing=1.4)
    fig.text(x, y + n * line + 0.014, title, fontsize=14.5, fontweight="bold", color=INK, ha="left", va="bottom")


def dot(ax, x, y, color, size=60, zorder=3, **kw):
    """Marker with a 2px surface ring."""
    return ax.scatter(x, y, s=size, color=color, edgecolors=SURFACE, linewidths=1.6, zorder=zorder, **kw)
