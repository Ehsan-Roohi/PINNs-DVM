"""Common layout of the eight-field shock-tube figures (figures 4, 10 and 11 of the manuscript).

Three columns drawn at the printed text width, so that the lettering appears at the stated point sizes; the
legend occupies the ninth cell, as in the five-field figure of the submitted manuscript.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator, MultipleLocator

WIDTH = 5.3  # inches: printed text width of the journal class
STYLE = {"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 8, "axes.labelsize": 9.5,
         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.45,
         "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.7, "pdf.fonttype": 42}
LABELS = [r"$\rho$", r"$u$", r"$T$", r"$q_x$", r"$\sigma_{xx}$", r"$m_{xxx}$", r"$R_{xx}$", r"$\Delta$"]
C_BLUE, C_GREY, C_GREEN, C_ORANGE = "#0072B2", "#7A7A7A", "#009E73", "#D55E00"
LW = 1.0
MARK = dict(marker="o", ls="none", mfc="none", mec=C_ORANGE, ms=2.9, mew=0.7)
NO_DATE = {"CreationDate": None}  # byte-reproducible PDF files


def marker_indices(x, y, core=0.10, step=0.05, min_gap=0.012):
    """Reference nodes shown as symbols: spacing `step` outside |x| <= core and step/2 inside, and the two
    extrema of the field itself. A regular node closer than `min_gap` to an extremum is left out."""
    targets = np.r_[np.arange(-0.5, 0.5 + 1e-9, step), np.arange(-core + step / 2, core, step)]
    regular = np.unique([int(np.argmin(np.abs(x - t))) for t in targets])
    keep = sorted({int(np.argmax(y)), int(np.argmin(y))})
    keep += [int(i) for i in regular if all(abs(x[i] - x[k]) >= min_gap for k in keep)]
    return np.array(sorted(keep))


def style_axis(ax, ylabel):
    ax.set_xlabel(r"$x$", labelpad=1.5)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.yaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
    ax.grid(alpha=0.22, lw=0.4)
    ax.tick_params(pad=2.5)


def grid(draw, legend_labels, height=4.75, **legend_kw):
    """draw(ax, k) plots field k = 0..7 and returns its legend handles; returns the figure."""
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, height), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.05, wspace=0.02, hspace=0.02)
    for k, ax in enumerate(axes.flat[:8]):
        handles = draw(ax, k)
        style_axis(ax, LABELS[k])
        ax.xaxis.set_major_locator(FixedLocator([-0.5, 0.0, 0.5]))  # the ends of the tube, as in the original figures
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    axes.flat[8].axis("off")
    for c in range(3):
        fig.align_ylabels(axes[:, c])
    # Legend centred in the ninth cell: from the left edge of the third column to the right edge of the figure,
    # at the height of the frames of the third row.
    fig.draw_without_rendering()
    inv = fig.transFigure.inverted()
    left = inv.transform_bbox(axes[1, 2].get_tightbbox()).x0
    row = axes[2, 1].get_position()
    kw = dict(loc="center", frameon=False, handlelength=2.1, handletextpad=0.6, labelspacing=1.0,
              borderaxespad=0.0, borderpad=0.0)
    kw.update(legend_kw)
    leg = fig.legend(handles, legend_labels, bbox_to_anchor=(0.5 * (left + 1.0), 0.5 * (row.y0 + row.y1)),
                     bbox_transform=fig.transFigure, **kw)
    leg.set_in_layout(False)
    fig.draw_without_rendering()
    box = inv.transform_bbox(leg.get_window_extent())
    assert box.x0 >= left - 0.005 and box.x1 <= 0.995, ("legend does not fit its cell", box.x0, left, box.x1)
    return fig
