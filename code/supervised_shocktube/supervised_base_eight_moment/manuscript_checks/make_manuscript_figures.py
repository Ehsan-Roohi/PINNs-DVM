"""Figures of the revised data-assisted shock-tube subsection, drawn from saved arrays and logs only.

  figures/supervised/ORIGINAL_MAIN_SHOCKTUBE_CLEAN.pdf      eight signed fields (replaces the five-field figure; same basename)
  figures/supervised/SHOCKTUBE_HIGHER_UPSTREAM_DETAIL.pdf   upstream detail of the third- and fourth-order moments
  figures/loss/compact/SUPERVISED_OUTPUT_LAYER.pdf          history of the output-layer refinement

usage: python make_manuscript_figures.py <ASSESSMENT_FIELDS.npz> <RUN dir with RUN_SETTINGS.json and ACCEPTED.jsonl>
                                         <figures dir> <STAGE0_FIELDS.npz> [--skip-history]

The field figures use the layout of eight_field_grid.py (print width, legend in the ninth cell).
--skip-history leaves the existing history figure untouched.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eight_field_grid as G

d = np.load(sys.argv[1], allow_pickle=False)
run, figs = Path(sys.argv[2]), Path(sys.argv[3])
d0 = np.load(sys.argv[4], allow_pickle=False)
x, old, new, ref = d["x"], d["model170"], d["stageB"], d["reference"]
five = d0["five_field"]
assert np.array_equal(d0["x"], x) and np.array_equal(d0["reference"], ref) and list(d0["columns"]) == list(d["columns"])
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 13,
                     "pdf.fonttype": 42, "axes.linewidth": 0.8})
COLS = [(0, r"$\rho$"), (1, r"$u$"), (2, r"$T$"), (3, r"$q_x$"), (4, r"$\sigma_{xx}$"),
        (5, r"$m_{xxx}$"), (6, r"$R_{xx}$"), (9, r"$\Delta$")]
C_NEW, C_OLD, C_FIVE, C_REF = "#0072B2", "#7A7A7A", "#009E73", "#D55E00"
LEG4 = ["BGK PINN, five fields", "BGK PINN, eight moments, original envelopes",
        "BGK PINN, eight moments, common-tail envelope, refined", "BGK DVM"]
LEG4_CELL = ["BGK PINN, five fields", "BGK PINN, eight moments,\noriginal envelopes",
             "BGK PINN, eight moments,\ncommon-tail envelope,\nrefined", "BGK DVM"]  # same wording, wrapped for the cell


def draw(ax, j, msk, marks, with_five):
    hs = []
    if with_five:
        hs += ax.plot(x[msk], five[msk, j], ":", color=C_FIVE, lw=G.LW + 0.1)
    hs += ax.plot(x[msk], old[msk, j], "--", dashes=(2.2, 1.1), color=C_OLD, lw=G.LW)
    hs += ax.plot(x[msk], new[msk, j], "-", color=C_NEW, lw=G.LW)
    hs += ax.plot(x[marks], ref[marks, j], **G.MARK)
    return hs


(figs / "supervised").mkdir(parents=True, exist_ok=True)
(figs / "loss" / "compact").mkdir(parents=True, exist_ok=True)
previews = Path(__file__).resolve().parent

# ---- eight signed fields: three columns at print width, legend in the ninth cell
full = np.ones_like(x, bool)
with plt.rc_context(G.STYLE):
    fig = G.grid(lambda ax, k: draw(ax, COLS[k][0], full, G.marker_indices(x, ref[:, COLS[k][0]]), True), LEG4_CELL)
    fig.savefig(figs / "supervised" / "ORIGINAL_MAIN_SHOCKTUBE_CLEAN.pdf", metadata=G.NO_DATE)
    fig.savefig(previews / "preview_main.png", dpi=220)
    plt.close(fig)

# ---- upstream detail of the third- and fourth-order moments
up = (x >= -0.32) & (x <= -0.08)
with plt.rc_context(G.STYLE):
    fig, axes = plt.subplots(1, 3, figsize=(G.WIDTH, 2.45), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.05, wspace=0.02)
    for ax, (j, lab) in zip(axes.flat, COLS[5:]):
        hs = draw(ax, j, up, np.flatnonzero(up)[::80], False)  # every 80th reference cell: spacing 0.0125
        G.style_axis(ax, lab)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    fig.legend(hs, LEG4[1:], loc="outside lower center", ncol=2, frameon=False, columnspacing=2.2, handlelength=2.6)
    fig.savefig(figs / "supervised" / "SHOCKTUBE_HIGHER_UPSTREAM_DETAIL.pdf", metadata=G.NO_DATE)
    fig.savefig(previews / "preview_detail.png", dpi=220)
    plt.close(fig)

if "--skip-history" in sys.argv:
    print("written under", figs, "(history figure left as it is)")
    raise SystemExit(0)

# ---- history of the output-layer refinement (accepted updates only)
s = json.loads((run / "RUN_SETTINGS.json").read_text(encoding="utf-8"))
src = np.array(s["source_components"][:15])
acc = [json.loads(line) for line in (run / "ACCEPTED.jsonl").read_text(encoding="utf-8").splitlines()]
hist = np.array([src] + [a["components"][:15] for a in acc]) / src
it = np.arange(len(hist))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.6))
styles = [("-", "#0072B2"), ("--", "#0072B2"), (":", "#0072B2"), ("-", "#009E73"), ("--", "#009E73"),
          ("-", "#D55E00"), ("--", "#D55E00"), (":", "#D55E00")]
labs = [r"$\rho$", r"$u$", r"$T$", r"$q_x$", r"$\sigma_{xx}$", r"$m_{xxx}$", r"$R_{xx}$", r"$\Delta$"]
for k in range(8):
    a1.semilogy(it, hist[:, k], styles[k][0], color=styles[k][1], lw=1.5, label=labs[k])
a1.set_xlabel("Accepted update")
a1.set_ylabel("Observation loss / initial value", fontsize=12)
a1.set_yticks([1.0, 0.5, 0.2, 0.1, 0.05])
a1.set_yticklabels(["1", "0.5", "0.2", "0.1", "0.05"])
a1.minorticks_off()
a1.grid(alpha=0.22, lw=0.5, which="both")
a1.legend(ncol=4, frameon=False, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.2), columnspacing=1.6)
pl = [(8, "Kinetic residual", "-", "#0072B2"), (9, "Relative residual", "--", "#0072B2"), (10, "Mass balance", "-", "#009E73"),
      (11, "Momentum balance", "--", "#009E73"), (12, "Energy balance", ":", "#009E73"), (13, "Initial residual", "-", "#D55E00"),
      (14, "Incoming boundary", "--", "#D55E00")]
for k, lab, ls, col in pl:
    a2.plot(it, hist[:, k], ls, color=col, lw=1.5, label=lab)
a2.set_xlabel("Accepted update")
a2.set_ylabel("Physical component / initial value", fontsize=12)
a2.grid(alpha=0.22, lw=0.5)
a2.legend(ncol=2, frameon=False, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.2), columnspacing=1.6)
fig.tight_layout(w_pad=3.0)
fig.savefig(figs / "loss" / "compact" / "SUPERVISED_OUTPUT_LAYER.pdf", bbox_inches="tight")
fig.savefig(figs.parent.parent / "work" / "preview_output_layer.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("max of any protected component relative to its initial value:", float(hist[:, [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14]].max()))
print("written under", figs)
