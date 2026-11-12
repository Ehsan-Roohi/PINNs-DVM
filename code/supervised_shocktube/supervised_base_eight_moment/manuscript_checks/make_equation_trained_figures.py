"""Figures 10 and 11 (equation-trained BGK and Shakhov shock tubes) in the three-column layout.

Re-drawn from the saved arrays read by the original plotting script (plot_refinement_saved.py): the model fields
at their 401 saved positions (PINN_FIELDS.npz) and the 3200-cell DVM fields (REFERENCE_3200_COMPARISON.npz).
No field value is changed; only the layout, the lettering size and the number of reference markers differ.

  figures/shocktube_physics/shocktube_bgk_physics.pdf
  figures/shocktube_physics/shocktube_shakhov_physics.pdf

usage: python make_equation_trained_figures.py <OUTPUT dir containing bgk/ and shakhov/> <figures dir> <check json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eight_field_grid as G

root, figs, check_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
C = ["rho", "u", "T", "q", "sigma", "m", "R", "Delta", "S"]
# Table 8 of the manuscript (relative L2 discrepancy in per cent on the 3200 reference nodes).
TABLE8 = {"bgk": [0.0096, 0.2042, 0.0092, 0.4100, 0.3604, 0.8263, 1.7145, 1.3976, 1.5759],
          "shakhov": [0.0096, 0.2029, 0.0092, 0.4101, 0.3606, 0.8284, 1.7125, 1.3979, 1.5748]}
check = {}
for fam, label in (("bgk", "BGK"), ("shakhov", "Shakhov")):
    own = np.load(root / fam / "PINN_FIELDS.npz", allow_pickle=False)
    z = np.load(root / fam / "REFERENCE_3200_COMPARISON.npz", allow_pickle=False)
    assert list(own["columns"]) == C and list(z["columns"]) == C
    x, ref, pred = z["x"], z["reference"], z["prediction"]
    # The saved arrays are those behind table 8: recompute its entries before drawing.
    l2 = [100 * np.linalg.norm(pred[:, j] - ref[:, j]) / np.linalg.norm(ref[:, j]) for j in range(9)]
    assert np.allclose(l2, TABLE8[fam], atol=6e-5), (fam, l2)
    counts = []

    def draw(ax, k):
        ir = G.marker_indices(x, ref[:, k])  # regular set, denser in the core, and the extrema of the field
        counts.append(int(len(ir)))
        hs = ax.plot(own["x"], own["fields"][:, k], "-", color=G.C_BLUE, lw=G.LW)
        hs += ax.plot(x[ir], ref[ir, k], **G.MARK)
        return hs

    with plt.rc_context(G.STYLE):
        fig = G.grid(draw, [label + " data-free PINN", label + " DVM"])
        out = figs / "shocktube_physics" / ("shocktube_%s_physics.pdf" % fam)
        fig.savefig(out, metadata=G.NO_DATE)
        fig.savefig(Path(__file__).resolve().parent / ("preview_%s_physics.png" % fam), dpi=220)
        plt.close(fig)
    check[fam] = {"recomputed_table8_percent": dict(zip(C, [round(float(v), 4) for v in l2])),
                  "table8_percent": dict(zip(C, TABLE8[fam])), "model_positions": int(len(own["x"])),
                  "reference_nodes": int(len(x)), "reference_markers_per_panel": counts, "figure": out.name}
    print(fam, "table 8 reproduced;", counts, "reference markers per panel ->", out.name)
check_path.write_text(json.dumps(check, indent=1), encoding="utf-8")
