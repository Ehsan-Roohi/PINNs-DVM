"""DVM refinement difference between the 3200-cell and the 6400-cell BGK references (same scheme and
48x20x20 velocity grid), with the convention of the manuscript's table tab:transient_physics:
the fine reference is linearly interpolated onto the coarse nodes; relative L2 difference in percent.

usage: python reference_refinement_3200_6400.py <REFERENCE_3200.npz> <REFERENCE6400_FIELDS.npz> <out json>
"""
import json
import sys

import numpy as np

c, f = np.load(sys.argv[1], allow_pickle=False), np.load(sys.argv[2], allow_pickle=False)
xc, yc, xf, yf = c["x"], c["fields"], f["x"], f["reference"]
cols_c, cols_f = [str(a) for a in c["columns"]], [str(a) for a in f["columns"]]
assert cols_c == cols_f, (cols_c, cols_f)
out = {"columns": cols_c, "coarse_cells": int(len(xc)), "fine_cells": int(len(xf)),
       "convention": "fine interpolated linearly onto coarse nodes; 100*||fine-coarse||_2/||coarse||_2"}
diff = {}
for k, n in enumerate(cols_c):
    g = np.interp(xc, xf, yf[:, k])
    diff[n] = float(100 * np.linalg.norm(g - yc[:, k]) / np.linalg.norm(yc[:, k]))
out["relative_L2_difference_percent"] = diff
out["heat_flux_peak"] = {"3200": [float(yc[:, 3].max()), float(xc[yc[:, 3].argmax()])], "6400": [float(yf[:, 3].max()), float(xf[yf[:, 3].argmax()])]}
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
