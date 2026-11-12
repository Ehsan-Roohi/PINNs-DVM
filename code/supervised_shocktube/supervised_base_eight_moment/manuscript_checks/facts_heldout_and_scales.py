"""Two checks for the manuscript text (saved arrays only):
  1. full-domain discrepancies when the two reference nodes that bracket each observation location are left out;
  2. whether the observation scales stored in the checkpoint equal the largest absolute target of each field.

usage: python facts_heldout_and_scales.py <ASSESSMENT_FIELDS.npz> <FACTS_FOR_METHODS.json> <out json>
"""
import json
import sys

import numpy as np

d = np.load(sys.argv[1], allow_pickle=False)
facts = json.load(open(sys.argv[2], encoding="utf-8"))
x, ref = d["x"], d["reference"]
cols = [str(c) for c in d["columns"]]
used = np.zeros(len(x), bool)
for name, fields in (("macro_x", (0, 1, 2)), ("moment_x", (3, 4, 5, 6, 9))):
    pass
out = {}
for stage in ("model170", "model170T", "stageB"):
    y = d[stage]
    row = {}
    for j, n in enumerate(cols):
        obs = d["macro_x"] if n in ("rho", "u", "T") else d["moment_x"]
        mask = np.ones(len(x), bool)
        for xo in obs:
            i = np.searchsorted(x, xo)
            mask[max(i - 1, 0):min(i + 1, len(x))] = False
        e = (y[:, j] - ref[:, j])[mask]
        row[n] = float(100 * np.linalg.norm(e) / np.linalg.norm(ref[mask, j]))
    out[stage] = row
res = {"L2_percent_excluding_nodes_bracketing_observations": out}
# observation scales
targ_macro = np.column_stack([np.interp(d["macro_x"], x, ref[:, k]) for k in (0, 1, 2)])
targ_mom = np.column_stack([np.interp(d["moment_x"], x, ref[:, k]) for k in (3, 4, 5, 6, 9)])
mx = np.r_[np.abs(targ_macro).max(0), np.abs(targ_mom).max(0)]
res["max_abs_target_per_field"] = mx.tolist()
res["stored_observation_scales"] = facts["observation_scales"]
res["scales_equal_max_abs_target"] = bool(np.allclose(mx, facts["observation_scales"], rtol=1e-12))
json.dump(res, open(sys.argv[3], "w", encoding="utf-8"), indent=1)
for n in ("q", "sigma", "m", "R", "Delta"):
    print("%-6s excluding observation neighbours: 170 %.4f  170T %.4f  B %.4f" % (n, out["model170"][n], out["model170T"][n], out["stageB"][n]))
print("scales equal max |target|:", res["scales_equal_max_abs_target"], "| ratio stored/max:", (np.array(facts["observation_scales"]) / mx).round(9).tolist())
