"""Numbers quoted in the revised manuscript text, computed from saved arrays and checkpoints (read-only).

usage: python -B facts_for_methods.py <project root> <plan dir> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np

plan = Path(sys.argv[2])
sys.path.insert(0, str(plan / "STAGE_B_TRAIN"))
from stageb_problem import Problem, FIELDS11  # noqa: E402

root, out = Path(sys.argv[1]), Path(sys.argv[3])
P = Problem(root, plan, plan / "STAGE_B_TRAIN" / "BOUND_CODE")
cp = P.cp
res = {}

# ---- model facts
res["q_eta"] = float(cp["model"]["q_eta"])
res["q_amplitude_factor"] = float(1 + 0.1 * np.tanh(res["q_eta"]))
res["parameters_total"] = int(sum(v.numel() for v in cp["model"].values()))
res["parameter_shapes"] = {k: list(v.shape) for k, v in cp["model"].items()}
res["observation_scales"] = P.scales.tolist()
res["normalizers"] = P.normal.tolist()
res["moment_x_range"] = [float(P.moment_x.min()), float(P.moment_x.max())]
res["macro_x_range"] = [float(P.macro_x.min()), float(P.macro_x.max())]
res["moment_x_spacing_minmax"] = [float(np.diff(P.moment_x).min()), float(np.diff(P.moment_x).max())]
for name in ("colloc", "fresh", "initial", "bt"):
    pts = getattr(P, name).numpy()
    res["points_" + name] = {"count": int(len(pts)), "t_over_tf_range": [float(pts[:, 0].min() / P.tf), float(pts[:, 0].max() / P.tf)],
                             "x_range": [float(pts[:, 1].min()), float(pts[:, 1].max())],
                             "inside_|x|<=0.135": int((np.abs(pts[:, 1]) <= 0.135).sum()),
                             "inside_window": int(((pts[:, 1] >= P.moment_x.min()) & (pts[:, 1] <= P.moment_x.max())).sum())}
res["points_initial_xy"] = cp["initial_physics"].tolist()
res["points_boundary_xy"] = cp["incoming_boundary"].tolist()
g = cp["grid"]
v = g["v"].numpy()
res["velocity_grid"] = {"nodes": int(len(v)), "unique_per_axis": [int(len(np.unique(v[:, k]))) for k in range(3)],
                        "range": [float(v.min()), float(v.max())], "weight_value": float(g["w"][0]), "weights_constant": bool((g["w"] == g["w"][0]).all())}

# ---- field numbers at the three stages (project convention, 6400-cell reference)
d = np.load(plan / "STAGE_B_TRAIN" / "RUN_S_STRICT" / "ASSESSMENT_FIELDS.npz", allow_pickle=False)
x, ref = d["x"], d["reference"]
stages = {"original_envelopes_model170": d["model170"], "common_tail_no_training_170T": d["model170T"], "after_output_layer_updates": d["stageB"]}
ix = {n: i for i, n in enumerate(FIELDS11)}
names = ["rho", "u", "T", "q", "sigma", "m", "R", "Delta", "S_raw"]
win = (x >= P.moment_x.min()) & (x <= P.moment_x.max())
up = x < P.moment_x.min()
i1, i2 = int(np.argmin(np.abs(x + 0.20))), int(np.argmin(np.abs(x + 0.25)))
tab = {}
for s, y in stages.items():
    row = {}
    for n in names:
        j = ix[n]
        e = y[:, j] - ref[:, j]
        row[n] = {"full_L2_percent": float(100 * np.linalg.norm(e) / np.linalg.norm(ref[:, j])),
                  "upstream_share_of_squared_error_percent": float(100 * (e[up] ** 2).sum() / (e ** 2).sum()),
                  "window_L2_percent": float(100 * np.linalg.norm(e[win]) / np.linalg.norm(ref[win, j])),
                  "upstream_decay_rate": float(np.log(abs(y[i1, j] / y[i2, j])) / (x[i1] - x[i2])) if n not in ("rho", "u", "T") else None,
                  "min": float(y[:, j].min()), "max": float(y[:, j].max())}
    tab[s] = row
res["stages"] = tab
res["reference"] = {n: {"min": float(ref[:, ix[n]].min()), "max": float(ref[:, ix[n]].max()),
                        "upstream_decay_rate": float(np.log(abs(ref[i1, ix[n]] / ref[i2, ix[n]])) / (x[i1] - x[i2])) if n not in ("rho", "u", "T") else None,
                        "x_of_max": float(x[ref[:, ix[n]].argmax()]), "x_of_min": float(x[ref[:, ix[n]].argmin()])} for n in names}

# ---- reference refinement: 3200 against 6400 cells (same scheme and velocity grid)
r32 = np.load(root.parent / "SUPERVISED_S0_CHARACTERISTIC_SHARED_20261107" / "REFERENCE_REFINEMENT_REDUCED" / "CPU_REFERENCE" / "REFERENCE_3200.npz", allow_pickle=False)
res["reference_3200_keys"] = {k: list(r32[k].shape) for k in r32.files}
print("3200 file keys:", res["reference_3200_keys"])
out.write_text(json.dumps(res, indent=1), encoding="utf-8")
print("q_eta %.6f  factor %.6f  parameters %d" % (res["q_eta"], res["q_amplitude_factor"], res["parameters_total"]))
print("velocity grid:", res["velocity_grid"])
for k in ("colloc", "fresh", "initial", "bt"):
    print("points", k, res["points_" + k])
print("moment x range", res["moment_x_range"], "spacing", res["moment_x_spacing_minmax"], "| macro x range", res["macro_x_range"])
print("field   full L2 %% (170 / 170T / B) | upstream share of squared error %% (170) | decay rate (DVM / 170 / 170T / B)")
for n in names:
    a, b, c = (tab[s][n] for s in stages)
    rr = res["reference"][n]["upstream_decay_rate"]
    print("  %-6s %9.6f / %9.6f / %9.6f | %5.1f | %s" % (n, a["full_L2_percent"], b["full_L2_percent"], c["full_L2_percent"], a["upstream_share_of_squared_error_percent"],
          "-" if rr is None else "%.1f / %.1f / %.1f / %.1f" % (rr, a["upstream_decay_rate"], b["upstream_decay_rate"], c["upstream_decay_rate"])))
