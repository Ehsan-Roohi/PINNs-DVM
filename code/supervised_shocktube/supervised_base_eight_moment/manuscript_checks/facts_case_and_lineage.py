"""Facts needed for the manuscript text (read-only on the project):
  1. the case of checkpoint 170 (Kn, t_final, initial jump) and of the DVM reference config;
  2. agreement between the 400-point DVM reference of the manuscript's Figure 4 and the 6400-point
     reference used in the higher-moment work;
  3. how model 170 was reached (accepted-update history stored in the checkpoint).

usage: python -B facts_case_and_lineage.py <project root> <plan dir> <out json>
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(sys.argv[2]) / "STAGE_B_TRAIN"))
from stageb_problem import Problem  # noqa: E402

root, plan, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
P = Problem(root, plan, Path(sys.argv[2]) / "STAGE_B_TRAIN" / "BOUND_CODE")
cp = P.cp
res = {"case_of_checkpoint_170": cp["config"]["case"], "velocity_grid_of_checkpoint_170": cp["config"].get("velocity")}
print("case:", json.dumps(cp["config"]["case"]))
print("velocity:", json.dumps(cp["config"].get("velocity")))
ref_cfg = json.loads((root / "REFERENCE_CONFIG.json").read_text())
res["REFERENCE_CONFIG"] = ref_cfg
print("REFERENCE_CONFIG.json:", json.dumps(ref_cfg)[:900])

# ---- 400-point reference of the manuscript base case vs the 6400-point reference
for name in ("TARGETS_ALL.npz", "TARGETS9.npz"):
    z = np.load(root / name, allow_pickle=False)
    print(name, {k: z[k].shape for k in z.files})
z = np.load(root / "TARGETS_ALL.npz", allow_pickle=False)
x4, r4 = z["x"], z["reference"]
cols = ["rho", "u", "T", "q", "sigma", "m", "R"]
cmp = {}
for k, n in enumerate(cols[:r4.shape[1]]):
    fine = np.interp(x4, P.rx, P.ry[:, k])
    cmp[n] = {"relative_L2_difference_percent": float(100 * np.linalg.norm(fine - r4[:, k]) / np.linalg.norm(r4[:, k])),
              "max_abs_difference": float(np.abs(fine - r4[:, k]).max()), "max_abs_400": float(np.abs(r4[:, k]).max())}
res["reference_6400_vs_400"] = cmp
print("6400-point reference interpolated to the 400 x-values, against TARGETS_ALL 'reference':")
for n, v in cmp.items():
    print("  %-6s rel L2 difference %.4f %%   max abs %.3e (field max %.3e)" % (n, v["relative_L2_difference_percent"], v["max_abs_difference"], v["max_abs_400"]))

# ---- lineage recorded in checkpoint 170
for k in ("schema", "accepted_updates", "attempts", "optimizer_kind", "controller_policy", "data_provenance", "phase",
          "all_parameters_trainable", "parent_accepted_updates", "cumulative_CPU_accepted_joint_cycles", "LM_mu", "function_radius"):
    v = cp.get(k)
    s = json.dumps(v, default=str)
    print("cp[%s] = %s" % (k, s if len(s) < 700 else s[:700] + " ..."))
    res["cp_" + k] = v if len(s) < 4000 else s[:4000]
gh = cp.get("guard_history", [])
print("guard_history entries:", len(gh), "| type of first:", type(gh[0]).__name__ if gh else None)
if gh and isinstance(gh[0], dict):
    print("keys of first entry:", sorted(gh[0].keys())[:40])
    print("keys of last entry:", sorted(gh[-1].keys())[:40])
    for key in ("optimizer", "optimizer_kind", "kind", "phase", "stage", "block", "controller", "accepted", "accepted_joint_update", "commit_status"):
        vals = [str(e.get(key)) for e in gh if key in e]
        if vals:
            print("  %s: %s" % (key, dict(Counter(vals).most_common(8))))
    res["guard_history_len"] = len(gh)
    res["guard_history_first"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(type(v))) for k, v in gh[0].items()}
    res["guard_history_last"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(type(v))) for k, v in gh[-1].items()}
for k in sorted(cp.keys()):
    if k.startswith("source") or k.endswith("sha256"):
        print("cp[%s] = %s" % (k, cp[k]))
out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
