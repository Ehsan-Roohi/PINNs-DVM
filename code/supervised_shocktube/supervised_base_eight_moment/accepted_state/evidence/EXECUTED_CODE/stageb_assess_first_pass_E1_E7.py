"""Endpoint assessment of a Stage B run against the held-out dense DVM reference.

Run once per finished training run. The criteria were written before any Stage B candidate was
assessed:
  E1-E5  the endpoint rules of the project's first trainer (train_all_moments.py, lines 193-198),
         with the transplanted model 170T as "before";
  E6     unobserved regions (no moment observation there): the squared error of q, sigma, m, R, Delta
         upstream and downstream of the observation window may not exceed 1.25 x its 170T value;
  E7     every protected training component within the limit the run declared.

usage: python -B stageb_assess.py <project root> <plan dir> <run dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stageb_problem import Problem, FIELDS11, NAMES, sha  # noqa: E402

root, plan, run = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
P = Problem(root, plan, Path(__file__).resolve().parent / "BOUND_CODE")
final_file = run / "STATE_FINAL.pt"
ck = torch.load(final_file, map_location="cpu", weights_only=False)
train = json.loads((run / "TRAIN_RESULT.json").read_text(encoding="utf-8"))
assert train["final_state_sha256"] == sha(final_file.read_bytes())

m170, mT, mB = P.model170(), P.model170T(), P.model170T(ck["model"])
changed = [k for k, v in mB.state_dict().items() if not torch.equal(v, mT.state_dict()[k])]
A170, p170, ref = P.assess(m170)
AT, pT, _ = P.assess(mT)
AB, pB, _ = P.assess(mB)
np.savez_compressed(run / "ASSESSMENT_FIELDS.npz", x=P.rx, model170=p170, model170T=pT, stageB=pB, reference=ref,
                    columns=np.array(FIELDS11), moment_x=P.moment_x, macro_x=P.macro_x)

idx = {n: i for i, n in enumerate(FIELDS11)}
lower, high = ["rho", "u", "T", "q", "sigma"], ["m", "R", "Delta"]
crit = {}
b, a = AT["full_L2_percent"], AB["full_L2_percent"]
e1 = {n: [b[idx[n]], a[idx[n]]] for n in lower}
crit["E1_lower_fields_full_L2"] = {"before_after": e1, "rule": "after <= before*(1+1e-3)+1e-5",
                                   "pass": all(v[1] <= v[0] * (1 + 1e-3) + 1e-5 for v in e1.values())}
e2 = {n: [b[idx[n]], a[idx[n]]] for n in high}
crit["E2_higher_fields_full_L2_strictly_lower"] = {"before_after": e2, "pass": all(v[1] < v[0] for v in e2.values())}
peaks = {}
for name, k, j in (("q_max", 3, 1), ("sigma_min", 4, 0), ("sigma_max", 4, 1)):
    r = AT["reference_extrema"][k][j]
    peaks[name] = [abs(AT["signed_extrema"][k][j] - r), abs(AB["signed_extrema"][k][j] - r)]
crit["E3_peak_errors"] = {"before_after": peaks, "rule": "after <= before*(1+1e-3)+1e-10",
                          "pass": all(v[1] <= v[0] * (1 + 1e-3) + 1e-10 for v in peaks.values())}
fb, fa = AT["fresh_physics"], AB["fresh_physics"]
e4 = {"raw_PDE": [fb["raw_PDE"], fa["raw_PDE"]], "relative_PDE": [fb["relative_PDE"], fa["relative_PDE"]]}
for k, n in enumerate(("mass", "momentum", "energy")):
    e4["balance_RMS_" + n] = [fb["balance_RMS"][k], fa["balance_RMS"][k]]
crit["E4_fresh_physics"] = {"before_after": e4, "rule": "after <= before*(1+1e-3)",
                            "pass": all(v[1] <= v[0] * (1 + 1e-3) for v in e4.values())}
crit["E5_S_raw_full_L2"] = {"before_after": [b[10], a[10]], "rule": "after <= before*(1+1e-3)",
                            "pass": a[10] <= b[10] * (1 + 1e-3)}
e6 = {}
for n in ["q", "sigma"] + high:
    for reg in ("upstream_of_window_x_lt_-0.104", "downstream_of_window"):
        e6[n + " " + reg] = [AT["regional"][n]["SSE"][reg], AB["regional"][n]["SSE"][reg]]
crit["E6_unobserved_regions_SSE"] = {"before_after": e6, "rule": "after <= 1.25*before",
                                     "pass": all(v[1] <= 1.25 * v[0] for v in e6.values())}
crit["E7_protected_training_components_within_declared_limits"] = {"pass": bool(train["protected_within_limits"])}

out = {"schema": "STAGE_B_ENDPOINT_ASSESSMENT_V1", "run": str(run), "final_state_sha256": train["final_state_sha256"],
       "accepted_updates": train["accepted_updates"], "stop_reason": train["stop_reason"],
       "state_entries_changed_vs_170T": changed, "criteria": crit,
       "all_criteria_pass": all(c["pass"] for c in crit.values()),
       "training_components": {"names": NAMES, "source_170T": train["source_components"], "final": train["final_components"]},
       "model170": A170, "model170T": AT, "stageB": AB, "paper_admission": False}
(run / "ASSESSMENT.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

print("run:", run.name, "| accepted updates", train["accepted_updates"], "| stop", train["stop_reason"])
print("full L2 %  (model 170 -> 170T -> Stage B):")
for n in lower + high + ["S_raw"]:
    print("  %-6s %10.6f -> %10.6f -> %10.6f" % (n, A170["full_L2_percent"][idx[n]], b[idx[n]], a[idx[n]]))
print("regional SSE, 170T -> Stage B (upstream of window | observed span | downstream of window):")
for n in ["q", "sigma"] + high:
    s0, s1 = AT["regional"][n]["SSE"], AB["regional"][n]["SSE"]
    print("  %-6s %.3e -> %.3e | %.3e -> %.3e | %.3e -> %.3e" % (
        n, s0["upstream_of_window_x_lt_-0.104"], s1["upstream_of_window_x_lt_-0.104"], s0["observed_span"], s1["observed_span"],
        s0["downstream_of_window"], s1["downstream_of_window"]))
print("fresh physics, 170T -> Stage B:")
for k, v in e4.items():
    print("  %-22s %.9e -> %.9e  (%+.4f %%)" % (k, v[0], v[1], 100 * (v[1] / v[0] - 1)))
for k, c in crit.items():
    print("%-58s %s" % (k, "PASS" if c["pass"] else "FAIL"))
print("all criteria pass:", out["all_criteria_pass"])
