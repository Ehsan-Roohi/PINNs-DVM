"""Facts requested by the independent scientific review, from saved arrays only (no model evaluation).

1. Like-for-like comparison on the 6400-cell BGK DVM: the equation-trained BGK state (401 saved positions) and the
   data-assisted final state at the same positions; also against the first-order extrapolate 2*DVM6400 - DVM3200.
2. Sign changes and genuine decay rates of the tails of the three eight-moment states.
3. Where the remaining q and sigma error of the final state sits.

usage: python facts_after_review.py <ASSESSMENT_FIELDS.npz> <REFERENCE_3200.npz> <equation-trained PINN_FIELDS.npz> <out json>
"""
import json
import sys

import numpy as np

d = np.load(sys.argv[1], allow_pickle=False)
c3 = np.load(sys.argv[2], allow_pickle=False)
pe = np.load(sys.argv[3], allow_pickle=True)
x, ref = d["x"], d["reference"]
cols = [str(a) for a in d["columns"]]                       # 11 columns of the assessment arrays
names9 = ["rho", "u", "T", "q", "sigma", "m", "R", "Delta", "S"]
idx11 = [cols.index(n) for n in ["rho", "u", "T", "q", "sigma", "m", "R", "Delta", "S_raw"]]
states = {"original_envelopes": d["model170"], "common_tail_same_weights": d["model170T"], "final": d["stageB"]}
out = {}

# ---- 1. like-for-like on the equation-trained positions
xe, fe = pe["x"], pe["fields"]
assert [str(a) for a in pe["columns"]] == names9
assert [str(a) for a in c3["columns"]] == names9
x3, f3 = c3["x"], c3["fields"]
ref_e = np.column_stack([np.interp(xe, x, ref[:, j]) for j in idx11])
ref3_e = np.column_stack([np.interp(xe, x3, f3[:, k]) for k in range(9)])
extrap_e = 2 * ref_e - ref3_e                                 # first-order Richardson extrapolate
fin_e = np.column_stack([np.interp(xe, x, states["final"][:, j]) for j in idx11])
l2 = lambda a, b: (100 * np.linalg.norm(a - b, axis=0) / np.linalg.norm(b, axis=0)).tolist()
out["positions"] = {"count": int(len(xe)), "range": [float(xe.min()), float(xe.max())]}
out["equation_trained_vs_6400"] = dict(zip(names9, l2(fe, ref_e)))
out["equation_trained_vs_3200"] = dict(zip(names9, l2(fe, ref3_e)))
out["data_assisted_final_vs_6400_same_positions"] = dict(zip(names9, l2(fin_e, ref_e)))
out["ratio_data_assisted_over_equation_trained"] = {n: out["data_assisted_final_vs_6400_same_positions"][n] / out["equation_trained_vs_6400"][n] for n in names9}
out["dvm_3200_vs_6400_same_positions"] = dict(zip(names9, l2(ref3_e, ref_e)))
out["equation_trained_vs_extrapolate"] = dict(zip(names9, l2(fe, extrap_e)))
out["data_assisted_final_vs_extrapolate"] = dict(zip(names9, l2(fin_e, extrap_e)))
out["heat_flux_maximum"] = {"equation_trained": float(fe[:, 3].max()), "data_assisted_final": float(states["final"][:, idx11[3]].max()),
                            "dvm_3200": float(f3[:, 3].max()), "dvm_6400": float(ref[:, idx11[3]].max()),
                            "extrapolate_at_saved_positions": float(extrap_e[:, 3].max())}

# ---- 2. tails: values at -0.20 and -0.25, zero crossings, rates on sign-definite intervals
def at(y, xx):
    return float(np.interp(xx, x, y))


def crossings(y, lo, hi):
    m = (x >= lo) & (x <= hi)
    xs, ys = x[m], y[m]
    k = np.where(np.sign(ys[:-1]) * np.sign(ys[1:]) < 0)[0]
    return [float(xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])) for i in k]


def rate(y, x_near, x_far):
    a, b = at(y, x_near), at(y, x_far)
    return float(np.log(abs(a / b)) / abs(x_near - x_far)) if a * b > 0 else None


tails = {}
for sname, arr in list(states.items()) + [("reference", ref)]:
    t = {}
    for n in ["q", "sigma", "m", "R", "Delta"]:
        y = arr[:, cols.index(n)]
        t[n] = {"value_at_-0.20": at(y, -0.20), "value_at_-0.25": at(y, -0.25),
                "fall_factor_-0.20_to_-0.25": abs(at(y, -0.20) / at(y, -0.25)),
                "rate_-0.25_-0.20": rate(y, -0.20, -0.25), "rate_-0.20_-0.16": rate(y, -0.16, -0.20),
                "zero_crossings_upstream_-0.45_-0.14": crossings(y, -0.45, -0.14),
                "zero_crossings_downstream_0.14_0.45": crossings(y, 0.14, 0.45),
                "value_at_0.25": at(y, 0.25), "ratio_to_reference_at_-0.25": at(y, -0.25) / at(ref[:, cols.index(n)], -0.25),
                "ratio_to_reference_at_-0.30": at(y, -0.30) / at(ref[:, cols.index(n)], -0.30)}
    tails[sname] = t
out["tails"] = tails

# ---- 3. location of the remaining q, sigma error of the final state
fin = states["final"]
loc = {}
for n in ["q", "sigma", "m", "R", "Delta"]:
    j = cols.index(n)
    e2 = (fin[:, j] - ref[:, j]) ** 2
    tot = e2.sum()
    loc[n] = {"share_upstream_of_window_x_lt_-0.10375": float(100 * e2[x < -0.10375].sum() / tot),
              "share_blend_zone_0.135_to_0.16_both_sides": float(100 * e2[(abs(x) > 0.135) & (abs(x) < 0.16)].sum() / tot),
              "share_upstream_blend_zone": float(100 * e2[(x < -0.135) & (x > -0.16)].sum() / tot),
              "share_inside_window": float(100 * e2[(x >= -0.10375) & (x <= 0.13375)].sum() / tot),
              "share_downstream_of_window": float(100 * e2[x > 0.13375].sum() / tot),
              "max_abs_error_over_field_max_percent": float(100 * np.sqrt(e2.max()) / abs(ref[:, j]).max()),
              "x_of_max_abs_error": float(x[e2.argmax()])}
out["final_error_location"] = loc
json.dump(out, open(sys.argv[4], "w", encoding="utf-8"), indent=1)
for k in ("equation_trained_vs_6400", "equation_trained_vs_3200", "data_assisted_final_vs_6400_same_positions", "ratio_data_assisted_over_equation_trained",
          "dvm_3200_vs_6400_same_positions", "equation_trained_vs_extrapolate", "data_assisted_final_vs_extrapolate"):
    print("%-46s" % k, " ".join("%s %.4f" % (n, v) for n, v in out[k].items()))
print("heat-flux maximum", out["heat_flux_maximum"])
for s in ("original_envelopes", "common_tail_same_weights", "final", "reference"):
    for n in ("q", "sigma", "R", "Delta", "m"):
        t = tails[s][n]
        print("%-26s %-6s v(-.20) %+.3e v(-.25) %+.3e fall %.1f rate[-.25,-.20] %s rate[-.20,-.16] %s up0 %s down0 %s v(.25) %+.3e r(-.25) %.2f r(-.30) %.2f" % (
            s, n, t["value_at_-0.20"], t["value_at_-0.25"], t["fall_factor_-0.20_to_-0.25"],
            "%.1f" % t["rate_-0.25_-0.20"] if t["rate_-0.25_-0.20"] else "sign",
            "%.1f" % t["rate_-0.20_-0.16"] if t["rate_-0.20_-0.16"] else "sign",
            ["%.3f" % a for a in t["zero_crossings_upstream_-0.45_-0.14"]], ["%.3f" % a for a in t["zero_crossings_downstream_0.14_0.45"]],
            t["value_at_0.25"], t["ratio_to_reference_at_-0.25"], t["ratio_to_reference_at_-0.30"]))
for n, v in loc.items():
    print(n, {k: round(a, 3) for k, a in v.items()})
