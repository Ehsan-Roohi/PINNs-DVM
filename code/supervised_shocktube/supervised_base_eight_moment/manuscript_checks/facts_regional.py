"""Regional squared errors and related numbers quoted in the manuscript, from the saved assessment only."""
import json
import sys
import numpy as np

a = json.load(open(sys.argv[1], encoding="utf-8"))
print("top-level keys:", list(a))
out = {}
for st in ("model170", "model170T", "stageB"):
    if st not in a:
        continue
    r = a[st]["regional"]
    out[st] = {f: {"SSE": r[f]["SSE"], "L2": r[f]["L2_percent"], "rate": r[f]["upstream_decay_rate"]} for f in ("q", "sigma", "m", "R", "Delta")}
    if "fresh_physics" in a[st]:
        fp = a[st]["fresh_physics"]
        out[st]["fresh"] = {"raw": fp["raw_PDE"], "rel": fp["relative_PDE"], "bal": fp["balance_RMS"]}
regions = list(out["model170"]["q"]["SSE"])
print("regions:", regions)
for f in ("q", "sigma", "m", "R", "Delta"):
    print(f)
    for reg in regions:
        s0, s1, s2 = (out[s][f]["SSE"][reg] for s in ("model170", "model170T", "stageB"))
        print("   %-34s SSE 170 %.4e  170T %.4e (%+.1f %%)  B %.4e (%+.1f %% vs 170T) | L2%% %.4f %.4f %.4f" % (
            reg, s0, s1, 100 * (s1 / s0 - 1), s2, 100 * (s2 / s1 - 1), *(out[s][f]["L2"][reg] for s in ("model170", "model170T", "stageB"))))
    tot = out["model170"][f]["SSE"]["full"]
    print("   share upstream of window (170): %.2f %% ; share x<-0.15: %.2f %%" % (
        100 * out["model170"][f]["SSE"]["upstream_of_window_x_lt_-0.104"] / tot, 100 * out["model170"][f]["SSE"]["upstream_shoulder_x_lt_-0.15"] / tot))
for st in out:
    if "fresh" in out[st]:
        print(st, "fresh physics", out[st]["fresh"])
other = {k: (v if not isinstance(v, dict) else list(v)) for k, v in a.items() if k not in ("model170", "model170T", "stageB")}
print(json.dumps(other, indent=1)[:3000])
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
