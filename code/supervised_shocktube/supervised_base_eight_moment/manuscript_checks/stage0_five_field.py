"""Stage 0 of the progression table: the five-field model currently shown in the manuscript's Figure 4
(job 66898929, weights a0734f...), evaluated on the 48x20x20 double-precision velocity grid and compared
with the 6400-cell DVM reference. Forward evaluation only; the project is only read.

usage: python -B stage0_five_field.py <project root> <plan dir> <out dir>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

root, plan, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(plan / "STAGE_B_TRAIN"))
from stageb_problem import Problem, FIELDS11  # noqa: E402

P = Problem(root, plan, plan / "STAGE_B_TRAIN" / "BOUND_CODE")
from all_moment_model import AllMomentHMM  # noqa: E402

inp = root / "INPUT.pt"
assert hashlib.sha256(inp.read_bytes()).hexdigest() == "a0734fccb6cc0766bf046c8a23f8f98bf15e2e03b39d6e96ed70e1a170d09003"
cp0 = torch.load(inp, map_location="cpu", weights_only=False)
m = AllMomentHMM(P.cfg).double().eval()
m.load_parent(cp0["model"])                      # five-field weights; added outputs are exactly zero
x = P.rx
pred, ref = P.profile(m, x), P.reference11(x)
l2 = 100 * np.linalg.norm(pred - ref, axis=0) / np.linalg.norm(ref, axis=0)
res = {"state_sha256": "a0734fccb6cc0766bf046c8a23f8f98bf15e2e03b39d6e96ed70e1a170d09003",
       "velocity_grid": "48x20x20 on [-7,7]^3, double precision", "reference": "6400-cell BGK DVM",
       "fields": FIELDS11, "full_L2_percent": l2.tolist(),
       "extrema": {n: [float(pred[:, k].min()), float(pred[:, k].max())] for k, n in enumerate(FIELDS11)}}
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "STAGE0_FIELDS.npz", x=x, five_field=pred, reference=ref, columns=np.array(FIELDS11))
(out / "STAGE0_FIVE_FIELD.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
for n, v in zip(FIELDS11, l2):
    print("  %-8s %.6f %%" % (n, v))
