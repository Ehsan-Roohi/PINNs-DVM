"""Absolute and relative size of the sampled physical residuals of the four states of the base case.

For the five-field state, the eight-moment state with the original envelopes (170 updates), the same
weights with the common-tail envelope, and the state after the output-layer refinement, this evaluates on
the 64 training points and on the 64 separate selection points:
  - RMS over points of the three transport balances b = sum_j w_j (1, v_x, |v|^2/2)(f_t + v_x f_x),
  - RMS over points of the time-derivative part and of the flux part of each balance,
  - the weighted mean-square kinetic residual and the weighted mean square of the streaming term
    f_t + v_x f_x and of the collision term (M[f] - f)/Kn, with the weights of the training loss.
Forward evaluation only; nothing is trained and the project is only read.

usage: python -B physics_magnitudes.py <project root> <plan dir> <out json>
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

root, plan, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(plan / "STAGE_B_TRAIN"))
from stageb_problem import Problem  # noqa: E402

P = Problem(root, plan, plan / "STAGE_B_TRAIN" / "BOUND_CODE")
from all_moment_model import AllMomentHMM  # noqa: E402

inp = root / "INPUT.pt"
assert hashlib.sha256(inp.read_bytes()).hexdigest() == "a0734fccb6cc0766bf046c8a23f8f98bf15e2e03b39d6e96ed70e1a170d09003"
five = AllMomentHMM(P.cfg).double().eval()
five.load_parent(torch.load(inp, map_location="cpu", weights_only=False)["model"])
pkg = plan / "ACCEPTED_SOURCE_170T_B35" / "STATE.pt"
assert hashlib.sha256(pkg.read_bytes()).hexdigest() == "0ba06f816850ff30362a981571203e3234d4506e2d596972a31250d00ff42676"
final = P.model170T(torch.load(pkg, map_location="cpu", weights_only=False)["model"])
models = {"five_field": five, "eight_moment_original_envelopes": P.model170(),
          "common_tail_same_weights": P.model170T(), "after_output_layer_refinement": final}
v, w, vw, kn = P.v, P.w, P.vw, P.kn
ww = w * vw


def evaluate(model, pts):
    bal, tim, flx, res, stream, coll = [], [], [], [], [], []
    for i in range(0, len(pts), 2):
        f, ft, fx, _ = model.grid_with_derivatives(pts[i:i + 2], v, create_graph=False)
        f, ft, fx = f.detach(), ft.detach(), fx.detach()
        adv = fx * v[:, 0]
        transport = ft + adv
        fw = f * w
        rho = fw.sum(1)
        vel = fw @ v / rho[:, None]
        e = ((v[None, :, :] - vel[:, None, :]) ** 2).sum(2)
        T = (fw * e).sum(1) / (3 * rho)
        Mf = rho[:, None] * torch.exp(-e / (2 * T[:, None])) / (2 * math.pi * T[:, None]) ** 1.5
        c = (Mf - f) / kn
        bal.append(((transport * w) @ P.kernel).numpy())
        tim.append(((ft * w) @ P.kernel).numpy())
        flx.append(((adv * w) @ P.kernel).numpy())
        res.append((((transport - c).square() * ww).sum(1) / ww.sum()).numpy())
        stream.append(((transport.square() * ww).sum(1) / ww.sum()).numpy())
        coll.append(((c.square() * ww).sum(1) / ww.sum()).numpy())
    bal, tim, flx = np.concatenate(bal), np.concatenate(tim), np.concatenate(flx)
    res, stream, coll = np.concatenate(res), np.concatenate(stream), np.concatenate(coll)
    rms = lambda a: np.sqrt(np.mean(a ** 2, axis=0))
    return {"balance_rms": rms(bal).tolist(), "time_term_rms": rms(tim).tolist(), "flux_term_rms": rms(flx).tolist(),
            "balance_rms_over_time_term_rms": (rms(bal) / rms(tim)).tolist(),
            "kinetic_residual_mean_square": float(res.mean()), "streaming_mean_square": float(stream.mean()),
            "collision_mean_square": float(coll.mean()),
            "residual_over_streaming_rms_ratio": float(math.sqrt(res.mean() / stream.mean())),
            "residual_over_collision_rms_ratio": float(math.sqrt(res.mean() / coll.mean()))}


audit = P.audit_points()          # 64 points that no training or selection step of any stage used (seed 20261111)
result = {"kernels": ["mass", "momentum", "energy"], "points": {"training": 64, "selection": 64, "unused": 64}, "states": {}}
for name, model in models.items():
    result["states"][name] = {"training": evaluate(model, P.colloc), "selection": evaluate(model, P.fresh),
                              "unused": evaluate(model, audit)}
    s = result["states"][name]
    print(name)
    for k in ("training", "selection", "unused"):
        print("  %-9s balance rms %s | / time term %s | res/stream %.4f res/coll %.4f" % (
            k, " ".join("%.4f" % a for a in s[k]["balance_rms"]),
            " ".join("%.4f" % a for a in s[k]["balance_rms_over_time_term_rms"]),
            s[k]["residual_over_streaming_rms_ratio"], s[k]["residual_over_collision_rms_ratio"]), flush=True)
out.write_text(json.dumps(result, indent=1), encoding="utf-8")
