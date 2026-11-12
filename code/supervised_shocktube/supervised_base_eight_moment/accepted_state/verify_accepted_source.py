"""Fresh-process reload check of an accepted common-tail source package.

Uses only files inside the package (STATE.pt, code/, inputs/, evidence/) and re-implements the
project's loss and profile definitions here, so it does not depend on the Stage B scripts.
It recomputes the 15 training components and the full-domain errors and compares them with the
values recorded when the source was trained and assessed, then writes RELOAD_VERIFICATION.json
and MANIFEST.json (SHA256 of every file in the package).

usage: python -B verify_accepted_source.py <package dir>
       (when called with three arguments, the third is the package dir)
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

pkg = Path(sys.argv[3] if len(sys.argv) > 3 else sys.argv[1]).resolve()
torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("STAGEB_THREADS", 8)))
sys.path.insert(0, str(pkg / "code"))
import legacy  # noqa: E402
from all_moment_model import moments_all  # noqa: E402
from common_tail_model import CommonTailHMM  # noqa: E402
import function_trust_proposal as trust  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()


st = torch.load(pkg / "STATE.pt", map_location="cpu", weights_only=False)
cfg = st["config"]
kn, tf = float(cfg["case"]["kn"]), float(cfg["case"]["t_final"])
model = CommonTailHMM(cfg, **st["architecture_args"]).double().eval()
model.load_state_dict(st["model"], strict=True)
assert [n for n, _ in model.named_parameters()] == st["parameter_names"]
assert sum(p.numel() for p in model.parameters()) == 52753
v, w, vw = (st["grid"][k].double() for k in ("v", "w", "vw"))
kernel = torch.stack([torch.ones(len(v)), v[:, 0], 0.5 * v.square().sum(1)], 1)
ref = np.load(pkg / "inputs" / "REFERENCE6400_FIELDS.npz", allow_pickle=False)
rx, ry = ref["x"], ref["reference"]
obs = np.load(pkg / "inputs" / "OBSERVATION_LOCATIONS_FROZEN.npz", allow_pickle=False)
scales, counts = st["observation_scales"].double(), np.asarray(st["observation_counts"])


def targets(xx):
    return np.column_stack([np.interp(xx, rx, ry[:, k]) for k in range(8)])


def values(xx):
    tx = torch.tensor(np.c_[np.full(len(xx), tf), xx])
    C = torch.stack([a for k, a in enumerate(model.coefficients(tx)) if k in (0, 1, 2, 3, 4, 6, 7, 8)], 1)
    f = model.grid_from_coefficients(C, v)[0]
    return trust.full_f_moment_chain_values(f, v, w, kn, moments_all)[0]


def physical(tx):
    f, ft, fx, _ = model.grid_with_derivatives(tx, v, create_graph=False)
    transport = ft + fx * v[:, 0]
    fw = f * w
    rho = fw.sum(1)
    vel = fw @ v / rho[:, None]
    e = ((v[None, :, :] - vel[:, None, :]) ** 2).sum(2)
    T = (fw * e).sum(1) / (3 * rho)
    Mf = rho[:, None] * torch.exp(-e / (2 * T[:, None])) / (2 * math.pi * T[:, None]) ** 1.5
    residual = transport - (Mf - f) / kn
    den = abs(Mf) + abs(f) + 1e-5
    ww = w * vw
    return ((residual.square() * ww).sum(1) / ww.sum()).detach(), (((residual / den).square() * ww).sum(1) / ww.sum()).detach(), \
        ((transport * w) @ kernel).detach()


comp = np.zeros(15)
with torch.no_grad():
    for xx, ks in ((obs["macro_x"], [0, 1, 2]), (obs["moment_x"], [3, 4, 5, 6, 7])):
        targ = torch.tensor(targets(xx))
        for i in range(0, len(xx), 4):
            rr = (values(xx[i:i + 4]) - targ[i:i + 4]) / scales
            for k in ks:
                comp[k] += float(rr[:, k].square().sum()) / counts[k]
colloc = st["training_physics"].double()
for i in range(0, 64, 2):
    raw, rel, bal = physical(colloc[i:i + 2])
    comp[8] += float(raw.sum()) / 64
    comp[9] += float(rel.sum()) / 64
    for k in range(3):
        comp[10 + k] += float(bal[:, k].square().sum()) / 64
init = st["initial_physics"].double().clone()
init[:, 0] = 1e-300                                   # see STATE['initial_time_note']
comp[13] = float(physical(init)[0].mean())
bt = st["incoming_boundary"].double()
with torch.no_grad():
    f, _ = model.grid(bt, v)
    left = bt[:, 1] < 0
    c = cfg["case"]
    rho = torch.where(left, float(c["left"]["rho"]), float(c["right"]["rho"]))
    u = torch.where(left, float(c["left"]["ux"]), float(c["right"]["ux"]))
    T = torch.where(left, float(c["left"]["T"]), float(c["right"]["T"]))
    M = legacy.maxwellian(rho[:, None], u[:, None], T[:, None], v)
    incoming = torch.where(bt[:, 1, None] < 0, v[:, 0][None, :] > 0, v[:, 0][None, :] < 0)
    comp[14] = float(((f - M).square() * w * incoming).sum() / (incoming * w).sum())

rows = []
with torch.no_grad():
    for i in range(0, len(rx), 8):
        z = values(rx[i:i + 8])
        rows.append(torch.stack([z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4], z[:, 5], z[:, 6],
                                 -z[:, 4] / kn, -2 * z[:, 3] / kn, z[:, 7], z[:, 6] + z[:, 7] / 3], 1).numpy())
pred = np.concatenate(rows)
t8 = ry[:, :8]
ref11 = np.c_[t8[:, :7], -t8[:, 4] / kn, -2 * t8[:, 3] / kn, t8[:, 7], t8[:, 6] + t8[:, 7] / 3]
l2 = 100 * np.linalg.norm(pred - ref11, axis=0) / np.linalg.norm(ref11, axis=0)

recorded = json.loads((pkg / "evidence" / "ASSESSMENT.json").read_text(encoding="utf-8"))["stageB"]
saved = np.load(pkg / "evidence" / "ASSESSMENT_FIELDS.npz", allow_pickle=False)["stageB"]
rc = np.asarray(st["components"])
out = {
    "schema": "COMMON_TAIL_SOURCE_RELOAD_VERIFICATION_V1",
    "package": pkg.name, "state_sha256": sha((pkg / "STATE.pt").read_bytes()),
    "torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads(),
    "component_names": st["component_names"],
    "components_recomputed": comp.tolist(), "components_recorded": rc.tolist(),
    "components_max_relative_difference": float(np.max(np.abs(comp - rc) / rc)),
    "full_L2_percent_recomputed": l2.tolist(), "full_L2_percent_recorded": recorded["full_L2_percent"],
    "full_L2_max_relative_difference": float(np.max(np.abs(l2 - np.asarray(recorded["full_L2_percent"])) / np.asarray(recorded["full_L2_percent"]))),
    "profile_max_abs_difference_to_saved_assessment_fields": float(np.abs(pred - saved).max()),
    "signed_extrema_identical_to_recorded": [[float(pred[:, k].min()), float(pred[:, k].max())] for k in range(11)] == recorded["signed_extrema"],
    "tolerance": "relative 1e-10 for components and L2; absolute 1e-12 for profiles",
}
out["passed"] = bool(out["components_max_relative_difference"] < 1e-10 and out["full_L2_max_relative_difference"] < 1e-10
                     and out["profile_max_abs_difference_to_saved_assessment_fields"] < 1e-12)
(pkg / "RELOAD_VERIFICATION.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
manifest = {str(p.relative_to(pkg)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
            for p in sorted(pkg.rglob("*")) if p.is_file() and p.name != "MANIFEST.json" and "__pycache__" not in p.parts}
(pkg / "MANIFEST.json").write_text(json.dumps({"members": len(manifest), "files": manifest}, indent=1), encoding="utf-8")
print("components: max relative difference to recorded %.3e" % out["components_max_relative_difference"])
print("full L2 %%: max relative difference to recorded %.3e" % out["full_L2_max_relative_difference"])
print("profiles: max abs difference to saved assessment fields %.3e" % out["profile_max_abs_difference_to_saved_assessment_fields"])
print("extrema identical:", out["signed_extrema_identical_to_recorded"], "| passed:", out["passed"], "| manifest members:", len(manifest))
