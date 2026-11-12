"""Shared definitions for Stage B (training from the transplanted model 170T).

Everything that defines the training problem is taken from the project, not invented here:
- checkpoint 170 (model weights, velocity grid, physics points, observation scales, normalizers)
  from HMM_FIELD_GN_FROM168_EXECUTION/NATIVE.zip;
- the 160 macro / 42 moment observation locations (OBSERVATION_LOCATIONS_FROZEN.npz);
- targets = np.interp of the dense reference at those locations, as in train_curvature_continuation.py;
- the 8 data losses, 7 physical losses, fresh-physics report and profile columns of
  train_curvature_continuation.py (helpers data_losses, physical_losses, incoming_loss,
  fresh_physics, full_columns) and physical_batch of protected_driver.py;
- the finite-grid moment convention full_f_moment_chain_values of function_trust_proposal.py.

The project tree is only read. Held-out reference values are used only by assess().
"""
import hashlib
import io
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

STATE_MEMBER = "OUTPUT/ACCEPTED_170/STATE.pt"
STATE_SHA = "bbee1d6481b6ee3ccdcf4a0472c408368272f571f4124937368eb602f0833b95"
REF_SHA = "c4fe0f1fbd0789b3793729791b9d9490465578b151c78bf3bac17571b3eeb090"
OBS_SHA = "c9724c94719342c116a81502a8fdb837235f8e6f8bc5d6d0c10dd17a2b1a4cff"
BOUND_SHA = {"all_moment_model.py": "dd5c7dd2435c919df463da2f3e64063481d13c3f9a0c6faacbda76de318f0eb3",
             "legacy.py": "bab5b7db84dcec760948b2a232e35ecddb1fa282764199af81301c3e2651e59e",
             "q_amplitude_adapter.py": "6f9bb5399e9c78a5656e4de6c5a1cb1e8990f7f7d822cbd2b4e03ec36feb15e1",
             "function_trust_proposal.py": "dbe5351a534e38b2efedbe72b72cae6dd03c3d5590019c03eac0162e73fb2b13"}
NAMES = ["rho", "u", "T", "q", "sigma", "m", "R", "Delta",
         "physical_raw_PDE", "physical_relative_PDE", "mass_balance", "momentum_balance", "energy_balance",
         "initial_physical_PDE", "incoming_boundary"]
FIELDS11 = ["rho", "u", "T", "q", "sigma", "m", "R", "Qxx_BGK", "Qx_BGK", "Delta", "S_raw"]
X1 = 0.135


def sha(b):
    return hashlib.sha256(b).hexdigest()


class Problem:
    def __init__(self, project_root, plan_dir, bound_dir, threads=None):
        import os
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(int(os.environ.get("STAGEB_THREADS", 8)) if threads is None else threads)
        root, plan = Path(project_root), Path(plan_dir)
        z = zipfile.ZipFile(root / "HMM_FIELD_GN_FROM168_EXECUTION" / "NATIVE.zip")
        state_bytes = z.read(STATE_MEMBER)
        assert sha(state_bytes) == STATE_SHA
        bound = Path(bound_dir)
        bound.mkdir(parents=True, exist_ok=True)
        for name, want in BOUND_SHA.items():
            b = z.read("OUTPUT/BOUND_CODE/" + name)
            assert sha(b) == want, name
            if not (bound / name).exists() or sha((bound / name).read_bytes()) != want:
                (bound / name).write_bytes(b)
        sys.path.insert(0, str(bound))
        sys.path.insert(0, str(plan))
        import legacy
        from all_moment_model import moments_all
        from q_amplitude_adapter import QAmplitudeAllMomentHMM
        from common_tail_model import CommonTailHMM
        import function_trust_proposal as trust
        self.legacy, self.moments_all, self.trust = legacy, moments_all, trust
        self.Parent, self.Child = QAmplitudeAllMomentHMM, CommonTailHMM
        self.code_sha = {"common_tail_model.py": sha((plan / "common_tail_model.py").read_bytes()),
                         "common_tail_envelope.py": sha((plan / "common_tail_envelope.py").read_bytes())}

        cp = torch.load(io.BytesIO(state_bytes), map_location="cpu", weights_only=False)
        assert cp["accepted_updates"] == 170
        self.cp, self.cfg = cp, cp["config"]
        self.kn, self.tf = float(self.cfg["case"]["kn"]), float(self.cfg["case"]["t_final"])
        assert self.kn == 1.0 and self.tf == 0.02
        g = cp["grid"]
        self.v, self.w, self.vw = g["v"].double(), g["w"].double(), g["vw"].double()
        assert self.v.shape == (19200, 3)
        self.kernel = torch.stack([torch.ones(len(self.v)), self.v[:, 0], 0.5 * self.v.square().sum(1)], 1)

        ref_path = root / "LOCAL_FIXED_DIAGNOSTIC_CURRENT" / "inputs" / "REFERENCE6400_FIELDS.npz"
        assert sha(ref_path.read_bytes()) == REF_SHA
        ref = np.load(ref_path, allow_pickle=False)
        self.rx, self.ry = ref["x"], ref["reference"]
        assert self.rx.shape == (6400,) and self.ry.shape == (6400, 9) and np.all(np.diff(self.rx) > 0)
        obs_path = root / "HMM_FUNCTION_TRUST_REPAIR_PREPARATION" / "OBSERVATION_LOCATIONS_FROZEN.npz"
        assert sha(obs_path.read_bytes()) == OBS_SHA
        obs = np.load(obs_path, allow_pickle=False)
        self.macro_x, self.moment_x = obs["macro_x"], obs["moment_x"]
        assert len(self.macro_x) == 160 and len(self.moment_x) == 42
        self.quality_x = np.load(root / "HMM_FUNCTION_TRUST_REPAIR_PREPARATION" / "QUALITY_LOCATIONS_FROZEN.npz",
                                 allow_pickle=False)["quality_x"]
        assert len(self.quality_x) == 399
        self.counts = np.array([160, 160, 160, 42, 42, 42, 42, 42])
        assert np.array_equal(self.counts, np.asarray(cp["observation_counts"]))
        self.scales = cp["observation_scales"].double()
        self.normal = np.asarray(cp["normalizers"], dtype=float).copy()
        self.limits170 = np.asarray(cp["component_limits"], dtype=float).copy()
        assert self.normal.shape == (15,) and self.limits170.shape == (15,)
        self.tm = torch.tensor(self.targets(self.macro_x))
        self.th = torch.tensor(self.targets(self.moment_x))
        self.families = [(self.macro_x, self.tm, [0, 1, 2]), (self.moment_x, self.th, [3, 4, 5, 6, 7])]
        self.colloc = cp["training_physics"].double()
        self.fresh = cp["selection_physics"].double()
        # The model computes tau = clamp(t/tf, 0, 1.5). The project's runs used a torch in which the
        # derivative of clamp at its bound is 1; torch 2.15 returns 0 there, which silently removes
        # d(tau)/dt at t = 0 (measured: 4.87e-7 instead of the recorded 2.8006887443947518e-08).
        # Evaluating the four initial points at t = 1e-300 restores the recorded value bit for bit
        # (probe_clamp_initial.py) without subnormal arithmetic. No other physics point sits on a
        # clamp bound.
        init = cp["initial_physics"].double().clone()
        assert bool((init[:, 0] == 0).all())
        init[:, 0] = 1e-300
        self.initial = init
        self.bt = cp["incoming_boundary"].double()
        assert self.colloc.shape == self.fresh.shape == (64, 2)
        for pts in (self.colloc, self.fresh, self.bt):
            assert bool((pts[:, 0] > 0).all()) and bool((pts[:, 0] < 1.5 * self.tf).all())

    # ---- models -------------------------------------------------------------------------
    def model170(self):
        m = self.Parent(self.cfg).double().eval()
        m.load_state_dict(self.cp["model"], strict=True)
        return m

    def model170T(self, state=None):
        m = self.Child(self.cfg, x1=X1).double().eval()
        m.load_state_dict(self.cp["model"] if state is None else state, strict=True)
        assert sum(p.numel() for p in m.parameters()) == 52753
        return m

    # ---- targets (training observations only) ---------------------------------------------
    def targets(self, xx):
        assert np.all(xx >= self.rx[0]) and np.all(xx <= self.rx[-1])
        return np.column_stack([np.interp(xx, self.rx, self.ry[:, k]) for k in range(8)])

    # ---- exact project definitions --------------------------------------------------------
    def coefficients(self, model, tx):
        return torch.stack([a for k, a in enumerate(model.coefficients(tx)) if k in (0, 1, 2, 3, 4, 6, 7, 8)], 1)

    def values_from_C(self, model, C):
        f = model.grid_from_coefficients(C, self.v)[0]
        return self.trust.full_f_moment_chain_values(f, self.v, self.w, self.kn, self.moments_all)[0]

    def model_values(self, model, xx):
        tx = torch.tensor(np.c_[np.full(len(xx), self.tf), xx])
        C = self.coefficients(model, tx)
        return tx, C, self.values_from_C(model, C)

    def full_columns(self, z):
        kn = self.kn
        return torch.stack([z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4], z[:, 5], z[:, 6],
                            -z[:, 4] / kn, -2 * z[:, 3] / kn, z[:, 7], z[:, 6] + z[:, 7] / 3], 1)

    def profile(self, model, xx):
        rows = []
        with torch.no_grad():
            for i in range(0, len(xx), 8):
                rows.append(self.full_columns(self.model_values(model, xx[i:i + 8])[2]).numpy())
        return np.concatenate(rows)

    def physical(self, model, tx, graph):
        v, w, vw, kn = self.v, self.w, self.vw, self.kn
        f, ft, fx, Mlatent = model.grid_with_derivatives(tx, v, create_graph=graph)
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
        raw = (residual.square() * ww).sum(1) / ww.sum()
        relative = ((residual / den).square() * ww).sum(1) / ww.sum()
        balance = (transport * w) @ self.kernel
        return raw, relative, balance

    def incoming_loss(self, model):
        cfg, bt, v, w = self.cfg, self.bt, self.v, self.w
        f, _ = model.grid(bt, v)
        left = bt[:, 1] < 0
        rho = torch.where(left, float(cfg["case"]["left"]["rho"]), float(cfg["case"]["right"]["rho"]))
        u = torch.where(left, float(cfg["case"]["left"]["ux"]), float(cfg["case"]["right"]["ux"]))
        T = torch.where(left, float(cfg["case"]["left"]["T"]), float(cfg["case"]["right"]["T"]))
        M = self.legacy.maxwellian(rho[:, None], u[:, None], T[:, None], v)
        incoming = torch.where(bt[:, 1, None] < 0, v[:, 0][None, :] > 0, v[:, 0][None, :] < 0)
        return ((f - M).square() * w * incoming).sum() / (incoming * w).sum()

    def physical_losses(self, model, params=None):
        """7 values; with params also the 7 x npar gradient matrix."""
        values = np.zeros(7)
        G = torch.zeros(7, sum(p.numel() for p in params)) if params is not None else None
        graph = params is not None

        def add(loss, k):
            values[k] += float(loss.detach())
            if graph:
                gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                G[k] += torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, gs)])
        for i in range(0, 64, 2):
            raw, relative, balance = self.physical(model, self.colloc[i:i + 2], graph)
            add(raw.sum() / 64, 0)
            add(relative.sum() / 64, 1)
            for k in range(3):
                add(balance[:, k].square().sum() / 64, k + 2)
        raw, _, _ = self.physical(model, self.initial, graph)
        add(raw.mean(), 5)
        add(self.incoming_loss(model), 6)
        return values, G

    def data_losses(self, model):
        values = np.zeros(8)
        with torch.no_grad():
            for xx, targ, ks in self.families:
                for i in range(0, len(xx), 4):
                    y = self.model_values(model, xx[i:i + 4])[2]
                    rr = (y - targ[i:i + 4]) / self.scales
                    for k in ks:
                        values[k] += float(rr[:, k].square().sum()) / self.counts[k]
        return values

    def components(self, model):
        with torch.enable_grad():
            phys = self.physical_losses(model)[0]
        return np.concatenate([self.data_losses(model), phys])

    def fresh_physics(self, model):
        raw, rel, bal = [], [], []
        for i in range(0, 64, 2):
            a, c, d = self.physical(model, self.fresh[i:i + 2], False)
            raw.extend(a.detach().numpy()); rel.extend(c.detach().numpy()); bal.extend(d.detach().numpy())
        return dict(raw_PDE=float(np.mean(raw)), relative_PDE=float(np.mean(rel)),
                    balance_RMS=np.sqrt(np.mean(np.asarray(bal) ** 2, axis=0)).tolist(),
                    per_point_raw=[float(a) for a in raw])

    # ---- held-out assessment (never used for a training decision) --------------------------
    def reference11(self, xx):
        t = self.targets(xx)
        return np.c_[t[:, :7], -t[:, 4] / self.kn, -2 * t[:, 3] / self.kn, t[:, 7], t[:, 6] + t[:, 7] / 3]

    def assess(self, model):
        x = self.rx
        pred, ref = self.profile(model, x), self.reference11(self.rx)
        regions = {"upstream_shoulder_x_lt_-0.15": x < -0.15, "upstream_of_window_x_lt_-0.104": x < self.moment_x.min(),
                   "observed_span": (x >= self.moment_x.min()) & (x <= self.moment_x.max()),
                   "downstream_of_window": x > self.moment_x.max(), "downstream_x_gt_0.15": x > 0.15,
                   "full": np.ones_like(x, bool)}
        i1, i2 = int(np.argmin(np.abs(x + 0.20))), int(np.argmin(np.abs(x + 0.25)))
        out = {"fields": FIELDS11,
               "full_L2_percent": (100 * np.linalg.norm(pred - ref, axis=0) / np.linalg.norm(ref, axis=0)).tolist(),
               "signed_extrema": [[float(pred[:, k].min()), float(pred[:, k].max())] for k in range(11)],
               "reference_extrema": [[float(ref[:, k].min()), float(ref[:, k].max())] for k in range(11)],
               "regional": {}}
        for j, name in enumerate(FIELDS11):
            e = pred[:, j] - ref[:, j]
            out["regional"][name] = {
                "SSE": {k: float((e[m] ** 2).sum()) for k, m in regions.items()},
                "L2_percent": {k: float(100 * np.linalg.norm(e[m]) / np.linalg.norm(ref[m, j])) for k, m in regions.items()},
                "upstream_decay_rate": float(np.log(abs(pred[i1, j] / pred[i2, j])) / (x[i1] - x[i2])) if pred[i2, j] != 0 else None}
        out["fresh_physics"] = self.fresh_physics(model)
        return out, pred, ref
