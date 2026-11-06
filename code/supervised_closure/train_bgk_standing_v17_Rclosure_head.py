#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_bgk_standing_v8_highmom_Rorth import (
    seed_all,
    load_ref_high,
    build_vgrid,
    build_scales,
    MLP,
    macro_anchor_loss,
    qsig_anchor_loss,
    flux_loss,
    boundary_loss,
    pde_loss,
    sample_from_pool,
    integrate_moments,
    maxwellian_grid,
)


EPS = 1.0e-8


def safe_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def rel_l2(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1.0e-14)


def finite_slope(y, x):
    return (y[1:] - y[:-1]) / torch.clamp(x[1:] - x[:-1], min=1.0e-12)


def normalize_m(m, rho, T):
    return m / (rho * torch.clamp(T, min=EPS) ** 1.5 + EPS)


def normalize_R(R, rho, T):
    return R / (rho * torch.clamp(T, min=EPS) ** 2.0 + EPS)


def ramp(step, s0, s1, wmax):
    if step <= s0:
        return 0.0
    if step >= s1:
        return wmax
    return wmax * (step - s0) / float(max(1, s1 - s0))


def zero_last_layer(mlp):
    mods = list(mlp.net)
    for m in reversed(mods):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
            return


def copy_partial(model, init_path, device):
    if not init_path or not Path(init_path).exists():
        print("[init] no checkpoint found; training from scratch", flush=True)
        return

    print(f"[init] loading partial state from {init_path}", flush=True)
    ck = torch.load(init_path, map_location=device)
    old = ck.get("model", ck)
    new = model.state_dict()

    copied = 0
    partial = 0

    for k, v in old.items():
        if k not in new:
            continue
        if new[k].shape == v.shape:
            new[k].copy_(v)
            copied += 1
        elif new[k].ndim == v.ndim and new[k].shape[1:] == v.shape[1:]:
            # final layer rows; useful when old net has fewer/more output rows
            new[k].zero_()
            n = min(new[k].shape[0], v.shape[0])
            new[k][:n].copy_(v[:n])
            partial += 1
        elif new[k].ndim == 1 and v.ndim == 1:
            new[k].zero_()
            n = min(new[k].shape[0], v.shape[0])
            new[k][:n].copy_(v[:n])
            partial += 1

    model.load_state_dict(new, strict=True)
    print(f"[init] copied={copied}, partial={partial}", flush=True)


class StandingPositiveHMMDecoupled(nn.Module):
    """
    Decoupled architecture:
      net    -> rho, u, T, q, sigma base modes
      m_head -> third-order closure mode
      R_head -> R-orthogonal fourth-order closure mode
    """
    def __init__(
        self,
        ref,
        psi_max=2.8,
        macro_amp=0.25,
        q_amp=1.6,
        sig_amp=1.6,
        h3_amp=1.4,
        h4_amp=2.8,
    ):
        super().__init__()
        self.ref = ref
        self.psi_max = psi_max
        self.macro_amp = macro_amp
        self.q_amp = q_amp
        self.sig_amp = sig_amp
        self.h3_amp = h3_amp
        self.h4_amp = h4_amp

        st = ref["states"]
        self.rho1 = st["rho1"]
        self.u1 = st["u1"]
        self.T1 = st["T1"]
        self.rho2 = st["rho2"]
        self.u2 = st["u2"]
        self.T2 = st["T2"]

        self.xmin = -0.5
        self.xmax = 0.5
        self.x0 = ref["x0"]
        self.width = ref["width"]

        # Keep old name `net` so v7/v8/v11 weights can partially load.
        # Outputs: drho, du, dT, aq, asig, unused_m, unused_R
        self.net = MLP(5, 7, width=96, depth=5)

        # Separate small heads for m and R
        self.m_head = MLP(5, 1, width=64, depth=4)
        self.R_head = MLP(5, 1, width=64, depth=4)

        # Start high heads from zero so importing v7 keeps primary fields intact.
        zero_last_layer(self.m_head)
        zero_last_layer(self.R_head)


        # -----------------------------------------------------------------
        # v17: explicit closure correction for R_xx^cl outside f.
        # This does NOT modify f, rho, q, sigma, or m directly.
        # It supplies a shock-local fourth-order tail closure:
        #     R_closure = R_from_f + R_tail_head(x)
        # Centers correspond to x/lambda_1 =
        # [-8,-6,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6,8]
        # multiplied by Kn_eff=1/80.
        # -----------------------------------------------------------------
        self.register_buffer(
            "Rtail_centers",
            torch.tensor(
                [-0.1000, -0.0750, -0.0500, -0.0375, -0.0250,
                 -0.01875, -0.0125, -0.00625, 0.0, 0.00625,
                  0.0125, 0.01875, 0.0250, 0.0375, 0.0500,
                  0.0750, 0.1000],
                dtype=torch.float32,
            ),
        )
        self.Rtail_coef = nn.Parameter(torch.zeros(17, dtype=torch.float32))
        self.Rtail_log_width = nn.Parameter(torch.tensor(math.log(0.018), dtype=torch.float32))
        self.Rtail_scale = 3.0

    def base_switch(self, x):
        return 0.5 * (1.0 + torch.tanh((x - self.x0) / self.width))

    def gates(self, x):
        gd = ((x - self.xmin) * (self.xmax - x)) / 0.25
        gd = torch.clamp(gd, min=0.0, max=1.0)
        gs = torch.exp(-((x - self.x0) / (3.0 * self.width)) ** 2)
        return gd, gs

    def features(self, x):
        S = self.base_switch(x)
        gd, gs = self.gates(x)
        return torch.stack([
            x,
            x - self.x0,
            torch.tanh((x - self.x0) / self.width),
            S,
            gd * gs,
        ], dim=1), S, gd, gs


    def Rtail_head(self, x):
        """
        Shock-local closure correction for R_xx^cl.
        Bounded by tanh to avoid runaway while allowing O(1-3) correction.
        """
        x = x.reshape(-1)
        centers = self.Rtail_centers.to(device=x.device, dtype=x.dtype)
        coef = self.Rtail_coef.to(device=x.device, dtype=x.dtype)

        width = torch.exp(self.Rtail_log_width.to(device=x.device, dtype=x.dtype))
        width = torch.clamp(
            width,
            min=torch.tensor(0.004, device=x.device, dtype=x.dtype),
            max=torch.tensor(0.080, device=x.device, dtype=x.dtype),
        )

        phi = torch.exp(-((x[:, None] - centers[None, :]) / width) ** 2)
        raw = torch.sum(phi * coef[None, :], dim=1)

        # Local shock gate; do not call self.gates(x), because in this model
        # gates() may return a tuple/list.
        x0 = self.x0.to(device=x.device, dtype=x.dtype) if torch.is_tensor(self.x0) else torch.tensor(float(self.x0), device=x.device, dtype=x.dtype)
        w0 = self.width.to(device=x.device, dtype=x.dtype) if torch.is_tensor(self.width) else torch.tensor(float(self.width), device=x.device, dtype=x.dtype)
        shock_gate = torch.exp(-((x - x0) / torch.clamp(3.0 * w0, min=torch.tensor(1.0e-6, device=x.device, dtype=x.dtype))) ** 2)

        return self.Rtail_scale * shock_gate * torch.tanh(raw)


    def forward_params(self, x):
        feat, S, gd, gs = self.features(x)

        log_rho_base = (1 - S) * math.log(self.rho1) + S * math.log(self.rho2)
        ux_base = (1 - S) * self.u1 + S * self.u2
        log_T_base = (1 - S) * math.log(self.T1) + S * math.log(self.T2)

        y = self.net(feat)
        drho, du, dT, aq, asig = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]

        a3_raw = self.m_head(feat).squeeze(-1)
        a4_raw = self.R_head(feat).squeeze(-1)

        gmacro = gd * gs

        rho = torch.exp(log_rho_base + self.macro_amp * gmacro * torch.tanh(drho))
        T = torch.exp(log_T_base + self.macro_amp * gmacro * torch.tanh(dT))

        du_scale = abs(self.u1 - self.u2) + 1.0e-8
        ux = ux_base + 0.30 * du_scale * gmacro * torch.tanh(du)

        aq = self.q_amp * gs * torch.tanh(aq)
        asig = self.sig_amp * gs * torch.tanh(asig)

        a3 = self.h3_amp * gs * torch.tanh(a3_raw)
        a4 = self.h4_amp * gs * torch.tanh(a4_raw)

        return rho, ux, T, aq, asig, a3, a4

    def f_flat(self, x, v):
        rho, ux, T, aq, asig, a3, a4 = self.forward_params(x)

        sqrtT = torch.sqrt(T)
        cx = (v[:, 0] - ux) / sqrtT
        cy = v[:, 1] / sqrtT
        cz = v[:, 2] / sqrtT
        c2 = cx**2 + cy**2 + cz**2

        Hq = cx * (c2 - 5.0)
        Hs = cx**2 - c2 / 3.0

        # third-order traceless mode
        H3 = cx**3 - (3.0 / 5.0) * cx * c2

        # closure-orthogonal fourth-order R mode
        HR = (c2 - 7.0) * (cx**2 - c2 / 3.0)

        raw = (
            0.08 * aq * Hq
            + 0.18 * asig * Hs
            + 0.055 * a3 * H3
            + 0.018 * a4 * HR
        )

        psi = self.psi_max * torch.tanh(raw / self.psi_max)

        c2phys = (v[:, 0] - ux)**2 + v[:, 1]**2 + v[:, 2]**2
        M = rho / torch.clamp((2.0 * math.pi * T) ** 1.5, min=1.0e-30)
        M = M * torch.exp(-c2phys / torch.clamp(2.0 * T, min=1.0e-30))

        f = M * torch.exp(psi)
        f = torch.clamp(f, min=1.0e-30, max=1.0e6)

        return f, M, rho, ux, T, aq, asig, a3, a4

    def f_grid(self, x, v):
        B = x.numel()
        Nv = v.shape[0]
        xx = x[:, None].expand(B, Nv).reshape(-1)
        vv = v[None, :, :].expand(B, Nv, 3).reshape(-1, 3)
        f, *_ = self.f_flat(xx, vv)
        return f.reshape(B, Nv)


def compute_high(model, x, v, v2, w):
    f = model.f_grid(x, v)
    rho, ux, T, qx, sig, *_ = integrate_moments(f, v, v2, w)

    M = maxwellian_grid(v, rho, ux, T)
    G = f - M

    Cx = v[None, :, 0] - ux[:, None]
    Cy = v[None, :, 1]
    Cz = v[None, :, 2]
    C2 = Cx**2 + Cy**2 + Cz**2

    mxxx = torch.sum(G * w[None, :] * (Cx**3 - (3.0 / 5.0) * Cx * C2), dim=1)
    Rxx_neq = torch.sum(G * w[None, :] * C2 * (Cx**2 - C2 / 3.0), dim=1)
    Rxx_cl = Rxx_neq - 7.0 * T * sig


    # v17: add shock-local R-closure correction outside f.

    if hasattr(model, "Rtail_head"):

        Rxx_cl = Rxx_cl + model.Rtail_head(x).reshape_as(Rxx_cl)


    return rho, ux, T, qx, sig, mxxx, Rxx_cl



def ensure_ref_aliases(ref):
    """
    Robust reference aliases:
      old DVM files may contain sig instead of sigma_xx.
    """
    if "sigma_xx" not in ref and "sig" in ref:
        ref["sigma_xx"] = ref["sig"]
    if "sig" not in ref and "sigma_xx" in ref:
        ref["sig"] = ref["sigma_xx"]
    if "Rxx_closure" not in ref and "Rxx" in ref:
        ref["Rxx_closure"] = ref["Rxx"]
    if "Rxx" not in ref and "Rxx_closure" in ref:
        ref["Rxx"] = ref["Rxx_closure"]
    return ref






def load_micro_ref(path, device, dtype, ref=None):
    if path is None or str(path).strip() == "":
        print("[micro] no micro-anchor file used", flush=True)
        return None

    z = np.load(path)

    micro = {
        "x": torch.tensor(z["x_micro"], device=device, dtype=dtype),
        "vx": torch.tensor(z["vx_micro"], device=device, dtype=dtype),
        "vy": torch.tensor(z["vy_micro"], device=device, dtype=dtype),
        "vz": torch.tensor(z["vz_micro"], device=device, dtype=dtype),
        "target_log": torch.tensor(z["logfM_micro"], device=device, dtype=dtype),
        "score": torch.tensor(z["score_micro"], device=device, dtype=dtype),
        "f_dvm": torch.tensor(z["f_micro"], device=device, dtype=dtype),
        "M_dvm": torch.tensor(z["M_micro"], device=device, dtype=dtype),
    }

    if "ix_micro" in z.files:
        micro["ix"] = torch.tensor(z["ix_micro"], device=device, dtype=torch.long)
    else:
        micro["ix"] = None

    micro["score"] = torch.clamp(micro["score"], min=torch.tensor(1.0e-30, device=device, dtype=dtype))
    micro["score"] = micro["score"] / torch.clamp(
        torch.mean(micro["score"]),
        min=torch.tensor(1.0e-30, device=device, dtype=dtype),
    )

    # DVM HR kernel and target contribution HR*(f-M)
    if ref is not None and micro["ix"] is not None:
        ix = micro["ix"]
        ux_ref = ref["ux"][ix]
        T_ref = torch.clamp(ref["T"][ix], min=torch.tensor(1.0e-12, device=device, dtype=dtype))
    elif ref is not None:
        # fallback nearest x
        xr = ref["x"]
        xm = micro["x"]
        dist = torch.abs(xm[:, None] - xr[None, :])
        ix = torch.argmin(dist, dim=1)
        ux_ref = ref["ux"][ix]
        T_ref = torch.clamp(ref["T"][ix], min=torch.tensor(1.0e-12, device=device, dtype=dtype))
    else:
        # last fallback: use M_dvm only, but HR target will be less exact
        ux_ref = torch.zeros_like(micro["vx"])
        T_ref = torch.ones_like(micro["vx"])

    sqrtT = torch.sqrt(T_ref)
    cx = (micro["vx"] - ux_ref) / sqrtT
    cy = micro["vy"] / sqrtT
    cz = micro["vz"] / sqrtT
    c2 = cx**2 + cy**2 + cz**2

    HR = (c2 - 7.0) * (cx**2 - c2 / 3.0)
    micro["HR_dvm"] = HR
    micro["target_HR"] = HR * (micro["f_dvm"] - micro["M_dvm"])

    micro["scale_HR"] = torch.clamp(
        torch.max(torch.abs(micro["target_HR"])),
        min=torch.tensor(1.0e-10, device=device, dtype=dtype),
    )

    print(f"[micro-v17] loaded {path}", flush=True)
    print(
        f"[micro-v17] N={micro['x'].numel()}, "
        f"logfM=({float(torch.min(micro['target_log'])):.3e},{float(torch.max(micro['target_log'])):.3e}), "
        f"HRtarget max={float(micro['scale_HR']):.3e}",
        flush=True,
    )
    return micro


def micro_losses_v15(model, micro, batch=512, score_power=0.5, clip=8.0):
    """
    Two sparse velocity-space anchors:
      1) log(f_theta/M_theta) ~= log(f_DVM/M_DVM)
      2) H_R(c_theta)*(f_theta-M_theta) ~= H_R(c_DVM)*(f_DVM-M_DVM)

    The second term is the R-kernel-aware correction.
    """
    if micro is None:
        dev = next(model.parameters()).device
        z = torch.tensor(0.0, device=dev)
        return z, z

    N = int(micro["x"].numel())
    if N == 0 or batch <= 0:
        z = torch.tensor(0.0, device=micro["x"].device, dtype=micro["x"].dtype)
        return z, z

    nb = min(int(batch), N)
    idx = torch.randint(0, N, (nb,), device=micro["x"].device)

    xb = micro["x"][idx]
    vb = torch.stack([micro["vx"][idx], micro["vy"][idx], micro["vz"][idx]], dim=1)

    target_log = torch.clamp(micro["target_log"][idx], min=-float(clip), max=float(clip))
    target_HR = micro["target_HR"][idx]

    weight = torch.clamp(micro["score"][idx] ** float(score_power), min=0.2, max=5.0)

    pred_log = torch.empty_like(target_log)
    pred_HR = torch.empty_like(target_HR)

    xu = torch.unique(xb)

    for xv in xu:
        mask = xb == xv
        vi = vb[mask]

        out_f = model.f_flat(xv.reshape(1), vi)
        if isinstance(out_f, (tuple, list)):
            fi = out_f[0]
        else:
            fi = out_f
        fi = fi.reshape(-1)
        fi = torch.clamp(fi, min=torch.tensor(1.0e-30, device=fi.device, dtype=fi.dtype))

        rho, ux, T, *_ = model.forward_params(xv.reshape(1))
        rho = torch.clamp(rho.reshape(()), min=torch.tensor(1.0e-30, device=fi.device, dtype=fi.dtype))
        ux = ux.reshape(())
        T = torch.clamp(T.reshape(()), min=torch.tensor(1.0e-12, device=fi.device, dtype=fi.dtype))

        c2_phys = (vi[:, 0] - ux)**2 + vi[:, 1]**2 + vi[:, 2]**2
        M = rho / ((2.0 * math.pi * T) ** 1.5) * torch.exp(-c2_phys / (2.0 * T))
        M = torch.clamp(M, min=torch.tensor(1.0e-30, device=M.device, dtype=M.dtype))

        pred_log[mask] = torch.log(fi / M)

        sqrtT = torch.sqrt(T)
        cx = (vi[:, 0] - ux) / sqrtT
        cy = vi[:, 1] / sqrtT
        cz = vi[:, 2] / sqrtT
        c2 = cx**2 + cy**2 + cz**2

        HR = (c2 - 7.0) * (cx**2 - c2 / 3.0)
        pred_HR[mask] = HR * (fi - M)

    pred_log = torch.clamp(pred_log, min=-float(clip), max=float(clip))

    Llog = torch.mean(weight * (pred_log - target_log)**2)
    LHR = torch.mean(weight * ((pred_HR - target_HR) / micro["scale_HR"])**2)

    return Llog, LHR


def smart_indices(ref, n_center=120, n_peak=20, n_tail=20):
    x = ref["x"].detach().cpu().numpy().reshape(-1)
    rho = ref["rho"].detach().cpu().numpy().reshape(-1)
    m = ref["mxxx"].detach().cpu().numpy().reshape(-1)
    R = ref["Rxx_closure"].detach().cpu().numpy().reshape(-1)

    idx = set()

    # shock center region based on density gradient
    center_i = int(np.argmax(np.abs(np.gradient(rho, x))))
    for j in range(max(0, center_i - n_center // 2), min(len(x), center_i + n_center // 2 + 1)):
        idx.add(j)

    def add_neighborhood(i0, r):
        i0 = int(i0)
        for j in range(max(0, i0 - r), min(len(x), i0 + r + 1)):
            idx.add(j)

    add_neighborhood(np.argmax(m), n_peak)
    add_neighborhood(np.argmin(m), n_peak)
    add_neighborhood(np.argmax(R), n_peak)
    add_neighborhood(np.argmin(R), n_peak)

    # zero crossings of m and R
    for arr in [m, R]:
        s = np.sign(arr)
        zc = np.where(s[:-1] * s[1:] < 0)[0]
        for j in zc:
            add_neighborhood(j, 8)

    for j in np.linspace(0, max(0, len(x)//7 - 1), n_tail, dtype=int):
        idx.add(int(j))
    for j in np.linspace(6*len(x)//7, len(x)-1, n_tail, dtype=int):
        idx.add(int(j))

    return torch.tensor(sorted(idx), dtype=torch.long, device=ref["x"].device)


def make_pools(ref, use_smart=True):
    device = ref["x"].device
    idx_all = torch.arange(ref["x"].numel(), device=device)

    m = torch.abs(ref["mxxx"])
    R = torch.abs(ref["Rxx_closure"])
    q = torch.abs(ref["qx"])
    s = torch.abs(ref["sig"])

    signal = (
        m / (torch.max(m) + 1e-30)
        + R / (torch.max(R) + 1e-30)
        + q / (torch.max(q) + 1e-30)
        + s / (torch.max(s) + 1e-30)
    )

    pool = idx_all[signal > 0.025]
    if pool.numel() < 64:
        pool = idx_all

    if use_smart:
        shape_idx = smart_indices(ref)
    else:
        shape_idx = torch.linspace(0, ref["x"].numel() - 1, 220, device=device).long()

    return idx_all, pool, shape_idx


def raw_feature_loss(x, P, R, scale):
    Pmax, Rmax = torch.max(P), torch.max(R)
    Pmin, Rmin = torch.min(P), torch.min(R)

    beta = 35.0
    xpmax = torch.sum(torch.softmax(beta * P.reshape(-1), dim=0) * x.reshape(-1))
    xrmax = torch.sum(torch.softmax(beta * R.reshape(-1), dim=0) * x.reshape(-1))

    xpmin = torch.sum(torch.softmax(-beta * P.reshape(-1), dim=0) * x.reshape(-1))
    xrmin = torch.sum(torch.softmax(-beta * R.reshape(-1), dim=0) * x.reshape(-1))

    areaP = torch.trapz(P.reshape(-1), x.reshape(-1))
    areaR = torch.trapz(R.reshape(-1), x.reshape(-1))

    return (
        ((Pmax - Rmax) / scale) ** 2
        + ((Pmin - Rmin) / scale) ** 2
        + 0.4 * (xpmax - xrmax) ** 2
        + 0.4 * (xpmin - xrmin) ** 2
        + 0.2 * ((areaP - areaR) / scale) ** 2
    )


def high_point_losses(model, ref, pool, batch, v, v2, w, scale_m, scale_R):
    idx = sample_from_pool(pool, batch)
    x = ref["x"][idx]

    rho, ux, T, qx, sig, m, R = compute_high(model, x, v, v2, w)

    mr = ref["mxxx"][idx]
    Rr = ref["Rxx_closure"][idx]

    wm = 1.0 + 4.0 * torch.abs(mr) / (scale_m + 1e-12)
    wR = 1.0 + 8.0 * torch.abs(Rr) / (scale_R + 1e-12)

    Lm_raw = torch.mean(wm * ((m - mr) / scale_m) ** 2)
    LR_raw = torch.mean(wR * ((R - Rr) / scale_R) ** 2)

    mhat = normalize_m(m, rho, T)
    mhat_r = normalize_m(mr, ref["rho"][idx], ref["T"][idx])

    Rhat = normalize_R(R, rho, T)
    Rhat_r = normalize_R(Rr, ref["rho"][idx], ref["T"][idx])

    Lm_norm = F.huber_loss(mhat, mhat_r, delta=0.025)
    LR_norm = F.huber_loss(Rhat, Rhat_r, delta=0.025)

    return Lm_raw, LR_raw, Lm_norm, LR_norm


def high_shape_losses(model, ref, idx_shape, v, v2, w, scale_m, scale_R):
    idx = idx_shape
    x = ref["x"][idx]
    sort = torch.argsort(x)
    idx = idx[sort]
    x = ref["x"][idx]

    rho, ux, T, qx, sig, m, R = compute_high(model, x, v, v2, w)
    mr = ref["mxxx"][idx]
    Rr = ref["Rxx_closure"][idx]

    Lm_shape = F.huber_loss(finite_slope(m, x) / scale_m, finite_slope(mr, x) / scale_m, delta=0.05)
    LR_shape = F.huber_loss(finite_slope(R, x) / scale_R, finite_slope(Rr, x) / scale_R, delta=0.05)

    Lm_feat = raw_feature_loss(x, m, mr, scale_m)
    LR_feat = raw_feature_loss(x, R, Rr, scale_R)

    return Lm_shape, LR_shape, Lm_feat, LR_feat





def tail_equilibrium_loss(model, ref, v, v2, w, n_each=48):
    """
    Far-field equilibrium protection with robust sig/sigma naming.
    """
    x = ref["x"]
    n = x.numel()
    k = min(int(n_each), max(4, n // 8))

    idx_left = torch.arange(0, k, device=x.device)
    idx_right = torch.arange(n-k, n, device=x.device)
    idx = torch.cat([idx_left, idx_right], dim=0)

    xb = x[idx]
    rho, ux, T, qx, sig, m, R = compute_high(model, xb, v, v2, w)

    sig_ref_all = ref["sigma_xx"] if "sigma_xx" in ref else ref["sig"]
    R_ref_all = ref["Rxx_closure"] if "Rxx_closure" in ref else ref["Rxx"]

    q_ref = ref["qx"][idx]
    sig_ref = sig_ref_all[idx]
    m_ref = ref["mxxx"][idx]
    R_ref = R_ref_all[idx]

    q_scale = torch.clamp(torch.max(torch.abs(ref["qx"])), min=torch.tensor(1e-6, device=x.device, dtype=x.dtype))
    sig_scale = torch.clamp(torch.max(torch.abs(sig_ref_all)), min=torch.tensor(1e-6, device=x.device, dtype=x.dtype))
    m_scale = torch.clamp(torch.max(torch.abs(ref["mxxx"])), min=torch.tensor(1e-6, device=x.device, dtype=x.dtype))
    R_scale = torch.clamp(torch.max(torch.abs(R_ref_all)), min=torch.tensor(1e-6, device=x.device, dtype=x.dtype))

    Lq = torch.mean(((qx - q_ref) / q_scale)**2)
    Lsig = torch.mean(((sig - sig_ref) / sig_scale)**2)
    Lm = torch.mean(((m - m_ref) / m_scale)**2)
    LR = torch.mean(((R - R_ref) / R_scale)**2)

    return Lq, Lsig, Lm, LR

def set_stage_requires(model, stage):
    # stage 1: fit m head only
    # stage 2: fit R head while m head stays trainable/protected
    # stage 3: unfreeze low base with very small LR
    for p in model.net.parameters():
        p.requires_grad_(stage >= 3)
    for p in model.m_head.parameters():
        p.requires_grad_(True)
    for p in model.R_head.parameters():
        p.requires_grad_(stage >= 2)


def train(args):
    seed_all(args.seed)

    device = safe_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    ref = load_ref_high(args.ref, device, dtype)
    ref = ensure_ref_aliases(ref)
    micro = load_micro_ref(args.micro_ref, device, dtype, ref=ref)
    v, v2, w = build_vgrid(args.nvx, args.nvy, args.nvz, args.vmax, device, dtype)

    scales = build_scales(ref)
    idx_all, high_pool, idx_shape = make_pools(ref, args.use_smart_highmom_anchors)

    scale_m = torch.clamp(torch.max(torch.abs(ref["mxxx"])), min=torch.tensor(1e-6, device=device, dtype=dtype))
    scale_R = torch.clamp(torch.max(torch.abs(ref["Rxx_closure"])), min=torch.tensor(1e-6, device=device, dtype=dtype))

    model = StandingPositiveHMMDecoupled(
        ref,
        psi_max=args.psi_max,
        macro_amp=args.macro_amp,
        q_amp=args.q_amp,
        sig_amp=args.sig_amp,
        h3_amp=args.h3_amp,
        h4_amp=args.h4_amp,
    ).to(device=device, dtype=dtype)

    copy_partial(model, args.init, device)

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = vars(args).copy()
    cfg["ref_used"] = args.ref
    cfg["scale_m"] = float(scale_m.detach().cpu())
    cfg["scale_R"] = float(scale_R.detach().cpu())
    cfg["num_shape_anchors"] = int(idx_shape.numel())
    cfg["device_used"] = str(device)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    opt = torch.optim.AdamW(
        [
            {"params": model.net.parameters(), "lr": args.lr_low},
            {"params": model.m_head.parameters(), "lr": args.lr_m},
            {"params": model.R_head.parameters(), "lr": args.lr_R},
        ],
        weight_decay=args.weight_decay,
    )

    # v17: add R-closure-head parameters to optimizer with a separate faster LR.

    try:

        _existing_params = set()

        for _g in opt.param_groups:

            for _p in _g.get("params", []):

                _existing_params.add(id(_p))

        _rtail_params = []

        for _p in [getattr(model, "Rtail_coef", None), getattr(model, "Rtail_log_width", None)]:

            if _p is not None and id(_p) not in _existing_params:

                _rtail_params.append(_p)

        if len(_rtail_params) > 0:

            opt.add_param_group({"params": _rtail_params, "lr": args.lr_Rhead, "weight_decay": 0.0})

            print(f"[v17] added R-closure-head params to optimizer, lr={args.lr_Rhead:.3e}", flush=True)

    except Exception as _e:

        print("[v17] could not add R-closure-head params:", repr(_e), flush=True)



    best = 1e99
    print(f"[train] run={args.run_name}", flush=True)
    print(f"[scale] m={float(scale_m):.6e}, R={float(scale_R):.6e}", flush=True)
    print(f"[anchors] pool={high_pool.numel()}, shape={idx_shape.numel()}", flush=True)

    for step in range(1, args.steps + 1):
        if step <= args.stage_m:
            stage = 1
        elif step <= args.stage_R:
            stage = 2
        else:
            stage = 3

        set_stage_requires(model, stage)

        opt.zero_grad(set_to_none=True)

        lmacro, mp = macro_anchor_loss(model, ref, idx_all, args.batch_macro, v, v2, w, scales)
        lqsig, qp = qsig_anchor_loss(model, ref, high_pool, args.batch_qsig, v, v2, w, scales)
        lflux, fp = flux_loss(model, ref, args.batch_flux, v, v2, w, scales, device, dtype)
        lbc = boundary_loss(model, ref, v, w)

        if stage >= 3 and (args.w_pde > 0.0 or args.w_pde_rel > 0.0):
            lpde, lpder = pde_loss(model, ref, args.batch_pde, v, v2, w, ref["kn_eff"], device, dtype)
        else:
            lpde = torch.zeros((), device=device, dtype=dtype)
            lpder = torch.zeros((), device=device, dtype=dtype)

        Lm_raw, LR_raw, Lm_norm, LR_norm = high_point_losses(
            model, ref, high_pool, args.batch_high, v, v2, w, scale_m, scale_R
        )
        # Memory-safe shape/feature anchors:
        # using the full idx_shape at once can exceed 32 GB VRAM.
        n_shape = min(int(args.batch_shape), int(idx_shape.numel()))
        idx_shape_use = sample_from_pool(idx_shape, n_shape)
        Lm_shape, LR_shape, Lm_feat, LR_feat = high_shape_losses(
            model, ref, idx_shape_use, v, v2, w, scale_m, scale_R
        )

        Ltail_q, Ltail_sig, Ltail_m, Ltail_R = tail_equilibrium_loss(
            model, ref, v, v2, w, n_each=args.batch_tail
        )

        if stage == 1:
            wm_raw = args.w_m_raw
            wm_norm = args.w_m_norm
            wm_shape = args.w_m_shape
            wm_feat = args.w_m_feat
            wR_raw = 0.0
            wR_norm = 0.0
            wR_shape = 0.0
            wR_feat = 0.0
        elif stage == 2:
            # keep m protected while ramping R
            rr = ramp(step, args.stage_m, args.stage_R, 1.0)
            wm_raw = args.w_m_raw_protect
            wm_norm = args.w_m_norm
            wm_shape = args.w_m_shape_protect
            wm_feat = args.w_m_feat_protect
            wR_raw = args.w_R_raw * rr
            wR_norm = args.w_R_norm * rr
            wR_shape = args.w_R_shape * rr
            wR_feat = args.w_R_feat * rr
        else:
            wm_raw = args.w_m_raw_final
            wm_norm = args.w_m_norm
            wm_shape = args.w_m_shape_final
            wm_feat = args.w_m_feat_final
            wR_raw = args.w_R_raw_final
            wR_norm = args.w_R_norm
            wR_shape = args.w_R_shape_final
            wR_feat = args.w_R_feat_final

        Lmicro_log, Lmicro_HR = micro_losses_v15(
            model,
            micro,
            batch=args.batch_micro,
            score_power=args.micro_score_power,
            clip=args.micro_clip,
        )

        loss = (
            args.w_macro * lmacro
            + args.w_qsig * lqsig
            + args.w_flux * lflux
            + args.w_bc * lbc
            + args.w_pde * lpde
            + args.w_pde_rel * lpder
            + wm_raw * Lm_raw
            + wm_norm * Lm_norm
            + wm_shape * Lm_shape
            + wm_feat * Lm_feat
            + wR_raw * LR_raw
            + wR_norm * LR_norm
            + wR_shape * LR_shape
            + wR_feat * LR_feat
            + args.w_tail_q * Ltail_q
            + args.w_tail_sig * Ltail_sig
            + args.w_tail_m * Ltail_m
            + args.w_tail_R * Ltail_R
            + args.w_micro_log * Lmicro_log + args.w_micro_HR * Lmicro_HR
        )

        if not torch.isfinite(loss):
            print("[error] non-finite loss", flush=True)
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            print(
                f"[log] step={step:06d} stage={stage} total={float(loss):.6e} "
                f"macro={float(lmacro):.3e} qsig={float(lqsig):.3e} flux={float(lflux):.3e} "
                f"pde={float(lpde):.3e} pder={float(lpder):.3e} "
                f"m_raw={float(Lm_raw):.3e} m_norm={float(Lm_norm):.3e} "
                f"m_sh={float(Lm_shape):.3e} m_feat={float(Lm_feat):.3e} "
                f"R_raw={float(LR_raw):.3e} R_norm={float(LR_norm):.3e} "
                f"R_sh={float(LR_shape):.3e} R_feat={float(LR_feat):.3e} micro_log={float(Lmicro_log):.3e} micro_HR={float(Lmicro_HR):.3e} "
                f"rho={float(mp[0]):.2e} ux={float(mp[1]):.2e} T={float(mp[2]):.2e} "
                f"q={float(qp[0]):.2e} sig={float(qp[1]):.2e}",
                flush=True,
            )

        if step % args.save_every == 0 or step == args.steps:
            ck = {"model": model.state_dict(), "step": step, "config": cfg}
            torch.save(ck, run_dir / "ckpt_last.pt")
            torch.save(ck, run_dir / f"ckpt_step{step:06d}.pt")
            if float(loss) < best:
                best = float(loss)
                torch.save(ck, run_dir / "ckpt_best.pt")
            print(f"[save] step={step}, best={best:.6e}", flush=True)


def predict_all(model, ref, v, v2, w, chunk=64):
    out = {k: [] for k in ["rho", "ux", "T", "qx", "sigma_xx", "mxxx", "Rxx_closure", "Qxx_bgk", "Qx_bgk"]}

    with torch.no_grad():
        x = ref["x"]
        for i in range(0, x.numel(), chunk):
            xb = x[i:i+chunk]
            rho, ux, T, qx, sig, m, R = compute_high(model, xb, v, v2, w)
            vals = {
                "rho": rho,
                "ux": ux,
                "T": T,
                "qx": qx,
                "sigma_xx": sig,
                "mxxx": m,
                "Rxx_closure": R,
                "Qxx_bgk": -sig,
                "Qx_bgk": -2.0 * qx,
            }
            for k, val in vals.items():
                out[k].append(val.detach().cpu().numpy())

    return {k: np.concatenate(vv) for k, vv in out.items()}


def plot(args):
    device = safe_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck.get("config", {})

    ref_path = args.ref if args.ref else cfg.get("ref_used")
    ref = load_ref_high(ref_path, device, dtype)

    eval_nvx = int(args.eval_nvx) if int(args.eval_nvx) > 0 else int(cfg.get("nvx", args.nvx))
    eval_nvy = int(args.eval_nvy) if int(args.eval_nvy) > 0 else int(cfg.get("nvy", args.nvy))
    eval_nvz = int(args.eval_nvz) if int(args.eval_nvz) > 0 else int(cfg.get("nvz", args.nvz))
    eval_vmax = float(args.eval_vmax) if float(args.eval_vmax) > 0 else float(cfg.get("vmax", args.vmax))

    print(f"[eval grid] Nv={eval_nvx*eval_nvy*eval_nvz} = {eval_nvx}x{eval_nvy}x{eval_nvz}, vmax={eval_vmax}", flush=True)

    v, v2, w = build_vgrid(
        eval_nvx,
        eval_nvy,
        eval_nvz,
        eval_vmax,
        device,
        dtype,
    )

    model = StandingPositiveHMMDecoupled(
        ref,
        psi_max=float(cfg.get("psi_max", args.psi_max)),
        macro_amp=float(cfg.get("macro_amp", args.macro_amp)),
        q_amp=float(cfg.get("q_amp", args.q_amp)),
        sig_amp=float(cfg.get("sig_amp", args.sig_amp)),
        h3_amp=float(cfg.get("h3_amp", args.h3_amp)),
        h4_amp=float(cfg.get("h4_amp", args.h4_amp)),
    ).to(device=device, dtype=dtype)

    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    pred = predict_all(model, ref, v, v2, w, chunk=args.plot_chunk)

    z = np.load(ref_path)
    xplot = z["x_mfp"] if "x_mfp" in z.files else z["x"]

    dvm = {
        "rho": z["rho"],
        "ux": z["ux"],
        "T": z["T"],
        "qx": z["qx"],
        "sigma_xx": z["sigma_xx"] if "sigma_xx" in z.files else z["sig"],
        "mxxx": z["mxxx"],
        "Rxx_closure": z["Rxx_closure"] if "Rxx_closure" in z.files else z["Rxx"],
    }
    dvm["Qxx_bgk"] = -dvm["sigma_xx"]
    dvm["Qx_bgk"] = -2.0 * dvm["qx"]

    items = [
        ("rho", r"$\rho$"),
        ("ux", r"$u_x$"),
        ("T", r"$T$"),
        ("qx", r"$q_x$"),
        ("sigma_xx", r"$\sigma_{xx}$"),
        ("mxxx", r"$m_{xxx}^{cl}$"),
        ("Rxx_closure", r"$R_{xx}^{cl}$"),
        ("Qxx_bgk", r"$Q_{xx}^{BGK}$"),
        ("Qx_bgk", r"$Q_x^{BGK}$"),
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    errors = {}

    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5), sharex=True)
    for ax, (k, lab) in zip(axes.ravel(), items):
        P = pred[k]
        R = dvm[k]
        e = rel_l2(P, R)
        errors[k] = e
        ax.plot(xplot, P, lw=2.2, label="PINN-v17")
        ax.plot(xplot, R, "--", lw=2.0, label="DVM")
        ax.set_title(f"{lab}, relL2={e:.2e}")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(r"$x/\lambda_1$")
        ax.legend(fontsize=8)

    fig.suptitle("Standing BGK normal shock: decoupled-head PINN-v17 vs DVM")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print("[plot]", out)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    for ax, (k, lab) in zip(axes.ravel(), items[5:]):
        P = pred[k]
        R = dvm[k]
        e = rel_l2(P, R)
        ax.plot(xplot, P, lw=2.4, label="PINN-v17")
        ax.plot(xplot, R, "--", lw=2.1, label="DVM")
        ax.set_title(f"{lab}, relL2={e:.2e}")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(r"$x/\lambda_1$")
        ax.legend(fontsize=8)

    fig.suptitle("Closure-level moments and BGK production terms: PINN-v17 vs DVM")
    fig.tight_layout()
    out2 = out.with_name(out.stem + "_closure.png")
    fig.savefig(out2, dpi=300)
    print("[plot]", out2)

    def ddx(y, x):
        return np.gradient(y, x, edge_order=2)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    dsig_p = ddx(pred["sigma_xx"], xplot)
    dsig_r = ddx(dvm["sigma_xx"], xplot)
    dq_p = ddx(pred["qx"], xplot)
    dq_r = ddx(dvm["qx"], xplot)

    axes[0,0].plot(dsig_p, pred["mxxx"], lw=2.2, label="PINN-v17")
    axes[0,0].plot(dsig_r, dvm["mxxx"], "--", lw=2.0, label="DVM")
    axes[0,0].set_xlabel(r"$\partial_x\sigma_{xx}$")
    axes[0,0].set_ylabel(r"$m_{xxx}^{cl}$")

    axes[0,1].plot(dq_p, pred["Rxx_closure"], lw=2.2, label="PINN-v17")
    axes[0,1].plot(dq_r, dvm["Rxx_closure"], "--", lw=2.0, label="DVM")
    axes[0,1].set_xlabel(r"$\partial_x q_x$")
    axes[0,1].set_ylabel(r"$R_{xx}^{cl}$")

    axes[1,0].plot(pred["sigma_xx"], pred["Qxx_bgk"], lw=2.2, label="PINN-v17")
    axes[1,0].plot(dvm["sigma_xx"], dvm["Qxx_bgk"], "--", lw=2.0, label="DVM")
    axes[1,0].set_xlabel(r"$\sigma_{xx}$")
    axes[1,0].set_ylabel(r"$Q_{xx}^{BGK}$")

    axes[1,1].plot(pred["qx"], pred["Qx_bgk"], lw=2.2, label="PINN-v17")
    axes[1,1].plot(dvm["qx"], dvm["Qx_bgk"], "--", lw=2.0, label="DVM")
    axes[1,1].set_xlabel(r"$q_x$")
    axes[1,1].set_ylabel(r"$Q_x^{BGK}$")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Phase-space closure relations: PINN-v17 vs DVM")
    fig.tight_layout()
    out3 = out.with_name(out.stem + "_phase.png")
    fig.savefig(out3, dpi=300)
    print("[plot]", out3)

    csv = out.with_suffix(".csv")
    with open(csv, "w") as f:
        f.write("quantity,relL2\n")
        for k, e in errors.items():
            f.write(f"{k},{e:.8e}\n")
    print("[csv]", csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "plot"], default="train")
    ap.add_argument("--ref", default="ref/standing_M2_x40_hmom_dvm.npz")
    ap.add_argument("--run-name", default="standing_M2_x40_v12_decoupled")
    ap.add_argument("--init", default="runs/standing_M2_x40_v7/ckpt_last.pt")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="figures/standing_M2_x40_v12_decoupled_compare.png")

    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=1212)

    ap.add_argument("--nvx", type=int, default=44)
    ap.add_argument("--nvy", type=int, default=16)
    ap.add_argument("--nvz", type=int, default=16)
    ap.add_argument("--vmax", type=float, default=12.0)

    ap.add_argument("--steps", type=int, default=18000)
    ap.add_argument("--stage-m", type=int, default=5000)
    ap.add_argument("--stage-R", type=int, default=13000)

    ap.add_argument("--lr-low", type=float, default=4.0e-7)
    ap.add_argument("--lr-m", type=float, default=4.0e-6)
    ap.add_argument("--lr-Rhead", type=float, default=5.0e-4)
    ap.add_argument("--lr-R", type=float, default=3.0e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-8)
    ap.add_argument("--grad-clip", type=float, default=0.045)

    ap.add_argument("--batch-macro", type=int, default=128)
    ap.add_argument("--batch-qsig", type=int, default=128)
    ap.add_argument("--batch-high", type=int, default=192)
    ap.add_argument("--batch-flux", type=int, default=128)
    ap.add_argument("--batch-pde", type=int, default=4)
    ap.add_argument("--batch-shape", type=int, default=64)
    ap.add_argument("--batch-tail", type=int, default=48)
    ap.add_argument("--w-tail-q", type=float, default=0.0)
    ap.add_argument("--w-tail-sig", type=float, default=0.0)
    ap.add_argument("--w-tail-m", type=float, default=0.0)
    ap.add_argument("--w-tail-R", type=float, default=0.0)

    ap.add_argument("--psi-max", type=float, default=2.8)
    ap.add_argument("--macro-amp", type=float, default=0.25)
    ap.add_argument("--q-amp", type=float, default=1.6)
    ap.add_argument("--sig-amp", type=float, default=1.6)
    ap.add_argument("--h3-amp", type=float, default=1.4)
    ap.add_argument("--h4-amp", type=float, default=2.8)

    ap.add_argument("--w-macro", type=float, default=220.0)
    ap.add_argument("--w-qsig", type=float, default=160.0)
    ap.add_argument("--w-flux", type=float, default=20.0)
    ap.add_argument("--w-bc", type=float, default=2.0)
    ap.add_argument("--w-pde", type=float, default=0.001)
    ap.add_argument("--w-pde-rel", type=float, default=0.0005)

    ap.add_argument("--w-m-raw", type=float, default=36.0)
    ap.add_argument("--w-m-norm", type=float, default=2.0)
    ap.add_argument("--w-m-shape", type=float, default=12.0)
    ap.add_argument("--w-m-feat", type=float, default=4.0)

    ap.add_argument("--w-m-raw-protect", type=float, default=44.0)
    ap.add_argument("--w-m-shape-protect", type=float, default=14.0)
    ap.add_argument("--w-m-feat-protect", type=float, default=5.0)

    ap.add_argument("--w-R-raw", type=float, default=55.0)
    ap.add_argument("--w-R-norm", type=float, default=4.0)
    ap.add_argument("--w-R-shape", type=float, default=14.0)
    ap.add_argument("--w-R-feat", type=float, default=8.0)

    ap.add_argument("--w-m-raw-final", type=float, default=30.0)
    ap.add_argument("--w-m-shape-final", type=float, default=10.0)
    ap.add_argument("--w-m-feat-final", type=float, default=3.0)

    ap.add_argument("--w-R-raw-final", type=float, default=38.0)
    ap.add_argument("--w-R-shape-final", type=float, default=10.0)
    ap.add_argument("--w-R-feat-final", type=float, default=5.0)

    ap.add_argument("--use-smart-highmom-anchors", action="store_true")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--plot-chunk", type=int, default=64)
    ap.add_argument("--eval-nvx", type=int, default=0)
    ap.add_argument("--eval-nvy", type=int, default=0)
    ap.add_argument("--eval-nvz", type=int, default=0)
    ap.add_argument("--eval-vmax", type=float, default=0.0)


    ap.add_argument("--micro-ref", type=str, default="")
    ap.add_argument("--batch-micro", type=int, default=512)
    ap.add_argument("--w-micro-log", type=float, default=0.0)
    ap.add_argument("--w-micro-HR", type=float, default=0.0)
    ap.add_argument("--micro-score-power", type=float, default=0.5)
    ap.add_argument("--micro-clip", type=float, default=8.0)

    args = ap.parse_args()

    if args.mode == "train":
        train(args)
    else:
        if not args.ckpt:
            args.ckpt = f"runs/{args.run_name}/ckpt_last.pt"
        plot(args)


if __name__ == "__main__":
    main()
