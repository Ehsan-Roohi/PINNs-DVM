#!/usr/bin/env python
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trap_weights(n, h, device, dtype):
    w = torch.ones(n, device=device, dtype=dtype) * h
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def build_vgrid(nvx, nvy, nvz, vmax, device, dtype):
    vx = torch.linspace(-vmax, vmax, nvx, device=device, dtype=dtype)
    vy = torch.linspace(-vmax, vmax, nvy, device=device, dtype=dtype)
    vz = torch.linspace(-vmax, vmax, nvz, device=device, dtype=dtype)

    wx = trap_weights(nvx, 2.0*vmax/(nvx-1), device, dtype)
    wy = trap_weights(nvy, 2.0*vmax/(nvy-1), device, dtype)
    wz = trap_weights(nvz, 2.0*vmax/(nvz-1), device, dtype)

    VX, VY, VZ = torch.meshgrid(vx, vy, vz, indexing="ij")
    W = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]

    v = torch.stack([VX.reshape(-1), VY.reshape(-1), VZ.reshape(-1)], dim=1)
    w = W.reshape(-1)
    v2 = v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2

    print(f"[quad] Nv={v.shape[0]} = {nvx}x{nvy}x{nvz}, vmax={vmax}")
    return v, v2, w


def maxwellian(v, rho, ux, T):
    # rho, ux, T: [N], v: [N,3] or broadcast-compatible
    c2 = (v[:, 0] - ux)**2 + v[:, 1]**2 + v[:, 2]**2
    pref = rho / torch.clamp((2.0*math.pi*T)**1.5, min=1.0e-30)
    return pref * torch.exp(-c2 / torch.clamp(2.0*T, min=1.0e-30))


def maxwellian_grid(v, rho, ux, T):
    # rho, ux, T: [B], v: [Nv,3] => [B,Nv]
    c2 = (v[None, :, 0] - ux[:, None])**2 + v[None, :, 1]**2 + v[None, :, 2]**2
    pref = rho[:, None] / torch.clamp((2.0*math.pi*T[:, None])**1.5, min=1.0e-30)
    return pref * torch.exp(-c2 / torch.clamp(2.0*T[:, None], min=1.0e-30))


def integrate_moments(f, v, v2, w):
    # f: [B,Nv]
    rho = torch.sum(f*w[None, :], dim=1)
    mom = torch.sum(f*v[None, :, 0]*w[None, :], dim=1)
    ux = mom / torch.clamp(rho, min=1.0e-30)

    e2 = torch.sum(f*v2[None, :]*w[None, :], dim=1)
    T = (e2/torch.clamp(rho, min=1.0e-30) - ux**2)/3.0
    T = torch.clamp(T, min=1.0e-8)

    cx = v[None, :, 0] - ux[:, None]
    cy = v[None, :, 1]
    cz = v[None, :, 2]
    c2 = cx**2 + cy**2 + cz**2

    qx = 0.5 * torch.sum(c2*cx*f*w[None, :], dim=1)
    sig = torch.sum((cx**2 - c2/3.0)*f*w[None, :], dim=1)

    # steady collision-invariant fluxes
    Fm = torch.sum(v[None, :, 0]*f*w[None, :], dim=1)
    Fp = torch.sum((v[None, :, 0]**2)*f*w[None, :], dim=1)
    Fe = torch.sum(0.5*v2[None, :]*v[None, :, 0]*f*w[None, :], dim=1)

    return rho, ux, T, qx, sig, Fm, Fp, Fe


def rel_mse(a, b, scale):
    return torch.mean(((a-b)/scale)**2)


def load_ref(path, device, dtype):
    z = np.load(path)
    d = {k: z[k] for k in z.files}

    x = d["x"] if "x" in d else d["x_scaled"]
    rho = d["rho"]
    ux = d["ux"]
    T = d["T"]
    qx = d["qx"]
    sig = d["sigma_xx"] if "sigma_xx" in d else d["sig"]

    states = d["states"]
    rho1, u1, T1, rho2, u2, T2 = [float(a) for a in states[:6]]

    kn_eff = float(d["kn_eff"]) if "kn_eff" in d else 0.0125

    ref = {
        "x": torch.tensor(x, device=device, dtype=dtype),
        "rho": torch.tensor(rho, device=device, dtype=dtype),
        "ux": torch.tensor(ux, device=device, dtype=dtype),
        "T": torch.tensor(T, device=device, dtype=dtype),
        "qx": torch.tensor(qx, device=device, dtype=dtype),
        "sig": torch.tensor(sig, device=device, dtype=dtype),
        "states": dict(rho1=rho1, u1=u1, T1=T1, rho2=rho2, u2=u2, T2=T2),
        "kn_eff": kn_eff,
    }

    # estimate shock center/width from density profile
    xr = x
    rr = rho
    mid = 0.5*(rho1+rho2)
    k = int(np.argmin(np.abs(rr-mid)))
    x0 = float(xr[k])

    r10 = rho1 + 0.10*(rho2-rho1)
    r90 = rho1 + 0.90*(rho2-rho1)
    k10 = int(np.argmin(np.abs(rr-r10)))
    k90 = int(np.argmin(np.abs(rr-r90)))
    width = abs(float(xr[k90]-xr[k10]))/4.0
    width = max(width, 0.025)

    ref["x0"] = x0
    ref["width"] = width

    print(f"[ref] path={path}")
    print(f"[ref] kn_eff={kn_eff:.6e}, x0={x0:.5f}, width={width:.5f}")
    print(f"[ref] left=(rho={rho1:.6f}, u={u1:.6f}, T={T1:.6f})")
    print(f"[ref] right=(rho={rho2:.6f}, u={u2:.6f}, T={T2:.6f})")

    return ref


# -----------------------------
# Positive macro-micro model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=96, depth=5):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            layers.append(nn.SiLU())
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class StandingPositiveHMM(nn.Module):
    def __init__(self, ref, psi_max=2.5, macro_amp=0.25, q_amp=1.6, sig_amp=1.6, h3_amp=0.4):
        super().__init__()
        self.ref = ref
        self.psi_max = psi_max
        self.macro_amp = macro_amp
        self.q_amp = q_amp
        self.sig_amp = sig_amp
        self.h3_amp = h3_amp

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

        # input features: x, centered x, tanh shock coordinate, base switch, domain gate
        self.net = MLP(5, 6, width=96, depth=5)

    def base_switch(self, x):
        return 0.5*(1.0 + torch.tanh((x - self.x0)/self.width))

    def gates(self, x):
        # domain gate vanishes at boundaries
        gd = ((x-self.xmin)*(self.xmax-x))/0.25
        gd = torch.clamp(gd, min=0.0, max=1.0)

        # shock-local gate
        gs = torch.exp(-((x-self.x0)/(3.0*self.width))**2)
        return gd, gs

    def forward_params(self, x):
        # x shape [N]
        S = self.base_switch(x)
        gd, gs = self.gates(x)

        log_rho_base = (1-S)*math.log(self.rho1) + S*math.log(self.rho2)
        ux_base = (1-S)*self.u1 + S*self.u2
        log_T_base = (1-S)*math.log(self.T1) + S*math.log(self.T2)

        xin = torch.stack([
            x,
            x - self.x0,
            torch.tanh((x-self.x0)/self.width),
            S,
            gd*gs,
        ], dim=1)

        y = self.net(xin)
        drho, du, dT, aq, asig, a3 = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5]

        # macro correction is local and boundary-vanishing
        gmacro = gd * gs
        rho = torch.exp(log_rho_base + self.macro_amp*gmacro*torch.tanh(drho))
        T = torch.exp(log_T_base + self.macro_amp*gmacro*torch.tanh(dT))

        du_scale = abs(self.u1 - self.u2) + 1.0e-8
        ux = ux_base + 0.30*du_scale*gmacro*torch.tanh(du)

        # non-equilibrium amplitudes localize to shock layer
        aq = self.q_amp * gs * torch.tanh(aq)
        asig = self.sig_amp * gs * torch.tanh(asig)
        a3 = self.h3_amp * gs * torch.tanh(a3)

        return rho, ux, T, aq, asig, a3

    def f_flat(self, x, v):
        # x [N], v [N,3]
        rho, ux, T, aq, asig, a3 = self.forward_params(x)

        cx = (v[:,0] - ux)/torch.sqrt(T)
        cy = v[:,1]/torch.sqrt(T)
        cz = v[:,2]/torch.sqrt(T)
        c2 = cx**2 + cy**2 + cz**2

        Hq = cx*(c2 - 5.0)
        Hs = cx**2 - c2/3.0
        H3 = cx**3 - (3.0/5.0)*cx*c2

        raw = 0.08*aq*Hq + 0.18*asig*Hs + 0.06*a3*H3
        psi = self.psi_max * torch.tanh(raw/self.psi_max)

        c2phys = (v[:,0]-ux)**2 + v[:,1]**2 + v[:,2]**2
        M = rho / torch.clamp((2.0*math.pi*T)**1.5, min=1.0e-30)
        M = M * torch.exp(-c2phys/torch.clamp(2.0*T, min=1.0e-30))

        f = M * torch.exp(psi)
        f = torch.clamp(f, min=1.0e-30, max=1.0e6)
        return f, M, rho, ux, T, aq, asig, a3

    def f_grid(self, x, v):
        # x [B], v [Nv,3] => [B,Nv]
        B = x.numel()
        Nv = v.shape[0]
        xx = x[:, None].expand(B, Nv).reshape(-1)
        vv = v[None, :, :].expand(B, Nv, 3).reshape(-1, 3)
        f, M, *_ = self.f_flat(xx, vv)
        return f.reshape(B, Nv)


# -----------------------------
# Sampling pools
# -----------------------------
def make_pools(ref, device):
    x = ref["x"]
    q = torch.abs(ref["qx"])
    s = torch.abs(ref["sig"])

    signal = q/(torch.max(q)+1e-30) + s/(torch.max(s)+1e-30)
    mask = signal > 0.05

    idx_all = torch.arange(x.numel(), device=device)
    idx_shock = idx_all[mask]
    if idx_shock.numel() < 32:
        idx_shock = idx_all

    return idx_all, idx_shock


def sample_from_pool(pool, n):
    j = torch.randint(0, pool.numel(), (n,), device=pool.device)
    return pool[j]


def sample_pde_x(ref, n, device, dtype):
    x0 = ref["x0"]
    width = ref["width"]

    n1 = int(0.75*n)
    n2 = n - n1

    xs = torch.empty(n, device=device, dtype=dtype)
    xs[:n1] = torch.randn(n1, device=device, dtype=dtype)*(3.0*width) + x0
    xs[:n1] = torch.clamp(xs[:n1], -0.5, 0.5)
    xs[n1:] = -0.5 + torch.rand(n2, device=device, dtype=dtype)
    return xs


# -----------------------------
# Losses
# -----------------------------
def eval_moments_at_x(model, x, v, v2, w):
    f = model.f_grid(x, v)
    return integrate_moments(f, v, v2, w)


def macro_anchor_loss(model, ref, pool, batch, v, v2, w, scales):
    idx = sample_from_pool(pool, batch)
    x = ref["x"][idx]

    rho, ux, T, qx, sig, Fm, Fp, Fe = eval_moments_at_x(model, x, v, v2, w)

    lrho = rel_mse(rho, ref["rho"][idx], scales["rho"])
    lux  = rel_mse(ux,  ref["ux"][idx],  scales["ux"])
    lT   = rel_mse(T,   ref["T"][idx],   scales["T"])

    return lrho + lux + lT, (lrho.detach(), lux.detach(), lT.detach())


def qsig_anchor_loss(model, ref, pool, batch, v, v2, w, scales):
    idx = sample_from_pool(pool, batch)
    x = ref["x"][idx]

    rho, ux, T, qx, sig, Fm, Fp, Fe = eval_moments_at_x(model, x, v, v2, w)

    lq = rel_mse(qx, ref["qx"][idx], scales["qx"])
    ls = rel_mse(sig, ref["sig"][idx], scales["sig"])
    return lq + ls, (lq.detach(), ls.detach())


def flux_loss(model, ref, batch, v, v2, w, scales, device, dtype):
    x = sample_pde_x(ref, batch, device, dtype)
    rho, ux, T, qx, sig, Fm, Fp, Fe = eval_moments_at_x(model, x, v, v2, w)

    st = ref["states"]
    rho1, u1, T1 = st["rho1"], st["u1"], st["T1"]

    Fm0 = rho1*u1
    Fp0 = rho1*u1*u1 + rho1*T1
    Fe0 = u1*(0.5*rho1*u1*u1 + 2.5*rho1*T1)

    l1 = torch.mean(((Fm - Fm0)/scales["Fm"])**2)
    l2 = torch.mean(((Fp - Fp0)/scales["Fp"])**2)
    l3 = torch.mean(((Fe - Fe0)/scales["Fe"])**2)
    return l1 + l2 + l3, (l1.detach(), l2.detach(), l3.detach())


def boundary_loss(model, ref, v, w):
    device = v.device
    dtype = v.dtype

    st = ref["states"]
    pos = v[:,0] > 0
    neg = v[:,0] < 0

    xL = torch.full((pos.sum().item(),), -0.5, device=device, dtype=dtype)
    xR = torch.full((neg.sum().item(),),  0.5, device=device, dtype=dtype)

    fL, ML, *_ = model.f_flat(xL, v[pos])
    fR, MR, *_ = model.f_flat(xR, v[neg])

    MLt = maxwellian(
        v[pos],
        torch.full_like(xL, st["rho1"]),
        torch.full_like(xL, st["u1"]),
        torch.full_like(xL, st["T1"]),
    )
    MRt = maxwellian(
        v[neg],
        torch.full_like(xR, st["rho2"]),
        torch.full_like(xR, st["u2"]),
        torch.full_like(xR, st["T2"]),
    )

    lL = torch.mean(((fL - MLt)/(MLt + 1.0e-8))**2)
    lR = torch.mean(((fR - MRt)/(MRt + 1.0e-8))**2)
    return lL + lR


def pde_loss(model, ref, batch, v, v2, w, kn_eff, device, dtype):
    B = batch
    Nv = v.shape[0]

    x = sample_pde_x(ref, B, device, dtype)
    xx = x[:, None].expand(B, Nv).reshape(-1).clone().detach().requires_grad_(True)
    vv = v[None, :, :].expand(B, Nv, 3).reshape(-1, 3)

    f, M0, *_ = model.f_flat(xx, vv)
    dfdx = torch.autograd.grad(
        f, xx,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    fmat = f.reshape(B, Nv)
    rho, ux, T, qx, sig, Fm, Fp, Fe = integrate_moments(fmat, v, v2, w)
    Meq = maxwellian_grid(v, rho, ux, T).reshape(-1)

    vx = vv[:, 0]
    res = vx*dfdx - (Meq - f)/kn_eff

    # velocity-weighted residual; this was important in BGK PINN studies
    ww = w[None, :].expand(B, Nv).reshape(-1)
    weight = ww * (1.0 + 0.01*(vv[:,0]**2 + vv[:,1]**2 + vv[:,2]**2)**2)

    abs_scale = torch.mean(weight*((Meq/kn_eff)**2 + 1.0e-12)).detach()
    lpde_abs = torch.sum(weight*res**2)/torch.sum(weight) / (abs_scale + 1.0e-12)

    den = (torch.abs(vx*dfdx) + torch.abs((Meq-f)/kn_eff) + 1.0e-7).detach()
    lpde_rel = torch.sum(weight*(res/den)**2)/torch.sum(weight)

    return lpde_abs, lpde_rel


# -----------------------------
# Training / plotting
# -----------------------------
def build_scales(ref):
    scales = {}
    for k in ["rho", "ux", "T", "qx", "sig"]:
        val = ref[k]
        scales[k if k != "sig" else "sig"] = torch.clamp(torch.max(torch.abs(val)), min=torch.tensor(1.0e-3, device=val.device, dtype=val.dtype))

    st = ref["states"]
    rho1, u1, T1 = st["rho1"], st["u1"], st["T1"]

    scales["Fm"] = torch.tensor(abs(rho1*u1)+1e-6, device=ref["x"].device, dtype=ref["x"].dtype)
    scales["Fp"] = torch.tensor(abs(rho1*u1*u1 + rho1*T1)+1e-6, device=ref["x"].device, dtype=ref["x"].dtype)
    scales["Fe"] = torch.tensor(abs(u1*(0.5*rho1*u1*u1 + 2.5*rho1*T1))+1e-6, device=ref["x"].device, dtype=ref["x"].dtype)

    # make q/sig slightly protected from tiny denominators
    scales["qx"] = torch.clamp(scales["qx"], min=torch.tensor(1.0e-2, device=ref["x"].device, dtype=ref["x"].dtype))
    scales["sig"] = torch.clamp(scales["sig"], min=torch.tensor(1.0e-2, device=ref["x"].device, dtype=ref["x"].dtype))
    return scales


def train(args):
    seed_all(args.seed)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    run_dir = Path("runs") / args.run_name
    fig_dir = Path("figures")
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ref = load_ref(args.ref, device, dtype)
    v, v2, w = build_vgrid(args.nvx, args.nvy, args.nvz, args.vmax, device, dtype)

    idx_all, idx_shock = make_pools(ref, device)
    scales = build_scales(ref)

    model = StandingPositiveHMM(
        ref,
        psi_max=args.psi_max,
        macro_amp=args.macro_amp,
        q_amp=args.q_amp,
        sig_amp=args.sig_amp,
        h3_amp=args.h3_amp,
    ).to(device=device, dtype=dtype)

    if args.init and Path(args.init).exists():
        print(f"[init] loading {args.init}")
        ck = torch.load(args.init, map_location=device)
        sd = ck.get("model", ck)
        model.load_state_dict(sd, strict=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config["device_used"] = str(device)
    config["kn_eff"] = ref["kn_eff"]
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    best_loss = 1.0e99

    print("[train] starting")
    print(f"[train] run_dir={run_dir}")

    for step in range(1, args.steps+1):
        opt.zero_grad(set_to_none=True)

        # Warmup first: moment/flux constraints dominate.
        if step <= args.warmup:
            wpde = 0.0
            wpder = 0.0
            wmacro = args.w_macro * 1.5
            wqsig = args.w_qsig * 1.5
            wflux = args.w_flux
        else:
            ramp = min(1.0, (step - args.warmup)/max(1, args.pde_ramp))
            wpde = args.w_pde * ramp
            wpder = args.w_pde_rel * ramp
            wmacro = args.w_macro
            wqsig = args.w_qsig
            wflux = args.w_flux

        lmacro, macro_parts = macro_anchor_loss(
            model, ref, idx_all, args.batch_macro, v, v2, w, scales
        )

        lqsig, qsig_parts = qsig_anchor_loss(
            model, ref, idx_shock, args.batch_qsig, v, v2, w, scales
        )

        lflux, flux_parts = flux_loss(
            model, ref, args.batch_flux, v, v2, w, scales, device, dtype
        )

        lbc = boundary_loss(model, ref, v, w)

        if wpde > 0:
            lpde, lpder = pde_loss(
                model, ref, args.batch_pde, v, v2, w, ref["kn_eff"], device, dtype
            )
        else:
            lpde = torch.tensor(0.0, device=device, dtype=dtype)
            lpder = torch.tensor(0.0, device=device, dtype=dtype)

        loss = (
            wpde*lpde
            + wpder*lpder
            + wmacro*lmacro
            + wqsig*lqsig
            + wflux*lflux
            + args.w_bc*lbc
        )

        if not torch.isfinite(loss):
            print("[error] non-finite loss")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            print(
                f"[log] step={step:06d} "
                f"total={float(loss):.6e} "
                f"pde={float(lpde):.3e} pde_rel={float(lpder):.3e} "
                f"macro={float(lmacro):.3e} qsig={float(lqsig):.3e} "
                f"flux={float(lflux):.3e} bc={float(lbc):.3e} "
                f"rho={float(macro_parts[0]):.2e} ux={float(macro_parts[1]):.2e} T={float(macro_parts[2]):.2e} "
                f"q={float(qsig_parts[0]):.2e} sig={float(qsig_parts[1]):.2e}",
                flush=True
            )

        if step % args.save_every == 0 or step == args.steps:
            ck = {
                "model": model.state_dict(),
                "step": step,
                "config": config,
                "ref_meta": {
                    "x0": ref["x0"],
                    "width": ref["width"],
                    "states": ref["states"],
                    "kn_eff": ref["kn_eff"],
                },
            }
            torch.save(ck, run_dir / "ckpt_last.pt")
            if float(loss) < best_loss:
                best_loss = float(loss)
                torch.save(ck, run_dir / "ckpt_best.pt")
            print(f"[save] step={step}, best={best_loss:.6e}")

    torch.save({"model": model.state_dict(), "config": config}, run_dir / "ckpt_last.pt")
    print(f"[done] saved {run_dir/'ckpt_last.pt'}")


def predict_profiles(model, x, v, v2, w, chunk=64):
    outs = []
    for i in range(0, x.numel(), chunk):
        xb = x[i:i+chunk]
        f = model.f_grid(xb, v)
        mom = integrate_moments(f, v, v2, w)[:5]
        outs.append([m.detach().cpu().numpy() for m in mom])

    arrs = []
    for j in range(5):
        arrs.append(np.concatenate([o[j] for o in outs]))
    return arrs


def plot(args):
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck.get("config", {})
    ref_path = args.ref if args.ref else cfg.get("ref", "ref/standing_M2_x40_hmom_dvm.npz")

    ref = load_ref(ref_path, device, dtype)

    nvx = int(cfg.get("nvx", args.nvx))
    nvy = int(cfg.get("nvy", args.nvy))
    nvz = int(cfg.get("nvz", args.nvz))
    vmax = float(cfg.get("vmax", args.vmax))

    v, v2, w = build_vgrid(nvx, nvy, nvz, vmax, device, dtype)

    model = StandingPositiveHMM(
        ref,
        psi_max=float(cfg.get("psi_max", args.psi_max)),
        macro_amp=float(cfg.get("macro_amp", args.macro_amp)),
        q_amp=float(cfg.get("q_amp", args.q_amp)),
        sig_amp=float(cfg.get("sig_amp", args.sig_amp)),
        h3_amp=float(cfg.get("h3_amp", args.h3_amp)),
    ).to(device=device, dtype=dtype)

    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    x = ref["x"]
    with torch.no_grad():
        rho, ux, T, qx, sig = predict_profiles(model, x, v, v2, w, chunk=args.plot_chunk)

    xr = x.detach().cpu().numpy()
    ref_np = {k: ref[k].detach().cpu().numpy() for k in ["rho", "ux", "T", "qx", "sig"]}

    keys = [
        ("rho", rho, ref_np["rho"], r"$\rho$"),
        ("ux", ux, ref_np["ux"], r"$u_x$"),
        ("T", T, ref_np["T"], r"$T$"),
        ("qx", qx, ref_np["qx"], r"$q_x$"),
        ("sig", sig, ref_np["sig"], r"$\sigma_{xx}$"),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(9, 10.5), sharex=True)
    errors = {}
    for ax, (name, pred, truth, lab) in zip(axes, keys):
        rel = np.linalg.norm(pred-truth)/(np.linalg.norm(truth)+1e-14)
        errors[name] = rel
        ax.plot(xr, pred, lw=2.2, label="PINN-v7")
        ax.plot(xr, truth, "--", lw=2.0, label="DVM")
        ax.text(0.02, 0.80, f"relL2={rel:.3e}", transform=ax.transAxes)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel(r"scaled coordinate $x/(2L_\lambda)$")
    fig.suptitle("Standing BGK normal shock: positive flux-locked moment-anchored PINN v7")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print("[plot]", out)

    csv = out.with_suffix(".csv")
    with open(csv, "w") as f:
        f.write("variable,relL2\n")
        for k, v_ in errors.items():
            f.write(f"{k},{v_:.8e}\n")
    print("[csv]", csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "plot"], default="train")

    ap.add_argument("--ref", default="ref/standing_M2_x40_hmom_dvm.npz")
    ap.add_argument("--run-name", default="standing_M2_x40_v7")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="figures/standing_M2_x40_v7_profiles.png")
    ap.add_argument("--init", default="")

    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--nvx", type=int, default=44)
    ap.add_argument("--nvy", type=int, default=16)
    ap.add_argument("--nvz", type=int, default=16)
    ap.add_argument("--vmax", type=float, default=12.0)

    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=6000)
    ap.add_argument("--pde-ramp", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=4.0e-5)
    ap.add_argument("--weight-decay", type=float, default=1.0e-8)
    ap.add_argument("--grad-clip", type=float, default=0.15)

    ap.add_argument("--batch-pde", type=int, default=5)
    ap.add_argument("--batch-macro", type=int, default=96)
    ap.add_argument("--batch-qsig", type=int, default=96)
    ap.add_argument("--batch-flux", type=int, default=96)

    ap.add_argument("--psi-max", type=float, default=2.5)
    ap.add_argument("--macro-amp", type=float, default=0.25)
    ap.add_argument("--q-amp", type=float, default=1.6)
    ap.add_argument("--sig-amp", type=float, default=1.6)
    ap.add_argument("--h3-amp", type=float, default=0.4)

    ap.add_argument("--w-pde", type=float, default=0.25)
    ap.add_argument("--w-pde-rel", type=float, default=0.05)
    ap.add_argument("--w-macro", type=float, default=40.0)
    ap.add_argument("--w-qsig", type=float, default=18.0)
    ap.add_argument("--w-flux", type=float, default=12.0)
    ap.add_argument("--w-bc", type=float, default=3.0)

    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--plot-chunk", type=int, default=64)

    args = ap.parse_args()

    if args.mode == "train":
        train(args)
    else:
        if not args.ckpt:
            args.ckpt = f"runs/{args.run_name}/ckpt_last.pt"
        plot(args)


if __name__ == "__main__":
    main()
