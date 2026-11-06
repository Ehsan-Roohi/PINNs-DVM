#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_bgk_standing_v7 import (
    seed_all,
    build_vgrid,
    load_ref,
    build_scales,
    integrate_moments,
    maxwellian_grid,
    macro_anchor_loss,
    qsig_anchor_loss,
    flux_loss,
    boundary_loss,
    pde_loss,
    sample_from_pool,
    MLP,
)


def rel_l2(a, b):
    return np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-14)


def load_ref_high(path, device, dtype):
    ref = load_ref(path, device, dtype)
    z = np.load(path)

    if "mxxx" not in z.files:
        raise RuntimeError("DVM ref does not contain mxxx. Use standing_M2_x40_hmom_dvm*.npz")

    ref["mxxx"] = torch.tensor(z["mxxx"], device=device, dtype=dtype)

    if "Rxx_closure" in z.files:
        R = z["Rxx_closure"]
    elif "Rxx" in z.files:
        R = z["Rxx"]
    else:
        raise RuntimeError("DVM ref does not contain Rxx_closure/Rxx")

    ref["Rxx_closure"] = torch.tensor(R, device=device, dtype=dtype)
    return ref


def make_high_pool(ref):
    m = torch.abs(ref["mxxx"])
    r = torch.abs(ref["Rxx_closure"])
    s = m/(torch.max(m)+1e-30) + r/(torch.max(r)+1e-30)
    idx = torch.arange(ref["x"].numel(), device=ref["x"].device)
    pool = idx[s > 0.04]
    if pool.numel() < 32:
        pool = idx
    return pool


class StandingPositiveHMMHigh(nn.Module):
    def __init__(
        self,
        ref,
        psi_max=2.5,
        macro_amp=0.25,
        q_amp=1.6,
        sig_amp=1.6,
        h3_amp=0.9,
        h4_amp=0.9,
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

        # drho, du, dT, aq, asig, a3, a4
        self.net = MLP(5, 7, width=96, depth=5)

    def base_switch(self, x):
        return 0.5*(1.0 + torch.tanh((x-self.x0)/self.width))

    def gates(self, x):
        gd = ((x-self.xmin)*(self.xmax-x))/0.25
        gd = torch.clamp(gd, min=0.0, max=1.0)
        gs = torch.exp(-((x-self.x0)/(3.0*self.width))**2)
        return gd, gs

    def forward_params(self, x):
        S = self.base_switch(x)
        gd, gs = self.gates(x)

        log_rho_base = (1-S)*math.log(self.rho1) + S*math.log(self.rho2)
        ux_base = (1-S)*self.u1 + S*self.u2
        log_T_base = (1-S)*math.log(self.T1) + S*math.log(self.T2)

        xin = torch.stack([
            x,
            x-self.x0,
            torch.tanh((x-self.x0)/self.width),
            S,
            gd*gs,
        ], dim=1)

        y = self.net(xin)
        drho, du, dT, aq, asig, a3, a4 = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6]

        gmacro = gd*gs
        rho = torch.exp(log_rho_base + self.macro_amp*gmacro*torch.tanh(drho))
        T = torch.exp(log_T_base + self.macro_amp*gmacro*torch.tanh(dT))

        du_scale = abs(self.u1-self.u2) + 1e-8
        ux = ux_base + 0.30*du_scale*gmacro*torch.tanh(du)

        aq = self.q_amp*gs*torch.tanh(aq)
        asig = self.sig_amp*gs*torch.tanh(asig)
        a3 = self.h3_amp*gs*torch.tanh(a3)
        a4 = self.h4_amp*gs*torch.tanh(a4)

        return rho, ux, T, aq, asig, a3, a4

    def f_flat(self, x, v):
        rho, ux, T, aq, asig, a3, a4 = self.forward_params(x)

        sqrtT = torch.sqrt(T)
        cx = (v[:,0]-ux)/sqrtT
        cy = v[:,1]/sqrtT
        cz = v[:,2]/sqrtT
        c2 = cx**2 + cy**2 + cz**2

        Hq = cx*(c2-5.0)
        Hs = cx**2 - c2/3.0
        H3 = cx**3 - (3.0/5.0)*cx*c2
        H4 = (c2 - 7.0) * (cx**2 - c2/3.0)

        raw = (
            0.08*aq*Hq
            + 0.18*asig*Hs
            + 0.055*a3*H3
            + 0.018*a4*H4
        )

        psi = self.psi_max*torch.tanh(raw/self.psi_max)

        c2phys = (v[:,0]-ux)**2 + v[:,1]**2 + v[:,2]**2
        M = rho/torch.clamp((2.0*math.pi*T)**1.5, min=1e-30)
        M = M*torch.exp(-c2phys/torch.clamp(2.0*T, min=1e-30))

        f = M*torch.exp(psi)
        f = torch.clamp(f, min=1e-30, max=1e6)
        return f, M, rho, ux, T, aq, asig, a3, a4

    def f_grid(self, x, v):
        B = x.numel()
        Nv = v.shape[0]
        xx = x[:,None].expand(B, Nv).reshape(-1)
        vv = v[None,:,:].expand(B, Nv, 3).reshape(-1, 3)
        f, *_ = self.f_flat(xx, vv)
        return f.reshape(B, Nv)


def copy_v7_to_v8(model, init_path, device):
    if not init_path or not Path(init_path).exists():
        print("[init] no v7 checkpoint used")
        return

    print(f"[init] loading v7 checkpoint into v8: {init_path}")
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
            # final layer: copy old output rows, zero the new high-moment row
            new[k].zero_()
            n = min(new[k].shape[0], v.shape[0])
            new[k][:n].copy_(v[:n])
            partial += 1
        elif new[k].ndim == 1 and v.ndim == 1:
            new[k].zero_()
            n = min(new[k].shape[0], v.shape[0])
            new[k][:n].copy_(v[:n])
            partial += 1

    model.load_state_dict(new)
    print(f"[init] copied={copied}, partial={partial}")


def compute_high(model, x, v, v2, w):
    f = model.f_grid(x, v)
    rho, ux, T, qx, sig, *_ = integrate_moments(f, v, v2, w)
    M = maxwellian_grid(v, rho, ux, T)
    G = f - M

    Cx = v[None,:,0] - ux[:,None]
    Cy = v[None,:,1]
    Cz = v[None,:,2]
    C2 = Cx**2 + Cy**2 + Cz**2

    mxxx = torch.sum(G*w[None,:]*(Cx**3 - (3.0/5.0)*Cx*C2), dim=1)
    Rxx_neq = torch.sum(G*w[None,:]*C2*(Cx**2 - C2/3.0), dim=1)
    Rxx_closure = Rxx_neq - 7.0*T*sig

    return mxxx, Rxx_closure


def high_loss(model, ref, pool, batch, v, v2, w, sm, sR):
    idx = sample_from_pool(pool, batch)
    x = ref["x"][idx]

    m, R = compute_high(model, x, v, v2, w)
    lm = torch.mean(((m-ref["mxxx"][idx])/sm)**2)
    lR = torch.mean(((R-ref["Rxx_closure"][idx])/sR)**2)
    return lm + lR, lm.detach(), lR.detach()


def train(args):
    seed_all(args.seed)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    ref = load_ref_high(args.ref, device, dtype)
    v, v2, w = build_vgrid(args.nvx, args.nvy, args.nvz, args.vmax, device, dtype)

    idx_all = torch.arange(ref["x"].numel(), device=device)
    qsig_pool = make_high_pool(ref)
    high_pool = make_high_pool(ref)

    scales = build_scales(ref)

    sm = torch.clamp(torch.max(torch.abs(ref["mxxx"])), min=torch.tensor(1e-3, device=device, dtype=dtype))
    sR = torch.clamp(torch.max(torch.abs(ref["Rxx_closure"])), min=torch.tensor(1e-3, device=device, dtype=dtype))

    model = StandingPositiveHMMHigh(
        ref,
        psi_max=args.psi_max,
        macro_amp=args.macro_amp,
        q_amp=args.q_amp,
        sig_amp=args.sig_amp,
        h3_amp=args.h3_amp,
        h4_amp=args.h4_amp,
    ).to(device=device, dtype=dtype)

    copy_v7_to_v8(model, args.init, device)

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = vars(args).copy()
    cfg["ref_used"] = args.ref
    cfg["device_used"] = str(device)
    (run_dir/"config.json").write_text(json.dumps(cfg, indent=2))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = 1e99
    print(f"[train] run={args.run_name}")

    for step in range(1, args.steps+1):
        opt.zero_grad(set_to_none=True)

        lmacro, mp = macro_anchor_loss(model, ref, idx_all, args.batch_macro, v, v2, w, scales)
        lqsig, qp = qsig_anchor_loss(model, ref, qsig_pool, args.batch_qsig, v, v2, w, scales)
        lhigh, lm, lR = high_loss(model, ref, high_pool, args.batch_high, v, v2, w, sm, sR)
        lflux, fp = flux_loss(model, ref, args.batch_flux, v, v2, w, scales, device, dtype)
        lbc = boundary_loss(model, ref, v, w)

        if args.w_pde > 0:
            lpde, lpder = pde_loss(model, ref, args.batch_pde, v, v2, w, ref["kn_eff"], device, dtype)
        else:
            lpde = torch.tensor(0.0, device=device, dtype=dtype)
            lpder = torch.tensor(0.0, device=device, dtype=dtype)

        loss = (
            args.w_macro*lmacro
            + args.w_qsig*lqsig
            + args.w_high*lhigh
            + args.w_flux*lflux
            + args.w_bc*lbc
            + args.w_pde*lpde
            + args.w_pde_rel*lpder
        )

        if not torch.isfinite(loss):
            print("[error] non-finite loss")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            print(
                f"[log] step={step:06d} total={float(loss):.6e} "
                f"macro={float(lmacro):.3e} qsig={float(lqsig):.3e} "
                f"high={float(lhigh):.3e} m={float(lm):.3e} R={float(lR):.3e} "
                f"flux={float(lflux):.3e} pde={float(lpde):.3e} pder={float(lpder):.3e} "
                f"rho={float(mp[0]):.2e} ux={float(mp[1]):.2e} T={float(mp[2]):.2e} "
                f"q={float(qp[0]):.2e} sig={float(qp[1]):.2e}",
                flush=True
            )

        if step % args.save_every == 0 or step == args.steps:
            ck = {"model": model.state_dict(), "step": step, "config": cfg}
            torch.save(ck, run_dir/"ckpt_last.pt")
            if float(loss) < best:
                best = float(loss)
                torch.save(ck, run_dir/"ckpt_best.pt")
            print(f"[save] step={step}, best={best:.6e}", flush=True)


def predict_all(model, ref, v, v2, w, chunk=64):
    out = {k: [] for k in ["rho","ux","T","qx","sigma_xx","mxxx","Rxx_closure","Qxx_bgk","Qx_bgk"]}

    with torch.no_grad():
        x = ref["x"]
        for i in range(0, x.numel(), chunk):
            xb = x[i:i+chunk]
            f = model.f_grid(xb, v)
            rho, ux, T, qx, sig, *_ = integrate_moments(f, v, v2, w)
            M = maxwellian_grid(v, rho, ux, T)
            G = f - M

            Cx = v[None,:,0] - ux[:,None]
            Cy = v[None,:,1]
            Cz = v[None,:,2]
            C2 = Cx**2 + Cy**2 + Cz**2

            mxxx = torch.sum(G*w[None,:]*(Cx**3 - (3.0/5.0)*Cx*C2), dim=1)
            Rxx_neq = torch.sum(G*w[None,:]*C2*(Cx**2 - C2/3.0), dim=1)
            Rxx_closure = Rxx_neq - 7.0*T*sig

            vals = {
                "rho": rho,
                "ux": ux,
                "T": T,
                "qx": qx,
                "sigma_xx": sig,
                "mxxx": mxxx,
                "Rxx_closure": Rxx_closure,
                "Qxx_bgk": -sig,
                "Qx_bgk": -2.0*qx,
            }

            for k, v_ in vals.items():
                out[k].append(v_.detach().cpu().numpy())

    return {k: np.concatenate(v) for k, v in out.items()}


def plot(args):
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck.get("config", {})

    ref_path = args.ref if args.ref else cfg["ref_used"]
    ref = load_ref_high(ref_path, device, dtype)

    v, v2, w = build_vgrid(
        int(cfg.get("nvx", args.nvx)),
        int(cfg.get("nvy", args.nvy)),
        int(cfg.get("nvz", args.nvz)),
        float(cfg.get("vmax", args.vmax)),
        device,
        dtype
    )

    model = StandingPositiveHMMHigh(
        ref,
        psi_max=float(cfg.get("psi_max", args.psi_max)),
        macro_amp=float(cfg.get("macro_amp", args.macro_amp)),
        q_amp=float(cfg.get("q_amp", args.q_amp)),
        sig_amp=float(cfg.get("sig_amp", args.sig_amp)),
        h3_amp=float(cfg.get("h3_amp", args.h3_amp)),
        h4_amp=float(cfg.get("h4_amp", args.h4_amp)),
    ).to(device=device, dtype=dtype)

    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    pred = predict_all(model, ref, v, v2, w, chunk=args.plot_chunk)

    z = np.load(ref_path)
    x = z["x_mfp"] if "x_mfp" in z.files else z["x"]

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
    dvm["Qx_bgk"] = -2.0*dvm["qx"]

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

    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5), sharex=True)
    errors = {}

    for ax, (k, lab) in zip(axes.ravel(), items):
        P = pred[k]
        R = dvm[k]
        e = rel_l2(P, R)
        errors[k] = e
        ax.plot(x, P, lw=2.2, label="PINN-v8")
        ax.plot(x, R, "--", lw=2.0, label="DVM")
        ax.set_title(f"{lab}, relL2={e:.2e}")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(r"$x/\lambda_1$")
        ax.legend(fontsize=8)

    fig.suptitle("Standing BGK normal shock: PINN-v8 high-moment anchored vs DVM")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print("[plot]", out)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    for ax, (k, lab) in zip(axes.ravel(), items[5:]):
        P = pred[k]
        R = dvm[k]
        e = rel_l2(P, R)
        ax.plot(x, P, lw=2.4, label="PINN-v8")
        ax.plot(x, R, "--", lw=2.1, label="DVM")
        ax.set_title(f"{lab}, relL2={e:.2e}")
        ax.set_xlabel(r"$x/\lambda_1$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Closure-level moments and BGK production terms: PINN-v8 vs DVM")
    fig.tight_layout()
    out2 = out.with_name(out.stem + "_closure.png")
    fig.savefig(out2, dpi=300)
    print("[plot]", out2)

    csv = out.with_suffix(".csv")
    with open(csv, "w") as f:
        f.write("quantity,relL2\n")
        for k,e in errors.items():
            f.write(f"{k},{e:.8e}\n")
    print("[csv]", csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","plot"], default="train")
    ap.add_argument("--ref", default="ref/standing_M2_x40_hmom_dvm.npz")
    ap.add_argument("--run-name", default="standing_M2_x40_v8_highmom")
    ap.add_argument("--init", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="figures/standing_M2_x40_v8_highmom.png")

    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=4321)

    ap.add_argument("--nvx", type=int, default=44)
    ap.add_argument("--nvy", type=int, default=16)
    ap.add_argument("--nvz", type=int, default=16)
    ap.add_argument("--vmax", type=float, default=12.0)

    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-8)
    ap.add_argument("--grad-clip", type=float, default=0.10)

    ap.add_argument("--batch-macro", type=int, default=128)
    ap.add_argument("--batch-qsig", type=int, default=128)
    ap.add_argument("--batch-high", type=int, default=128)
    ap.add_argument("--batch-flux", type=int, default=128)
    ap.add_argument("--batch-pde", type=int, default=4)

    ap.add_argument("--psi-max", type=float, default=2.5)
    ap.add_argument("--macro-amp", type=float, default=0.25)
    ap.add_argument("--q-amp", type=float, default=1.6)
    ap.add_argument("--sig-amp", type=float, default=1.6)
    ap.add_argument("--h3-amp", type=float, default=0.9)
    ap.add_argument("--h4-amp", type=float, default=0.9)

    ap.add_argument("--w-macro", type=float, default=35.0)
    ap.add_argument("--w-qsig", type=float, default=18.0)
    ap.add_argument("--w-high", type=float, default=8.0)
    ap.add_argument("--w-flux", type=float, default=8.0)
    ap.add_argument("--w-bc", type=float, default=2.0)
    ap.add_argument("--w-pde", type=float, default=0.02)
    ap.add_argument("--w-pde-rel", type=float, default=0.01)

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
