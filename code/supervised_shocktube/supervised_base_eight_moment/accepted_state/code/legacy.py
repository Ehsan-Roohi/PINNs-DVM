#!/usr/bin/env python
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# -----------------------------
# utilities
# -----------------------------
def load_json(path):
    return json.loads(Path(path).read_text())

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))

def get_device(dev):
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)

def smooth_step(x, k):
    return 0.5 * (1.0 + torch.tanh(k * x))

def case_state_torch(cfg, x):
    c = cfg["case"]
    k = float(c.get("smooth_k", 40.0))
    H = smooth_step(x, k)
    L, R = c["left"], c["right"]
    rho = float(L["rho"]) * (1-H) + float(R["rho"]) * H
    ux  = float(L.get("ux",0.0)) * (1-H) + float(R.get("ux",0.0)) * H
    T   = float(L["T"]) * (1-H) + float(R["T"]) * H
    return rho, ux, T

def case_state_np(cfg, x):
    k = float(cfg["case"].get("smooth_k", 40.0))
    H = 0.5 * (1.0 + np.tanh(k * x))
    L, R = cfg["case"]["left"], cfg["case"]["right"]
    rho = float(L["rho"]) * (1-H) + float(R["rho"]) * H
    ux  = float(L.get("ux",0.0)) * (1-H) + float(R.get("ux",0.0)) * H
    T   = float(L["T"]) * (1-H) + float(R["T"]) * H
    return rho, ux, T

def build_vgrid(cfg, device, dtype):
    vcfg = cfg["velocity"]
    vmax = float(vcfg["vmax"])
    nvx, nvy, nvz = int(vcfg["nvx"]), int(vcfg["nvy"]), int(vcfg["nvz"])
    vx = torch.linspace(-vmax, vmax, nvx, device=device, dtype=dtype)
    vy = torch.linspace(-vmax, vmax, nvy, device=device, dtype=dtype)
    vz = torch.linspace(-vmax, vmax, nvz, device=device, dtype=dtype)
    dvx = (2*vmax)/(nvx-1)
    dvy = (2*vmax)/(nvy-1)
    dvz = (2*vmax)/(nvz-1)
    X,Y,Z = torch.meshgrid(vx,vy,vz, indexing="ij")
    v = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)
    qw = torch.full((v.shape[0],), dvx*dvy*dvz, device=device, dtype=dtype)
    v2 = torch.sum(v*v, dim=1)
    alpha = float(vcfg.get("weight_alpha", 0.1))
    beta  = float(vcfg.get("weight_beta", 4.0))
    vw = 1.0 + alpha * torch.pow(v2 + 1e-30, 0.5*beta)
    if bool(vcfg.get("normalize_weight", True)):
        vw = vw / torch.mean(vw)
    print(f"[quad] Nv={v.shape[0]} = {nvx}x{nvy}x{nvz}, vmax={vmax}")
    return v, qw, vw

def maxwellian(rho, ux, T, v):
    vx, vy, vz = v[:,0], v[:,1], v[:,2]
    T = torch.clamp(T, min=1e-6)
    rho = torch.clamp(rho, min=1e-9)
    c2 = (vx-ux)**2 + vy**2 + vz**2
    return rho / torch.pow(2.0*math.pi*T, 1.5) * torch.exp(-0.5*c2/T)

def find_key(d, candidates):
    keys = list(d.keys())
    lower = {k.lower(): k for k in keys}
    for c in candidates:
        if c in d:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def load_npz_ref(path):
    d = np.load(path)
    out = {}
    kx = find_key(d, ["x","xs","xgrid","x_grid"])
    kr = find_key(d, ["rho","density","n","RHO"])
    ku = find_key(d, ["ux","u","U","u_x"])
    kT = find_key(d, ["T","temp","temperature"])
    kq = find_key(d, ["qx","q_x","q","heat_flux","heatflux"])
    ks = find_key(d, ["sigma_xx","sig_xx","sxx","sigma","stress"])
    if kr is None or ku is None or kT is None:
        raise RuntimeError(f"Could not find rho/ux/T keys in {path}. keys={list(d.keys())}")
    n = len(d[kr])
    out["x"] = d[kx] if kx is not None else np.linspace(-0.5,0.5,n)
    out["rho"] = d[kr]
    out["ux"] = d[ku]
    out["T"] = d[kT]
    out["qx"] = d[kq] if kq is not None else np.zeros_like(out["x"])
    out["sigma_xx"] = d[ks] if ks is not None else np.zeros_like(out["x"])
    return out

def make_dvm_anchor(cfg):
    ref_path = Path(cfg["anchor"].get("dvm_ref", "ref/dvm_smoke_fixed.npz"))
    out_path = Path(cfg["anchor"].get("path", "anchors/dvm_sparse_v6d_kn1_t0p02.npz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref = load_npz_ref(ref_path)
    x = np.asarray(ref["x"])
    xmin = float(cfg["anchor"].get("x_min", -0.10))
    xmax = float(cfg["anchor"].get("x_max",  0.14))
    npts = int(cfg["anchor"].get("n_points", 36))
    mask = (x >= xmin) & (x <= xmax)
    idx = np.where(mask)[0]
    if len(idx) > npts:
        pick = np.linspace(0, len(idx)-1, npts).round().astype(int)
        idx = idx[pick]
    floor_q = float(cfg["anchor"].get("q_std_floor", 0.003))
    floor_s = float(cfg["anchor"].get("sig_std_floor", 0.003))
    np.savez(
        out_path,
        anchor_x=x[idx],
        anchor_qx=np.asarray(ref["qx"])[idx],
        anchor_sigma_xx=np.asarray(ref["sigma_xx"])[idx],
        anchor_sig_xx=np.asarray(ref["sigma_xx"])[idx],
        anchor_qx_std=np.full(len(idx), floor_q),
        anchor_sigma_xx_std=np.full(len(idx), floor_s),
        anchor_sig_xx_std=np.full(len(idx), floor_s),
    )
    print(f"[v6d-anchor] wrote {out_path}; n={len(idx)}, x=[{x[idx].min():+.4f},{x[idx].max():+.4f}]")
    return str(out_path)

def load_anchor(path, device, dtype):
    d = np.load(path)
    x = d["anchor_x"].astype(np.float64)
    q = d["anchor_qx"].astype(np.float64)
    if "anchor_sigma_xx" in d:
        s = d["anchor_sigma_xx"].astype(np.float64)
    else:
        s = d["anchor_sig_xx"].astype(np.float64)
    qstd = d["anchor_qx_std"].astype(np.float64) if "anchor_qx_std" in d else np.full_like(q, 0.003)
    if "anchor_sigma_xx_std" in d:
        sstd = d["anchor_sigma_xx_std"].astype(np.float64)
    elif "anchor_sig_xx_std" in d:
        sstd = d["anchor_sig_xx_std"].astype(np.float64)
    else:
        sstd = np.full_like(s, 0.003)
    out = {
        "x": torch.tensor(x, device=device, dtype=dtype),
        "q": torch.tensor(q, device=device, dtype=dtype),
        "s": torch.tensor(s, device=device, dtype=dtype),
        "qstd": torch.tensor(qstd, device=device, dtype=dtype),
        "sstd": torch.tensor(sstd, device=device, dtype=dtype),
    }
    print(f"[anchor] loaded {path}; n={len(x)}, max|q|={np.max(np.abs(q)):.4e}, max|sig|={np.max(np.abs(s)):.4e}")
    return out

# -----------------------------
# model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, layers=4):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        mods += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*mods)
    def forward(self, x):
        return self.net(x)

class PositiveHMM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg["model"]
        hidden = int(mcfg.get("hidden", 128))
        layers = int(mcfg.get("layers", 4))
        self.net = MLP(8, 5, hidden, layers)
        self.psi_max = float(mcfg.get("psi_max", 0.65))
        self.gate_width = float(mcfg.get("gate_width", 0.13))
        self.q_amp = float(mcfg.get("q_amp", 0.22))
        self.s_amp = float(mcfg.get("sig_amp", 0.22))

    def features(self, tx):
        t = tx[:,0:1]
        x = tx[:,1:2]
        tf = float(self.cfg["case"]["t_final"])
        tau = t / tf
        return torch.cat([
            tau, x,
            torch.sin(2*math.pi*x), torch.cos(2*math.pi*x),
            torch.sin(4*math.pi*x), torch.cos(4*math.pi*x),
            torch.sin(math.pi*tau), torch.cos(math.pi*tau),
        ], dim=1)

    def forward_moments(self, tx):
        t = tx[:,0]
        x = tx[:,1]
        tf = float(self.cfg["case"]["t_final"])
        tau = torch.clamp(t/tf, 0.0, 1.5)
        raw = self.net(self.features(tx))
        rho0, ux0, T0 = case_state_torch(self.cfg, x)
        # hard initial condition for macros
        rho = rho0 * torch.exp(0.45*tau*torch.tanh(raw[:,0]))
        ux  = ux0  + 0.42*tau*torch.tanh(raw[:,1])
        T   = T0   * torch.exp(0.35*tau*torch.tanh(raw[:,2]))
        # shock-local micro gate
        w = self.gate_width
        gate = torch.exp(-((x)/w)**4)
        qcoef = tau * gate * self.q_amp * torch.tanh(raw[:,3])
        scoef = tau * gate * self.s_amp * torch.tanh(raw[:,4])
        return rho, ux, T, qcoef, scoef, gate

    def f_flat(self, tx, v):
        rho, ux, T, qcoef, scoef, gate = self.forward_moments(tx)
        vx, vy, vz = v[:,0], v[:,1], v[:,2]
        T = torch.clamp(T, min=1e-6)
        sqrtT = torch.sqrt(T)
        cx = (vx - ux) / sqrtT
        cy = vy / sqrtT
        cz = vz / sqrtT
        c2 = cx*cx + cy*cy + cz*cz
        # normalized-ish Hermite shapes to avoid exponential blow-up
        phi_q = ((0.5*c2 - 2.5) * cx) / 8.0
        phi_s = (cx*cx - c2/3.0) / 4.0
        h = qcoef * phi_q + scoef * phi_s
        psi = self.psi_max * torch.tanh(h / self.psi_max)
        M = maxwellian(rho, ux, T, v)
        f = M * torch.exp(psi)
        return f, M, torch.exp(psi), rho, ux, T, qcoef, scoef

# -----------------------------
# moment integration
# -----------------------------
def integrated_profiles(model, cfg, x, vgrid, qweight, device, dtype):
    tf = float(cfg["case"]["t_final"])
    n = x.numel()
    Nv = vgrid.shape[0]
    tx = torch.stack([torch.full_like(x, tf), x], dim=1)
    tx_flat = tx[:,None,:].expand(n,Nv,2).reshape(n*Nv,2)
    v_flat = vgrid[None,:,:].expand(n,Nv,3).reshape(n*Nv,3)
    f, M, fac, *_ = model.f_flat(tx_flat, v_flat)
    f = f.reshape(n,Nv)
    vx = vgrid[:,0][None,:]
    vy = vgrid[:,1][None,:]
    vz = vgrid[:,2][None,:]
    w = qweight[None,:]
    rho = torch.sum(f*w, dim=1)
    ux = torch.sum(f*vx*w, dim=1) / torch.clamp(rho, min=1e-12)
    cx = vx - ux[:,None]
    c2 = cx*cx + vy*vy + vz*vz
    T = torch.sum(f*c2*w, dim=1) / torch.clamp(3*rho, min=1e-12)
    qx = 0.5 * torch.sum(f*c2*cx*w, dim=1)
    sig = torch.sum(f*(cx*cx - c2/3.0)*w, dim=1)
    return rho, ux, T, qx, sig

# -----------------------------
# sampling and losses
# -----------------------------
def sample_tx(cfg, n, device, dtype):
    c = cfg["case"]
    xmin, xmax = float(c["x_min"]), float(c["x_max"])
    tf = float(c["t_final"])
    tr = cfg["train"]
    frac = float(tr.get("shock_fraction", 0.55))
    ns = int(n*frac)
    nu = n - ns
    xu = xmin + (xmax-xmin)*torch.rand(nu, device=device, dtype=dtype)
    xs = torch.randn(ns, device=device, dtype=dtype) * float(tr.get("shock_sigma", 0.09))
    xs = torch.clamp(xs, xmin, xmax)
    x = torch.cat([xu,xs], dim=0)
    t = tf * torch.rand(n, device=device, dtype=dtype)
    return torch.stack([t,x], dim=1)

def pde_loss(model, cfg, vgrid, qweight, vweight):
    device, dtype = vgrid.device, vgrid.dtype
    B = int(cfg["train"].get("batch_pde", 8))
    Nv = vgrid.shape[0]
    tx = sample_tx(cfg, B, device, dtype)
    tx_flat = tx[:,None,:].expand(B,Nv,2).reshape(B*Nv,2).detach().clone().requires_grad_(True)
    v_flat = vgrid[None,:,:].expand(B,Nv,3).reshape(B*Nv,3)
    f, M, fac, *_ = model.f_flat(tx_flat, v_flat)
    g = torch.autograd.grad(f.sum(), tx_flat, create_graph=True)[0]
    dfdt = g[:,0]
    dfdx = g[:,1]
    kn = float(cfg["case"]["kn"])
    res = dfdt + v_flat[:,0]*dfdx - (M - f)/kn
    qw = qweight.repeat(B)
    vw = vweight.repeat(B)
    loss = torch.sum(qw*vw*res*res) / torch.sum(qw*vw)
    scale = torch.abs(M) + torch.abs(f) + float(cfg["train"].get("rel_eps", 1e-5))
    loss_rel = torch.sum(qw*vw*(res/scale)**2) / torch.sum(qw*vw)
    loss_pos = torch.mean(torch.relu(float(cfg["train"].get("fac_floor", 0.03)) - fac)**2)
    return loss, loss_rel, loss_pos

def bc_loss(model, cfg, vgrid, qweight):
    device, dtype = vgrid.device, vgrid.dtype
    c = cfg["case"]
    tf = float(c["t_final"])
    xmin, xmax = float(c["x_min"]), float(c["x_max"])
    nb = int(cfg["train"].get("batch_bc", 4))
    times = tf * torch.rand(nb, device=device, dtype=dtype)
    losses = []
    for side, xval, mask, state in [
        ("L", xmin, vgrid[:,0] > 0, c["left"]),
        ("R", xmax, vgrid[:,0] < 0, c["right"]),
    ]:
        if mask.sum() == 0:
            continue
        vv = vgrid[mask]
        ww = qweight[mask]
        Nv = vv.shape[0]
        x = torch.full((nb,), xval, device=device, dtype=dtype)
        tx = torch.stack([times, x], dim=1)
        tx_flat = tx[:,None,:].expand(nb,Nv,2).reshape(nb*Nv,2)
        v_flat = vv[None,:,:].expand(nb,Nv,3).reshape(nb*Nv,3)
        f, *_ = model.f_flat(tx_flat, v_flat)
        rho = torch.full((nb*Nv,), float(state["rho"]), device=device, dtype=dtype)
        ux = torch.full_like(rho, float(state.get("ux",0.0)))
        T = torch.full_like(rho, float(state["T"]))
        Mt = maxwellian(rho, ux, T, v_flat)
        w = ww.repeat(nb)
        losses.append(torch.sum(w*(f-Mt)**2)/torch.sum(w))
    if not losses:
        return torch.tensor(0.0, device=device, dtype=dtype)
    return sum(losses)/len(losses)

def anchor_and_macro_loss(model, cfg, anchor, ref, vgrid, qweight, device, dtype):
    # anchor q/sigma on sparse shock layer
    x = anchor["x"]
    rho, ux, T, qx, sig = integrated_profiles(model, cfg, x, vgrid, qweight, device, dtype)
    qscale = float(cfg["train"].get("q_scale", max(float(torch.max(torch.abs(anchor["q"])).detach().cpu()), 1e-3)))
    sscale = float(cfg["train"].get("sig_scale", max(float(torch.max(torch.abs(anchor["s"])).detach().cpu()), 1e-3)))
    lq = torch.mean(((qx-anchor["q"])/(qscale + anchor["qstd"]))**2)
    ls = torch.mean(((sig-anchor["s"])/(sscale + anchor["sstd"]))**2)

    # macro-lock on a moderate grid against DVM
    nlock = int(cfg["train"].get("macro_lock_points", 96))
    x_np = ref["x"]
    idx = np.linspace(0, len(x_np)-1, nlock).round().astype(int)
    xl = torch.tensor(x_np[idx], device=device, dtype=dtype)
    rr = torch.tensor(ref["rho"][idx], device=device, dtype=dtype)
    uu = torch.tensor(ref["ux"][idx], device=device, dtype=dtype)
    TT = torch.tensor(ref["T"][idx], device=device, dtype=dtype)
    rho_l, ux_l, T_l, _, _ = integrated_profiles(model, cfg, xl, vgrid, qweight, device, dtype)
    lrho = torch.mean(((rho_l-rr)/(torch.max(torch.abs(rr))+1e-8))**2)
    lux  = torch.mean(((ux_l-uu)/(torch.max(torch.abs(uu))+1e-4))**2)
    lT   = torch.mean(((T_l-TT)/(torch.max(torch.abs(TT))+1e-8))**2)
    lmacro = lrho + float(cfg["train"].get("macro_lock_u_weight", 10.0))*lux + lT
    stats = {
        "anchor_q": lq.detach(),
        "anchor_sig": ls.detach(),
        "macro_lock": lmacro.detach(),
        "pbgk_q_mean": torch.mean(qx).detach(),
        "pbgk_sig_mean": torch.mean(sig).detach(),
        "pbgk_q_tgt_mean": torch.mean(anchor["q"]).detach(),
        "pbgk_sig_tgt_mean": torch.mean(anchor["s"]).detach(),
    }
    return lq, ls, lmacro, stats

# -----------------------------
# train
# -----------------------------
def train(cfg):
    dtype = torch.float64 if cfg.get("dtype","float32") == "float64" else torch.float32
    device = get_device(cfg["train"].get("device","auto"))
    torch.set_default_dtype(dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg["train"].get("allow_tf32", True))

    vgrid, qweight, vweight = build_vgrid(cfg, device, dtype)

    # make and load DVM anchor
    apath = make_dvm_anchor(cfg)
    cfg["anchor"]["path"] = apath
    anchor = load_anchor(apath, device, dtype)
    ref = load_npz_ref(cfg["anchor"].get("dvm_ref", "ref/dvm_smoke_fixed.npz"))

    model = PositiveHMM(cfg).to(device=device, dtype=dtype)
    print(f"[info] v6d device={device}, dtype={dtype}, run_dir=runs/{cfg['run']['name']}")
    print(f"[info] model parameters: {sum(p.numel() for p in model.parameters()):,}")

    init = cfg["train"].get("init_model", "")
    if init and Path(init).exists():
        data = torch.load(init, map_location=device)
        sd = data.get("model", data)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[v6d] initialized from {init}; missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print("[v6d] training from scratch")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay",0.0)))
    steps = int(cfg["train"]["steps"])
    min_lr = float(cfg["train"].get("min_lr", 1e-6))
    lr0 = float(cfg["train"]["lr"])
    run_dir = Path("runs")/cfg["run"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, run_dir/"config_used.json")

    t0 = time.time()
    for step in range(1, steps+1):
        frac = (step-1)/max(steps-1,1)
        lr = min_lr + 0.5*(lr0-min_lr)*(1+math.cos(math.pi*frac))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        lpde, lrel, lpos = pde_loss(model, cfg, vgrid, qweight, vweight)
        lbc = bc_loss(model, cfg, vgrid, qweight)
        lq, ls, lmacro, st = anchor_and_macro_loss(model, cfg, anchor, ref, vgrid, qweight, device, dtype)

        tr = cfg["train"]
        total = (
            float(tr.get("lambda_pde", 1.0))*lpde
            + float(tr.get("lambda_pde_rel", 0.02))*lrel
            + float(tr.get("lambda_pos", 5.0))*lpos
            + float(tr.get("lambda_bc", 1.0))*lbc
            + float(tr.get("lambda_anchor_q", 1.0))*lq
            + float(tr.get("lambda_anchor_sig", 1.0))*ls
            + float(tr.get("lambda_macro_lock", 10.0))*lmacro
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("grad_clip", 0.5)))
        opt.step()

        if step == 1 or step % int(tr.get("log_every",100)) == 0:
            print(
                f"[log] step={step} lr={lr:.8e} total={float(total.detach()):.8e} "
                f"pde={float(lpde.detach()):.8e} pde_rel={float(lrel.detach()):.8e} "
                f"macro_lock={float(st['macro_lock']):.8e} bc={float(lbc.detach()):.8e} pos={float(lpos.detach()):.8e} "
                f"anchor_q={float(st['anchor_q']):.8e} anchor_sig={float(st['anchor_sig']):.8e} "
                f"fac_min={float(model.f_flat(torch.tensor([[cfg['case']['t_final'],0.0]],device=device,dtype=dtype).repeat(1,1), torch.tensor([[0.0,0.0,0.0]],device=device,dtype=dtype))[2].min().detach()):.8e} "
                f"pbgk_q_mean={float(st['pbgk_q_mean']):.8e} pbgk_sig_mean={float(st['pbgk_sig_mean']):.8e} "
                f"pbgk_q_tgt_mean={float(st['pbgk_q_tgt_mean']):.8e} pbgk_sig_tgt_mean={float(st['pbgk_sig_tgt_mean']):.8e}",
                flush=True
            )

        if step % int(tr.get("save_every",1000)) == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, run_dir/"ckpt_last.pt")
            print(f"[info] saved checkpoint at step {step}", flush=True)

    torch.save({"model": model.state_dict(), "cfg": cfg, "step": steps}, run_dir/"ckpt_last.pt")
    print(f"[done] v6d completed. Last checkpoint: {run_dir/'ckpt_last.pt'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default="")
    ap.add_argument("--init-model", default="")
    args = ap.parse_args()
    cfg = load_json(args.config)
    if args.run_name:
        cfg["run"]["name"] = args.run_name
    if args.init_model:
        cfg["train"]["init_model"] = args.init_model
    train(cfg)

if __name__ == "__main__":
    main()
