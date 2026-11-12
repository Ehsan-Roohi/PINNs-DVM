"""Stage B (version 3): guarded Gauss-Newton training of the two output layers (the five lower
outputs and the three higher-moment outputs; the trunk and the temporal head stay frozen), from the
transplanted model 170T.

Why version 3: version 2 (stageb_train_v2_headonly.py) trained higher_head alone. Its smoke test
showed that fitting the higher moments with frozen q and sigma coefficients raises the q and sigma
data losses through the moment coupling (first-order +250 % and +60 % for a full step), so the
admissible descent ran out after four updates. Version 3 lets the lower output layer co-adjust,
with a minimal-function-change regulariser on all eight outputs.

Why version 2: the first version (stageb_train_v1_allparams.py) trained all 52,753 parameters.
Its two-attempt smoke test showed that such steps move the macroscopic outputs at times other
than t_final and that the balance components grow about fivefold per doubling of the step, so
only steps of 1-2 % of the Gauss-Newton step were admissible. Version 2 trained only higher_head (387 parameters).

Source      : weights of accepted checkpoint 170 in CommonTailHMM (Stage A).
Data        : the existing 160 macro + 42 moment observations only (no new label).
Held-out    : never read here. Assessment is a separate script run after training.
Direction   : damped Gauss-Newton on the 126 higher-moment observation residuals, with two
              data-free regularisers on the bounded outputs tanh(output_k): (i) tail preservation
              of the q, sigma, m, R, Delta outputs for |x| >= x2, (ii) a weak function-change
              penalty on all eight outputs over a dense space-time set; then a small quadratic
              programme that keeps the first-order change of every protected component in its room.
Acceptance  : every protected component (rho, u, T, q, sigma data losses and the 7 physical
              components) <= source value * (1 + 1e-6); higher-moment objective decreases;
              joint normalized objective decreases; tail change within its cap.
All settings below were fixed before the first run of this version.

usage: python -B stageb_train.py <project root> <plan dir> <out dir> [--max-attempts N] [--wall-seconds S]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stageb_problem import Problem, NAMES, sha  # noqa: E402

SET = {
    "trainable": "net.net[-1] (5x128 + 5) and higher_head (3x128 + 3): 1032 parameters; trunk, temporal_head, q_eta frozen",
    "max_accepted": 60,
    "max_attempts": 150,
    "wall_seconds": 3 * 3600,
    "mu0": 1e-3, "mu_min": 1e-10, "mu_max": 1e6,
    "backtracks": 14,
    "protected": [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14],
    "limit_rule": "source_170T * (1 + 1e-6) + 1e-24, never reset",
    "descent_reversal": 0.1,
    "lower_row_balance": 1.0,
    "tail_x_abs": [round(0.16 + 0.01 * i, 2) for i in range(35)],
    "tail_times_over_tf": [0.5, 1.0],
    "tail_unit_rms": 0.02,
    "tail_cap_rms": 0.02,
    "dense_x": [round(-0.45 + 0.01 * i, 2) for i in range(91)],
    "dense_times_over_tf": [0.25, 0.5, 0.75, 1.0],
    "dense_unit_rms": 0.5,
    "regulariser_weight": "each regulariser at its unit RMS costs as much as the whole source higher-moment misfit",
    "stall": "stop when the higher-moment objective fell by < 0.1 % over the last 8 accepted updates",
}


TAIL_CH = [3, 4, 5, 6, 7]      # q, sigma, m, R, Delta: the outputs with no observation outside the window
ALL_CH = list(range(8))


def flat(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def assign(params, z):
    with torch.no_grad():
        k = 0
        for p in params:
            p.copy_(z[k:k + p.numel()].view_as(p))
            k += p.numel()


def flat_grad(scalar, params):
    gs = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, gs)])


def head_parts(model, tx):
    """hidden features, raw (5) and high (3) exactly as AllMomentHMM.coefficients builds them."""
    hidden = model.features(tx)
    for layer in list(model.net.net)[:-1]:
        hidden = layer(hidden)
    raw = model.net.net[-1](hidden)
    high = model.higher_head(hidden)
    c = model.cfg["case"]
    tau = tx[:, 0] / float(c["t_final"])
    lo, hi = float(c["x_min"]), float(c["x_max"])
    xi = 2 * (tx[:, 1] - (lo + hi) / 2) / (hi - lo)
    corr = ((1 - tau) * (1 - xi * xi) ** 2)[:, None] * model.temporal_head(hidden)
    return hidden, raw + corr[:, :5], high + corr[:, 5:]


def check_head_parts(P, model):
    tx = torch.tensor([[P.tf, -0.30], [P.tf, -0.05], [0.4 * P.tf, 0.02], [P.tf, 0.21], [0.7 * P.tf, -0.17]])
    with torch.no_grad():
        _, raw, high = head_parts(model, tx)
        a = model.coefficients_from_raw(tx, raw, high)
        b = model.coefficients(tx)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


NPAR = 1032          # flat order: W5 (5x128), b5 (5), W3 (3x128), b3 (3)


def head_tanh(model, tx):
    hidden, raw, high = head_parts(model, tx)
    return hidden, torch.tanh(torch.cat([raw, high], 1))      # 8 channels: rho, u, T, q, sigma, m, R, Delta


def reg_block(model, tx, u_src, s, channels, unit, weight, with_jac=True):
    """Rows for (tanh(output_k) - source)/s_k, k in channels, on a point set; analytic Jacobian
    with respect to the two output layers (the trunk is frozen, so the outputs are linear in them)."""
    with torch.no_grad():
        hidden, u = head_tanh(model, tx)
        u = u[:, channels]
        rel = (u - u_src) / s[channels]
        rms = float(rel.square().mean().sqrt())
        if not with_jac:
            return rms, None, None
        n, nc = len(tx), len(channels)
        scale = math.sqrt(weight) / (unit * math.sqrt(nc * n))
        J = torch.zeros(n, nc, NPAR)
        for a, k in enumerate(channels):
            g = (1 - u[:, a] ** 2) / s[k]
            if k < 5:
                J[:, a, k * 128:(k + 1) * 128] = g[:, None] * hidden
                J[:, a, 640 + k] = g
            else:
                J[:, a, 645 + (k - 5) * 128:645 + (k - 4) * 128] = g[:, None] * hidden
                J[:, a, 1029 + (k - 5)] = g
        return rms, (rel * scale).reshape(-1), (J * scale).reshape(nc * n, NPAR)


def data_system(P, model, params):
    """690 residual rows in observation-scale units and their Jacobian w.r.t. params."""
    rows, jac, field = [], [], []
    for xx, targ, ks in P.families:
        for i in range(0, len(xx), 16):
            tx = torch.tensor(np.c_[np.full(len(xx[i:i + 16]), P.tf), xx[i:i + 16]])
            C = P.coefficients(model, tx)
            Cl = C.detach().requires_grad_(True)
            y = P.values_from_C(model, Cl)
            local = torch.stack([torch.autograd.grad(y[:, k].sum(), Cl, retain_graph=True)[0] for k in ks], 1)
            y = y.detach()
            for p in range(len(tx)):
                for a, k in enumerate(ks):
                    factor = 1.0 / (float(P.scales[k]) * math.sqrt(P.counts[k]))
                    rows.append(float(y[p, k] - targ[i + p, k]) * factor)
                    jac.append(flat_grad((C[p] * local[p, a]).sum() * factor, params))
                    field.append(k)
    r, J, field = torch.tensor(rows), torch.stack(jac), np.asarray(field)
    losses = np.array([float(r[field == k].square().sum()) for k in range(8)])
    G = torch.stack([2 * (r[field == k, None] * J[field == k]).sum(0) for k in range(8)])
    return r, J, field, losses, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("plan"); ap.add_argument("out")
    ap.add_argument("--max-attempts", type=int, default=SET["max_attempts"])
    ap.add_argument("--wall-seconds", type=float, default=SET["wall_seconds"])
    ap.add_argument("--physics-limit-rel", type=float, default=1e-6,
                    help="relative room above the source value for the 7 physical components (data components always 1e-6)")
    a = ap.parse_args()
    t0 = time.monotonic()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "STATES").mkdir()
    P = Problem(Path(a.root), Path(a.plan), Path(__file__).resolve().parent / "BOUND_CODE")
    model = P.model170T()
    source_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.requires_grad_(False)
    last = model.net.net[-1]
    params = [last.weight, last.bias, model.higher_head.weight, model.higher_head.bias]
    for p in params:
        p.requires_grad_(True)
    npar = sum(p.numel() for p in params)
    assert npar == NPAR and tuple(last.weight.shape) == (5, 128) and tuple(model.higher_head.weight.shape) == (3, 128)
    check_head_parts(P, model)

    # ---- frozen quantities of the source ------------------------------------------------
    tail_x = np.array([s * x for s in (-1.0, 1.0) for x in SET["tail_x_abs"]])
    tail_tx = torch.tensor([[f * P.tf, x] for f in SET["tail_times_over_tf"] for x in tail_x])
    dense_tx = torch.tensor([[f * P.tf, x] for f in SET["dense_times_over_tf"] for x in SET["dense_x"]])
    with torch.no_grad():
        u_tail = head_tanh(model, tail_tx)[1][:, TAIL_CH].clone()
        u_dense = head_tanh(model, dense_tx)[1].clone()
        win = torch.tensor(np.c_[np.full(42, P.tf), P.moment_x])
        s_head = head_tanh(model, win)[1].square().mean(0).sqrt().clone()                  # in-window RMS per channel
    source = P.components(model)
    limits = source * (1 + 1e-6) + 1e-24
    limits[8:] = source[8:] * (1 + a.physics_limit_rel) + 1e-24
    normal = P.normal
    prot = SET["protected"]
    w_reg = float(source[5:8].sum())                       # whole source higher-moment misfit, scale units

    def joint(c):
        return float((c / normal).sum())

    def higher(c):
        return float((c[5:8] / normal[5:8]).sum())

    def log(name, obj):
        with (out / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    settings = dict(SET, max_attempts=a.max_attempts, wall_seconds=a.wall_seconds, npar=npar,
                    physics_limit_rel=a.physics_limit_rel,
                    limit_rule_applied="data components: source*(1+1e-6); physical components: source*(1+%g); never reset" % a.physics_limit_rel,
                    source_components=source.tolist(), limits=limits.tolist(), normalizers=normal.tolist(),
                    names=NAMES, tail_points=int(len(tail_tx)), dense_points=int(len(dense_tx)),
                    head_channel_scales=s_head.tolist(), regulariser_weight_value=w_reg,
                    torch=torch.__version__, threads=torch.get_num_threads(),
                    code_sha256=dict(P.code_sha, stageb_train=sha(Path(__file__).read_bytes()),
                                     stageb_problem=sha((Path(__file__).resolve().parent / "stageb_problem.py").read_bytes())),
                    source_state_sha256="bbee1d6481b6ee3ccdcf4a0472c408368272f571f4124937368eb602f0833b95",
                    held_out_reference_read_by_this_script=False)
    (out / "RUN_SETTINGS.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
    print("source components:", " ".join("%s=%.6e" % t for t in zip(NAMES, source)), flush=True)

    def gradients():
        r, J, field, dl, Gd = data_system(P, model, params)
        pv, Gp = P.physical_losses(model, params)
        return r, J, field, np.concatenate([dl, pv]), torch.cat([Gd, Gp])

    def regs(with_jac):
        t = reg_block(model, tail_tx, u_tail, s_head, TAIL_CH, SET["tail_unit_rms"], w_reg, with_jac)
        d = reg_block(model, dense_tx, u_dense, s_head, ALL_CH, SET["dense_unit_rms"], w_reg, with_jac)
        return t, d

    # ---- derivative checks ------------------------------------------------------------------
    r, J, field, cur, G = gradients()
    assert np.allclose(cur, source, rtol=1e-9, atol=0), "gradient pass and value pass disagree at the source"
    origin = flat(params)
    live = [k for k in range(15) if float(G[k].norm()) > 0]
    direction = sum(G[k] / G[k].norm() for k in (5, 6, 7, 8, 9))
    direction = direction / direction.norm()
    eps = 1e-5
    assign(params, origin + eps * direction); cp_ = P.components(model)
    assign(params, origin - eps * direction); cm_ = P.components(model)
    assign(params, origin)
    fd = (cp_ - cm_) / (2 * eps)
    ad = (G @ direction).numpy()
    gnorm = G.norm(dim=1).numpy()
    err = np.abs(fd - ad) / np.maximum(np.abs(ad), 1e-4 * gnorm + 1e-300)
    err[gnorm == 0] = np.abs(fd[gnorm == 0]) / np.maximum(source[gnorm == 0], 1e-300)     # must stay exactly unchanged
    # analytic regulariser Jacobian against autograd on a few rows
    u = head_tanh(model, tail_tx[:4])[1]
    _, _, Jt = reg_block(model, tail_tx[:4], u_tail[:4], s_head, TAIL_CH, SET["tail_unit_rms"], w_reg)
    scale = math.sqrt(w_reg) / (SET["tail_unit_rms"] * math.sqrt(4 * len(TAIL_CH)))
    Ja = torch.stack([flat_grad(u[j, k] / s_head[k] * scale, params) for j in range(4) for k in TAIL_CH])
    reg_err = float((Ja - Jt).abs().max() / Jt.abs().max())
    check = {"fd": fd.tolist(), "ad": ad.tolist(), "relative_error": err.tolist(), "gradient_norms": gnorm.tolist(),
             "components_with_nonzero_gradient": [NAMES[k] for k in live],
             "error_definition": "|fd-ad| / max(|ad|, 1e-4*|grad|); for zero-gradient components |fd|/value",
             "regulariser_jacobian_relative_error": reg_err, "names": NAMES}
    (out / "GRADIENT_CHECK.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
    print("gradient check, relative error per component:", " ".join("%.1e" % e for e in err), "| regulariser %.1e" % reg_err, flush=True)
    assert err.max() < 5e-3 and reg_err < 1e-10, "derivative check failed"

    mu, accepted, history, reason, stale = SET["mu0"], 0, [], "MAX_ATTEMPTS", False
    torch.save({"model": source_state, "accepted": 0, "components": source.tolist()}, out / "STATES" / "STATE_000_SOURCE.pt")
    attempt = 0
    for attempt in range(1, a.max_attempts + 1):
        if time.monotonic() - t0 > a.wall_seconds:
            reason = "WALL_TIME"
            break
        tick = time.monotonic()
        if stale:
            r, J, field, cur, G = gradients()
            stale = False
        origin = flat(params)
        (rms_t, rt, Jt), (rms_d, rd, Jd) = regs(True)
        # Rows of the model: the 126 higher-moment residuals at weight 1, and the 564 residuals of the five
        # lower fields, each field weighted so that its current misfit counts as much as the whole source
        # higher-moment misfit. A protected least-squares loss cannot be held by its gradient alone: the
        # first version-3 smoke test satisfied every slope constraint and still multiplied the rho and T
        # losses, because the step changed those fields pointwise by several times their residual in
        # directions orthogonal to it. With the rows in the model the compensation is pointwise.
        wrow = torch.ones(len(r))
        for k in range(5):
            wrow[torch.tensor(field == k)] = math.sqrt(SET["lower_row_balance"] * w_reg / max(float(cur[k]), 1e-300))
        A = torch.cat([J * wrow[:, None], Jt, Jd])
        res = torch.cat([r * wrow, rt, rd])
        N = A.T @ A
        damp = mu * torch.diag(N).mean().clamp_min(1e-30)
        H = N + damp * torch.eye(npar)
        d0 = -torch.linalg.solve(H, A.T @ res)
        predicted0 = float((res + A @ d0).square().sum() / res.square().sum())
        # Component-descent step as a small quadratic programme: minimise the damped Gauss-Newton model
        # subject to G_k.d <= s_k for the protected components. s_k is a quarter of the room left under the
        # limit (the rule of train_all_moments.py); where d0 would use more, the excess is reversed by the
        # fraction descent_reversal. The correction is taken in the metric of the model Hessian
        # (d = d0 - H^-1 G^T lambda), i.e. along directions that cost the data fit least; a Euclidean
        # projection (the first two smoke tests) destroyed the descent of the R and Delta residuals.
        act = [k for k in prot if float(G[k].norm()) > 0]
        Ga = G[act]
        HiGt = torch.linalg.solve(H, Ga.T)
        metric = (Ga @ HiGt).numpy()
        gd0 = (Ga @ d0).numpy()
        room = np.maximum((limits - cur)[act] * 0.25, 0.0)
        slack = room - SET["descent_reversal"] * np.maximum(gd0 - room, 0.0)
        q = gd0 - slack
        lam = np.zeros(len(act))
        for pit in range(200000):
            prev = lam.copy()
            for k in range(len(lam)):
                lam[k] = max(0.0, lam[k] + (q[k] - metric[k] @ lam) / max(metric[k, k], 1e-300))
            if np.max(np.abs(lam - prev)) <= 1e-12 * max(1e-300, np.max(np.abs(lam))):
                break
        d = d0 - HiGt @ torch.tensor(lam)
        predicted = float((res + A @ d).square().sum() / res.square().sum())
        slopes0, slopes = (G @ d0).numpy(), (G @ d).numpy()
        log("DIRECTIONS.jsonl", {"attempt": attempt, "mu": mu, "norm_d0": float(d0.norm()), "norm_d": float(d.norm()),
                                 "predicted_objective_ratio_unconstrained": predicted0,
                                 "predicted_objective_ratio_at_alpha_1": predicted,
                                 "relative_slopes_unconstrained": (slopes0 / cur).tolist(),
                                 "relative_slopes": (slopes / cur).tolist(),
                                 "projection_iterations": pit, "active": [int(act[k]) for k in range(len(lam)) if lam[k] > 0],
                                 "seconds_gradients_and_solve": time.monotonic() - tick})
        base_joint, base_high = joint(cur), higher(cur)
        ok = False
        for back in range(SET["backtracks"]):
            alpha = 2.0 ** (-back)
            assign(params, origin + alpha * d)
            dl = P.data_losses(model)
            (rms_t, _, _), (rms_d, _, _) = regs(False)
            failed = [NAMES[k] for k in (0, 1, 2, 3, 4) if dl[k] > limits[k]]
            if (dl[5:8] / normal[5:8]).sum() >= base_high:
                failed.append("higher_moments_not_decreased")
            if rms_t > SET["tail_cap_rms"]:
                failed.append("tail_cap")
            cand = None
            if not failed:                                   # physics is the expensive part: only if data passed
                cand = np.concatenate([dl, P.physical_losses(model)[0]])
                failed += [NAMES[k] for k in prot if k >= 8 and cand[k] > limits[k]]
                if joint(cand) >= base_joint:
                    failed.append("joint_objective_not_decreased")
            log("TRIALS.jsonl", {"attempt": attempt, "alpha": alpha, "data": dl.tolist(),
                                 "components": None if cand is None else cand.tolist(), "tail_rms": rms_t, "dense_rms": rms_d,
                                 "failed": failed, "accepted": not failed})
            if not failed:
                ok = True
                break
        if not ok:
            assign(params, origin)
            mu *= 10
            print("attempt %d rejected at every alpha; mu -> %.1e (%.0f s)" % (attempt, mu, time.monotonic() - tick), flush=True)
            if mu > SET["mu_max"]:
                reason = "LINE_SEARCH_REJECTED"
                break
            continue
        accepted += 1
        stale = True
        mu = max(mu / 3, SET["mu_min"])
        history.append(higher(cand))
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        state_file = out / "STATES" / ("STATE_%03d.pt" % accepted)
        torch.save({"model": state, "accepted": accepted, "components": cand.tolist(), "tail_rms": rms_t, "dense_rms": rms_d}, state_file)
        log("ACCEPTED.jsonl", {"accepted": accepted, "attempt": attempt, "alpha": alpha, "mu_next": mu, "components": cand.tolist(),
                               "tail_rms": rms_t, "dense_rms": rms_d, "higher_objective": history[-1], "joint_objective": joint(cand),
                               "elapsed": time.monotonic() - t0, "state_sha256": sha(state_file.read_bytes())})
        print("accepted %d (attempt %d, alpha %.4g): m %.4e R %.4e Delta %.4e | tail %.2e dense %.2e | %.0f s, total %.0f s" % (
            accepted, attempt, alpha, cand[5], cand[6], cand[7], rms_t, rms_d, time.monotonic() - tick, time.monotonic() - t0), flush=True)
        if accepted >= SET["max_accepted"]:
            reason = "MAX_ACCEPTED"
            break
        if len(history) > 8 and history[-1] > history[-9] * (1 - 1e-3):
            reason = "STALL"
            break
    final = P.components(model)
    (rms_t, _, _), (rms_d, _, _) = regs(False)
    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    changed = [k for k in final_state if not torch.equal(final_state[k], source_state[k])]
    torch.save({"model": final_state, "accepted": accepted, "components": final.tolist(), "tail_rms": rms_t, "dense_rms": rms_d,
                "settings": settings, "stop_reason": reason, "architecture": "CommonTailHMM(x1=0.135)",
                "source_state_sha256": settings["source_state_sha256"]}, out / "STATE_FINAL.pt")
    result = {"schema": "STAGE_B_GUARDED_GN_OUTPUT_LAYERS_FROM170T_V3", "stop_reason": reason, "accepted_updates": accepted,
              "attempts": attempt, "elapsed_seconds": time.monotonic() - t0, "names": NAMES,
              "source_components": source.tolist(), "final_components": final.tolist(), "limits": limits.tolist(),
              "protected_within_limits": bool(all(final[k] <= limits[k] for k in prot)),
              "state_entries_changed": changed, "final_tail_rms": rms_t, "final_dense_rms": rms_d,
              "final_state_sha256": sha((out / "STATE_FINAL.pt").read_bytes()),
              "held_out_reference_read_by_this_script": False, "paper_admission": False}
    (out / "TRAIN_RESULT.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("stop_reason", "accepted_updates", "attempts", "elapsed_seconds",
                                             "protected_within_limits", "state_entries_changed")}), flush=True)


if __name__ == "__main__":
    main()
