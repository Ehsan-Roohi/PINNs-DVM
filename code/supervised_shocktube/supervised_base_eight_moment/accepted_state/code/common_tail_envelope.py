"""Common exponential-tail support for the non-equilibrium coefficients of the
positive macro-micro BGK model (numpy reference implementation and self-check).

Existing model (all_moment_model.py / legacy.py), x in [-0.5, 0.5]:
    q, sigma coefficients    ~ G_q(x) = exp(-(x/w_q)^4),                 w_q = 0.13
    m, R, Delta coefficients ~ G_h(x) = exp(-(x/w_h)^4) * (1-(2x)^2)^2,  w_h = 0.22

Proposed envelope E_g (g = q or h), written through its logarithmic slope
L(a) = -d ln E / d|x|, a = |x|:

    a <= x1        : E_g = G_g            (same expression, same floating-point path)
    x1 < a < x2    : L = (1 - s) L_old + s * 2k,  s = smootherstep((a - x1)/(x2 - x1))
    a >= x2        : L = 2k               (E_g = exp(C_g - 2k a))

so ln E_g(a) = ln G_g(a) - P_g(a),  P_g(a) = int_{x1}^{a} s (2k - L_old) da'.
The slope goes monotonically from the old gate's slope to 2k, E_g is C3, and
there is no overshoot of the value blend.

Every constant comes from the case definition or from the existing gates:
    k  = case.smooth_k   (the initial jump is tanh(k x); for t << Kn every
                          non-equilibrium moment decays as exp(-2k|x|))
    x1 = first value beyond the outermost moment observation (|x| = 0.13375)
    x2 = x1 + 1/(2k)     (one e-folding length)
No reference (DVM) value enters.
"""
import numpy as np

K = 20.0            # case.smooth_k
RATE = 2.0 * K
W_Q = 0.13          # model.gate_width
W_H = 0.22          # AllMomentHMM.higher_width
X1 = 0.135
X2 = X1 + 1.0 / RATE
_XI, _WI = np.polynomial.legendre.leggauss(16)


def ln_gate(x, group):
    x = np.asarray(x, dtype=float)
    if group == "q":
        return -(x / W_Q) ** 4
    return -(x / W_H) ** 4 + 2.0 * np.log(np.clip(1.0 - (2.0 * x) ** 2, 1e-300, None))


def slope_old(a, group):
    """-d ln G / da for a = |x| > 0."""
    if group == "q":
        return 4.0 * a ** 3 / W_Q ** 4
    return 4.0 * a ** 3 / W_H ** 4 + 16.0 * a / (1.0 - 4.0 * a * a)


def smoother(u):
    return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


def _P(a, group, x1=X1, x2=X2):
    a = np.clip(np.atleast_1d(np.asarray(a, dtype=float)), x1, x2)
    half = 0.5 * (a - x1)
    nodes = x1 + half[:, None] * (1.0 + _XI[None, :])
    f = smoother((nodes - x1) / (x2 - x1)) * (RATE - slope_old(nodes, group))
    return half * (f @ _WI)


def tail_constant(group, x1=X1, x2=X2):
    return float(ln_gate(x2, group) - _P(x2, group, x1, x2)[0] + RATE * x2)


def ln_envelope(x, group, x1=X1, x2=X2):
    """group = 'q' (q and sigma coefficients) or 'h' (m, R, Delta coefficients)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    a = np.abs(x)
    old = ln_gate(x, group)
    mid = ln_gate(np.clip(a, x1, x2), group) - _P(a, group, x1, x2)
    tail = tail_constant(group, x1, x2) - RATE * a
    return np.where(a <= x1, old, np.where(a < x2, mid, tail))


def self_check():
    out = {"constants": {"k": K, "rate": RATE, "x1": X1, "x2": X2,
                         "C_q": tail_constant("q"), "C_h": tail_constant("h")}}
    x = np.linspace(-0.4995, 0.4995, 99901)
    h = x[1] - x[0]
    a = np.abs(x)
    for g in ("q", "h"):
        y = ln_envelope(x, g)
        inside = a <= X1
        s = smoother(np.clip((a - X1) / (X2 - X1), 0.0, 1.0))
        slope_exact = np.where(inside, slope_old(np.maximum(a, 1e-12), g),
                               (1.0 - s) * slope_old(np.clip(a, X1, X2), g) + s * RATE)
        slope_fd = -np.gradient(y, h) * np.sign(x)
        keep = (a > 2e-3) & (a < 0.495)
        res = {
            "identical_inside_x1": bool(np.array_equal(y[inside], ln_gate(x[inside], g))),
            "max_abs_slope_error_fd_vs_formula": float(np.abs(slope_fd - slope_exact)[keep].max()),
            "slope_at_x1_mid_x2": [float(slope_old(X1, g)),
                                   float(0.5 * slope_old(0.5 * (X1 + X2), g) + 0.5 * RATE), RATE],
            "max_slope_in_blend": float(slope_exact[(a > X1) & (a < X2)].max()),
            "monotone_decay_for_x_gt_0": bool(np.all(np.diff(y[x > 0.0]) < 0.0)),
            "continuity_jump_at_x1_x2": [float(abs(ln_envelope(X1 + 1e-12, g)[0] - ln_envelope(X1, g)[0])),
                                         float(abs(ln_envelope(X2 - 1e-12, g)[0] - ln_envelope(X2, g)[0]))],
            "E_new_over_E_old_at_0p16_0p20_0p25_0p30": [
                float(np.exp(ln_envelope(v, g)[0] - ln_gate(v, g))) for v in (0.16, 0.20, 0.25, 0.30)],
        }
        assert res["identical_inside_x1"]
        assert res["monotone_decay_for_x_gt_0"]
        assert res["max_abs_slope_error_fd_vs_formula"] < 5e-3      # finite-difference noise level
        assert max(res["continuity_jump_at_x1_x2"]) < 1e-9
        out[g] = res
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(self_check(), indent=1))
