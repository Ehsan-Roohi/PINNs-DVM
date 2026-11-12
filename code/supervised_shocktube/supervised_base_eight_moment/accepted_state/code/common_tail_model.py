"""Model 170's class with the common exponential-tail envelope (PyTorch).

STATUS: written on a machine without torch, so this file has NOT been executed.
test_common_tail_model.py must pass before any use (Stage A preflight in PLAN_FA.md).
The numerical definition of the envelope is common_tail_envelope.py, which has
been executed and self-checked; this file must reproduce it.

Design
- Subclass of QAmplitudeAllMomentHMM (the class of accepted checkpoint 170).
- No new parameter and no new buffer: the state_dict keys and the 52,753
  parameters are unchanged, so checkpoint 170 loads with strict=True.
- For |x| <= x1 every coefficient is the parent's own value (torch.where keeps
  the parent tensor), so values and autograd derivatives are those of model 170.
- For |x| > x1 the two x-gates are replaced by E_q and E_h, which share the
  decay rate 2k of the initial condition beyond x2.
"""
import numpy as np
import torch

from q_amplitude_adapter import QAmplitudeAllMomentHMM

_XI, _WI = np.polynomial.legendre.leggauss(16)


class CommonTailHMM(QAmplitudeAllMomentHMM):
    def __init__(self, cfg, x1=0.135, width=None):
        super().__init__(cfg)
        case = cfg["case"]
        lo, hi = float(case["x_min"]), float(case["x_max"])
        assert lo == -hi, "the envelope assumes a domain centred on the initial jump"
        self.tail_rate = 2.0 * float(case["smooth_k"])
        self.tail_x1 = float(x1)
        self.tail_x2 = float(x1) + (1.0 / self.tail_rate if width is None else float(width))
        self.tail_half_domain = hi
        assert 0.0 < self.tail_x1 < self.tail_x2 < hi
        # plain floats: not parameters, not buffers
        self.tail_Cq = self._tail_constant("q")
        self.tail_Ch = self._tail_constant("h")

    # ---- old gates, written in a = |x| -------------------------------------------------
    def _ln_old(self, a, group):
        if group == "q":
            return -(a / self.gate_width) ** 4
        xi = a / self.tail_half_domain
        return -(a / self.higher_width) ** 4 + 2.0 * torch.log(torch.clamp(1.0 - xi * xi, min=1e-300))

    def _slope_old(self, a, group):
        """-d ln G / da."""
        if group == "q":
            return 4.0 * a ** 3 / self.gate_width ** 4
        L = self.tail_half_domain
        return 4.0 * a ** 3 / self.higher_width ** 4 + 4.0 * a / (L * L - a * a)

    # ---- P(a) = int_{x1}^{a} s (rate - slope_old) da', a clamped to [x1, x2] ------------
    def _P(self, a, group):
        xi = torch.as_tensor(_XI, dtype=a.dtype, device=a.device)
        wi = torch.as_tensor(_WI, dtype=a.dtype, device=a.device)
        half = 0.5 * (a - self.tail_x1)
        nodes = self.tail_x1 + half[:, None] * (1.0 + xi[None, :])
        u = (nodes - self.tail_x1) / (self.tail_x2 - self.tail_x1)
        s = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
        return half * ((s * (self.tail_rate - self._slope_old(nodes, group))) @ wi)

    def _tail_constant(self, group):
        a2 = torch.tensor([self.tail_x2], dtype=torch.float64)
        return float(self._ln_old(a2, group) - self._P(a2, group) + self.tail_rate * self.tail_x2)

    def envelope(self, x, group):
        """E_g(x); meaningful for |x| > x1 (callers keep the parent value for |x| <= x1)."""
        a = torch.abs(x)
        ac = torch.clamp(a, self.tail_x1, self.tail_x2)
        mid = self._ln_old(ac, group) - self._P(ac, group)
        tail = (self.tail_Cq if group == "q" else self.tail_Ch) - self.tail_rate * a
        return torch.exp(torch.where(a < self.tail_x2, mid, tail))

    # ---- coefficients ----------------------------------------------------------------
    def coefficients_from_raw(self, tx, raw, high):
        # rho, ux, T, qcoef, scoef, gate, mc, rc, dc exactly as model 170 computes them
        values = list(super().coefficients_from_raw(tx, raw, high))
        x = tx[:, 1]
        tau = torch.clamp(tx[:, 0] / float(self.cfg["case"]["t_final"]), 0.0, 1.5)
        outside = torch.abs(x) > self.tail_x1
        Eq = self.envelope(x, "q")
        Eh = self.envelope(x, "h") * self.higher_amp
        q_new = tau * Eq * self.q_amp * torch.tanh(raw[:, 3]) * (1.0 + 0.1 * torch.tanh(self.q_eta))
        s_new = tau * Eq * self.s_amp * torch.tanh(raw[:, 4])
        m_new = tau ** 3 * Eh * torch.tanh(high[:, 0])
        r_new = tau ** 2 * Eh * torch.tanh(high[:, 1])
        d_new = tau ** 2 * Eh * torch.tanh(high[:, 2])
        values[3] = torch.where(outside, q_new, values[3])
        values[4] = torch.where(outside, s_new, values[4])
        values[5] = torch.where(outside, Eq, values[5])
        values[6] = torch.where(outside, m_new, values[6])
        values[7] = torch.where(outside, r_new, values[7])
        values[8] = torch.where(outside, d_new, values[8])
        return tuple(values)
