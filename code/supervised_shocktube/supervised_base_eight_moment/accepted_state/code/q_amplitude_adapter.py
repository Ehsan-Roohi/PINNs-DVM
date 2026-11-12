"""One bounded amplitude parameter in the existing positive macro–micro family."""
import torch
from torch import nn
from all_moment_model import AllMomentHMM

class QAmplitudeAllMomentHMM(AllMomentHMM):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.q_eta = nn.Parameter(next(self.parameters()).new_zeros(()))

    def coefficients_from_raw(self, tx, raw, high):
        values = list(super().coefficients_from_raw(tx, raw, high))
        # Recompute on every call: eta remains in f, f_t and f_x's AD graph.
        values[3] = values[3] * (1.0 + 0.1 * torch.tanh(self.q_eta))
        return tuple(values)

    @classmethod
    def from_original_source(cls, cfg, original_state):
        base = AllMomentHMM(cfg)
        base.load_parent(original_state)
        child = cls(cfg)
        missing = child.load_state_dict(base.state_dict(), strict=False)
        assert missing.missing_keys == ['q_eta']
        assert missing.unexpected_keys == []
        assert torch.equal(child.q_eta.detach(), child.q_eta.new_zeros(()))
        assert all(torch.equal(value, child.state_dict()[name])
                   for name, value in base.state_dict().items())
        assert sum(p.numel() for p in child.parameters()) == sum(p.numel() for p in base.parameters()) + 1
        return base, child
