"""Prepared optimizer helpers only. Importing this module never trains or changes a model.

The caller supplies the existing AllMomentHMM coefficient chain and inherited
acceptance checks. No distribution, network architecture or observation is added.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class FunctionTrustSettings:
    radius: float = 0.10
    terminal_time: float = 0.02
    spatial_length: float = 0.13
    coefficient_scales: tuple = (1.0, 1.0, 1.0, 0.25, 0.25, 0.20, 0.20, 0.20)


def physical_coefficients(model, tx, *, graph):
    """Use the unmodified bounded network's eight physical coefficients."""
    import torch
    if not tx.requires_grad:
        tx = tx.detach().clone().requires_grad_(True)
    all_c = model.coefficients(tx)
    C = torch.stack([all_c[k] for k in (0, 1, 2, 3, 4, 6, 7, 8)], dim=1)
    dc = torch.stack([
        torch.autograd.grad(C[:, k].sum(), tx, create_graph=True,
                            retain_graph=True)[0] for k in range(8)
    ], dim=1)
    if not graph:
        C, dc = C.detach(), dc.detach()
    return C, dc[:, :, 0], dc[:, :, 1]


def function_report(delta_f, delta_C, delta_Ct, delta_Cx,
                    frozen_f_denominator, frozen_velocity_weights,
                    settings=FunctionTrustSettings()):
    """Dimensionless finite-calibration-point trust gauge, not a global bound.

    Both the denominator and coefficient scales stay fixed to original a073.
    C_t and C_x changes are normalized over the physical time/length scales.
    """
    import torch
    scales = delta_C.new_tensor(settings.coefficient_scales)
    weight = frozen_velocity_weights / frozen_velocity_weights.sum()
    z = delta_f / frozen_f_denominator
    values = {
        'f_weighted_rms': ((z.square() * weight).sum(dim=-1).mean()).sqrt(),
        'f_scaled_max': z.abs().max(),
        'C_scaled_max': (delta_C / scales).abs().max(),
        'Ct_scaled_max': (settings.terminal_time * delta_Ct / scales).abs().max(),
        'Cx_scaled_max': (settings.spatial_length * delta_Cx / scales).abs().max(),
    }
    assert all(bool(torch.isfinite(x)) for x in values.values())
    return {k: float(v.detach()) for k, v in values.items()}


def predicted_trust_scale(projected_parameter_direction, directional_feature_jvp,
                          frozen_normalizers, settings=FunctionTrustSettings()):
    """Call after the existing component-descent projection, before line search.

    directional_feature_jvp uses eight-coefficient JVPs, never an Nv x Ntheta
    transport Jacobian. The parameter norm is logged, but never caps the step.
    """
    changes = directional_feature_jvp(projected_parameter_direction)
    report = function_report(*changes, *frozen_normalizers, settings)
    gauge = max(report.values())
    alpha = 1.0 if gauge == 0 else min(1.0, settings.radius / gauge)
    return projected_parameter_direction * alpha, {
        'predicted_function_blocks_before_scale': report,
        'predicted_function_gauge_before_scale': gauge,
        'functional_initial_scale': alpha,
        'parameter_norm_diagnostic_only': float(projected_parameter_direction.norm()),
    }


def actual_trust_acceptance(actual_changes, frozen_normalizers,
                            inherited_failures, settings=FunctionTrustSettings()):
    """Add trust diagnostics; do not discard any physical/field/peak guard."""
    report = function_report(*actual_changes, *frozen_normalizers, settings)
    failures = list(inherited_failures)
    for name, value in report.items():
        if value > settings.radius:
            failures.append('actual_function_trust:' + name)
    return not failures, failures, report


def append_pointwise_moment_rows(C, local_moment_jacobian, prediction, target,
                                 active_columns, counts, scales, normalizers,
                                 parameters, residual_rows, jacobian_rows):
    """Exact local-eight-C chain for every scalar observation residual.

    local_moment_jacobian has shape (batch, 8 moments, 8 coefficients), computed
    on the exact 48x20x20 endpoint grid using the existing physical distribution.
    No fit in unconstrained physical-C space occurs; C remains a function of all
    original NN parameters. Different field families retain counts 160 or42.
    """
    import math
    import torch
    for point, field in active_columns:
        factor = 1.0 / (scales[field] * math.sqrt(counts[field] * normalizers[field]))
        residual_rows.append(((prediction[point, field] - target[point, field]) * factor).detach())
        coeff_gradient = local_moment_jacobian[point, field].detach() * factor
        scalar = (C[point] * coeff_gradient).sum()
        gs = torch.autograd.grad(scalar, parameters, retain_graph=True, allow_unused=True)
        jacobian_rows.append(torch.cat([
            (torch.zeros_like(p) if g is None else g).reshape(-1)
            for p, g in zip(parameters, gs)
        ]).detach())


def local_function_whitener(normalized_function_jacobian,
                            measured_chain_fd_relative_error):
    """Optional diagnostic, not the default repair or a raw-logit fit.

    Input columns are physical C (8), or a deliberately small reduced direction
    basis; never form a Ntheta-squared parameter metric. Singular modes below
    measured numerical resolution are reported and excluded, with no cap change.
    A later implementation needs separate review before replacing the direction.
    """
    import torch
    A = normalized_function_jacobian
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    rel = max(torch.finfo(A.dtype).eps * max(A.shape),
              10.0 * float(measured_chain_fd_relative_error))
    keep = S > S[0] * rel
    whitener = Vh[keep].T / S[keep]
    return whitener, {'singular_values': S.detach().cpu().tolist(),
                      'retained_rank': int(keep.sum()),
                      'relative_numerical_cutoff': rel,
                      'method_executed_as_optimizer': False}


def make_directional_feature_jvp(model, calibration_tx, velocity):
    """Prepared exact parameter→C/Ct/Cx→f directional chain.

    This evaluates one direction, not a full transport/parameter Jacobian.
    The temporary functional-call view does not add trainable heads or modify
    the original model. CPU finite differences must validate it before use.
    """
    import torch
    from torch import nn
    from torch.func import functional_call

    class CoefficientView(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.model = original
        def forward(self, tx):
            return physical_coefficients(self.model, tx, graph=True)

    view = CoefficientView(model)
    names = [name for name, _ in view.named_parameters()]
    parameters = tuple(p.detach().requires_grad_(True) for p in view.parameters())
    tx = calibration_tx.detach().clone().requires_grad_(True)

    def values(*weights):
        return functional_call(view, dict(zip(names, weights)), (tx,))

    def directional(flat_direction):
        pieces = []
        offset = 0
        for p in parameters:
            pieces.append(flat_direction[offset:offset + p.numel()].view_as(p))
            offset += p.numel()
        assert offset == flat_direction.numel()
        (C, _, _), (dC, dCt, dCx) = torch.autograd.functional.jvp(
            values, parameters, tuple(pieces), create_graph=False, strict=False)
        _, df = torch.autograd.functional.jvp(
            lambda c: model.grid_from_coefficients(c, velocity)[0],
            C.detach(), dC.detach(), create_graph=False, strict=False)
        return df, dC, dCt, dCx

    return directional


def full_f_moment_chain_values(f, velocity, weights, kn, legacy_moments_all):
    """Exact finite-grid conversion; offsets remain differentiable for the chain."""
    import math
    import torch
    old = legacy_moments_all(f, velocity, weights, kn)
    fw = f * weights
    rho = fw.sum(dim=1)
    u = (fw @ velocity) / rho[:, None]
    c = velocity[None, :, :] - u[:, None, :]
    e = c.square().sum(dim=2)
    T = (fw * e).sum(dim=1) / (3 * rho)
    M = rho[:, None] * torch.exp(-e / (2 * T[:, None])) / (2 * math.pi * T[:, None]) ** 1.5
    Em = (M * weights * (c[:, :, 0] ** 3 - .6 * c[:, :, 0] * e)).sum(dim=1)
    ER = (M * weights * e * (c[:, :, 0].square() - e / 3)).sum(dim=1)
    values = torch.stack([old[:, 0], old[:, 1], old[:, 2], old[:, 3], old[:, 4],
                          old[:, 5] + Em, old[:, 6] + ER, old[:, 9]], dim=1)
    return values, {'Em': Em, 'ER': ER, 'raw_S': old[:, 10]}
