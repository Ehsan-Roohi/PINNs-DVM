"""Deterministic, differentiable physics objective for full-batch line searches.

The only external implementation dependency is the campaign's immutable
``base_solver.py``. No reference solution is read. The velocity envelope and
tail weighting are frozen ONCE at the initial, independently trained physics
checkpoint; a restart must supply the saved ``frozen_state()``. Central moment
kernels remain functions of the current distribution and are differentiated.

``value_and_grad(flat)`` sets model parameters, accumulates the exact objective
gradient in spatial chunks, and returns (Python float, flat float64 tensor).
It performs no clipping, optimizer step, stochastic sampling or ramp changes.
The phase condition contributes once, independently of spatial chunk count.
"""
import copy
import hashlib
import json
import math

from pathlib import Path
import torch
import base_solver as base


def fixed_points(c, n_uniform=64, n_shock=32, n_boundary_each=16, device='cpu'):
    """Declared RH-scale spatial design; repeated endpoints are intentional."""
    L = c['L']
    return torch.cat((
        torch.linspace(-L, L, n_uniform, dtype=torch.float64, device=device),
        torch.linspace(-L/3, L/3, n_shock, dtype=torch.float64, device=device),
        torch.linspace(-L, -5*L/6, n_boundary_each, dtype=torch.float64, device=device),
        torch.linspace(5*L/6, L, n_boundary_each, dtype=torch.float64, device=device),
    )).sort().values


def make_points(c, device='cpu'):
    design=json.loads((Path(__file__).parent/'spatial_grid.json').read_text())
    assert len(design['x'])==c['nx']==187
    return torch.tensor(design['x'],dtype=torch.float64,device=device)


def _config_hash(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _state_hash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        h.update(name.encode())
        if torch.is_tensor(value):
            a = value.detach().cpu().contiguous()
            h.update(str(a.dtype).encode())
            h.update(str(tuple(a.shape)).encode())
            h.update(a.numpy().tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
    return h.hexdigest()


class DeterministicObjective:
    VERSION = 'frozen-spatial-weighted-physics-v2'

    def __init__(self, model, c, grid, x, chunk_size=8, frozen_state=None, spatial_weights=None):
        self.model = model
        # The runner may supply its QN manifest or the physical model config.
        # All equations/scales/weights always come from model.c.
        c = model.c
        self.c = copy.deepcopy(c)
        self.config_hash = _config_hash(c)
        assert _config_hash(model.c) == self.config_hash, 'Model and objective config mismatch'
        assert c['weights']['boundary'] == 0, 'Finite-domain outgoing boundary penalty must remain disabled'
        self.params = tuple(model.parameters())
        assert self.params and all(p.dtype == torch.float64 for p in self.params)
        self.device = self.params[0].device
        self.grid = tuple(z.detach().to(self.device).clone() for z in grid)
        self.x = x.detach().to(device=self.device, dtype=torch.float64).flatten().clone()
        assert len(self.x) and bool(torch.isfinite(self.x).all())
        assert bool((abs(self.x) <= c['L']).all())
        self.chunk_size = int(chunk_size)
        assert self.chunk_size > 0
        self.weights = copy.deepcopy(c['weights'])  # final ramp=1, forever
        self.state = base.rh(c)
        self.last_terms = {}
        self.last_metrics = {}
        v, r, w = self.grid
        if frozen_state is None:
            if spatial_weights is None:
                design=json.loads((Path(__file__).parent/'spatial_grid.json').read_text())
                if len(self.x)==len(design['x']):
                    assert torch.equal(self.x,torch.tensor(design['x'],dtype=torch.float64,device=self.device))
                    spatial_weights=torch.tensor(design['weights'],dtype=torch.float64,device=self.device)
                else:spatial_weights=torch.ones_like(self.x)
            spatial_weights=spatial_weights.to(self.device,dtype=torch.float64)
            assert spatial_weights.shape==self.x.shape and bool((spatial_weights>0).all())
            spatial_weights=spatial_weights/spatial_weights.sum()
            # Envelope is analytic RH; local u affects only the initially frozen
            # high-order weighting, never the objective during optimization.
            fixed = base.prescribed_envelope(self.x, v, r, c).detach()
            env = fixed + fixed.amax(1, keepdim=True)*1e-7
            with torch.no_grad():
                initial = torch.cat([base.profiles(model, z, self.grid)
                                     for z in self.x.split(self.chunk_size)])
                e0 = (v[None, :]-initial[:, 1, None])**2+r[None, :]**2
                tail = 1+(e0/(5*self.state['T2']))**4
            frozen_state = {
                'version': self.VERSION, 'config_sha256': self.config_hash,
                'spatial_weight':spatial_weights, 'x': self.x, 'v': v, 'r': r, 'w': w,
                'rh_envelope': fixed, 'residual_envelope': env, 'tail_multiplier': tail,
                'kinetic_weight': fixed*w/(env**2*(fixed*w).sum(1, keepdim=True)),
                'tail_weight': fixed*w*tail/(env**2*(fixed*w*tail).sum(1, keepdim=True)),
                'initial_u': initial[:, 1],
            }
        self._frozen = {k: (z.detach().to(self.device).clone() if torch.is_tensor(z) else copy.deepcopy(z))
                        for k, z in frozen_state.items()}
        assert self._frozen['version'] == self.VERSION
        assert self._frozen['config_sha256'] == self.config_hash
        for name, current in [('x', self.x), ('v', v), ('r', r), ('w', w)]:
            assert torch.equal(self._frozen[name], current), 'Frozen '+name+' mismatch'
        expected_shape = (len(self.x), len(v))
        for name in ['kinetic_weight', 'tail_weight']:
            a = self._frozen[name]
            assert a.shape == expected_shape and bool(torch.isfinite(a).all())
            assert bool((a >= 0).all()) and not a.requires_grad
        self.spatial_weight=self._frozen['spatial_weight']
        assert self.spatial_weight.shape==self.x.shape and bool((self.spatial_weight>0).all())
        assert abs(float(self.spatial_weight.sum())-1)<1e-12
        self.frozen_hash = _state_hash(self._frozen)
        self.metadata = dict(version=self.VERSION, config_sha256=self.config_hash,
                             frozen_sha256=self.frozen_hash, spatial_count=len(self.x),
                             velocity_count=len(v), parameter_count=sum(p.numel() for p in self.params),
                             weights=self.weights,
                             phase='rho(0)=(rho1+rho2)/2; evaluated once',
                             tail='frozen once using initial own-physics model; restored verbatim',
                             information='RH states and ordinary BGK (Pr=1, constant tau=1) equation only; no external profiles')

    def frozen_state(self):
        return {k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                for k, v in self._frozen.items()}

    def get_flat(self):
        return torch.cat([p.detach().reshape(-1) for p in self.params]).clone()

    def set_flat(self, flat):
        flat = flat.detach().to(device=self.device, dtype=torch.float64).flatten()
        assert flat.numel() == sum(p.numel() for p in self.params)
        if not bool(torch.isfinite(flat).all()):
            raise FloatingPointError('Nonfinite parameter candidate')
        offset = 0
        with torch.no_grad():
            for p in self.params:
                p.copy_(flat[offset:offset+p.numel()].view_as(p))
                offset += p.numel()

    def _chunk_terms(self, start, end):
        v, r, w = self.grid
        residual, f, _base, S, a = base.equation(self.model, self.x[start:end], self.grid, True)
        rho, u, T, q, sigma = a[:, :5].unbind(1)
        rs, ts = self.state['rho2'], self.state['T2']
        vs = math.sqrt(ts)
        sw=self.spatial_weight[start:end]
        def mean(a):return (a*sw).sum()/sw.sum()
        kinetic = mean((residual**2*self._frozen['kinetic_weight'][start:end]).sum(1))
        tail = mean((residual**2*self._frozen['tail_weight'][start:end]).sum(1))
        raw = mean((residual**2*w).sum(1)/(rs*rs))
        u1 = self.state['u1']
        flux = torch.stack((rho*u/u1,
                            (rho*u*u+rho*T+sigma)/(u1*u1+1),
                            (.5*rho*u**3+2.5*rho*T*u+u*sigma+q)/(.5*u1**3+2.5*u1)), 1)
        cx = v-u[:, None]
        e = cx**2+r**2
        # Keep these state-dependent moment kernels differentiable: freezing
        # them would define a different objective and break its true gradient.
        kernels = [torch.ones_like(cx), v.expand_as(cx)/vs, (v*v+r*r).expand_as(cx)/ts,
                   (cx*cx-e/3)/ts, .5*cx*e/(ts*vs),
                   (e-7*T[:, None])*(cx*cx-e/3)/(ts*ts), e**2/(ts*ts)]
        projected = torch.stack([(residual*w*k).sum(1)/rs for k in kernels], 1)
        terms = {'kinetic': kinetic, 'tail': tail, 'raw': raw,
                 'flux': mean(((flux-1)**2).mean(1)), 'invariant': mean((projected[:, :3]**2).mean(1)),
                 'stress': mean(projected[:, 3]**2), 'heat': mean(projected[:, 4]**2),
                 'R_projection': mean(projected[:, 5]**2),
                 'Delta_projection': mean(projected[:, 6]**2)}
        with torch.no_grad():
            metrics = dict(flux_max=float(abs(flux-1).max()),
                           projection_max=float(abs(projected).max()),
                           rho_min=float(rho.min()), T_min=float(T.min()),
                           f_negative_mass=float((torch.clamp(-f, min=0)*w).sum(1).div((f.abs()*w).sum(1)).max()),
                           target_negative_mass=float((torch.clamp(-S, min=0)*w).sum(1).div((S.abs()*w).sum(1)).max()))
        return terms, metrics

    def value_and_grad(self, flat=None):
        if _config_hash(self.model.c) != self.config_hash:
            raise ValueError('Model configuration changed after objective freeze')
        if flat is not None:
            self.set_flat(flat)
        self.model.zero_grad(set_to_none=True)
        total_value = 0.
        raw_terms = {}
        metrics = {}
        N = len(self.x)
        for start in range(0, N, self.chunk_size):
            end = min(start+self.chunk_size, N)
            terms, local_metrics = self._chunk_terms(start, end)
            share = float(self.spatial_weight[start:end].sum())
            contribution = share*sum(self.weights[k]*z for k, z in terms.items())
            if not bool(torch.isfinite(contribution)):
                raise FloatingPointError('Nonfinite objective candidate')
            contribution.backward()
            total_value += float(contribution.detach())
            for k, z in terms.items():
                raw_terms[k] = raw_terms.get(k, 0.)+share*float(z.detach())
            for k, z in local_metrics.items():
                if k not in metrics:
                    metrics[k] = z
                else:
                    metrics[k] = min(metrics[k], z) if k.endswith('_min') else max(metrics[k], z)
        pin = base.profiles(self.model, torch.zeros(1, dtype=torch.float64, device=self.device), self.grid)[0, 0]
        midpoint = (1+self.state['rho2'])/2
        phase = ((pin-midpoint)/midpoint)**2
        phase_loss = self.weights['phase']*phase
        if not bool(torch.isfinite(phase_loss)):
            raise FloatingPointError('Nonfinite phase objective')
        phase_loss.backward()
        total_value += float(phase_loss.detach())
        raw_terms['phase'] = float(phase.detach())
        raw_terms['boundary'] = 0.  # diagnostic only, disabled in scalar objective
        grad = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1))
                          for p in self.params]).clone()
        if not math.isfinite(total_value) or not bool(torch.isfinite(grad).all()):
            raise FloatingPointError('Nonfinite objective or gradient')
        self.last_terms, self.last_metrics = raw_terms, metrics
        return total_value, grad

    loss_and_grad = value_and_grad

