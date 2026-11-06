"""Analytic physics and read-only velocity-resolution tests for the new model.

Only RH states, analytic distributions and the user's own no-label checkpoint
are consumed. A checkpoint's velocity-zero residual is diagnostic, not an
initialization acceptance test: the new model must subsequently be solved.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import base_solver as b


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def scales(c):
    a = b.rh(c)
    rho, T = a['rho2'], a['T2']
    return np.array([rho, a['u1'], T, rho*T**1.5, rho*T, rho*T*T, rho*T*T, rho*T*T])


def ladder(model, c, device, orders):
    x = torch.tensor([-c['L'], -c['L']/3, 0., c['L']/3, c['L']], device=device)
    rows = {}
    with torch.no_grad():
        for order in orders:
            grid = b.quadrature(order, c['nr'], c['vmax'], device)
            rows[order] = b.profiles(model, x, grid).cpu().numpy()
        radial = b.profiles(model, x, b.quadrature(orders[-1], max(96,c['nr']+16), c['vmax'], device)).cpu().numpy()
    fine = rows[orders[-1]]
    errors = {str(n): np.max(abs(a-fine)/scales(c), axis=0).tolist() for n,a in rows.items()}
    good = [n for n in orders[:-1] if max(errors[str(n)]) < 1e-4]
    return dict(x=x.cpu().tolist(), total_axial_orders=orders,
                fields={str(n): a.tolist() for n,a in rows.items()},
                max_normalized_difference_to_finest=errors,
                radial_refinement_max=np.max(abs(radial-fine)/scales(c),axis=0).tolist(),
                recommended_total_nvx=min(good) if good else None)


def velocity_zero(model, c, grid, device):
    xx = torch.tensor([-c['L']+1, -c['L']/3, 0., c['L']/3, c['L']-1],device=device)
    rr = torch.tensor([0., 1., 2.],device=device)
    X,R = torch.meshgrid(xx,rr,indexing='ij')
    vp=torch.full_like(X,1e-8)
    vm=-vp
    with torch.no_grad():
        fp=model(X,vp,R)[0]
        fm=model(X,vm,R)[0]
        f0=model(X,torch.zeros_like(X),R)[0]
        a=b.profiles(model,xx,grid)
        rho,u,T,q=a[:,:4].unbind(1)
        cx=-u[:,None]
        e=cx**2+R**2
        M=rho[:,None]/(2*math.pi*T[:,None])**1.5*torch.exp(-e/(2*T[:,None]))
        S0=M*(1+(1-c['Pr'])*q[:,None]*cx/(5*rho[:,None]*T[:,None]**2)*(e/T[:,None]-5))
        tau=c['mu1']*T**(c['omega']-1)/rho
        residual0=(f0-S0)/tau[:,None]
    return dict(x=xx.cpu().tolist(),r=rr.cpu().tolist(),
                plus=fp.cpu().tolist(),minus=fm.cpu().tolist(),zero=f0.cpu().tolist(),
                jump_absolute_max=float(abs(fp-fm).max()),
                collision_residual_vx0=residual0.cpu().tolist(),
                collision_residual_vx0_max=float(abs(residual0).max()),
                note='Interior continuity is structural; zero-velocity collision residual must be reduced by new training.')


def main():
    ap=argparse.ArgumentParser()
    root=Path(__file__).resolve().parent
    ap.add_argument('--config', default=str(root/'model_config.json'))
    ap.add_argument('--checkpoint', action='append', default=[])
    ap.add_argument('--out', default=str(root/'PHYSICS_GATE.json'))
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--orders', default='160,192,256,320,384')
    args=ap.parse_args()
    start=time.monotonic()
    c=json.loads(Path(args.config).read_text())
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    if args.gpu:
        assert torch.cuda.is_available()
    device='cuda' if args.gpu else 'cpu'
    assert c['nvx']%2==0
    grid=b.quadrature(c['nvx'],c['nr'],c['vmax'],device)
    v,r,w=grid
    assert bool((v!=0).all()) and int((v<0).sum())==int((v>0).sum())
    # Independent integration of an exact separable radial polynomial-Gaussian.
    # The RH and collision checks below test the complete cylindrical measure.
    a=b.rh(c)
    rho=torch.tensor([1.,a['rho2']],device=device)
    u=torch.tensor([a['u1'],a['u2']],device=device)
    T=torch.tensor([1.,a['T2']],device=device)
    M=b.maxwellian(v,r,rho,u,T)
    obs=b.integrate(M,v,r,w)
    equilibrium_macro_error=float(abs(obs[:,:3]-torch.stack((rho,u,T),1)).max())
    equilibrium_high_error=float(abs(obs[:,3:]).max())
    assert max(equilibrium_macro_error,equilibrium_high_error)<1e-9
    f=(.4*M[0]+.6*M[1])[None,:]
    S,tau,obs=b.target(f,v,r,w,c)
    C=(S-f)/tau[:,None]
    cx=v-obs[:,1,None];e=cx**2+r**2
    invariant_error=float(torch.stack(((C*w).sum(1),(C*w*v).sum(1),(C*w*(v*v+r*r)).sum(1)),1).abs().max())
    q_error=float(abs(.5*(C*w*cx*e).sum(1)+c['Pr']*obs[:,3]/tau).max())
    stress_error=float(abs((C*w*(cx*cx-e/3)).sum(1)+obs[:,4]/tau).max())
    assert max(invariant_error,q_error,stress_error)<1e-8
    b.seed_all(c['seed']);model=b.Distribution(c).to(device)
    # Structural incoming gate tests independent of a learned checkpoint.
    X=torch.tensor([[-c['L'],-c['L'],c['L'],c['L'],0.,0.]],device=device)
    V=torch.tensor([[.1,3.,-.1,-3.,-1e-12,1e-12]],device=device)
    gate=b.incoming_gate(X,V,c)
    assert float(abs(gate[0,:4]).max())<1e-14
    assert float(abs(gate[0,4:]-1).max())<1e-12
    zz=velocity_zero(model,c,grid,device)
    assert zz['jump_absolute_max']<1e-7
    boundary_errors=[]
    for side,vel,k in [(-1,.1,0),(1,-.1,1)]:
        vb=torch.tensor([[vel]],device=device);rb=torch.tensor([[.4]],device=device)
        actual=model(torch.full_like(vb,side*c['L']),vb,rb)[0]
        expected=rho[k]/(2*math.pi*T[k])**1.5*torch.exp(-((vb-u[k])**2+rb**2)/(2*T[k]))
        boundary_errors.append(float(abs(actual-expected).max()))
    assert max(boundary_errors)<1e-13
    # Derivative of the new gate/model in x, away from the exact boundary corner.
    xp=torch.tensor([[.7]],device=device,requires_grad=True)
    vp=torch.tensor([[.31]],device=device);rp=torch.tensor([[.6]],device=device)
    dx=torch.autograd.grad(model(xp,vp,rp)[0].sum(),xp)[0]
    fd=(model(xp.detach()+1e-5,vp,rp)[0]-model(xp.detach()-1e-5,vp,rp)[0])/(2e-5)
    derivative_error=float(abs(dx-fd).max())
    assert derivative_error<1e-8
    orders=sorted(set(int(n) for n in args.orders.split(',')))
    assert all(n%2==0 for n in orders)
    checkpoints=[]
    for path in args.checkpoint:
        ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        model.load_state_dict(ck['model'])
        row=dict(checkpoint=str(path),checkpoint_sha256=sha(path),step=ck.get('step'),
                 ladder=ladder(model,c,device,orders),velocity_zero=velocity_zero(model,c,grid,device),
                 interpretation='Own no-label weights in NEW characteristic-gate model, not an exact old-model restart')
        checkpoints.append(row)
        del ck
    result=dict(PASS=True,seconds=time.monotonic()-start,device=device,
                equilibrium_macro_error=equilibrium_macro_error,equilibrium_high_error=equilibrium_high_error,
                invariant_error=invariant_error,q_decay_error=q_error,stress_decay_error=stress_error,
                incoming_boundary_errors=boundary_errors,derivative_error=derivative_error,
                structural_vx0=zz,checkpoints=checkpoints,
                config_sha256=sha(args.config),solver_sha256=sha(b.__file__),gate_sha256=sha(__file__),
                scope='Analytic implementation gate plus read-only initialization-resolution diagnostics; not convergence proof')
    Path(args.out).write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
