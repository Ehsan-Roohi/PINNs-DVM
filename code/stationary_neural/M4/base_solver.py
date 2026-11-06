"""Fresh Mach4 physics-only stationary Shakhov normal-shock PINN. No reference reader."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch import nn

STOP = False


def stop(*_):
    global STOP
    STOP = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rh(c):
    m = c['mach']
    density = 4*m*m/(m*m+3)
    return dict(rho1=1., u1=m*math.sqrt(5/3), T1=1., rho2=density,
                u2=m*math.sqrt(5/3)/density, T2=(5*m*m-1)/(4*density))


def validate_config(c):
    # Fail closed: the training API accepts only declared physical/numerical inputs.
    allowed = {'mach','Pr','omega','mu1','L','nvx','nr','vmax','width','depth',
               'representation','psi_max','seed','batch','steps','lr','wall_seconds',
               'checkpoint_every','log_every','gradient_every','validation_every',
               'validation_n','profile_n','refine_nvx','refine_nr','weights',
               'acceptance','initial_width','gradient_clip'}
    assert not set(c)-allowed, 'Unknown input fields: '+str(set(c)-allowed)
    assert c['mach'] == 4 and abs(c['Pr']-2/3)<1e-12
    assert c['omega']==1 and c['mu1']==1
    assert c['representation'] in ('signed','positive')


def quadrature(nvx, nr, vmax, device='cpu'):
    """Split axial half ranges: nvx is the TOTAL number of axial nodes.

    Separate half-range rules resolve the kinetic incoming/outgoing structure
    near zero rather than relying on cancellation across a full-range rule.
    """
    assert nvx>=4 and nvx%2==0, 'Split axial quadrature requires even total nvx'
    a, aw = np.polynomial.legendre.leggauss(nvx//2)
    b, bw = np.polynomial.legendre.leggauss(nr)
    axial=np.concatenate((vmax*(a-1)/2,vmax*(a+1)/2))
    axial_weights=np.concatenate((vmax*aw/2,vmax*aw/2))
    v, r = np.meshgrid(axial, vmax*(b+1)/2, indexing='ij')
    w = axial_weights[:,None]*vmax*bw[None,:]/2*2*np.pi*r
    return tuple(torch.as_tensor(z.ravel(), dtype=torch.float64, device=device) for z in (v,r,w))


def incoming_gate(x,v,c):
    """Characteristic-distance gate with the upstream RH collision time.

    g=1-exp(-d_in/(|vx|*tau1)), tau1=mu1*T1**(omega-1)/rho1.
    This is an ansatz scale, not replacement of the local Shakhov tau in the
    equation. At any strict interior point both vx->0 limits equal one.
    The zero-speed boundary corner is a measure-zero kinetic compatibility
    point; choose zero on its sign-selected incoming endpoint. No hard
    equilibrium region or imposed interior profile is introduced.
    """
    state=rh(c)
    tau1=c['mu1']*state['T1']**(c['omega']-1)/state['rho1']
    distance=torch.where(v>=0,x+c['L'],c['L']-x)
    speed=v.abs()
    denominator=torch.where(speed>0,speed,torch.ones_like(speed))*tau1
    ordinary=-torch.expm1(-distance/denominator)
    zero=torch.where(distance>0,torch.ones_like(distance),torch.zeros_like(distance))
    return torch.where(speed>0,ordinary,zero)


def maxwellian(v, r, rho, u, T):
    return rho[:,None]/(2*math.pi*T[:,None])**1.5*torch.exp(-((v-u[:,None])**2+r**2)/(2*T[:,None]))


def integrate(f, v, r, w):
    fw=f*w
    rho=fw.sum(1)
    if bool((rho.detach()<=0).any()):
        raise FloatingPointError('Nonpositive density; no clipping')
    u=(fw*v).sum(1)/rho
    cx=v-u[:,None]; e=cx**2+r**2
    T=(fw*e).sum(1)/(3*rho)
    if bool((T.detach()<=0).any()):
        raise FloatingPointError('Nonpositive temperature; no clipping')
    q=.5*(fw*cx*e).sum(1)
    sigma=(fw*(cx**2-e/3)).sum(1)
    R=(fw*(e-7*T[:,None])*(cx**2-e/3)).sum(1)
    delta=(fw*e**2).sum(1)-15*rho*T**2
    return torch.stack((rho,u,T,q,sigma,R,delta,R+delta/3),1)


def target(f,v,r,w,c):
    a=integrate(f,v,r,w)
    rho,u,T,q=a[:,:4].unbind(1)
    cx=v-u[:,None]; e=cx**2+r**2
    M=maxwellian(v,r,rho,u,T)
    S=M*(1+(1-c['Pr'])*q[:,None]*cx/(5*rho[:,None]*T[:,None]**2)*(e/T[:,None]-5))
    tau=c['mu1']*T**(c['omega']-1)/rho
    return S,tau,a


class Distribution(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c=c
        layers=[nn.Linear(5,c['width']),nn.Tanh()]
        for _ in range(c['depth']-1):
            layers += [nn.Linear(c['width'],c['width']),nn.Tanh()]
        layers += [nn.Linear(c['width'],1)]
        self.net=nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight,std=1e-3)
        nn.init.zeros_(self.net[-1].bias)
        self.log_width=nn.Parameter(torch.tensor(math.log(c['initial_width']),dtype=torch.float64))

    def forward(self,x,v,r):
        c=self.c; state=rh(c); L=c['L']
        width=self.log_width.exp()
        left=torch.sigmoid(-torch.as_tensor(L,device=x.device)/width)
        right=torch.sigmoid(torch.as_tensor(L,device=x.device)/width)
        s=(torch.sigmoid(x/width)-left)/(right-left)
        ml=(2*math.pi)**-1.5*torch.exp(-((v-state['u1'])**2+r*r)/2)
        mr=state['rho2']/(2*math.pi*state['T2'])**1.5*torch.exp(-((v-state['u2'])**2+r*r)/(2*state['T2']))
        base=(1-s)*ml+s*mr
        features=torch.stack((x/L,torch.tanh(x/c['initial_width']),v/math.sqrt(state['T2']),r*r/state['T2'],s),-1)
        correction=c['psi_max']*torch.tanh(self.net(features).squeeze(-1)/c['psi_max'])
        gate=incoming_gate(x,v,c)
        z=gate*correction
        return base*(1+z if c['representation']=='signed' else torch.exp(z)),base


def profiles(model,x,grid):
    v,r,w=grid
    xx=x[:,None].expand(-1,len(v))
    f,_=model(xx,v.expand_as(xx),r.expand_as(xx))
    return integrate(f,v,r,w)


def equation(model,x,grid,create_graph=True):
    v,r,w=grid
    xx=x[:,None].expand(-1,len(v)).clone().requires_grad_(True)
    f,base=model(xx,v.expand_as(xx),r.expand_as(xx))
    fx=torch.autograd.grad(f.sum(),xx,create_graph=create_graph)[0]
    S,tau,a=target(f,v,r,w,model.c)
    residual=v*fx-(S-f)/tau[:,None]
    return residual,f,base,S,a


def prescribed_envelope(x,v,r,c):
    """Fixed analytic RH mixture; independent of every trainable parameter."""
    state=rh(c); L=c['L']; width=c['initial_width']
    left=1/(1+math.exp(L/width));right=1/(1+math.exp(-L/width))
    s=(torch.sigmoid(x[:,None]/width)-left)/(right-left)
    ml=(2*math.pi)**-1.5*torch.exp(-((v-state['u1'])**2+r*r)/2)
    mr=state['rho2']/(2*math.pi*state['T2'])**1.5*torch.exp(-((v-state['u2'])**2+r*r)/(2*state['T2']))
    return (1-s)*ml+s*mr


def loss_terms(model,x,grid,create_graph=True):
    c=model.c; state=rh(c); v,r,w=grid
    residual,f,base,S,a=equation(model,x,grid,create_graph)
    rho,u,T,q,sigma=a[:,:5].unbind(1)
    # Frozen analytic RH scales, not observed profile amplitudes.
    rs=state['rho2']; ts=state['T2']; vs=math.sqrt(ts)
    fixed=prescribed_envelope(x,v,r,c)
    floor=fixed.amax(1,keepdim=True)*1e-7
    env=fixed+floor
    mass=(fixed*w).sum(1)
    relative=residual/env
    kinetic=((relative**2*fixed*w).sum(1)/mass).mean()
    cx=v-u[:,None]; e=cx**2+r**2
    # c8 weights expose relative residual in high-order tails; base retains integrability.
    tail_weight=1+(e.detach()/(5*ts))**4
    tail=((relative**2*fixed*w*tail_weight).sum(1)/(fixed*w*tail_weight).sum(1)).mean()
    raw=((residual**2*w).sum(1)/(rs*rs)).mean()
    flux=torch.stack((rho*u/state['u1'],(rho*u*u+rho*T+sigma)/(state['u1']**2+1),
                      (.5*rho*u**3+2.5*rho*T*u+u*sigma+q)/(.5*state['u1']**3+2.5*state['u1'])),1)
    # Polynomials project the same kinetic equation, including its streaming term.
    kernels=[torch.ones_like(cx),v.expand_as(cx)/vs,(v*v+r*r).expand_as(cx)/ts,
             (cx*cx-e/3)/ts,.5*cx*e/(ts*vs),
             (e-7*T[:,None])*(cx*cx-e/3)/(ts*ts),e**2/(ts*ts)]
    projected=torch.stack([(residual*w*k).sum(1)/rs for k in kernels],1)
    terms={'kinetic':kinetic,'tail':tail,'raw':raw,'flux':((flux-1)**2).mean(),
           'invariant':(projected[:,:3]**2).mean(),'stress':(projected[:,3]**2).mean(),
           'heat':(projected[:,4]**2).mean(),'R_projection':(projected[:,5]**2).mean(),
           'Delta_projection':(projected[:,6]**2).mean()}
    pin=profiles(model,torch.zeros(1,device=x.device),grid)[0,0]
    terms['phase']=((pin-(1+rs)/2)/((1+rs)/2))**2
    endpoints=torch.tensor([-c['L'],c['L']],device=x.device,dtype=x.dtype)
    xx=endpoints[:,None].expand(-1,len(v))
    boundary,_=model(xx,v.expand_as(xx),r.expand_as(xx))
    exact=maxwellian(v,r,torch.tensor([1.,rs],device=x.device),
                      torch.tensor([state['u1'],state['u2']],device=x.device),
                      torch.tensor([1.,ts],device=x.device))
    be=exact+exact.amax(1,keepdim=True)*1e-7
    terms['boundary']=((((boundary-exact)/be)**2*exact*w).sum(1)/(exact*w).sum(1)).mean()
    with torch.no_grad():
        mismatch=(boundary-exact)/be
        incoming=torch.stack(((v>=0),(v<0)))
        outgoing=~incoming
        collision=(S-f)*rho[:,None]/(c['mu1']*T[:,None]**(c['omega']-1))
        collision_invariants=torch.stack(((collision*w).sum(1)/rs,
                         (collision*w*v).sum(1)/(rs*vs),
                         (collision*w*(v*v+r*r)).sum(1)/(rs*ts)),1)
        metrics={'flux_max':float((flux-1).abs().max()),
                 'projection_max':float(projected.abs().max()),
                 'f_negative_mass':float((torch.clamp(-f,min=0)*w).sum(1).div((f.abs()*w).sum(1)).max()),
                 'target_negative_mass':float((torch.clamp(-S,min=0)*w).sum(1).div((S.abs()*w).sum(1)).max()),
                 'rho_min':float(rho.min()),'T_min':float(T.min()),
                 'incoming_max':float(mismatch[incoming].abs().max()),
                 'outgoing_left_rms':float(((mismatch[0,outgoing[0]]**2*exact[0,outgoing[0]]*w[outgoing[0]]).sum()/(exact[0,outgoing[0]]*w[outgoing[0]]).sum()).sqrt()),
                 'outgoing_right_rms':float(((mismatch[1,outgoing[1]]**2*exact[1,outgoing[1]]*w[outgoing[1]]).sum()/(exact[1,outgoing[1]]*w[outgoing[1]]).sum()).sqrt()),
                 'collision_invariant_defect_max':float(collision_invariants.abs().max())}
    return terms,metrics


def weights(c,step):
    out=dict(c['weights'])
    ramp=.2+.8*min(step/1000,1.)
    for name in ('kinetic','tail','raw','invariant','stress','heat','R_projection','Delta_projection'):
        out[name]*=ramp
    return out


def train_step(model,opt,c,grid,step,gradient_report=False):
    device=grid[0].device; B=c['batch']
    x=(2*torch.rand(B,device=device)-1)*c['L']
    # Shock-focused locations depend on declared RH length scale, never a reference.
    x[:B//2]=(2*torch.rand(B//2,device=device)-1)*c['L']/3
    # Explicitly resolve an outgoing relaxation layer at either boundary.
    if B>=4:
        sides=torch.where(torch.rand(2,device=device)>.5,1.,-1.)
        x[-2:]=sides*(c['L']-torch.rand(2,device=device)*c['L']/6)
    opt.zero_grad(set_to_none=True)
    terms,metrics=loss_terms(model,x,grid)
    ww=weights(c,step)
    weighted={k:ww[k]*value for k,value in terms.items()}
    total=sum(weighted.values())
    grads={}
    if gradient_report:
        for k,value in weighted.items():
            g=torch.autograd.grad(value,tuple(model.parameters()),retain_graph=True,allow_unused=True)
            grads[k]=math.sqrt(sum(float((z.detach()**2).sum()) for z in g if z is not None))
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite loss')
    total.backward()
    norm=nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'])
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Nonfinite gradient')
    opt.step()
    return dict(loss=float(total.detach()),raw_terms={k:float(v.detach()) for k,v in terms.items()},
                weighted_terms={k:float(v.detach()) for k,v in weighted.items()},
                gradient_norm_before_clip=float(norm),component_gradient_norms=grads,**metrics)


def atomic_checkpoint(path,model,opt,step,signature):
    path=Path(path); tmp=path.with_suffix('.tmp')
    state=dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,signature=signature,
               python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    torch.save(state,tmp)
    with tmp.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(tmp,path)


def restore(path,model,opt,signature):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['signature']==signature,'Checkpoint/source/config mismatch'
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
    random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng'])
    if torch.cuda.is_available() and ck['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
    return ck['step']


def validation(model,grid,n=None):
    c=model.c
    # Fixed uniform grid plus deterministic off-grid points independent of training RNG.
    x=torch.linspace(-c['L'],c['L'],n or c['validation_n'],device=grid[0].device)
    count=n or c['validation_n']
    offsets=((torch.arange(count,device=x.device,dtype=x.dtype)*.6180339887498949+.137)%1)*2*c['L']-c['L']
    x=torch.cat((x,offsets)).sort().values
    rows=[]
    for batch in x.split(2):
        terms,metrics=loss_terms(model,batch,grid,create_graph=False)
        rows.append((len(batch),{k:float(v.detach()) for k,v in terms.items()},metrics))
    total=sum(a for a,_,_ in rows)
    result={'terms':{k:sum(n*z[k] for n,z,_ in rows)/total for k in rows[0][1]},
            'metrics':{k:(min(z[k] for _,_,z in rows) if k.endswith('_min') else max(z[k] for _,_,z in rows)) for k in rows[0][2]}}
    return result


def save_profiles(model,grid,out):
    c=model.c; device=grid[0].device
    x=torch.linspace(-c['L'],c['L'],c['profile_n'],device=device)
    with torch.no_grad():
        pred=torch.cat([profiles(model,z,grid) for z in x.split(4)]).cpu().numpy()
        refined=quadrature(c['refine_nvx'],c['refine_nr'],c['vmax'],device)
        fine=torch.cat([profiles(model,z,refined) for z in x.split(2)]).cpu().numpy()
        extended_nvx=2*math.ceil(c['refine_nvx']*1.125/2)
        extended=quadrature(extended_nvx,math.ceil(c['refine_nr']*1.125),c['vmax']*1.125,device)
        extension=torch.cat([profiles(model,z,extended) for z in x.split(2)]).cpu().numpy()
    # Outputs only. No data archive is ever read by training.
    np.savez(Path(out)/'physics_profiles.npz',x=x.cpu().numpy(),fields=pred,refined_fields=fine,extended_velocity_fields=extension,
             columns=np.array(['rho','u','T','q','sigma','R','Delta','S']))
    scale=np.array([rh(c)['rho2'],rh(c)['u1'],rh(c)['T2'],rh(c)['rho2']*rh(c)['T2']**1.5,
                    rh(c)['rho2']*rh(c)['T2'],rh(c)['rho2']*rh(c)['T2']**2,
                    rh(c)['rho2']*rh(c)['T2']**2,rh(c)['rho2']*rh(c)['T2']**2])
    return dict(normalized_refinement_max=np.max(abs(fine-pred)/scale,axis=0).tolist(),
                normalized_velocity_extension_max=np.max(abs(extension-fine)/scale,axis=0).tolist(),
                velocity_extension_vmax=c['vmax']*1.125,
                velocity_extension_nvx=extended_nvx,
                velocity_extension_nr=math.ceil(c['refine_nr']*1.125))


def acceptance(c,val,refinement):
    limits=c['acceptance']; m=val['metrics']; t=val['terms']
    checks={'flux':m['flux_max']<limits['flux_max'],
            'projection':m['projection_max']<limits['projection_max'],
            'kinetic':t['kinetic']<limits['kinetic'],
            'tail':t['tail']<limits['tail'],
            'phase':t['phase']<limits['phase'],
            'quadrature':max(refinement['normalized_refinement_max'])<limits['quadrature'],
            'positive_macros':m['rho_min']>0 and m['T_min']>0}
    return {'physics_checks':checks,'local_physics_gate_pass':all(checks.values()),
            'CONVERGED':False,
            'endpoint_equilibrium_diagnostic':t['boundary'],
            'interpretation':'Necessary diagnostics only; domain independence and an independent solution check remain required.'}


def run(c,out,wall_seconds=None):
    global STOP
    STOP=False;validate_config(c)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    seed_all(c['seed']);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=Distribution(c).to(device);opt=torch.optim.Adam(model.parameters(),lr=c['lr'])
    grid=quadrature(c['nvx'],c['nr'],c['vmax'],device)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    protocol=Path(__file__).with_name('protocol.json')
    protocol_hash=digest(protocol) if protocol.exists() else None
    signature=hashlib.sha256((json.dumps(c,sort_keys=True)+digest(__file__)+str(protocol_hash)).encode()).hexdigest()
    cp=out/'checkpoint.pt';step=restore(cp,model,opt,signature) if cp.exists() else 0
    write_json(out/'config.json',c)
    write_json(out/'information_ledger.json',dict(training_information='RH states, Shakhov equation, boundary states, density phase gauge, analytic scales only',
               external_profiles=False,learned_initialization=False,signature=signature,protocol_sha256=protocol_hash))
    with (out/'hardware.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(job=os.getenv('SLURM_JOB_ID'),device=device,torch=torch.__version__,
                     gpu=torch.cuda.get_device_name() if device=='cuda' else None,start_step=step))+'\n')
    start=time.monotonic();budget=wall_seconds if wall_seconds is not None else c['wall_seconds']
    while step<c['steps'] and not STOP and time.monotonic()-start<budget:
        fraction=step/max(c['steps'],1)
        lr=c['lr']*(1 if fraction<.5 else .3 if fraction<.8 else .1)
        for group in opt.param_groups: group['lr']=lr
        record=train_step(model,opt,c,grid,step,step%c['gradient_every']==0)
        step+=1
        if step%c['log_every']==0 or step==1 or (step-1)%c['gradient_every']==0:
            record.update(step=step,lr=lr,seconds=time.monotonic()-start)
            with (out/'history.jsonl').open('a') as handle: handle.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if step%c['checkpoint_every']==0: atomic_checkpoint(cp,model,opt,step,signature)
        if step%c['validation_every']==0:
            check=validation(model,grid);check['step']=step
            with (out/'validation.jsonl').open('a') as handle:handle.write(json.dumps(check)+'\n')
    atomic_checkpoint(cp,model,opt,step,signature)
    val=validation(model,grid);refinement=save_profiles(model,grid,out)
    status=dict(step=step,finished=step>=c['steps'],signature=signature,validation=val,
                refinement=refinement,acceptance=acceptance(c,val,refinement),checkpoint_sha256=digest(cp),
                peak_gpu_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0)
    write_json(out/'STATUS.json',status)
    if status['finished']:write_json(out/'DONE.json',status)
    return status


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    ap.add_argument('--out',required=True);ap.add_argument('--wall-seconds',type=int);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();c=json.loads(Path(args.config).read_text())
    if args.smoke:c.update(nvx=40,nr=16,width=12,depth=2,batch=2,steps=2,validation_n=3,profile_n=5,refine_nvx=48,refine_nr=20)
    signal.signal(signal.SIGTERM,stop)
    if hasattr(signal,'SIGUSR1'):signal.signal(signal.SIGUSR1,stop)
    result=run(c,args.out,args.wall_seconds)
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['finished'] else 75)


if __name__=='__main__':main()
