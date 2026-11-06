"""Import-safe reduced BGK characteristic engine. No checkpoint/reference readers."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_k]='4'
import argparse,hashlib,json,time,types
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
import historical_math

class ArrayAdapter:
    def __init__(self,a): self.a=np.asarray(a)
    def numpy(self): return self.a

def rh(mach):
    gamma=5/3; J=mach*np.sqrt(gamma)
    r=(gamma+1)*mach**2/((gamma-1)*mach**2+2)
    pressure=(2*gamma*mach**2-(gamma-1))/(gamma+1)
    return J,r,pressure/r

class Engine:
    def __init__(self,cfg,deadline):
        self.cfg=cfg; self.deadline=deadline
        self.J,self.rs,self.Ts=rh(cfg['mach'])
        self.P=2*cfg['basis']-1
        # Match radial-integrated axial weights of cylindrical quadrature.
        z,zw=np.polynomial.legendre.leggauss(cfg['nvx']//2);C=cfg['vmax'];v=np.r_[C*(z-1)/2,C*(z+1)/2];w=np.r_[C*zw/2,C*zw/2]
        if np.any(v==0): raise ValueError('Even axial order required')
        self.grid=[ArrayAdapter(v),ArrayAdapter(np.zeros_like(v)),ArrayAdapter(w*np.pi*cfg['vmax']**2)]
        ns=dict(historical_math.__dict__)
        ns.update(np=np,cfg=cfg,J=self.J,rs=self.rs,Ts=self.Ts,Pnum=self.P,
                  USE_FULL_REPORT=False,budget=self.budget,rh={'u2':self.J/self.rs})
        self.transport=types.FunctionType(historical_math.transport.__code__,ns, argdefs=historical_math.transport.__defaults__)
    def budget(self):
        if time.monotonic()>self.deadline: raise TimeoutError('Fixed CPU budget exhausted')
    def envelope(self,x):
        L=self.cfg['L']; s=(abs(x)-(L-self.J))/self.J
        return np.where(abs(x)<=L-self.J,1.,np.where(abs(x)>=L,0.,1-10*s**3+15*s**4-6*s**5))
    def setup(self,seed_x,seed_fields,x,knots):
        n=self.cfg['basis']; E=self.envelope(x)
        B=BSpline(knots,np.eye(n),3)(np.tanh(x/self.J))*E[:,None]
        b0=BSpline(knots,np.eye(n),3)(np.array([0.]))
        null=np.linalg.svd(b0,full_matrices=True)[2][1:].T
        Cu=np.c_[self.J*(B@null),np.zeros((len(x),n))]
        CT=np.c_[np.zeros((len(x),n-1)),self.Ts*B]
        u0=E*np.interp(x,seed_x,seed_fields[:,1])+(1-E)*np.where(x<0,self.J,self.J/self.rs)
        T0=E*np.interp(x,seed_x,seed_fields[:,2])+(1-E)*np.where(x<0,1.,self.Ts)
        # Seed must already carry analytically defined phase: never fit to reference.
        target=2*self.J/(1+self.rs)
        if abs(np.interp(0.,x,u0)-target)>self.cfg['seed_phase_tolerance']:
            raise ValueError('Own macro seed not admitted at analytic RH phase')
        return u0,T0,Cu,CT
    def profile(self,c,geo):
        u0,T0,Cu,CT=geo;u=u0+Cu@c;T=T0+CT@c
        if np.min(u)<=0 or np.min(T)<=0: raise ValueError('Nonpositive macro proposal')
        return np.c_[self.J/u,u,T]

def run(args):
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if cfg['mode'] not in ('pilot','production_proposal'): raise ValueError('mode')
    if cfg['mode']=='production_proposal': raise ValueError('Production blocked: full native diagnostic adapter and approval review pending')
    assert cfg['max_steps']<=2 and cfg['wall_seconds']<=60
    assert cfg['basis']>=4 and cfg['nvx']%2==0 and cfg['mach'] in (7,8)
    seedpath=Path(args.seed); actual=hashlib.sha256(seedpath.read_bytes()).hexdigest()
    assert actual==cfg['seed_sha256']
    out=Path(args.output);out.mkdir(exist_ok=False)
    start=time.monotonic(); deadline=start+cfg['wall_seconds']; history=[]
    def save(name,z): (out/name).write_text(json.dumps(z,indent=2),encoding='utf-8')
    save('INTENT.json',dict(config=cfg,seed_sha256=actual,no_reference_access=True,neural_updates=0))
    try:
        with np.load(seedpath,allow_pickle=False) as z:
            sx=z['x'].copy();sf=z['fields'].copy()
        x=np.asarray(cfg['x'],float);knots=np.asarray(cfg['knots'],float)
        assert np.all(np.diff(x)>0) and x[0]==-cfg['L'] and x[-1]==cfg['L'] and np.any(x==0)
        assert sf.shape==(len(sx),3) and np.all(np.diff(sx)>0) and np.all(np.isfinite(sf))
        assert sx[0]<=x[0] and sx[-1]>=x[-1]
        e=Engine(cfg,deadline);geo=e.setup(sx,sf,x,knots);c=np.zeros(e.P);sw=historical_math.weights_space(x)
        for step in range(cfg['max_steps']):
            e.budget();_,a,g,j=e.transport(x,e.profile(c,geo),e.grid,geo[2],geo[3]);A=(j*np.sqrt(sw)[:,None,None]).reshape(-1,e.P);r=(g*np.sqrt(sw)[:,None]).ravel();loss=float(r@r)
            d,_,rank,_=np.linalg.lstsq(A,-r,rcond=1e-12)
            if rank!=e.P: raise ValueError('Rank deficient; no ridge fallback')
            trust=max(abs(geo[2]@d).max()/e.J,abs(geo[3]@d).max()/e.Ts)
            if trust<1e-7: break
            d*=min(1.,.005/trust);slope=float((A.T@r)@d);trials=[];accepted=False
            for k in range(8):
                e.budget();alpha=2.**(-k)
                try:ma=e.profile(c+alpha*d,geo)
                except ValueError:trials.append(dict(alpha=alpha,positive=False));continue
                _,_,g2,_=e.transport(x,ma,e.grid);new=float(np.sum(sw[:,None]*g2*g2));trials.append(dict(alpha=alpha,loss=new))
                if new<=loss+1e-4*alpha*slope:c+=alpha*d;accepted=True;break
            history.append(dict(step=step+1,loss_before=loss,loss_after=new if accepted else loss,accepted=accepted,trials=trials))
            save('HISTORY.json',history)
            if not accepted:break
            np.savez(out/f'STATE_{step+1:03d}.npz',coefficients=c,x=x,macro=e.profile(c,geo))
        save('RESULT.json',dict(scope='Reduced software/physics pilot only',history=history,seconds=time.monotonic()-start,full_kinetic_validation_executed=False,convergence_certified=False,neural_updates=0))
    except BaseException as ex:
        save('FAILURE.json',dict(type=type(ex).__name__,message=str(ex),history=history,seconds=time.monotonic()-start));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--seed',required=True);p.add_argument('--output',required=True);run(p.parse_args())
