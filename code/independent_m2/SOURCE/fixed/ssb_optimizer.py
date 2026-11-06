"""Dense FP64 self-scaled Broyden with verified strong-Wolfe acceptance.

The self-scaling coefficients follow the user's cavity implementation. Dense
storage avoids its accumulated-generator scaling. No gradient modification,
resampling or objective definitions belong in this module.
"""
import math

import torch


class UpdateRejected(FloatingPointError):
    pass


def ssb_scalars(P, ys, ykHkyk, sg, alpha, eps=1e-30):
    if P<=1 or not all(math.isfinite(z) for z in (ys,ykHkyk,sg,alpha)):
        raise UpdateRejected('Invalid coefficient inputs')
    if ys<=eps or ykHkyk<=eps or sg>=0 or alpha<=0:
        raise UpdateRejected('Nonpositive curvature or invalid descent step')
    rho=1./ys
    hk=ykHkyk*rho
    bk=-alpha*rho*sg
    ak=bk*hk-1.
    denominator=(1.+ak) or eps
    rad=max(abs(ak)/denominator,0.)
    rhom=min(1.,hk*(1.-math.sqrt(rad)))
    theta_min=0. if abs(ak)<eps else (rhom-1.)/ak
    theta_max=1./rhom if abs(rhom)>eps else 1./eps
    inner=theta_max if abs(bk)<eps else (1.-bk)/bk
    theta=max(theta_min,min(theta_max,inner))
    rhok=min(1.,1./bk if abs(bk)>eps else 1./eps)
    sigma=1.+theta*ak
    sigma_power=abs(sigma)**(1./(1.-float(P))) if abs(sigma)>0 else 0.
    tau=(min(rhok*sigma_power,sigma) if theta<=0 else rhok*min(sigma_power,1./theta))
    phi=(1.-theta)/((1.+ak*theta) or eps)
    if not all(math.isfinite(z) for z in (rho,phi,tau)) or tau<=eps:
        raise UpdateRejected('Invalid self-scaling coefficients')
    return rho,phi,tau


def dense_update(H,s,y,g,alpha):
    """Hnew=(H-uu'/q+phi*q*vv')/tau+rho*ss', u=Hy, v=rho*s-u/q."""
    ys=float(torch.dot(y,s));u=H@y;q=float(torch.dot(y,u))
    rho,phi,tau=ssb_scalars(s.numel(),ys,q,float(torch.dot(s,g)),float(alpha))
    v=rho*s-u/q
    new=H.clone()
    new.addmm_(u[:,None],u[None,:],beta=1.,alpha=-1./q)
    new.addmm_(v[:,None],v[None,:],beta=1.,alpha=phi*q)
    new.div_(tau)
    new.addmm_(s[:,None],s[None,:],beta=1.,alpha=rho)
    new=(new+new.T)*.5
    if not bool(torch.isfinite(new).all()):
        raise UpdateRejected('Nonfinite dense inverse Hessian')
    # The exact update satisfies Hnew*y=s. Check the computed matrix as well.
    secant=float(torch.linalg.vector_norm(new@y-s)/max(float(torch.linalg.vector_norm(s)),1e-30))
    if not math.isfinite(secant) or secant>1e-6:
        raise UpdateRejected('Dense update lost secant accuracy')
    return new,dict(rho=rho,phi=phi,tau=tau,ys=ys,yHy=q,secant_relative=secant)


class DenseSSB:
    def __init__(self,n,device='cpu',dtype=torch.float64,c1=1e-4,c2=.9,
                 max_evals=40,max_step=1.,initial_step=1.,gtol=1e-12):
        assert n>1 and dtype==torch.float64
        assert 0<c1<c2<1 and max_evals>=2 and max_step>0 and initial_step>0
        self.n=int(n);self.device=torch.device(device);self.dtype=dtype
        self.c1=float(c1);self.c2=float(c2);self.max_evals=int(max_evals)
        self.max_step=float(max_step);self.initial_step=float(initial_step);self.gtol=float(gtol)
        self.H=torch.eye(self.n,device=self.device,dtype=dtype)
        self.iteration=0;self.total_evaluations=0;self.rejections=0

    def state_dict(self):
        return dict(version=1,n=self.n,H=self.H.detach().cpu().clone(),iteration=self.iteration,
                    total_evaluations=self.total_evaluations,rejections=self.rejections,
                    settings={k:getattr(self,k) for k in ('c1','c2','max_evals','max_step','initial_step','gtol')})

    def load_state_dict(self,state):
        assert state['version']==1 and state['n']==self.n
        assert state['settings']=={k:getattr(self,k) for k in state['settings']},'Optimizer settings mismatch'
        H=state['H'].to(device=self.device,dtype=self.dtype).clone()
        assert H.shape==(self.n,self.n) and bool(torch.isfinite(H).all())
        self.H=H;self.iteration=int(state['iteration'])
        self.total_evaluations=int(state['total_evaluations']);self.rejections=int(state['rejections'])

    def step(self,x,eval_fg):
        """Attempt one coherent step; callback must set parameters to its argument.

        On rejection eval_fg(x) restores model parameters. Caller must inspect
        accepted/reason; there is no silent fallback or curvature reset.
        """
        x=x.detach().clone().to(device=self.device,dtype=self.dtype)
        assert x.shape==(self.n,) and bool(torch.isfinite(x).all())
        used=0

        def evaluate(point):
            nonlocal used
            used+=1;self.total_evaluations+=1
            try:
                value,gradient=eval_fg(point)
            except FloatingPointError:
                return math.inf,torch.full_like(point,math.nan)
            value=float(value)
            gradient=gradient.detach().clone().to(device=self.device,dtype=self.dtype)
            if gradient.shape!=x.shape:
                raise ValueError('Flat gradient has wrong shape')
            if not math.isfinite(value) or not bool(torch.isfinite(gradient).all()):
                return math.inf,torch.full_like(point,math.nan)
            return value,gradient

        f0,g0=evaluate(x)

        def reject(reason):
            self.rejections+=1
            # Restore callback-owned model even after a failed trial evaluation.
            restored_f,restored_g=evaluate(x)
            if not math.isfinite(restored_f):
                raise FloatingPointError('Cannot restore a finite base objective')
            return dict(accepted=False,x=x,loss=restored_f,gradient=restored_g,alpha=0.,
                        reason=reason,evaluations=used,wolfe=False,curvature=None)

        if not math.isfinite(f0):
            return reject('nonfinite_initial_objective')
        if float(torch.linalg.vector_norm(g0))<=self.gtol:
            return dict(accepted=False,x=x,loss=f0,gradient=g0,alpha=0.,reason='gradient_tolerance',
                        evaluations=used,wolfe=False,curvature=None)
        p=-(self.H@g0);d0=float(torch.dot(g0,p))
        if not bool(torch.isfinite(p).all()) or not math.isfinite(d0) or d0>=0:
            return reject('non_descent_direction')

        def trial(alpha):
            f,g=evaluate(x+alpha*p)
            d=float(torch.dot(g,p)) if math.isfinite(f) else math.nan
            return f,g,d

        def wolfe(alpha,f,d):
            return (math.isfinite(f) and math.isfinite(d) and
                    f<=f0+self.c1*alpha*d0 and abs(d)<=-self.c2*d0)

        def zoom(lo,hi,flo):
            while used<self.max_evals:
                alpha=.5*(lo+hi)
                if alpha==lo or alpha==hi:
                    return None
                f,g,d=trial(alpha)
                if not math.isfinite(f) or f>f0+self.c1*alpha*d0 or f>=flo:
                    hi=alpha
                else:
                    if wolfe(alpha,f,d):return alpha,f,g
                    if not math.isfinite(d):return None
                    if d*(hi-lo)>=0:hi=lo
                    lo=alpha;flo=f
            return None

        previous=0.;fprevious=f0;alpha=min(self.initial_step,self.max_step);candidate=None
        while used<self.max_evals:
            f,g,d=trial(alpha)
            if not math.isfinite(f) or f>f0+self.c1*alpha*d0 or (previous>0 and f>=fprevious):
                candidate=zoom(previous,alpha,fprevious);break
            if wolfe(alpha,f,d):candidate=(alpha,f,g);break
            if not math.isfinite(d):break
            if d>=0:
                candidate=zoom(alpha,previous,f);break
            next_alpha=min(2*alpha,self.max_step)
            if next_alpha==alpha:break
            previous=alpha;fprevious=f;alpha=next_alpha
        if candidate is None:
            return reject('strong_wolfe_search_failed')
        alpha,f1,g1=candidate;x1=x+alpha*p
        if not wolfe(alpha,f1,float(torch.dot(g1,p))):
            return reject('strong_wolfe_verification_failed')
        try:
            newH,curvature=dense_update(self.H,x1-x,g1-g0,g0,alpha)
        except (UpdateRejected,OverflowError,ZeroDivisionError) as error:
            return reject('curvature_update_rejected: '+str(error))
        self.H=newH;self.iteration+=1
        return dict(accepted=True,x=x1,loss=f1,gradient=g1,alpha=alpha,reason='accepted',
                    evaluations=used,wolfe=True,curvature=curvature)
