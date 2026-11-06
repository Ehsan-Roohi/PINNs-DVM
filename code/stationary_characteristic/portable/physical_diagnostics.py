import torch as t
import physical_base as base
from moment_features import features as complete9_features

def diagnostics(x,f,fx,grid):
 v,r,w=grid;a=base.integrate(f[None],v,r,w)[0];rho,u,T,q,sigma=a[:5];M=base.maxwellian(v,r,a[0:1],a[1:2],a[2:3])[0];R=v*fx+f-M;fixed=base.prescribed_envelope(t.tensor([float(x)]),v,r,cfg)[0];env=fixed+fixed.max()*1e-7;tw=1+(((v-u)**2+r*r)/(5*Ts))**4;cx=v-u;e=cx*cx+r*r
 kernels=t.stack((t.ones_like(v),v/Ts**.5,(v*v+r*r)/Ts,(cx*cx-e/3)/Ts,.5*cx*e/Ts**1.5,(e-7*T)*(cx*cx-e/3)/Ts**2,e*e/Ts**2),1);pr=kernels.T@(w*R)/rs
 flux=t.stack(((f*w*v).sum()/J,(f*w*v*v).sum()/Pi,(f*w*.5*v*(v*v+r*r)).sum()/energy))-1
 terms=t.stack((((R/env)**2*fixed*w).sum()/(fixed*w).sum(),((R/env)**2*fixed*w*tw).sum()/(fixed*w*tw).sum(),(R*R*w).sum()/rs**2,pr[3]**2,pr[4]**2,pr[5]**2,pr[6]**2))
 return dict(fields=a,R=R,terms=terms,flux=flux,negative_mass=(t.clamp(-f,min=0)*w).sum()/(f.abs()*w).sum(),full9=complete9_features(v,r,u,Ts).T@(w*R)/rs,projected=pr)
