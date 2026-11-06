import numpy as np
"""Verbatim AST-extracted mathematical functions; no historical imports or execution."""
def transport(x,a,grid,Cu=None,CT=None):
 if Cu is None and USE_FULL_REPORT:return full_transport(x,a,grid)
 budget();vv,rr,ww=[z.numpy()for z in grid];v,inv=np.unique(vv,return_inverse=True);C=float(cfg['vmax']);wv=np.bincount(inv,weights=ww)/(np.pi*C*C);nx=len(x);jac=Cu is not None
 raw=np.zeros((nx,3));draw=np.zeros((nx,3,Pnum))if jac else None
 def source1(aa,mask):
  vm=v[mask];g=aa[0]/np.sqrt(2*np.pi*aa[2])*np.exp(-(vm-aa[1])**2/(2*aa[2]));return g,2*aa[2]*g
 for direction in [1,-1]:
  mask=v>0 if direction==1 else v<0;vm=v[mask];wm=wv[mask];anchor=0 if direction==1 else nx-1
  fg,fh=source1(np.array([1.,J,1.])if direction==1 else np.array([rs,rh['u2'],Ts]),mask);oldg,oldh=source1(a[anchor],mask)
  if jac:dg=np.zeros((len(vm),Pnum));dh=dg.copy();oldDg=dg.copy();oldDh=dg.copy()
  raw[anchor]+=[wm@fg,(wm*vm)@fg,wm@(vm*vm*fg+fh)]
  for i in (range(1,nx)if direction==1 else range(nx-2,-1,-1)):
   if i%40==0:budget()
   prev=i-direction;q=abs(x[i]-x[prev])/abs(vm);alpha=np.exp(-q);z=-np.expm1(-q)/q;b=z-alpha;c=1-z;g,h=source1(a[i],mask)
   fg=alpha*fg+b*oldg+c*g;fh=alpha*fh+b*oldh+c*h;raw[i]+=[wm@fg,(wm*vm)@fg,wm@(vm*vm*fg+fh)]
   if jac:
    ru=(vm-a[i,1])/a[i,2]-1/a[i,1];rt=-.5/a[i,2]+(vm-a[i,1])**2/(2*a[i,2]**2);Dg=(g*ru)[:,None]*Cu[i]+(g*rt)[:,None]*CT[i];Dh=2*a[i,2]*Dg+2*g[:,None]*CT[i]
    dg=alpha[:,None]*dg+b[:,None]*oldDg+c[:,None]*Dg;dh=alpha[:,None]*dh+b[:,None]*oldDh+c[:,None]*Dh;draw[i,0]+=wm@dg;draw[i,1]+=(wm*vm)@dg;draw[i,2]+=(wm*vm*vm)@dg+wm@dh;oldDg,oldDh=Dg,Dh
   oldg,oldh=g,h
 rho=raw[:,0];u=raw[:,1]/rho;T=(raw[:,2]/rho-u*u)/3;actual=np.c_[rho,u,T];G=(actual-a)/np.array([rs,J,Ts]);Jac=None
 if jac:
  dr=draw[:,0];du=(draw[:,1]-u[:,None]*dr)/rho[:,None];dT=(draw[:,2]-(u*u+3*T)[:,None]*dr-2*(rho*u)[:,None]*du)/(3*rho[:,None]);Jac=np.stack(((dr+J/a[:,1,None]**2*Cu)/rs,(du-Cu)/J,(dT-CT)/Ts),axis=1)
 return None,actual,G,Jac

def weights_space(x):
 h=np.diff(x);w=np.r_[h[0]/2,(h[:-1]+h[1:])/2,h[-1]/2];return w/(x[-1]-x[0])
