"""Data-free transient BGK/Shakhov integral representation.
No field targets. Shakhov target is signed, never clipped. Discrete rectangular
velocity weights match historical S0; finite-grid equilibrium defects are retained.
"""
import math
import numpy as np
import torch
from torch import nn

class CharacteristicPINN(nn.Module):
 def __init__(self,collision='bgk',hidden=32,depth=2,time_nodes=16,grid=(32,12,12),vmax=7.,kn=1.,tf=.02,dtype=torch.float64):
  super().__init__();assert collision in ('bgk','shakhov');assert kn>0 and tf>0
  self.collision=collision;self.pr=1. if collision=='bgk' else 2/3;self.kn=float(kn);self.tf=float(tf);self.time_nodes=int(time_nodes)
  layers=[];n=2
  for _ in range(depth):layers += [nn.Linear(n,hidden),nn.Tanh()];n=hidden
  layers += [nn.Linear(n,4 if collision=='shakhov' else 3)];self.net=nn.Sequential(*layers).to(dtype=dtype)
  nn.init.zeros_(self.net[-1].bias)
  # Small independent initialization retains meaningful seed differences.
  with torch.no_grad():self.net[-1].weight.mul_(.05)
  for name,N in zip(('vx','vy','vz'),grid):
   assert N>=2;self.register_buffer(name,torch.linspace(-vmax,vmax,N,dtype=dtype));self.register_buffer('w'+name,torch.full((N,),2*vmax/(N-1),dtype=dtype))
  z,w=np.polynomial.legendre.leggauss(time_nodes);self.register_buffer('tau',torch.tensor((z+1)/2,dtype=dtype));self.register_buffer('wt',torch.tensor(w/2,dtype=dtype))
 @staticmethod
 def initial(x):
  H=.5*(1+torch.tanh(20*x));return {'rho':1-.875*H,'u':torch.zeros_like(x),'T':1-.2*H,'q':torch.zeros_like(x)}
 def macros(self,tx):
  shape=tx.shape[:-1];t,x=tx[...,0],tx[...,1];base=self.initial(x);raw=self.net(torch.stack([t/self.tf,2*x],-1).reshape(-1,2)).reshape(*shape,-1);a=t/self.tf
  return {'rho':base['rho']*torch.exp(.5*a*raw[...,0]),'u':a*raw[...,1], 'T':base['T']*torch.exp(.5*a*raw[...,2]),'q':a*base['rho']*base['T']**1.5*raw[...,3] if self.collision=='shakhov' else torch.zeros_like(t)}
 def paths(self,tx,vx):
  t=tx[:,0,None];x=tx[:,1,None];v=vx[None,:]
  safe=torch.where(v!=0,v,torch.ones_like(v));hit=torch.where(v>0,(x+.5)/safe,torch.where(v<0,(x-.5)/safe,torch.full_like(x+v,float('inf'))))
  ell=torch.minimum(t,hit).clamp_min(0);initial=hit>=t;xi=x-v*ell
  b=self.initial(xi)
  for key,L,R in [('rho',1.,.125),('u',0.,0.),('T',1.,.8),('q',0.,0.)]:b[key]=torch.where(initial,b[key],torch.where(v>0,torch.full_like(xi,L),torch.full_like(xi,R)))
  s=t[...,None]-ell[...,None]*(1-self.tau);xs=x[...,None]-v[...,None]*(t[...,None]-s)
  return ell,b,torch.stack([s,xs],-1)
 def _transverse(self,mac,vx,higher=False):
  rho,u,T,q=(mac[k] for k in ('rho','u','T','q'));cx=vx-u
  def sums(v,w):
   g=torch.exp(-v*v/(2*T[...,None]))/torch.sqrt(2*math.pi*T[...,None]);return [(g*w*v.pow(2*j)).sum(-1) for j in range(4 if higher else 3)]
  y=sums(self.vy,self.wvy);z=sums(self.vz,self.wvz);Z0=y[0]*z[0];Z1=y[1]*z[0]+y[0]*z[1];Z2=y[2]*z[0]+2*y[1]*z[1]+y[0]*z[2]
  gx=rho*torch.exp(-cx*cx/(2*T))/torch.sqrt(2*math.pi*T);A=(1-self.pr)*q*cx/(5*rho*T*T)
  F0=gx*(Z0+A*((cx*cx/T-5)*Z0+Z1/T));Fr=gx*(Z1+A*((cx*cx/T-5)*Z1+Z2/T))
  parts=[F0,vx*F0,vx*vx*F0,Fr,vx**3*F0,vx*Fr]
  if higher:
   Z3=y[3]*z[0]+3*y[2]*z[1]+3*y[1]*z[2]+y[0]*z[3]
   Fr2=gx*(Z2+A*((cx*cx/T-5)*Z2+Z3/T))
   parts += [vx**4*F0,vx*vx*Fr,Fr2]
  return torch.stack(parts,-1)
 def _raw_integrals(self,tx,higher=False):
  ell,b,st=self.paths(tx,self.vx);v=self.vx[None,:];base=self._transverse(b,v,higher);src=self._transverse(self.macros(st),v[...,None],higher);decay=torch.exp(-ell/self.kn)
  weight=ell[...,None]/self.kn*self.wt*torch.exp(-(tx[:,0,None,None]-st[...,0])/self.kn)
  integrated=decay[...,None]*base+(weight[...,None]*src).sum(-2);r=(integrated*self.wvx[None,:,None]).sum(1)
  return r
 def moments(self,tx):
  r=self._raw_integrals(tx)
  rho,px,vx2,rad,vx3,vxr=r.unbind(-1);u=px/rho;T=(vx2+rad-px*u)/(3*rho);q=.5*(vx3+vxr-u*(3*vx2+rad)+2*rho*u**3)
  return dict(rho=rho,px=px,energy=.5*(vx2+rad),u=u,T=T,q=q,raw=r)
 def higher_moments(self,tx):
  r=self._raw_integrals(tx,higher=True)
  rho,px,vx2,rad,vx3,vxr,vx4,vx2r,rad2=r.unbind(-1);u=px/rho
  cx2=vx2-rho*u*u;cx3=vx3-3*u*vx2+2*rho*u**3
  cx4=vx4-4*u*vx3+6*u*u*vx2-3*rho*u**4
  cxrad=vxr-u*rad;cx2rad=vx2r-2*u*vxr+u*u*rad
  T=(cx2+rad)/(3*rho);q=.5*(cx3+cxrad);sigma=cx2-(cx2+rad)/3
  trace4=cx4+2*cx2rad+rad2;contracted=cx4+cx2rad
  m=cx3-1.2*q;R=contracted-trace4/3-7*T*sigma;Delta=trace4-15*rho*T*T
  return dict(rho=rho,px=px,u=u,T=T,q=q,sigma=sigma,m=m,R=R,Delta=Delta,S=R+Delta/3,raw9=r)
 def self_consistency(self,tx):
  m=self.macros(tx);a=self.moments(tx);r={'rho':a['rho']-m['rho'],'px':a['px']-m['rho']*m['u'],'energy':a['energy']-.5*m['rho']*(m['u']**2+3*m['T'])}
  if self.collision=='shakhov':r['q']=a['q']-m['q']
  return r
 def _target(self,mac,v):
  rho,u,T,q=(mac[k] for k in ('rho','u','T','q'));cx=v[...,0]-u;e=cx*cx+v[...,1]**2+v[...,2]**2;M=rho/(2*math.pi*T)**1.5*torch.exp(-e/(2*T))
  return M*(1+(1-self.pr)*q*cx/(5*rho*T*T)*(e/T-5))
 def distribution(self,tx,velocities,chunk=128):
  """Actual f, signed for Shakhov. Endpoint caller chunks tx separately."""
  result=[]
  for v in velocities.split(chunk):
   ell,b,st=self.paths(tx,v[:,0]);base=self._target(b,v[None,:,:]);src=self._target(self.macros(st),v[None,:,None,:]);w=ell[...,None]/self.kn*self.wt*torch.exp(-(tx[:,0,None,None]-st[...,0])/self.kn)
   result.append(torch.exp(-ell/self.kn)*base+(w*src).sum(-1))
  return torch.cat(result,1)
 def velocity_grid(self):
  v=torch.stack(torch.meshgrid(self.vx,self.vy,self.vz,indexing='ij'),-1).reshape(-1,3);w=(self.wvx[:,None,None]*self.wvy[None,:,None]*self.wvz[None,None,:]).reshape(-1);return v,w
