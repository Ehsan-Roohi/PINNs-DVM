"""Mach7 protected-moment method port: analytic RH8 background, fixed u,T/flux.
Four degree3/4 nonconserved modes after five raw moment constraints; 12 splines.
Positive RH carrier with fixed exp(-e/16) damping bounds polynomial tail changes.
No DVM inputs. Primary projection defines a function held fixed on refined grids.
"""
from pathlib import Path
import os,sys,math,hashlib,copy
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W/'probe-runtime-2.10'))
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import torch as t,numpy as np
P=Path(__file__).resolve().parent/'physics_source';sys.path.insert(0,str(P/'bgk_pr1_tau1_M8'));import base_solver as base
from m7_complete9_metric import features as complete9_features
t.set_default_dtype(t.float64);t.set_num_threads(4)
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
SOURCE=W/'M8_M7_WEIGHT_WARMSTART_20261030/STEP0.pt'
SOURCE_SHA='c96a056246c5767f74a24fa3440bd43a2bc6c20f72c239cedaa5982f12289c9f'
assert sha(SOURCE)==SOURCE_SHA
state=t.load(SOURCE,map_location='cpu',weights_only=False);cfg=copy.deepcopy(state['model_config']);rh=base.rh(cfg);J=rh['u1'];Pi=J*J+1;energy=.5*J**3+2.5*J;rs=rh['rho2'];Ts=rh['T2']
# Only the own Mach7 learned background width is reused, in physical u1*tau units.
width=float(state['model']['log_width'].exp())*8/7
PRIMARY=base.quadrature(512,144,48.,'cpu')
def background(x,v,r):
 left=t.sigmoid(t.as_tensor(-cfg['L']/width));right=t.sigmoid(t.as_tensor(cfg['L']/width));s=(t.sigmoid(x/width)-left)/(right-left)
 ml=base.maxwellian(v,r,t.tensor([1.]),t.tensor([J]),t.tensor([1.]))[0];mr=base.maxwellian(v,r,t.tensor([rs]),t.tensor([rh['u2']]),t.tensor([Ts]))[0]
 sx=t.sigmoid(x/width)*(1-t.sigmoid(x/width))/width/(right-left)
 return (1-s)*ml+s*mr,sx*(mr-ml)
def polynomial(v,r,a):
 z=(v-a[1])/a[2].sqrt();s=r*r/a[2];e=z*z+s
 phi=t.stack((t.ones_like(z),z,(e-3)/math.sqrt(6),z*z-e/3,.5*z*e),1)
 psi=t.stack((z**3-.6*z*e,(e-7)*(z*z-e/3),e*e-15,z**4-6/7*z*z*e+3/35*e*e),1)
 return phi,psi,e
def projection(v,r,w,a,f):
 phi,psi,e=polynomial(v,r,a);d=f*t.exp(-e/16);G=phi.T@((w*d)[:,None]*phi);return t.linalg.solve(G,phi.T@((w*d)[:,None]*psi))
def kernel(v,r,a,f,A):
 phi,psi,e=polynomial(v,r,a);return (f*t.exp(-e/16))[:,None]*(psi-phi@A)
def spatial(x):
 xi=t.tanh(x/J);interior=t.linspace(-1,1,10)[1:-1];knots=t.cat((t.full((4,),-1.),interior,t.full((4,),1.)))
 b=[((xi>=knots[i])&(xi<knots[i+1])).to(x.dtype)for i in range(15)]
 for p in range(1,4):
  nxt=[]
  for i in range(15-p):
   z=t.zeros_like(x)
   if float(knots[i+p]-knots[i])>0:z+=(xi-knots[i])/(knots[i+p]-knots[i])*b[i]
   if float(knots[i+p+1]-knots[i+1])>0:z+=(knots[i+p+1]-xi)/(knots[i+p+1]-knots[i+1])*b[i+1]
   nxt.append(z)
  b=nxt
 plateau=35*8/7;edge=80*8/7;s=(x.abs()-plateau)/(edge-plateau);fade=1-10*s**3+15*s**4-6*s**5;E=t.where(x.abs()<=plateau,t.ones_like(x),t.where(x.abs()>=edge,t.zeros_like(x),fade))
 return E*t.stack(b)
def context(x,derivative=True):
 x=t.as_tensor(float(x));v,r,w=PRIMARY;f,fx=background(x,v,r);a,ax=t.autograd.functional.jvp(lambda q:base.integrate(q[None],v,r,w)[0],f,fx)
 if derivative:
  A,Ax=t.autograd.functional.jvp(lambda z:projection(v,r,w,a+z*ax,f+z*fx),t.tensor(0.),t.tensor(1.));B,Bx=t.autograd.functional.jvp(spatial,x,t.tensor(1.))
 else:A=projection(v,r,w,a,f);Ax=t.zeros_like(A);B=spatial(x);Bx=t.zeros_like(B)
 assert a[0]>0 and a[2]>.99 and t.isfinite(A).all()
 return dict(x=x,a=a,ax=ax,A=A,Ax=Ax,B=B,Bx=Bx,f=f,fx=fx)
def basis(ctx,grid):
 v,r,w=grid;f,fx=background(ctx['x'],v,r);a,ax,A,Ax=[ctx[k]for k in ['a','ax','A','Ax']]
 K,Kx=t.autograd.functional.jvp(lambda z:kernel(v,r,a+z*ax,f+z*fx,A+z*Ax),t.tensor(0.),t.tensor(1.));return f,fx,K,Kx
def affine(ctx,grid,coef):
 f,fx,K,Kx=basis(ctx,grid);C=t.as_tensor(coef).reshape(4,12);beta=C@ctx['B'];betax=C@ctx['Bx'];return f+K@beta,fx+Kx@beta+K@betax
def diagnostics(x,f,fx,grid):
 v,r,w=grid;a=base.integrate(f[None],v,r,w)[0];rho,u,T,q,sigma=a[:5];M=base.maxwellian(v,r,a[0:1],a[1:2],a[2:3])[0];R=v*fx+f-M;fixed=base.prescribed_envelope(t.tensor([float(x)]),v,r,cfg)[0];env=fixed+fixed.max()*1e-7;tw=1+(((v-u)**2+r*r)/(5*Ts))**4;cx=v-u;e=cx*cx+r*r
 kernels=t.stack((t.ones_like(v),v/Ts**.5,(v*v+r*r)/Ts,(cx*cx-e/3)/Ts,.5*cx*e/Ts**1.5,(e-7*T)*(cx*cx-e/3)/Ts**2,e*e/Ts**2),1);pr=kernels.T@(w*R)/rs
 flux=t.stack(((f*w*v).sum()/J,(f*w*v*v).sum()/Pi,(f*w*.5*v*(v*v+r*r)).sum()/energy))-1
 terms=t.stack((((R/env)**2*fixed*w).sum()/(fixed*w).sum(),((R/env)**2*fixed*w*tw).sum()/(fixed*w*tw).sum(),(R*R*w).sum()/rs**2,pr[3]**2,pr[4]**2,pr[5]**2,pr[6]**2))
 return dict(fields=a,R=R,terms=terms,flux=flux,negative_mass=(t.clamp(-f,min=0)*w).sum()/(f.abs()*w).sum(),full9=complete9_features(v,r,u,Ts).T@(w*R)/rs,projected=pr)
def velocity_quadratic_to_spatial(H,g,B,Bx):
 b=t.stack((B,Bx));full=t.einsum('aibj,ak,bl->ikjl',H.reshape(2,4,2,4),b,b).reshape(48,48);gg=t.einsum('ai,ak->ik',g.reshape(2,4),b).reshape(48);return full,gg
def bernstein_positive_map(ctx):
 # In each local (z,s=r^2/T) cell, exp(-e/16)<=exp(-min_e/16).
 # Bounding every polynomial Bernstein control below -exp(min_e/16) is sufficient.
 Z=(48+J)/math.sqrt(.99);S=48**2/.99;ze=[-Z,-16.,-12.,-8.,-4.,0.,4.,8.,12.,16.,Z];se=[0.,16.,64.,144.,256.,S]
 def matrix(n):
  u=np.linspace(0,1,n+1);return np.array([[math.comb(n,j)*z**j*(1-z)**(n-j)for j in range(n+1)]for z in u])
 inv4=t.tensor(np.linalg.inv(matrix(4)));inv2=t.tensor(np.linalg.inv(matrix(2)));rows=[];A=ctx['A']
 for z0,z1 in zip(ze[:-1],ze[1:]):
  for s0,s1 in zip(se[:-1],se[1:]):
   zz,ss=t.meshgrid(t.linspace(z0,z1,5),t.linspace(s0,s1,3),indexing='ij');v=ctx['a'][1]+ctx['a'][2].sqrt()*zz.reshape(-1);r=(ctx['a'][2]*ss.reshape(-1)).sqrt();phi,psi,_=polynomial(v,r,ctx['a']);values=(psi-phi@A).reshape(5,3,4);controls=t.einsum('ij,jlk,ml->imk',inv4,values,inv2).reshape(15,4)
   minz=0. if z0<=0<=z1 else min(abs(z0),abs(z1));maximum=math.exp(-(minz**2+s0)/16);rows.append((maximum*controls[:,:,None]*ctx['B'][None,None,:]).reshape(15,48))
 return t.cat(rows).numpy()
