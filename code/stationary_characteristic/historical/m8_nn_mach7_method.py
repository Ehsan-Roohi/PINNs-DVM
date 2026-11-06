"""Faithful Mach7 macro transport + six protected kinetic modes on actual NN8.
Domain-aware compact support: full RH repair except one u1*tau boundary layer.
No DVM, no copied optimizer, no alteration of the original source field.
"""
from m8_protected_moment_port import *
from m7_transport_kinetic_basis import feature as feature6
model=base.Distribution(copy.deepcopy(cfg));model.load_state_dict(state['model']);model.requires_grad_(False);model.eval()
def envelope(x):
 L=cfg['L'];s=(x.abs()-(L-J))/J;fade=1-10*s**3+15*s**4-6*s**5;return t.where(x.abs()<=L-J,t.ones_like(x),t.where(x.abs()>=L,t.zeros_like(x),fade))
def spatial(x):
 xi=t.tanh(x/J);interior=t.linspace(-1,1,10)[1:-1];knots=t.cat((t.full((4,),-1.),interior,t.full((4,),1.)));b=[((xi>=knots[i])&(xi<knots[i+1])).to(x.dtype)for i in range(15)]
 for p in range(1,4):
  nxt=[]
  for i in range(15-p):
   z=t.zeros_like(x)
   if float(knots[i+p]-knots[i])>0:z+=(xi-knots[i])/(knots[i+p]-knots[i])*b[i]
   if float(knots[i+p+1]-knots[i+1])>0:z+=(knots[i+p+1]-xi)/(knots[i+p+1]-knots[i+1])*b[i+1]
   nxt.append(z)
  b=nxt
 return envelope(x)*t.stack(b)
def raw_source(x,v,r):
 xx=t.full((1,len(v)),float(x),requires_grad=True);f,_=model(xx,v[None],r[None]);fx=t.autograd.grad(f.sum(),xx)[0][0];return f.detach()[0],fx.detach()
def macro_parameters(x,a):
 rho,u,T,q,sigma=a[:5];E=envelope(x);ua=u+E*(J/rho-u);pressure=rho*T+sigma;heat=ua*(2.5*rho*T+sigma);res=t.stack(((rho*ua*ua+pressure)/Pi-1,(.5*rho*ua**3+heat+q)/energy-1));der=t.stack((2*pressure/Pi,(2*heat+3*q)/energy));k=1-E*(res@der)/(der@der);return t.stack((ua,k))
def transported(x,v,r,a0,a0x,p,px):
 ua,k=p;uax,kx=px;assert k>0
 xx=t.full((1,len(v)),float(x),requires_grad=True);vv=(a0[1]+(v-ua)/k)[None].clone().requires_grad_(True);rr=(r/k)[None].clone().requires_grad_(True);f,b=model(xx,vv,rr);fx,fv,fr=t.autograd.grad(f.sum(),(xx,vv,rr),retain_graph=True);bx,bv,br=t.autograd.grad(b.sum(),(xx,vv,rr));vx=a0x[1]-uax/k-(v-ua)*kx/k**2;rx=-r*kx/k**2;fa=f/k**3;ba=b/k**3;fax=(fx+fv*vx+fr*rx)/k**3-3*kx/k*fa;bax=(bx+bv*vx+br*rx)/k**3-3*kx/k*ba;return fa.detach()[0],fax.detach()[0],ba.detach()[0],bax.detach()[0]
def projection6(v,r,w,a,carrier):
 phi,psi=feature6(v,r,a);e=((v-a[1])**2+r*r)/a[2];d=carrier*t.exp(-e/16);G=phi.T@((w*d)[:,None]*phi);return t.linalg.solve(G,phi.T@((w*d)[:,None]*psi))
def kernel6(v,r,a,carrier,A):
 phi,psi=feature6(v,r,a);e=((v-a[1])**2+r*r)/a[2];return (carrier*t.exp(-e/16))[:,None]*(psi-phi@A)
def context(x,derivative=True):
 x=t.as_tensor(float(x));v,r,w=PRIMARY;f0,fx0=raw_source(x,v,r);a0,a0x=t.autograd.functional.jvp(lambda f:base.integrate(f[None],v,r,w)[0],f0,fx0);p,px=t.autograd.functional.jvp(lambda z:macro_parameters(x+z,a0+z*a0x),t.tensor(0.),t.tensor(1.));f,fx,d,dx=transported(x,v,r,a0,a0x,p,px);a,ax=t.autograd.functional.jvp(lambda q:base.integrate(q[None],v,r,w)[0],f,fx)
 if derivative:A,Ax=t.autograd.functional.jvp(lambda z:projection6(v,r,w,a+z*ax,d+z*dx),t.tensor(0.),t.tensor(1.));B,Bx=t.autograd.functional.jvp(spatial,x,t.tensor(1.))
 else:A=projection6(v,r,w,a,d);Ax=t.zeros_like(A);B=spatial(x);Bx=t.zeros_like(B)
 return dict(x=x,a0=a0,a0x=a0x,p=p,px=px,f0=f0,fx0=fx0,a=a,ax=ax,f=f,fx=fx,carrier=d,carrier_x=dx,A=A,Ax=Ax,B=B,Bx=Bx)
def basis(ctx,grid):
 v,r,w=grid;f,fx,d,dx=transported(ctx['x'],v,r,ctx['a0'],ctx['a0x'],ctx['p'],ctx['px']);a,ax,A,Ax=[ctx[k]for k in ['a','ax','A','Ax']];K,Kx=t.autograd.functional.jvp(lambda z:kernel6(v,r,a+z*ax,d+z*dx,A+z*Ax),t.tensor(0.),t.tensor(1.));return f,fx,K,Kx
def affine(ctx,grid,coef):
 f,fx,K,Kx=basis(ctx,grid);C=t.as_tensor(coef).reshape(6,12);beta=C@ctx['B'];betax=C@ctx['Bx'];return f+K@beta,fx+Kx@beta+K@betax
def quadratic(H,g,B,Bx):
 b=t.stack((B,Bx));return t.einsum('aibj,ak,bl->ikjl',H.reshape(2,6,2,6),b,b).reshape(72,72),t.einsum('ai,ak->ik',g.reshape(2,6),b).reshape(72)
def positivity_sample_map(ctx):
 v1,r1=t.meshgrid(t.linspace(-48,48,33),t.linspace(0,48,17),indexing='ij');z,rr=t.meshgrid(t.tensor([-8.,-6.,-4.,-2.,0.,2.,4.,6.,8.]),t.tensor([0.,1.,2.,4.,6.,8.]),indexing='ij');v2=ctx['a'][1]+ctx['a'][2].sqrt()*z;r2=ctx['a'][2].sqrt()*rr;keep=(v2.abs()<=48)&(r2<=48);v=t.cat((v1.reshape(-1),v2[keep]));r=t.cat((r1.reshape(-1),r2[keep]));f,_,d,_=transported(ctx['x'],v,r,ctx['a0'],ctx['a0x'],ctx['p'],ctx['px']);phi,psi=feature6(v,r,ctx['a']);e=((v-ctx['a'][1])**2+r*r)/ctx['a'][2];ratio=f/d;pol=t.exp(-e/16)[:,None]*(psi-phi@ctx['A']);return -(pol[:,:,None]*ctx['B'][None,None,:]).reshape(-1,72).numpy(),-ratio.numpy()
