"""Positive shared-trunk eight-moment extension; no reference fields are read at inference."""
import math,torch
from torch import nn
import legacy
class AllMomentHMM(legacy.PositiveHMM):
 def __init__(self,cfg):
  super().__init__(cfg)
  self.higher_head=nn.Linear(self.net.net[-1].in_features,3)
  with torch.no_grad():self.higher_head.weight.zero_();self.higher_head.bias.zero_()
  self.temporal_head=nn.Linear(self.net.net[-1].in_features,8)
  with torch.no_grad():self.temporal_head.weight.zero_();self.temporal_head.bias.zero_()
  self.higher_amp=.20;self.higher_bound=.35;self.higher_width=.22
 def load_parent(self,state):
  old={k:v for k,v in self.state_dict().items() if not k.startswith(('higher_head.','temporal_head.'))};assert set(old)==set(state)
  self.load_state_dict(state,strict=False)
  with torch.no_grad():self.higher_head.weight.zero_();self.higher_head.bias.zero_();self.temporal_head.weight.zero_();self.temporal_head.bias.zero_()
 def coefficients_from_raw(self,tx,raw,high):
  t=tx[:,0];x=tx[:,1];tf=float(self.cfg['case']['t_final']);tau=torch.clamp(t/tf,0.,1.5)
  rho0,ux0,T0=legacy.case_state_torch(self.cfg,x)
  rho=rho0*torch.exp(.45*tau*torch.tanh(raw[:,0]));ux=ux0+.42*tau*torch.tanh(raw[:,1]);T=T0*torch.exp(.35*tau*torch.tanh(raw[:,2]))
  gate=torch.exp(-((x)/self.gate_width)**4)
  qcoef=tau*gate*self.q_amp*torch.tanh(raw[:,3]);scoef=tau*gate*self.s_amp*torch.tanh(raw[:,4])
  lo=float(self.cfg['case']['x_min']);hi=float(self.cfg['case']['x_max']);xi=2*(x-(lo+hi)/2)/(hi-lo);boundary=(1-xi*xi)**2
  higher=torch.exp(-(x/self.higher_width)**4)*boundary*self.higher_amp
  return rho,ux,T,qcoef,scoef,gate,tau**3*higher*torch.tanh(high[:,0]),tau**2*higher*torch.tanh(high[:,1]),tau**2*higher*torch.tanh(high[:,2])
 def coefficients(self,tx):
  hidden=self.features(tx)
  for layer in list(self.net.net)[:-1]:hidden=layer(hidden)
  raw=self.net.net[-1](hidden);high=self.higher_head(hidden)
  tau=tx[:,0]/float(self.cfg["case"]["t_final"]);x=tx[:,1];lo=float(self.cfg["case"]["x_min"]);hi=float(self.cfg["case"]["x_max"]);xi=2*(x-(lo+hi)/2)/(hi-lo);gate=(1-tau)*(1-xi*xi)**2;correction=gate[:,None]*self.temporal_head(hidden)
  raw=raw+correction[:,:5];high=high+correction[:,5:]
  return self.coefficients_from_raw(tx,raw,high)
 def forward_moments(self,tx):return self.coefficients(tx)[:6]
 def distribution(self,tx,v,coeff):
  rho,ux,T,qcoef,scoef,gate,mc,rc,dc=coeff;vx,vy,vz=v[:,0],v[:,1],v[:,2];T=torch.clamp(T,min=1e-6);sqrtT=torch.sqrt(T)
  cx=(vx-ux)/sqrtT;cy=vy/sqrtT;cz=vz/sqrtT;c2=cx*cx+cy*cy+cz*cz
  phi_q=((.5*c2-2.5)*cx)/8.;phi_s=(cx*cx-c2/3.)/4.
  oldh=qcoef*phi_q+scoef*phi_s;oldpsi=self.psi_max*torch.tanh(oldh/self.psi_max)
  phi_m=(cx**3-.6*cx*c2)/math.sqrt(12/5);phi_R=(c2-7)*(cx*cx-c2/3)/math.sqrt(56/3)
  phi_D=(c2*c2-10*c2+15)/math.sqrt(120)
  highh=mc*phi_m+rc*phi_R+dc*phi_D;psi=oldpsi+self.higher_bound*torch.tanh(highh/self.higher_bound)
  M=legacy.maxwellian(rho,ux,T,v);fac=torch.exp(psi);f=M*fac
  return f,M,fac,rho,ux,T,qcoef,scoef
 def f_flat(self,tx,v):return self.distribution(tx,v,self.coefficients(tx))
 def grid_from_coefficients(self,C,v):
  n=len(C);nv=len(v);cf=[C[:,k,None].expand(n,nv).reshape(-1) for k in range(8)];gate=C.new_ones(n*nv)
  coeff=(cf[0],cf[1],cf[2],cf[3],cf[4],gate,cf[5],cf[6],cf[7])
  out=self.distribution(None,v[None,:,:].expand(n,nv,3).reshape(-1,3),coeff)
  return out[0].reshape(n,nv),out[1].reshape(n,nv)
 def grid(self,tx,v):
  coeff=self.coefficients(tx);C=torch.stack([coeff[k]for k in [0,1,2,3,4,6,7,8]],1)
  return self.grid_from_coefficients(C,v)
 def grid_with_derivatives(self,tx,v,create_graph=True):
  if not tx.requires_grad:tx=tx.clone().requires_grad_(True)
  coeff=self.coefficients(tx);C=torch.stack([coeff[k]for k in [0,1,2,3,4,6,7,8]],1)
  derivatives=torch.stack([torch.autograd.grad(C[:,k].sum(),tx,create_graph=create_graph,retain_graph=True)[0]for k in range(8)],1)
  if not create_graph:C=C.detach();derivatives=derivatives.detach()
  fun=lambda z:self.grid_from_coefficients(z,v)[0]
  f,ft=torch.autograd.functional.jvp(fun,C,derivatives[:,:,0],create_graph=create_graph)
  _,fx=torch.autograd.functional.jvp(fun,C,derivatives[:,:,1],create_graph=create_graph)
  M=self.grid_from_coefficients(C,v)[1]
  return f,ft,fx,M
 def f_from_raw(self,tx,v,z):return self.distribution(tx,v,self.coefficients_from_raw(tx,z[:,:5],z[:,5:]))[0]
def moments_all(f,v,w,kn):
 fw=f*w;rho=fw.sum(1);vel=(fw@v)/rho[:,None];c=v[None,:,:]-vel[:,None,:];e=(c*c).sum(2);cx=c[:,:,0];T=(fw*e).sum(1)/(3*rho);q=.5*(fw*cx*e).sum(1);dev=cx*cx-e/3;sig=(fw*dev).sum(1)
 M=rho[:,None]*torch.exp(-e/(2*T[:,None]))/(2*math.pi*T[:,None])**1.5;dfw=(f-M)*w
 m=(dfw*(cx**3-.6*cx*e)).sum(1);R=(dfw*e*dev).sum(1)-7*T*sig
 delta=(fw*e*e).sum(1)-15*rho*T*T
 ER=(M*w*e*dev).sum(1)
 return torch.stack([rho,vel[:,0],T,q,sig,m,R,-sig/kn,-2*q/kn,delta,R+ER+delta/3],1)
def profiles_all(model,cfg,x,v,w):
 tx=torch.stack([x*0+float(cfg['case']['t_final']),x],1)
 f=model.grid(tx,v)[0]
 return moments_all(f,v,w,float(cfg['case']['kn']))
