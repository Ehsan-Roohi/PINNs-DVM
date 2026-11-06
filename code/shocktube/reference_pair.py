"""Separate post-solve FP64 DVM. Never imported/read by PINN training.
Production: --collision bgk|shakhov --nx 800|1600 --grid 48 20 20 --device cuda.
--nx20 requires --test-only; no positivity clipping or convergence certification.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,json,time,signal,hashlib,math
from pathlib import Path
import numpy as np
import torch
STOP=False
def stop(*_):
 global STOP;STOP=True
for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,stop)
if hasattr(signal,'SIGUSR1'):signal.signal(signal.SIGUSR1,stop)
def atomic(p,**a):
 q=p.with_suffix('.partial')
 with q.open('wb') as f:np.savez(f,**{k:v.detach().cpu().numpy() if torch.is_tensor(v) else v for k,v in a.items()});f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def moments(f,v,w):
 fw=f*w;rho=fw.sum(1);u=(fw@v)/rho[:,None];c=v[None]-u[:,None];e=(c*c).sum(2);T=(fw*e).sum(1)/(3*rho)
 if not bool(torch.isfinite(f).all()) or bool((rho<=0).any()) or bool((T<=0).any()):raise FloatingPointError('Invalid state; no clipping')
 q=.5*(fw*c[:,:,0]*e).sum(1);sig=(fw*(c[:,:,0]**2-e/3)).sum(1)
 return rho,u,T,q,sig,c,e
def equilibrium(rho,u,T,v):return rho[:,None]/(2*math.pi*T[:,None])**1.5*torch.exp(-((v[None]-u[:,None])**2).sum(2)/(2*T[:,None]))
def source(rr,uu,tt,qq,c,e,v,Pr):return equilibrium(rr,uu,tt,v)*(1+(1-Pr)*qq[:,None]*c[:,:,0]/(5*rr[:,None]*tt[:,None]**2)*(e/tt[:,None]-5))
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--collision',choices=['bgk','shakhov'],required=True);p.add_argument('--nx',type=int,choices=[20,400,800,1600,3200],required=True);p.add_argument('--grid',type=int,nargs=3,default=[32,12,12]);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--chunk',type=int,choices=[64,128],default=64);p.add_argument('--seconds',type=float,default=110);p.add_argument('--resume',action='store_true');p.add_argument('--test-only',action='store_true');a=p.parse_args();assert a.nx!=20 or a.test_only
 torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False);dev=torch.device(a.device);dtype=torch.float64;start=time.monotonic();out=Path(a.out)
 cfg={'collision':a.collision,'Pr':1. if a.collision=='bgk' else 2/3,'Kn':1.,'nx':a.nx,'grid':a.grid,'vmax':7.,'x':[-.5,.5],'tf':.02,'smooth_k':20.,'dtype':'float64','scheme':'upwind then exponential relaxation to frozen post-transport analytic source','CFL':.4,'device':a.device,'chunk':a.chunk,'test_only':a.test_only,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
 if a.resume:assert out.is_dir() and not (out/'COMPLETE.json').exists()
 else:out.mkdir(parents=True,exist_ok=False)
 lock=out/'WRITER.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 try:
  axes=[torch.linspace(-7,7,n,device=dev,dtype=dtype)for n in a.grid];v=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3);w=torch.full((len(v),),math.prod([14/(n-1)for n in a.grid]),device=dev,dtype=dtype);dx=1/a.nx;x=-.5+(torch.arange(a.nx,device=dev,dtype=dtype)+.5)*dx;nsteps=math.ceil(.02/(.4*dx/7));dt=.02/nsteps
  h=.5*(1+torch.tanh(20*x));rho=1-.875*h;T=1-.2*h;f=torch.empty((a.nx,len(v)),device=dev,dtype=dtype)
  for i in range(0,a.nx,a.chunk):f[i:i+a.chunk]=equilibrium(rho[i:i+a.chunk],torch.zeros((len(rho[i:i+a.chunk]),3),device=dev,dtype=dtype),T[i:i+a.chunk],v)
  def boundary(r,t):return equilibrium(torch.tensor([r],device=dev,dtype=dtype),torch.zeros((1,3),device=dev,dtype=dtype),torch.tensor([t],device=dev,dtype=dtype),v)[0]
  ML=boundary(1.,1.);MR=boundary(.125,.8);step=0;elapsed=0.;maxdef=0.;mintarget=float('inf')
  if a.resume:
   z=np.load(out/'CHECKPOINT.npz',allow_pickle=False);assert str(z['signature'])==signature;f=torch.as_tensor(z['f'],device=dev,dtype=dtype);step=int(z['step']);elapsed=float(z['elapsed']);maxdef=float(z['max_collision_invariant_defect']);mintarget=float(z['min_target'])
  (out/'CONFIG.json').write_text(json.dumps(cfg,indent=2));cn=dt*v[:,0]/dx;pos=v[:,0]>0;E=math.exp(-dt)
  def checkpoint():atomic(out/'CHECKPOINT.npz',f=f,step=np.array(step),elapsed=np.array(elapsed+time.monotonic()-start),signature=np.array(signature),max_collision_invariant_defect=np.array(maxdef),min_target=np.array(mintarget))
  while step<nsteps and not STOP and elapsed+time.monotonic()-start<a.seconds-15:
   fs=torch.empty_like(f)
   # Chunked conservative upwind update avoids full-grid temporary vector distributions.
   for i in range(0,a.nx,a.chunk):
    j=min(i+a.chunk,a.nx);ff=f[i:j];left=torch.cat([ML[None] if i==0 else f[i-1:i],f[i:j-1]],0);right=torch.cat([f[i+1:j],MR[None] if j==a.nx else f[j:j+1]],0);fs[i:j]=ff-cn*torch.where(pos[None],ff-left,right-ff)
   for i in range(0,a.nx,a.chunk):
    sl=slice(i,i+a.chunk);rr,uu,tt,qq,ss,c,e=moments(fs[sl],v,w);target=source(rr,uu,tt,qq,c,e,v,cfg['Pr']);mintarget=min(mintarget,float(target.min()));delta=(target-fs[sl])*w;invariants=torch.cat([delta.sum(1)[:,None],delta@v,(delta@(v*v).sum(1))[:,None]],1);maxdef=max(maxdef,float(invariants.abs().max()));f[sl]=E*fs[sl]+(1-E)*target
   step+=1
   if step%50==0:checkpoint()
  checkpoint();rows=[]
  for i in range(0,a.nx,a.chunk):
   ff=f[i:i+a.chunk];rr,uu,tt,qq,ss,c,e=moments(ff,v,w);fw=ff*w;cx=c[:,:,0];m=(fw*(cx**3-.6*cx*e)).sum(1);R=(fw*(e-7*tt[:,None])*(cx**2-e/3)).sum(1);D=(fw*e*e).sum(1)-15*rr*tt*tt;rows.append(torch.stack([rr,uu[:,0],tt,qq,ss,m,R,D,R+D/3],1).cpu())
  atomic(out/'FIELDS.npz',x=x,fields=torch.cat(rows),columns=np.array(['rho','u','T','q','sigma','mxxx','R_closure','Delta','S']));atomic(out/'FULLSTATE.npz',x=x,f=f,vgrid=v,qweight=w,step=np.array(step),dt=np.array(dt),nsteps=np.array(nsteps))
  result={'completed':step==nsteps,'test_only':a.test_only,'step':step,'nsteps':nsteps,'dt':dt,'elapsed':elapsed+time.monotonic()-start,'f_min':float(f.min()),'negative_count':int((f<0).sum()),'negative_mass':float((torch.clamp(-f,min=0)*w).sum()*dx),'min_target':mintarget,'max_collision_invariant_defect':maxdef,'convergence_certified':False,'no_clipping_of_solution':True,'post_solve_only':True,'signature':signature,'peak_cuda_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else 0,'files':{q.name:hashlib.sha256(q.read_bytes()).hexdigest()for q in out.glob('*.npz')}};(out/('COMPLETE.json' if step==nsteps else 'INTERRUPTED.json')).write_text(json.dumps(result,indent=2))
 finally:os.close(fd);lock.unlink()
if __name__=='__main__':main()
