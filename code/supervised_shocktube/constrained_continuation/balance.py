"""Bounded supervised continuation; fixed endpoint, original source untouched."""
import os,json,time,random,signal,hashlib,importlib.util
from pathlib import Path
import numpy as np
import torch
root=Path('/task'); arm='LM_PHYSICS';out=root/('OUTPUT_'+arm);out.mkdir(exist_ok=False)
start=time.monotonic();stop=False

def sig(*a):
 global stop
 stop=True
for s in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(s,sig)
torch.set_num_threads(2);torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32=False
random.seed(606);np.random.seed(606);torch.manual_seed(606);torch.cuda.manual_seed_all(606)
spec=importlib.util.spec_from_file_location('legacy',root/'legacy.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cfg=json.loads((root/'CONFIG.json').read_text());dev=torch.device('cuda');dtype=torch.float64
model=m.PositiveHMM(cfg).to(device=dev,dtype=dtype);cp=torch.load(root/'INPUT.pt',map_location=dev,weights_only=False);model.load_state_dict(cp['model'],strict=True);assert all(torch.equal(z,cp['model'][n]) for n,z in model.state_dict().items())
v,w,vw=m.build_vgrid(cfg,dev,torch.float32);v=v.double();w=w.double();vw=vw.double();ref=m.load_npz_ref(str(root/'REFERENCE.npz'));x=np.asarray(ref['x'])
# Fixed 42 locations, as in the source configuration; no new peak observations.
pool=np.flatnonzero((x>=cfg['anchor']['x_min'])&(x<=cfg['anchor']['x_max']));ai=pool[np.linspace(0,len(pool)-1,42).round().astype(int)];mi=np.linspace(0,len(x)-1,160).round().astype(int)
truth=np.column_stack([ref[k] for k in ['rho','ux','T','qx','sigma_xx']]);scale=np.max(abs(truth),axis=0);scale[1]+=1e-4;scale[3:]=np.maximum(np.max(abs(truth[ai,3:]),axis=0),1e-3)+.003
algorithm={};opt=None
def checkpoint(step):
 p=out/'STATE.tmp';torch.save({'model':model.state_dict(),'algorithm':algorithm,'optimizer':opt.state_dict() if opt is not None else None,'fresh_projected_stage':True,'step':step,'config':cfg,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'elapsed':time.monotonic()-start},p);os.replace(p,out/'STATE.pt')
def assess(tag):
 rows=[]
 with torch.no_grad():
  for i in range(0,len(x),4):rows.append(torch.stack(m.integrated_profiles(model,cfg,torch.tensor(x[i:i+4],device=dev,dtype=dtype),v,w,dev,dtype),1).cpu().numpy())
 y=np.concatenate(rows);np.savez(out/(tag+'.npz'),x=x,prediction=y,reference=truth,anchor_indices=ai,macro_indices=mi)
 held=np.ones(len(x),bool);held[np.unique(np.r_[ai,mi])]=False
 errors=lambda mask:(np.linalg.norm((y-truth)[mask],axis=0)/np.linalg.norm(truth[mask],axis=0)).tolist()
 # Identical random samples at both endpoints, including native legacy residual.
 rng=torch.get_rng_state();crng=torch.cuda.get_rng_state_all();torch.manual_seed(9106);torch.cuda.manual_seed_all(9106)
 losses=[m.pde_loss(model,cfg,v,w,vw)[:2] for _ in range(2)];pde=np.mean([[z.detach().item() for z in row] for row in losses],axis=0).tolist();torch.set_rng_state(rng);torch.cuda.set_rng_state_all(crng)
 r={'relative_L2':errors(slice(None)),'heldout_relative_L2':errors(held),'legacy_pde_losses':pde,'q_peak':float(y[:,3].max()),'q_reference_peak':float(truth[:,3].max()),'sigma_minmax':[float(y[:,4].min()),float(y[:,4].max())],'sigma_reference_minmax':[float(truth[:,4].min()),float(truth[:,4].max())]};(out/(tag+'.json')).write_text(json.dumps(r));return r

params=list(model.parameters())
def vector():return torch.cat([z.detach().reshape(-1) for z in params])
def assign(a):
 with torch.no_grad():
  pos=0
  for z in params:z.copy_(a[pos:pos+z.numel()].view_as(z));pos+=z.numel()
def components(grad=False):
 vals=np.zeros(12);gs=torch.zeros((12,sum(z.numel() for z in params)),device=dev,dtype=dtype) if grad else None
 def add(loss,k):
  vals[k]+=float(loss.detach())
  if grad:
   gg=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
   gs[k]+=torch.cat([(torch.zeros_like(z) if g is None else g).reshape(-1) for z,g in zip(params,gg)])
 # Physics-only training collocations are distinct from the frozen independent audit.
 gen=torch.Generator().manual_seed(87317)
 tx=torch.stack([torch.rand(32,generator=gen,dtype=dtype)*.02,torch.cat([torch.linspace(-.5,.5,16,dtype=dtype),torch.linspace(-.12,.14,16,dtype=dtype)])],1).to(dev)
 nv=len(v);kern=torch.stack([torch.ones(nv,device=dev),v[:,0],.5*(v*v).sum(1)],1)
 for ii in range(0,len(tx),2):
  t=tx[ii:ii+2];n=len(t);tt=t[:,None,:].expand(n,nv,2).reshape(-1,2).clone().requires_grad_(True);vv=v[None,:,:].expand(n,nv,3).reshape(-1,3)
  f,M,*_=model.f_flat(tt,vv);fg=torch.autograd.grad(f.sum(),tt,create_graph=True)[0];transport=(fg[:,0]+vv[:,0]*fg[:,1]).reshape(n,nv);f=f.reshape(n,nv);M=M.reshape(n,nv)
  legacy=transport-(M-f)/cfg['case']['kn'];den=abs(M)+abs(f)+cfg['train'].get('rel_eps',1e-5);ww=w*vw
  add(((legacy**2*ww).sum(1)/ww.sum()).sum()/32,5)
  add((((legacy/den)**2*ww).sum(1)/ww.sum()).sum()/32,6)
  rho=(f*w).sum(1);u=(f*v[:,0]*w).sum(1)/rho;ee=(v[None,:,0]-u[:,None])**2+v[None,:,1]**2+v[None,:,2]**2;T=(f*ee*w).sum(1)/(3*rho);Mf=m.maxwellian(rho[:,None],u[:,None],T[:,None],v)
  physical=transport-(Mf-f)/cfg['case']['kn'];add(((physical**2*ww).sum(1)/ww.sum()).sum()/32,8)
  balances=(transport[:,:,None]*kern[None,:,:]*w[None,:,None]).sum(1)
  for k in range(3):add((balances[:,k]**2).sum()/32,9+k)
 rr=torch.get_rng_state();cc=torch.cuda.get_rng_state_all();torch.manual_seed(87318);torch.cuda.manual_seed_all(87318)
 add(m.bc_loss(model,cfg,v,w),7);torch.set_rng_state(rr);torch.cuda.set_rng_state_all(cc)
 for ids,cols in [(mi,[0,1,2]),(ai,[3,4])]:
  for i in range(0,len(ids),4):
   ix=ids[i:i+4];pred=torch.stack(m.integrated_profiles(model,cfg,torch.tensor(x[ix],device=dev,dtype=dtype),v,w,dev,dtype),1)
   target=torch.tensor(truth[ix],device=dev);sc=torch.tensor(scale,device=dev)
   for k in cols:add((((pred[:,k]-target[:,k])/sc[k])**2).sum()/len(ids),k)
 assert np.isfinite(vals).all()
 return vals,gs

fields=['rho','u','T','q','sigma','raw_PDE','relative_PDE','boundary','moment_equilibrium_PDE','mass_balance','momentum_balance','energy_balance']
def log(name,record):
 with (out/name).open('a') as f:f.write(json.dumps(record)+'\n');f.flush()

torch.set_default_dtype(dtype);fp32=assess('START_IMPROVED')
random.setstate(cp['python_rng']);np.random.set_state(cp['numpy_rng']);torch.set_rng_state(cp['cpu_rng'].cpu());torch.cuda.set_rng_state_all([z.cpu() for z in cp['cuda_rng']])
before=assess('BEFORE');initial,_=components();original=vector().clone();accepted=0;mu=1e-3;reason='MAX_ATTEMPTS'
algorithm.update(method=arm,accepted=0,initial_losses=initial.tolist(),fresh_optimizer=True,precision='FP64 original grid cast',fixed_physical_seed=42106)
opt=torch.optim.LBFGS(params,lr=1.,max_iter=1,history_size=10,line_search_fn=None)
def residual_jacobian():
 rs=[];js=[]
 for ids,cols in [(mi,[0,1,2]),(ai,[3,4])]:
  for ii in range(0,len(ids),4):
   ix=ids[ii:ii+4];pred=torch.stack(m.integrated_profiles(model,cfg,torch.tensor(x[ix],device=dev,dtype=dtype),v,w,dev,dtype),1);target=torch.tensor(truth[ix],device=dev,dtype=dtype)
   for k in cols:
    weight=(100. if arm=='LM_MACRO' and k<3 else 1.)
    rr=(pred[:,k]-target[:,k])/scale[k]/np.sqrt(len(ids)*initial[k]/weight)
    for val in rr:
     gg=torch.autograd.grad(val,params,retain_graph=True,allow_unused=True)
     rs.append(val.detach());js.append(torch.cat([(torch.zeros_like(z) if g is None else g).reshape(-1) for z,g in zip(params,gg)]))
 # Include each physical component separately in the least-squares direction.
 for k in range(5,12):
  val=np.sqrt(current[k]/initial[k]);rs.append(torch.tensor(val,device=dev));js.append(G[k]/(2*np.sqrt(current[k]*initial[k])))
 return torch.stack(rs),torch.stack(js)
checkpoint(0)
# Independent finite-difference check of the newly differentiated physical losses.
val,gg=components(True);basevec=vector().clone();direction=gg[5]/gg[5].norm();eps=1e-6
assign(basevec+eps*direction);vp,_=components();assign(basevec-eps*direction);vm,_=components();assign(basevec)
fd=(vp-vm)/(2*eps);ad=(gg@direction).cpu().numpy();ids=[5,8,9,10,11];err=abs(fd-ad)/np.maximum(abs(ad),1e-10)
log('GRADIENT_CHECK.jsonl',{'finite_difference':fd.tolist(),'autodiff':ad.tolist(),'relative_error':err.tolist()})
assert max(err[ids])<.005,('Physical derivative check failed',err)
for attempt in range(1,101):
 if stop or time.monotonic()-start>850:reason='TIME_OR_SIGNAL';break
 origin=vector().clone();current,G=components(True)
 if arm.startswith('LM'):
  r,J=residual_jacobian()
  # Normalize dual damping by the observed Jacobian Gram diagonal.
  K=J@J.T;damp=mu*float(torch.diagonal(K).mean());coef=torch.linalg.solve(K+damp*torch.eye(len(r),device=dev),r);d=-J.T@coef
  pred_reduction=float((r*r).sum()-((r+J@d)**2).sum())
  # Project in the full damped Gauss-Newton metric; preserve all six descent conditions.
  U=G/G.norm(dim=1)[:,None]
  invU=(U.T-J.T@torch.linalg.solve(K+damp*torch.eye(len(r),device=dev),J@U.T))/damp
  metric=(U@invU).cpu().numpy();q=(U@d).cpu().numpy()+1e-4*max(float(d.norm()),1e-8);lam=np.zeros(12)
  for projection_it in range(20000):
   prev=lam.copy()
   for k in range(12):lam[k]=max(0.,lam[k]+(q[k]-metric[k]@lam)/metric[k,k])
   if max(abs(lam-prev))<1e-12:break
  d=d-invU@torch.tensor(lam,device=dev)
  log('PROJECTION.jsonl',{'attempt':attempt,'damping':damp,'normalized_slopes':(U@d).cpu().tolist(),'iterations':projection_it})
  del J,K,r,invU
 else:
  def closure():
   opt.zero_grad(set_to_none=True);vv,gg=components(True);total=float((vv/initial).sum());flat=(gg/torch.tensor(initial,device=dev)[:,None]).sum(0);pos=0
   for z in params:z.grad=flat[pos:pos+z.numel()].view_as(z).clone();pos+=z.numel()
   return torch.tensor(total,device=dev)
  opt.step(closure);d=vector()-origin;assign(origin);pred_reduction=None
 dn=float(d.norm())
 if dn>.1:d*=.1/dn
 slope=(G@d).cpu().numpy();ok=False
 log('DIRECTIONS.jsonl',{'attempt':attempt,'method':arm,'mu':mu,'norm_before_cap':dn,'slopes':slope.tolist(),'data_linear_predicted_reduction_before_cap':pred_reduction})
 for back in range(16):
  if stop or time.monotonic()-start>940:break
  alpha=2.**(-back);assign(origin+alpha*d);candidate,_=components();limits=initial.copy();limits[5:]=current[5:]
  failed=[fields[k] for k in range(12) if not candidate[k]<=limits[k]]
  if not np.all(candidate[[5,8,9,10,11]]<current[[5,8,9,10,11]]):failed.append('physics_strict')
  log('TRIALS.jsonl',{'attempt':attempt,'alpha':alpha,'before':current.tolist(),'after':candidate.tolist(),'limits':limits.tolist(),'predicted_change':(alpha*slope).tolist(),'actual_change':(candidate-current).tolist(),'failed_components':failed,'accepted':not failed})
  if not failed:ok=True;break
 if not ok:
  assign(origin)
  if arm.startswith('LM') and mu<1e3:mu*=10;checkpoint(accepted);continue
  reason='LINE_SEARCH_REJECTED';break
 accepted+=1;mu=max(mu/3,1e-8);algorithm.update(accepted=accepted,attempts=attempt,mu=mu,last_losses=candidate.tolist())
 record={'attempt':attempt,'accepted':accepted,'alpha':alpha,'before':current.tolist(),'after':candidate.tolist(),'elapsed':time.monotonic()-start};log('LOSS.jsonl',record);print(json.dumps(record),flush=True);checkpoint(accepted)
checkpoint(accepted);after=assess('AFTER');result={'arm':arm,'status':'FINISHED','stop_reason':reason,'accepted_updates':accepted,'original_fp32':fp32,'before':before,'after':after,'elapsed':time.monotonic()-start,'algorithm':algorithm,'same_reference_and_observations':True,'convergence_claim':False,'acceptance_scope':'all five field training losses bounded by improved start; raw PDE and three balances decrease; independent audit required'}
(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
