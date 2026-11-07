from pathlib import Path
import torch,numpy as np,json,time,hashlib,importlib.util
p=Path('/native');out=Path('/analysis');start=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.set_default_dtype(torch.float32)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
inputs={n:sha(p/n) for n in ['legacy.py','CONFIG.json','INPUT.pt','OUTPUT_LM_PEAK_PHYSICS/STATE.pt']}
spec=importlib.util.spec_from_file_location('legacy',p/'legacy.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cfg=json.loads((p/'CONFIG.json').read_text());v,w,vw=m.build_vgrid(cfg,torch.device('cpu'),torch.float32);nv=len(v)
g=torch.Generator().manual_seed(981176);tx=torch.stack([torch.rand(32,generator=g)*.02,torch.cat([torch.linspace(-.5,.5,16),torch.linspace(-.12,.14,16)])],1)
results={}
for name,cp in [('before','INPUT.pt'),('after','OUTPUT_LM_PEAK_PHYSICS/STATE.pt')]:
 dt=torch.float64;model=m.PositiveHMM(cfg).to(dtype=dt);model.load_state_dict(torch.load(p/cp,map_location='cpu',weights_only=False)['model'],strict=True);model.eval();rows=[]
 for i in range(0,len(tx),2):
  assert time.monotonic()-start<105
  t=tx[i:i+2].to(dtype=dt);n=len(t);tt=t[:,None,:].expand(n,nv,2).reshape(-1,2).clone().requires_grad_(True);vv=v.to(dtype=dt)[None,:,:].expand(n,nv,3).reshape(-1,3)
  f,M,*_=model.f_flat(tt,vv);grad=torch.autograd.grad(f.sum(),tt,create_graph=False)[0];transport=(grad[:,0]+vv[:,0]*grad[:,1]).reshape(n,nv).double();f=f.detach().reshape(n,nv).double();M=M.detach().reshape(n,nv).double();vd=v.double();wd=w.double()[None,:]
  rho=(f*wd).sum(1);u=(f*vd[None,:,0]*wd).sum(1)/rho;e=(vd[None,:,0]-u[:,None])**2+vd[None,:,1]**2+vd[None,:,2]**2;T=(f*e*wd).sum(1)/(3*rho)
  Mf=m.maxwellian(rho[:,None],u[:,None],T[:,None],vd)
  kernels=torch.stack([torch.ones(nv),v[:,0],.5*(v*v).sum(1)],1).double()
  coll=(M-f)/cfg['case']['kn'];legacy=transport-coll;physical=transport-(Mf-f)/cfg['case']['kn']
  for j in range(n):
   integ=lambda a:((a[:,None]*kernels)*w.double()[:,None]).sum(0).tolist()
   metric=lambda r:float((r*r*w.double()*vw.double()).sum()/(w.double()*vw.double()).sum())
   rows.append({'t':float(t[j,0]),'x':float(t[j,1]),'legacy_collision_invariants':integ(coll[j]),'moment_equilibrium_collision_invariants':integ((Mf[j]-f[j])/cfg['case']['kn']),'transport_invariant_residuals':integ(transport[j]),'legacy_residual_mse':metric(legacy[j]),'moment_equilibrium_residual_mse':metric(physical[j]),'rho':float(rho[j]),'T':float(T[j])})
 arr=lambda key:np.array([r[key] for r in rows]);results[name]={'samples':rows,'max_abs_legacy_collision_invariants':abs(arr('legacy_collision_invariants')).max(0).tolist(),'rms_transport_invariant_residuals':np.sqrt((arr('transport_invariant_residuals')**2).mean(0)).tolist(),'max_abs_moment_equilibrium_collision_invariants':abs(arr('moment_equilibrium_collision_invariants')).max(0).tolist(),'mean_legacy_residual_mse':float(arr('legacy_residual_mse').mean()),'mean_moment_equilibrium_residual_mse':float(arr('moment_equilibrium_residual_mse').mean())}
for n,h in inputs.items():assert sha(p/n)==h
r={'status':'COMPLETE_SAVED_DIAGNOSTIC','inputs':inputs,'optimizer_updates':0,'CPU_threads':2,'elapsed':time.monotonic()-start,'scope':'32 identical independent space-time points; improved-start FP64 and updated FP64 forward/derivatives, original grid values, FP64 integration; analytic Maxwellian from integrated moments has finite-quadrature defects, not exact discrete conservative projection','results':results}
(out/'RESULT.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:{a:b for a,b in r.items() if a!='samples'} for k,r in results.items()}))
