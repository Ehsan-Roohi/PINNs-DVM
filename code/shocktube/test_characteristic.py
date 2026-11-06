"""Small CPU mathematical admission tests; does not train or read reference fields."""
import json,torch
from characteristic import CharacteristicPINN

def run():
 torch.set_num_threads(2);torch.manual_seed(104);results={}
 for family in ['bgk','shakhov']:
  m=CharacteristicPINN(family,hidden=8,depth=1,grid=(9,7,7),vmax=5,time_nodes=16);tx=torch.tensor([[.012,-.03],[.019,.025]],dtype=torch.float64);v,w=m.velocity_grid()
  f=m.distribution(tx,v);vx=v[:,0];rad=v[:,1]**2+v[:,2]**2
  brute=torch.stack([(f*w).sum(1),(f*w*vx).sum(1),(f*w*vx**2).sum(1),(f*w*rad).sum(1),(f*w*vx**3).sum(1),(f*w*vx*rad).sum(1)],1)
  hm=m.higher_moments(tx);cx=vx[None,:]-hm['u'][:,None];e=cx*cx+rad[None,:];sig=(f*w*(cx*cx-e/3)).sum(1);third=(f*w*(cx**3-.6*cx*e)).sum(1);fourth=(f*w*e*(cx*cx-e/3)).sum(1)-7*hm['T']*sig;delta=(f*w*e**2).sum(1)-15*hm['rho']*hm['T']**2
  for k,z in [('m',third),('R',fourth),('Delta',delta),('S',fourth+delta/3)]:assert torch.allclose(hm[k],z,atol=1e-10,rtol=1e-10),(family,k,float((hm[k]-z).abs().max()))
  fact=m.moments(tx)['raw'];err=float((brute-fact).abs().max());assert torch.allclose(brute,fact,atol=1e-11,rtol=1e-11)
  t0=tx.clone();t0[:,0]=0;expected=m._target(m.initial(t0[:,1,None]),v[None,:,:]);assert torch.allclose(m.distribution(t0,v),expected,atol=1e-14,rtol=1e-14)
  boundary=torch.tensor([[.01,-.5],[.01,.5]],dtype=torch.float64);vv=torch.tensor([[1.,.2,0.],[-1.,.2,0.],[0.,.2,0.]],dtype=torch.float64);fb=m.distribution(boundary,vv)
  L={k:torch.tensor(z,dtype=torch.float64) for k,z in dict(rho=1.,u=0.,T=1.,q=0.).items()};R={k:torch.tensor(z,dtype=torch.float64) for k,z in dict(rho=.125,u=0.,T=.8,q=0.).items()}
  assert torch.allclose(fb[0,0],m._target(L,vv[0]),atol=1e-14);assert torch.allclose(fb[1,1],m._target(R,vv[1]),atol=1e-14);assert torch.isfinite(fb).all()
  oldkn=m.kn;m.kn=1e12;ell,b,_=m.paths(tx,v[:,0]);ball=m._target(b,v[None,:,:]);assert torch.allclose(m.distribution(tx,v),ball,atol=1e-12,rtol=1e-12);m.kn=oldkn
  params=list(m.parameters());dirs=[torch.randn_like(p) for p in params];loss=sum((r*r).mean() for r in m.self_consistency(tx).values());grads=torch.autograd.grad(loss,params);analytic=sum((g*d).sum() for g,d in zip(grads,dirs));orig=[p.detach().clone() for p in params];eps=1e-5;vals=[]
  with torch.no_grad():
   for sign in [1,-1]:
    for p,z,d in zip(params,orig,dirs):p.copy_(z+sign*eps*d)
    vals.append(sum((r*r).mean() for r in m.self_consistency(tx).values()))
   for p,z in zip(params,orig):p.copy_(z)
  fd=(vals[0]-vals[1])/(2*eps);rel=float(abs(fd-analytic)/(1+abs(analytic)));assert rel<1e-7,(family,rel)
  # Uniform equilibrium, no boundary hit: integral must reproduce constant f.
  oldmac=m.macros;oldinitial=m.initial
  def const(x):return dict(rho=torch.ones_like(x),u=torch.zeros_like(x),T=torch.ones_like(x),q=torch.zeros_like(x))
  m.initial=const;m.macros=lambda st:const(st[...,0]);eq=m.distribution(tx*torch.tensor([1.,0.]),v);expect=m._target(const(tx[:,0,None]),v[None,:,:]);assert torch.allclose(eq,expect,atol=1e-13,rtol=1e-13);m.macros=oldmac;m.initial=oldinitial
  if family=='bgk':assert (f>=0).all()
  # Explicitly certify no hidden clipping of a negative Shakhov target.
  if family=='shakhov':
   bad=dict(rho=torch.tensor(1.),u=torch.tensor(0.),T=torch.tensor(1.),q=torch.tensor(10.));assert m._target(bad,torch.tensor([-4.,0.,0.]))<0
  results[family]={'factorized_higher_vs_full':True,'factorized_max_abs_error':err,'directional_fd_scaled_error':rel,'IC':True,'inflow_signs_and_zero_velocity':True,'ballistic_limit':True,'uniform_equilibrium':True,'no_target_clipping':True}
 return {'status':'PASS','tests':results,'scope':'Small deterministic mathematical tests; no field accuracy or convergence claim.'}
if __name__=='__main__':print(json.dumps(run(),indent=2))
