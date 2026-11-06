"""Saved-candidate full kinetic evaluation. No reference readers or threshold defaults."""
import types
import numpy as np
import torch as t
import physical_base as base
import physical_diagnostics as pd
from stream_characteristic import full_sparse,reduced_fields

def evaluate(cfg,x,macro,query_x,query_weights,budget):
    """Caller supplies ORIGINAL heldout quadrature; no grid or limit invention."""
    t.set_default_dtype(t.float64);t.set_num_threads(int(cfg.get('diagnostic_threads',4)))
    query_x=np.asarray(query_x);query_weights=np.asarray(query_weights)
    assert len(query_x)==len(query_weights) and np.all(query_weights>0)
    assert query_x.min()>=x[0] and query_x.max()<=x[-1]
    rh=base.rh(cfg);J=rh['u1'];rs=rh['rho2'];Ts=rh['T2']
    grid=base.quadrature(cfg['nvx'],cfg['nr'],cfg['vmax'],'cpu');ng=[a.numpy() for a in grid]
    up=np.array([1.,J,1.]);down=np.array([rs,rh['u2'],Ts])
    queries=np.r_[query_x,0.,x[0],x[-1]]
    Q,MQ,actual=full_sparse(x,macro,ng,up,down,queries,budget)
    rf,rq=reduced_fields(x,macro,ng,cfg['vmax'],up,down,queries,budget)
    ns=dict(pd.__dict__);ns.update(cfg=cfg,J=J,rs=rs,Ts=Ts,Pi=J*J+1,energy=.5*J**3+2.5*J)
    diag=types.FunctionType(pd.diagnostics.__code__,ns)
    terms=[];proj=[];flux=[];full9=[]
    for i,z in enumerate(query_x):
        budget();f=t.from_numpy(Q[i]);fx=t.from_numpy((MQ[i]-Q[i])/ng[0]);d=diag(z,f,fx,grid)
        terms.append(d['terms'].numpy());proj.append(d['projected'].numpy());flux.append(d['flux'].numpy());full9.append(float(d['full9']@d['full9']))
    fields=base.integrate(t.from_numpy(Q),*grid).numpy()
    boundary=[]
    for i,state in [(-2,up),(-1,down)]:
        exact=base.maxwellian(grid[0],grid[1],t.tensor(state[0:1]),t.tensor(state[1:2]),t.tensor(state[2:3]))[0];den=exact+exact.max()*1e-7
        boundary.append(float((((t.from_numpy(Q[i])-exact)/den)**2*exact*grid[2]).sum()/(exact*grid[2]).sum()))
    terms=np.asarray(terms);sc=np.array([rs,J,Ts,rs*Ts**1.5,rs*Ts,rs*Ts**2,rs*Ts**2,rs*Ts**2])
    result=dict(kinetic_mean=float(query_weights@terms[:,0]),tail_mean=float(query_weights@terms[:,1]),raw_mean=float(query_weights@terms[:,2]),full9_mean=float(query_weights@np.array(full9)),max_flux=float(abs(np.array(flux)).max()),max_projection=float(abs(np.array(proj)).max()),phase=float(((fields[-3,0]-(1+rs)/2)/((1+rs)/2))**2),boundary=float(np.mean(boundary)),negative_distribution_value_max=float(np.maximum(-Q,0).max()),negative_mass_fraction_max=float(np.max((np.maximum(-Q,0)*ng[2][None,:]).sum(axis=1)/fields[:,0])),negative_mass_normalization='quadrature integral of negative part divided by positive net density at each query',rho_min=float(actual[:,0].min()),T_min=float(actual[:,2].min()),reduced_full_macro_max=float(abs(actual-rf[:,:3]).max()),reduced_full_query_normalized_max=(abs(fields-rq)/sc).max(axis=0).tolist(),thresholds_applied=False,convergence_certified=False,reference_access=False)
    return result,dict(x=x,macro=macro,query_x=queries,query_fields=fields,reduced_fields=rf,terms=terms,projected=np.array(proj),flux=np.array(flux))
