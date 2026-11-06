"""Bounded-memory exact characteristic sweeps. No fitting or file operations.

Full finite radial quadrature remains independent admission for reduced analytic
Maxwellian radial moments. Query values use original piecewise-linear SOURCE
Maxwellians, without changing source cells when a query is inserted.
"""
import numpy as np

def fields_from_raw(raw):
 rho,B1,B2,B3,B4,H0,H1,H2,K0=raw.T;u=B1/rho
 c2=B2-u*u*rho;c3=B3-3*u*B2+2*u**3*rho;c4=B4-4*u*B3+6*u*u*B2-3*u**4*rho
 h1=H1-u*H0;h2=H2-2*u*H1+u*u*H0;T=(c2+H0)/(3*rho)
 assert np.min(rho)>0 and np.min(T)>0
 q=.5*(c3+h1);sigma=(2*c2-H0)/3;R=(2*c4+h2-K0)/3-7*T*sigma;Delta=c4+2*h2+K0-15*rho*T*T
 return np.c_[rho,u,T,q,sigma,R,Delta,R+Delta/3]

def reduced_fields(x,a,grid,C,upstream,downstream,queries,budget):
 vv,rr,ww=[np.asarray(z)for z in grid];v,inv=np.unique(vv,return_inverse=True);w=np.bincount(inv,weights=ww)/(np.pi*C*C)
 queries=np.asarray(queries);raw=np.zeros((len(x),9));qr=np.zeros((len(queries),9))
 def M(aa,mask):
  g=aa[0]/np.sqrt(2*np.pi*aa[2])*np.exp(-(v[mask]-aa[1])**2/(2*aa[2]));return np.array([g,2*aa[2]*g,8*aa[2]**2*g])
 def pack(f,vm,wm):
  g,h,k=f;return np.array([wm@g,(wm*vm)@g,(wm*vm**2)@g,(wm*vm**3)@g,(wm*vm**4)@g,wm@h,(wm*vm)@h,(wm*vm**2)@h,wm@k])
 for direction in [1,-1]:
  mask=v>0 if direction==1 else v<0;vm=v[mask];wm=w[mask];anchor=0 if direction==1 else len(x)-1;f=M(upstream if direction==1 else downstream,mask);old=M(a[anchor],mask);raw[anchor]+=pack(f,vm,wm)
  for qi in np.where(abs(queries-x[anchor])<1e-12)[0]:qr[qi]+=pack(f,vm,wm)
  for i in (range(1,len(x))if direction==1 else range(len(x)-2,-1,-1)):
   if i%40==0:budget()
   prev=i-direction;new=M(a[i],mask);h=abs(x[i]-x[prev]);lo,hi=sorted((x[i],x[prev]));selected=np.where((queries>lo+1e-12)&(queries<hi-1e-12))[0]
   for qi in selected:
    distance=abs(queries[qi]-x[prev]);theta=distance/h;mq=(1-theta)*old+theta*new;q=distance/abs(vm);al=np.exp(-q);z=-np.expm1(-q)/q;fq=al*f+(z-al)*old+(1-z)*mq;qr[qi]+=pack(fq,vm,wm)
   q=h/abs(vm);al=np.exp(-q);z=-np.expm1(-q)/q;f=al*f+(z-al)*old+(1-z)*new;raw[i]+=pack(f,vm,wm)
   for qi in np.where(abs(queries-x[i])<1e-12)[0]:qr[qi]+=pack(f,vm,wm)
   old=new
 return fields_from_raw(raw),fields_from_raw(qr)if len(queries)else np.empty((0,8))

def full_sparse(x,a,grid,upstream,downstream,queries,budget):
 v,r,w=[np.asarray(z)for z in grid];queries=np.asarray(queries);Q=np.zeros((len(queries),len(v)));MQ=np.zeros_like(Q);raw=np.zeros((len(x),3))
 def M(aa,mask):return aa[0]/(2*np.pi*aa[2])**1.5*np.exp(-((v[mask]-aa[1])**2+r[mask]**2)/(2*aa[2]))
 for direction in [1,-1]:
  mask=v>0 if direction==1 else v<0;vm=v[mask];rm=r[mask];wm=w[mask];anchor=0 if direction==1 else len(x)-1;f=M(upstream if direction==1 else downstream,mask);old=M(a[anchor],mask);kernel=np.array([wm,wm*vm,wm*(vm*vm+rm*rm)])
  raw[anchor]+=kernel@f
  for qi in np.where(abs(queries-x[anchor])<1e-12)[0]:Q[qi,mask]=f;MQ[qi,mask]=old
  for i in (range(1,len(x))if direction==1 else range(len(x)-2,-1,-1)):
   if i%40==0:budget()
   prev=i-direction;new=M(a[i],mask);h=abs(x[i]-x[prev]);lo,hi=sorted((x[i],x[prev]));selected=np.where((queries>lo+1e-12)&(queries<hi-1e-12))[0]
   for qi in selected:
    distance=abs(queries[qi]-x[prev]);theta=distance/h;mq=(1-theta)*old+theta*new;q=distance/abs(vm);al=np.exp(-q);z=-np.expm1(-q)/q;Q[qi,mask]=al*f+(z-al)*old+(1-z)*mq;MQ[qi,mask]=mq
   q=h/abs(vm);al=np.exp(-q);z=-np.expm1(-q)/q;f=al*f+(z-al)*old+(1-z)*new;raw[i]+=kernel@f
   for qi in np.where(abs(queries-x[i])<1e-12)[0]:Q[qi,mask]=f;MQ[qi,mask]=new
   old=new
 rho=raw[:,0];u=raw[:,1]/rho;T=(raw[:,2]/rho-u*u)/3
 return Q,MQ,np.c_[rho,u,T]
