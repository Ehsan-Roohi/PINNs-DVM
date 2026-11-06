"""Frozen-source carrier and compact physical-length cubic B-spline basis.

Six complete axisymmetric nonconserved polynomial directions through degree4.
Carrier is positive part of frozen transported source, NOT a modified field.
All derivative chains (carrier, local frame, Gram, spatial spline) retained.
"""
import torch as t
import math
from m7_velocity_transport import envelope

def feature(v,r,a):
    rho,u,T=a[:3];z=(v-u)/T.sqrt();e=z*z+r*r/T
    phi=t.stack((t.ones_like(z),z,(e-3)/math.sqrt(6)),1)
    psi=t.stack((z*z-e/3,.5*z*e,z**3-.6*z*e,(e-7)*(z*z-e/3),e*e-15,
                 z**4-6/7*z*z*e+3/35*e*e),1)
    return phi,psi

def projection(v,r,w,a,f):
    phi,psi=feature(v,r,a);d=t.where(f>0,f,t.zeros_like(f))
    G=phi.T@((w*d)[:,None]*phi)
    return t.linalg.solve(G,phi.T@((w*d)[:,None]*psi))

def kinetic(v,r,a,f,A):
    phi,psi=feature(v,r,a);d=t.where(f>0,f,t.zeros_like(f))
    return d[:,None]*(psi-phi@A)

def spatial(x):
    # Fixed upstream streaming length u1*tau1, no observed/DVM width.
    xi=t.tanh(x/(7*math.sqrt(5/3)))
    interior=t.linspace(-1,1,10,dtype=x.dtype,device=x.device)[1:-1]
    knots=t.cat((t.full((4,),-1.,dtype=x.dtype,device=x.device),interior,
                 t.full((4,),1.,dtype=x.dtype,device=x.device)))
    b=[((xi>=knots[i])&(xi<knots[i+1])).to(x.dtype)for i in range(15)]
    for p in range(1,4):
        nxt=[]
        for i in range(15-p):
            value=t.zeros_like(x)
            if float(knots[i+p]-knots[i])>0:value=value+(xi-knots[i])/(knots[i+p]-knots[i])*b[i]
            if float(knots[i+p+1]-knots[i+1])>0:value=value+(knots[i+p+1]-xi)/(knots[i+p+1]-knots[i+1])*b[i+1]
            nxt.append(value)
        b=nxt
    return envelope(x)*t.stack(b)

def basis_with_x(v,r,w,a,ax,f,fx,x):
    A,Ax=t.autograd.functional.jvp(lambda z:projection(v,r,w,a+z*ax,f+z*fx),t.tensor(0.,dtype=f.dtype),t.tensor(1.,dtype=f.dtype))
    K,Kx=t.autograd.functional.jvp(lambda z:kinetic(v,r,a+z*ax,f+z*fx,A+z*Ax),t.tensor(0.,dtype=f.dtype),t.tensor(1.,dtype=f.dtype))
    B,Bx=t.autograd.functional.jvp(spatial,t.as_tensor(float(x),dtype=f.dtype),t.tensor(1.,dtype=f.dtype))
    return K,Kx,B,Bx,A,Ax

def kron_quadratic(H,g,B,Bx):
    # Velocity affine columns are [v*Kx+K, v*K]. Never materialize Nv x72.
    b=t.stack((B,Bx));Ht=H.reshape(2,6,2,6)
    full=t.einsum('aibj,ak,bl->ikjl',Ht,b,b).reshape(72,72)
    fullg=t.einsum('ai,ak->ik',g.reshape(2,6),b).reshape(72)
    return full,fullg
