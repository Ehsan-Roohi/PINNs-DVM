import math
import numpy as np
import torch

def rh(c):
    m = c['mach']
    density = 4*m*m/(m*m+3)
    return dict(rho1=1., u1=m*math.sqrt(5/3), T1=1., rho2=density,
                u2=m*math.sqrt(5/3)/density, T2=(5*m*m-1)/(4*density))

def quadrature(nvx, nr, vmax, device='cpu'):
    """Split axial half ranges: nvx is the TOTAL number of axial nodes.

    Separate half-range rules resolve the kinetic incoming/outgoing structure
    near zero rather than relying on cancellation across a full-range rule.
    """
    assert nvx>=4 and nvx%2==0, 'Split axial quadrature requires even total nvx'
    a, aw = np.polynomial.legendre.leggauss(nvx//2)
    b, bw = np.polynomial.legendre.leggauss(nr)
    axial=np.concatenate((vmax*(a-1)/2,vmax*(a+1)/2))
    axial_weights=np.concatenate((vmax*aw/2,vmax*aw/2))
    v, r = np.meshgrid(axial, vmax*(b+1)/2, indexing='ij')
    w = axial_weights[:,None]*vmax*bw[None,:]/2*2*np.pi*r
    return tuple(torch.as_tensor(z.ravel(), dtype=torch.float64, device=device) for z in (v,r,w))

def maxwellian(v, r, rho, u, T):
    return rho[:,None]/(2*math.pi*T[:,None])**1.5*torch.exp(-((v-u[:,None])**2+r**2)/(2*T[:,None]))

def integrate(f, v, r, w):
    fw=f*w
    rho=fw.sum(1)
    if bool((rho.detach()<=0).any()):
        raise FloatingPointError('Nonpositive density; no clipping')
    u=(fw*v).sum(1)/rho
    cx=v-u[:,None]; e=cx**2+r**2
    T=(fw*e).sum(1)/(3*rho)
    if bool((T.detach()<=0).any()):
        raise FloatingPointError('Nonpositive temperature; no clipping')
    q=.5*(fw*cx*e).sum(1)
    sigma=(fw*(cx**2-e/3)).sum(1)
    R=(fw*(e-7*T[:,None])*(cx**2-e/3)).sum(1)
    delta=(fw*e**2).sum(1)-15*rho*T**2
    return torch.stack((rho,u,T,q,sigma,R,delta,R+delta/3),1)

def prescribed_envelope(x,v,r,c):
    """Fixed analytic RH mixture; independent of every trainable parameter."""
    state=rh(c); L=c['L']; width=c['initial_width']
    left=1/(1+math.exp(L/width));right=1/(1+math.exp(-L/width))
    s=(torch.sigmoid(x[:,None]/width)-left)/(right-left)
    ml=(2*math.pi)**-1.5*torch.exp(-((v-state['u1'])**2+r*r)/2)
    mr=state['rho2']/(2*math.pi*state['T2'])**1.5*torch.exp(-((v-state['u2'])**2+r*r)/(2*state['T2']))
    return (1-s)*ml+s*mr
