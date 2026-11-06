"""Tiny tests prepared only; root decides execution. No files/models/reference inputs."""
import time,unittest
import numpy as np
from engine import Engine
class Tests(unittest.TestCase):
 def test_equilibrium(self):
  cfg=dict(mach=1.,basis=4,nvx=128,vmax=12.,L=4.)
  e=Engine(cfg,time.monotonic()+30);x=np.linspace(-4,4,9)
  a=np.tile([1.,e.J,1.],(len(x),1));_,actual,g,_=e.transport(x,a,e.grid)
  self.assertLess(np.max(abs(actual-a)),1e-10)
 def test_tangent(self):
  cfg=dict(mach=1.,basis=4,nvx=128,vmax=12.,L=4.)
  e=Engine(cfg,time.monotonic()+30);x=np.linspace(-4,4,9);u=np.full(9,e.J);T=np.ones(9)
  Cu=np.zeros((9,e.P));CT=Cu.copy();Cu[:,0]=.01*np.sin(np.pi*(x+4)/8);CT[:,0]=.02*np.sin(np.pi*(x+4)/8)
  a=np.c_[e.J/u,u,T];_,_,g,j=e.transport(x,a,e.grid,Cu,CT)
  eps=1e-4
  def f(sign):
   uu=u+sign*eps*Cu[:,0];tt=T+sign*eps*CT[:,0]
   return e.transport(x,np.c_[e.J/uu,uu,tt],e.grid)[2]
  fd=(f(1)-f(-1))/(2*eps)
  self.assertLess(np.max(abs(fd-j[:,:,0])),1e-8)
if __name__=='__main__':unittest.main()
