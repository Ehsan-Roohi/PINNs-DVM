import unittest
import numpy as np
from engine import Engine
import physical_base as base
import time
class QuadratureParity(unittest.TestCase):
 def test_axial(self):
  cfg=dict(mach=7,basis=4,nvx=32,nr=12,vmax=24,L=105)
  e=Engine(cfg,time.monotonic()+30);v,r,w=[z.numpy() for z in base.quadrature(32,12,24)]
  vv,ix=np.unique(v,return_inverse=True);ww=np.bincount(ix,weights=w)/(np.pi*24**2)
  self.assertTrue(np.array_equal(vv,e.grid[0].numpy()))
  self.assertLess(np.max(abs(ww-e.grid[2].numpy()/(np.pi*24**2))),1e-13)
if __name__=='__main__':unittest.main()
