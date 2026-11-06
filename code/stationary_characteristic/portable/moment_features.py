import math
import torch as t

def features(v,r,u,Tref):
 z=(v-u)/math.sqrt(Tref);e=z*z+r*r/Tref
 return t.stack((t.ones_like(v),z,(e-3)/math.sqrt(6),math.sqrt(3)/2*(z*z-e/3),z*(e-5)/math.sqrt(10),math.sqrt(5/12)*(z**3-.6*z*e),math.sqrt(3/56)*(e-7)*(z*z-e/3),(e*e-10*e+15)/math.sqrt(120),math.sqrt(35/192)*(z**4-6/7*z*z*e+3/35*e*e)),1)
