"""Mach12 ordinary BGK Pr1 CONSTANTtau1; no reference arrays or neural updates.
Same hash-bound equations/diagnostics as the verified Mach8 research correction.
"""
from pathlib import Path
import os,sys,math,hashlib,copy,ast,json
W=Path(__file__).parent;P=Path(__file__).resolve().parent/'physics_source'
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
sys.path.insert(0,str(W/'probe-convex-runtime'));import numpy as np
sys.path.insert(0,str(W/'probe-runtime-2.10'));import torch as t
sys.path.insert(0,str(P/'bgk_pr1_tau1_M8'));import base_solver as base
from m7_complete9_metric import features as complete9_features
t.set_default_dtype(t.float64);t.set_num_threads(4)
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
SOURCE=W/'M8_PORTABLE_FIELD_20261030/FROZEN_FIELD_STATE.npz';SOURCE_SHA='2b9c5bc31faf69ad8a3e43770ed5b64ef1c2b18f473cb95842f06548a0064f0a';assert sha(SOURCE)==SOURCE_SHA
cfg_source=W/'M8_ORIGINAL3975_MODEL_ONLY_20261030/NN_MODEL_ONLY.pt';assert sha(cfg_source)=='2e5d47b9091d074619a6e2b006bcf110feb73af464c8cbee150e991196b8d2f3'
state=t.load(cfg_source,map_location='cpu',weights_only=False);cfg=copy.deepcopy(state['model_config']);cfg.update(mach=12,L=180,nvx=768,nr=176,refine_nvx=960,refine_nr=216,vmax=72.);cfg['initial_width']*=1.5;base.validate_config(cfg)
rh=base.rh(cfg);J=rh['u1'];Pi=J*J+1;energy=.5*J**3+2.5*J;rs=rh['rho2'];Ts=rh['T2'];PRIMARY=base.quadrature(768,176,72.,'cpu')
for file,name in [('m8_protected_moment_port.py','diagnostics'),('m8_nn_mach7_method.py','envelope')]:
 tree=ast.parse((W/file).read_text());defs=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name];assert len(defs)==1;exec(compile(ast.Module(body=defs,type_ignores=[]),str(W/file)+'::Mach12-same-physics-function','exec'),globals())
REPORT_RUNTIME={'mach':12,'Pr':1,'tau':1,'RH':rh,'L':180,'C':72,'primary':[768,176],'refined':[960,216],'cutoff_test':[1080,240,81.],'physical_source_SHA':sha(P/'bgk_pr1_tau1_M8/base_solver.py'),'Mach8_source_state_SHA':SOURCE_SHA,'diagnostic_source_SHA':sha(W/'m8_protected_moment_port.py'),'reference_solution_values_read':False,'NN_optimizer_updates':0,'Torch_threads':t.get_num_threads()}
