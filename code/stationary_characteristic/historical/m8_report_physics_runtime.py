"""Pure reporting runtime: no inherited warmstart model or source side effects.
Physical functions are compiled verbatim from the frozen scientific sources.
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
SOURCE=W/'M8_ORIGINAL3975_MODEL_ONLY_20261030/NN_MODEL_ONLY.pt';SOURCE_SHA='2e5d47b9091d074619a6e2b006bcf110feb73af464c8cbee150e991196b8d2f3';assert sha(SOURCE)==SOURCE_SHA
state=t.load(SOURCE,map_location='cpu',weights_only=False);assert state['source_SHA']=='12ddfe5c344c4ff4a1d215ec7211f728dba9f1c66b80a0c2920f93be81f98ffc';cfg=copy.deepcopy(state['model_config']);cfg.update(nvx=512,nr=144,refine_nvx=640,refine_nr=176)
rh=base.rh(cfg);J=rh['u1'];Pi=J*J+1;energy=.5*J**3+2.5*J;rs=rh['rho2'];Ts=rh['T2'];PRIMARY=base.quadrature(512,144,48.,'cpu')
for file,name in [('m8_protected_moment_port.py','diagnostics'),('m8_nn_mach7_method.py','envelope')]:
 tree=ast.parse((W/file).read_text());defs=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name];assert len(defs)==1;exec(compile(ast.Module(body=defs,type_ignores=[]),str(W/file)+'::verbatim-report-function','exec'),globals())
REPORT_RUNTIME={'physical_source_SHA':sha(P/'bgk_pr1_tau1_M8/base_solver.py'),'diagnostics_AST_source_SHA':sha(W/'m8_protected_moment_port.py'),'envelope_AST_source_SHA':sha(W/'m8_nn_mach7_method.py'),'numpy_file':np.__file__,'requested_BLAS_threads_before_numpy_import':1,'Torch_threads':t.get_num_threads(),'inherited_warmstart_model_loaded':False,'DVM_access':False}
