"""Tiny isolated restart regression, never scientific training evidence."""
import argparse,importlib,json,os,pathlib,random,signal,subprocess,sys
P=pathlib.Path(__file__).resolve().parent
STOP=False
def stop(*a):
 global STOP
 STOP=True

def child(a):
 import torch,numpy as np
 sys.path.insert(0,str(P/'SOURCE'/a.stage));b=importlib.import_module('solver' if a.stage=='adam' else 'base_solver')
 c=json.loads((P/'SOURCE'/a.stage/('config.json' if a.stage=='adam' else 'model_config.json')).read_text(encoding='utf-8-sig'))
 c.update(width=8,depth=1,nvx=16,nr=8,batch=2,seed=141001)
 torch.set_default_dtype(torch.float64);torch.set_num_threads(2);b.seed_all(c['seed']);m=b.Distribution(c).to(a.device);g=b.quadrature(16,8,c['vmax'],a.device)
 cp=pathlib.Path(a.out);old=torch.load(cp,map_location='cpu',weights_only=False) if a.resume else None
 obj=None
 if a.stage=='adam':o=torch.optim.Adam(m.parameters(),lr=c['lr'])
 else:
  from objective import DeterministicObjective,make_points
  from ssb_optimizer import DenseSSB
  sc=json.loads((P/'SOURCE'/a.stage/'config.json').read_text());sc.update(chunk_size=2)
  if old:m.load_state_dict(old['model'])
  obj=DeterministicObjective(m,c,g,make_points(sc,a.device),chunk_size=2,frozen_state=old['frozen'] if old else None)
  o=DenseSSB(len(obj.get_flat()),device=a.device,**{k:sc[k] for k in ['c1','c2','max_evals','max_step','initial_step','gtol']})
 if old:
  m.load_state_dict(old['model']);o.load_state_dict(old['optimizer']);random.setstate(old['python_rng']);np.random.set_state(old['numpy_rng']);torch.set_rng_state(old['torch_rng'])
  if a.device=='cuda':torch.cuda.set_rng_state_all(old['cuda_rng'])
 n=old['step'] if old else 0
 accepted=old['accepted'] if old else []
 signal.signal(signal.SIGTERM,stop)
 while n<2 and not STOP:
  if obj:
   r=o.step(obj.get_flat(),obj.value_and_grad);accepted.append(bool(r['accepted']));assert r['accepted'], 'Fixture must perform actual SSB H updates'
  else:r=b.train_step(m,o,c,g,n,False)
  n+=1
  if a.signal_after and n==1:signal.raise_signal(signal.SIGTERM) if os.name=='nt' else os.kill(os.getpid(),signal.SIGTERM)
 state=dict(accepted=accepted,model=m.state_dict(),optimizer=o.state_dict(),step=n,frozen=obj.frozen_state() if obj else None,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if a.device=='cuda' else None)
 tmp=cp.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,cp)

def equal(a,b):
 import torch,numpy as np
 if type(a)!=type(b):return False
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def main(a):
 import torch
 out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False);results={}
 import hashlib
 sys.path.insert(0,str(P/'SOURCE'/'adam'));b=importlib.import_module('solver');c=json.loads((P/'SOURCE'/'adam'/'config.json').read_text());torch.set_default_dtype(torch.float64)
 digests=[]
 for seed in [141001,141002,141003,141001]:
  b.seed_all(seed);m=b.Distribution(c);h=hashlib.sha256()
  for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().numpy().tobytes())
  digests.append(h.hexdigest())
 assert digests[0]==digests[3] and len(set(digests[:3]))==3
 (out/'INITIALIZATION.json').write_text(json.dumps(dict(seeds=[141001,141002,141003,141001],digests=digests,PASS=True),indent=2))
 for stage in ['adam','ssb','fixed']:
  full=out/(stage+'_full.pt');split=out/(stage+'_split.pt')
  for dst,extra in [(full,[]),(split,['--signal-after']),(split,['--resume'])]:
   subprocess.run([sys.executable,'-B',__file__,'--stage',stage,'--out',str(dst),'--device',a.device]+extra,check=True,timeout=90)
  results[stage]=equal(torch.load(full,map_location='cpu',weights_only=False),torch.load(split,map_location='cpu',weights_only=False))
 (out/'RESULT.json').write_text(json.dumps(dict(PASS=all(results.values()),stages=results,actual_SIGTERM=True,new_process_resume=True,device=a.device,science=False),indent=2));assert all(results.values())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--stage',choices=['adam','ssb','fixed']);p.add_argument('--resume',action='store_true');p.add_argument('--signal-after',action='store_true');a=p.parse_args();child(a) if a.stage else main(a)



