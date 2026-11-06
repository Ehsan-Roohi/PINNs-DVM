"""Cold independent M2 lineage, original kernels and explicit optimizer phases.
No reference file is accepted. Root alone submits jobs. Linux shared workspace only.
"""
import argparse,hashlib,json,os,pathlib,shutil,signal,subprocess,sys,time
P=pathlib.Path(__file__).resolve().parent
ACTIVE=None
STOP_REQUESTED=False

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,z):
 p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(z,indent=2,allow_nan=False));os.replace(tmp,p)
def persistent_deadline(out,cap):
 budgetfile=pathlib.Path(out)/'BUDGET.json'
 if not budgetfile.exists():write(budgetfile,dict(deadline_epoch=time.time()+cap,cap_seconds=cap,downtime_counts_conservatively=True))
 data=json.loads(budgetfile.read_text());assert data['cap_seconds']==cap, 'Budget cap changed'
 return data['deadline_epoch']

def forward(sig,*_):
 global STOP_REQUESTED
 STOP_REQUESTED=True
 if ACTIVE is not None:ACTIVE.send_signal(sig)

def worker(a):
 import importlib,torch
 d=pathlib.Path(a.work);sys.path.insert(0,str(d));b=importlib.import_module('solver' if a.stage=='adam' else 'base_solver')
 torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
 assert torch.cuda.is_available();assert any(v in torch.cuda.get_device_name().upper() for v in ('V100','A100'))
 if a.stage=='adam':
  c=json.loads((d/'config.json').read_text());b.seed_all(c['seed']);m=b.Distribution(c)
  h=hashlib.sha256()
  for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().numpy().tobytes())
  write(d/'INITIALIZATION.json',dict(seed=c['seed'],initial_parameter_sha256=h.hexdigest(),cold=True,external_profiles=False))
  del m
  signal.signal(signal.SIGTERM,b.stop);signal.signal(signal.SIGUSR1,b.stop)
  result=b.run(c,d/'run',a.seconds)
  return 0 if result['finished'] else 75
 r=importlib.import_module('runner');signal.signal(signal.SIGTERM,r.stop);signal.signal(signal.SIGUSR1,r.stop);sys.argv=[str(d/'runner.py'),'--out','run']
 import threading
 timer=threading.Timer(a.seconds,r.stop);timer.daemon=True;timer.start()
 try:r.main()
 finally:timer.cancel()
 status=json.loads((d/'run/STATUS.json').read_text());return 0 if status['finished'] else 75

def main(a):
 global ACTIVE
 import fcntl
 out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 with (out/'lineage.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  source_files={str(p.relative_to(P/'SOURCE')):sha(p) for p in (P/'SOURCE').rglob('*') if p.is_file() and p.suffix in ('.py','.json')}
  identity=dict(seed=a.seed,stages=['adam20000','ssb500','fixed500'],source_files=source_files,runner_sha256=sha(__file__),test_sha256=sha(P/'test_restart.py'),max_seconds=a.seconds)
  binding=out/'LINEAGE.json'
  if binding.exists():assert json.loads(binding.read_text())==identity,'Lineage/source/config mismatch'
  else:write(binding,identity)
  start=time.monotonic()
  deadline=persistent_deadline(out,a.seconds)
  assert deadline-time.time()>300, 'Cumulative lineage budget exhausted'
  admission=out/'GPU_RESTART/RESULT.json'
  if not admission.exists():
   if STOP_REQUESTED:return 75
   ACTIVE=subprocess.Popen([sys.executable,'-B',str(P/'test_restart.py'),'--device','cuda','--out',str(out/'GPU_RESTART')])
   try:rc=ACTIVE.wait(timeout=180)
   except subprocess.TimeoutExpired:
    ACTIVE.terminate();ACTIVE.wait(timeout=15);raise
   finally:ACTIVE=None
   if STOP_REQUESTED:return 75
   assert rc==0,'GPU restart qualification failed'
  assert json.loads(admission.read_text())['PASS']
  for idx,stage in enumerate(['adam','ssb','fixed']):
   if STOP_REQUESTED:write(out/'STOPPED.json',dict(reason='signal_before_stage',stage=stage));return 75
   d=out/stage; prev=out/(['adam','ssb'][idx-1] if idx else 'none')/'run'
   if not d.exists():
    d.mkdir()
    for p in (P/'SOURCE'/stage).iterdir():
     if p.is_file() and p.suffix in ('.py','.json'):shutil.copy2(p,d/p.name)
    cpath=d/('config.json' if idx==0 else 'model_config.json');mc=json.loads(cpath.read_text(encoding='utf-8-sig'));mc['seed']=a.seed
    # LR/weight ramp and scientific steps remain historical. Wall budget administrative only.
    mc['wall_seconds']=a.seconds;write(cpath,mc)
    if idx:
     prior=json.loads((prev/'STATUS.json').read_text());assert prior['finished'],'Cannot advance incomplete parent'
     sc=json.loads((d/'config.json').read_text());sc['input_checkpoint_sha256']=sha(prev/'checkpoint.pt');sc['input_signature']=prior['signature'];sc['wall_seconds']=a.seconds;write(d/'config.json',sc)
     shutil.copy2(prev/'checkpoint.pt',d/'input_checkpoint.pt')
     write(d/'OWN_PARENT.json',dict(seed=a.seed,parent_stage=['adam','ssb'][idx-1],sha256=sc['input_checkpoint_sha256'],step=prior['step'],fresh_H=True,new_frozen_objective=True))
    hashes={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.suffix in ('.py','.json')};write(d/'source.sha256',hashes)
    write(d/'PREPARED.json',dict(source_files=hashes,seed=a.seed))
   assert (d/'PREPARED.json').exists(),'Partial stage preparation; reconcile manually'
   for name,expected in json.loads((d/'PREPARED.json').read_text())['source_files'].items():assert sha(d/name)==expected, 'Stage source/config changed'
   assert json.loads((d/'source.sha256').read_text())==json.loads((d/'PREPARED.json').read_text())['source_files']
   statuspath=d/'run/STATUS.json'
   if statuspath.exists():
    status=json.loads(statuspath.read_text())
    if status['finished']:continue
    assert status.get('reason','wall_budget') in ['wall_budget','signal_checkpoint'],'Scientific rejection is terminal, no automatic retry'
   remaining=deadline-time.time()
   if remaining<300:write(out/'STOPPED.json',dict(reason='budget_before_next_stage',stage=stage));return 75
   if STOP_REQUESTED:write(out/'STOPPED.json',dict(reason='signal_before_launch',stage=stage));return 75
   cmd=[sys.executable,'-B',__file__,'--stage',stage,'--work',str(d),'--seconds',str(int(remaining-240))]
   with (d/'controller.log').open('a') as log:
    ACTIVE=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    try:rc=ACTIVE.wait(timeout=remaining-120)
    except subprocess.TimeoutExpired:
     ACTIVE.send_signal(signal.SIGTERM)
     try:rc=ACTIVE.wait(timeout=100)
     except subprocess.TimeoutExpired:ACTIVE.kill();ACTIVE.wait();rc=124
    ACTIVE=None
   if rc:write(out/'STOPPED.json',dict(stage=stage,returncode=rc));return rc
  write(out/'COMPLETE.json',dict(seed=a.seed,adam_steps=20000,ssb_steps=500,fixed_ssb_steps=500,convergence_claim=False,external_field_training=False,elapsed=time.monotonic()-start));return 0
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--seed',type=int,choices=[141001,141002,141003]);p.add_argument('--seconds',type=int,default=13800);p.add_argument('--stage',choices=['adam','ssb','fixed']);p.add_argument('--work');a=p.parse_args()
 if not a.stage:assert a.out and a.seed
 signal.signal(signal.SIGTERM,forward)
 if hasattr(signal,'SIGUSR1'):signal.signal(signal.SIGUSR1,forward)
 raise SystemExit(worker(a) if a.stage else main(a))




