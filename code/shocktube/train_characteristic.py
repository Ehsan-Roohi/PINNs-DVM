"""Fresh physics-only characteristic PINN. No reference/parent checkpoint inputs."""
import argparse, copy, hashlib, json, math, os, random, signal, time
from pathlib import Path
import numpy as np
import torch
from characteristic import CharacteristicPINN
from ssb_optimizer import DenseSSB

SSB_SHA='69d05fb79c9b4c0179bbe1d6dc8078e25a506e75eb3adb653521807251da9a94'
STOP=False
class InterruptedStep(Exception): pass

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('wb') as f: torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def rng(): return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if s['cuda']:
        assert len(s['cuda'])==torch.cuda.device_count();torch.cuda.set_rng_state_all(s['cuda'])
def seed(n):
    random.seed(n);np.random.seed(n);torch.manual_seed(n)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(n)
def sample(n,device):
    z=torch.rand((n,2),dtype=torch.float64,device=device);z[:,0]*=.02;z[:,1]-=.5;z[:n//2,1]*=.3;return z

def objective(model,tx):
    # Upstream rho=1, thermal speed=sqrt(T1)=1; fixed dimensional scales.
    r=model.self_consistency(tx)
    expected={'rho','px','energy'}|({'q'} if model.collision=='shakhov' else set())
    assert set(r)==expected,(set(r),expected)
    parts={k:(v/1.).square().mean() for k,v in r.items()}
    loss=sum(parts.values())
    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite physics objective')
    return loss,parts

def run(cfg,out,device='cpu',resume=False,stop_after=None):
    global STOP
    STOP=False
    def handler(*_):
        global STOP
        STOP=True
    old={s:signal.signal(s,handler) for s in (signal.SIGTERM,signal.SIGINT)}
    if hasattr(signal,'SIGUSR1'):old[signal.SIGUSR1]=signal.signal(signal.SIGUSR1,handler)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'LOCK').open('a+b')
    if os.name=='nt':
        import msvcrt
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        source=Path(__file__).resolve().parent
        assert sha(source/'ssb_optimizer.py')==SSB_SHA
        binding={n:sha(source/n) for n in ('train_characteristic.py','characteristic.py','ssb_optimizer.py')}
        digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        ck=out/'STATE.pt'
        if resume:
            state=torch.load(ck,map_location='cpu',weights_only=False)
            assert state['config_sha']==digest and state['sources']==binding
            assert state['status']=='INTERRUPTED','Only interrupted trajectories may resume'
            deadline=state['deadline_epoch']
        else:
            assert not ck.exists() and not (out/'INTENT.json').exists(),'Existing run; reconcile instead of overwriting'
            deadline=time.time()+float(cfg['train_seconds'])
            with (out/'INTENT.json').open('x') as f:json.dump(dict(config=cfg,config_sha=digest,sources=binding,deadline_epoch=deadline,initialization='independent_random_no_parent_no_reference'),f,indent=2)
        seed(int(cfg['seed']))
        torch.set_num_threads(int(cfg.get('threads',2)))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=CharacteristicPINN(collision=cfg['collision'],**cfg['model']).to(device=device,dtype=torch.float64)
        params=list(model.parameters());n=sum(p.numel() for p in params)
        adam=torch.optim.Adam(params,lr=float(cfg['lr']))
        ssb=DenseSSB(n,device=device,**cfg.get('ssb',{}))
        frozen=sample(int(cfg['fixed_points']),device)
        history=[];a=b=0
        if resume:
            model.load_state_dict(state['model']);adam.load_state_dict(state['adam']);ssb.load_state_dict(state['ssb'])
            frozen=state['frozen'].to(device);history=state['history'];a=state['adam_steps'];b=state['ssb_steps'];restore_rng(state['rng'])
        def save(status,reason=None):
            atomic(ck,dict(version=1,config=cfg,config_sha=digest,sources=binding,model=model.state_dict(),adam=adam.state_dict(),ssb=ssb.state_dict(),rng=rng(),frozen=frozen.detach().cpu(),history=history,adam_steps=a,ssb_steps=b,deadline_epoch=deadline,status=status,reason=reason,complete_optimizer_boundary=True))
        def check():
            if STOP or time.time()>=deadline:raise InterruptedStep()
        def vector():return torch.nn.utils.parameters_to_vector(params).detach().clone()
        def assign(x):torch.nn.utils.vector_to_parameters(x,params)
        def fg(x):
            check();assign(x);adam.zero_grad(set_to_none=True);total=0.
            for tx in frozen.split(int(cfg.get('chunk',64))):
                check();loss,_=objective(model,tx);weighted=loss*(len(tx)/len(frozen));weighted.backward();total+=float(weighted.detach())
            g=torch.cat([p.grad.reshape(-1) for p in params])
            if not bool(torch.isfinite(g).all()):raise FloatingPointError('Nonfinite gradient')
            return total,g
        save('INTERRUPTED','initialized')
        status='COMPLETE';reason=None
        while a<int(cfg['adam_steps']) or b<int(cfg['ssb_steps']):
            snapshot=(copy.deepcopy(model.state_dict()),copy.deepcopy(adam.state_dict()),copy.deepcopy(ssb.state_dict()),rng())
            try:
                check()
                if a<int(cfg['adam_steps']):
                    tx=sample(int(cfg['batch']),device);adam.zero_grad(set_to_none=True);loss,parts=objective(model,tx);loss.backward()
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),'Nonfinite/missing gradient'
                    adam.step()
                    assert all(bool(torch.isfinite(p).all()) for p in params),'Nonfinite parameters'
                    a+=1;history.append(dict(stage='adam',step=a,loss_before=float(loss.detach()),components_before={k:float(v.detach()) for k,v in parts.items()}))
                else:
                    result=ssb.step(vector(),fg)
                    if not result['accepted']:
                        # Explicit rollback independent of the optimizer callback's restoration.
                        model.load_state_dict(snapshot[0]);adam.load_state_dict(snapshot[1]);ssb.load_state_dict(snapshot[2]);restore_rng(snapshot[3])
                        status='STOPPED';reason=result['reason'];save(status,reason);break
                    assign(result['x']);b+=1;history.append(dict(stage='ssb',step=b,loss_after=result['loss'],alpha=result['alpha'],wolfe=result['wolfe'],evaluations=result['evaluations']))
                save('INTERRUPTED','completed_update')
                if stop_after is not None and a+b>=stop_after:raise InterruptedStep()
            except InterruptedStep:
                # A completed update is already durable; an unfinished trial rolls back.
                committed=torch.load(ck,map_location='cpu',weights_only=False)
                model.load_state_dict(committed['model']);adam.load_state_dict(committed['adam']);ssb.load_state_dict(committed['ssb']);restore_rng(committed['rng'])
                a=committed['adam_steps'];b=committed['ssb_steps'];history=committed['history']
                status='INTERRUPTED';reason='signal_or_budget_or_test_boundary';break
            except Exception:
                model.load_state_dict(snapshot[0]);adam.load_state_dict(snapshot[1]);ssb.load_state_dict(snapshot[2]);restore_rng(snapshot[3]);save('FAILED','uncommitted_step_failure');raise
        save(status,reason)
        result=dict(status=status,reason=reason,adam_steps=a,ssb_steps=b,checkpoint_sha=sha(ck),config_sha=digest,deadline_epoch=deadline,training_complete=status=='COMPLETE',no_reference_access=True,initialization='independent_random')
        (out/'RESULT.json').write_text(json.dumps(result,indent=2));return result
    finally:
        lock.close()
        for s,h in old.items():signal.signal(s,h)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true');a=p.parse_args()
    r=run(json.loads(Path(a.config).read_text()),a.out,a.device,a.resume)
    print(json.dumps(r),flush=True)
    raise SystemExit(0 if r['status']=='COMPLETE' else 75 if r['status']=='INTERRUPTED' else 2)


