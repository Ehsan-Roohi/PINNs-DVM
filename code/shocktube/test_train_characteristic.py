"""Tiny CPU software tests; no DVM, no production training."""
import copy,json,tempfile,signal
from pathlib import Path
import torch
import train_characteristic as t

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if hasattr(a,'shape'):return bool((a==b).all())
    return a==b

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=dict(seed=141101,collision='bgk',model=dict(hidden=4,depth=1,time_nodes=2,grid=[4,4,4],vmax=5),lr=1e-4,adam_steps=2,ssb_steps=0,batch=2,fixed_points=2,chunk=2,threads=1,train_seconds=90)
    t.run(cfg,out/'full')
    original_step=torch.optim.Adam.step
    def signaled_step(self,*args,**kwargs):
        result=original_step(self,*args,**kwargs);signal.raise_signal(signal.SIGTERM);return result
    torch.optim.Adam.step=signaled_step
    try:t.run(cfg,out/'split')
    finally:torch.optim.Adam.step=original_step
    t.run(cfg,out/'split',resume=True)
    a=torch.load(out/'full/STATE.pt',weights_only=False);b=torch.load(out/'split/STATE.pt',weights_only=False)
    for k in ('model','adam','ssb','rng','frozen','history','adam_steps','ssb_steps'):assert equal(a[k],b[k]),k
    # Explicit cumulative deadline must survive resume; expired trajectory makes zero updates.
    t.run(cfg,out/'expired',stop_after=1)
    st=torch.load(out/'expired/STATE.pt',weights_only=False);st['deadline_epoch']=0;t.atomic(out/'expired/STATE.pt',st)
    r=t.run(cfg,out/'expired',resume=True);assert r['adam_steps']==1 and r['status']=='INTERRUPTED'
    c=copy.deepcopy(cfg);c['adam_steps']=3
    try:t.run(c,out/'expired',resume=True)
    except AssertionError:pass
    else:raise AssertionError('Changed configuration accepted')
    # Strong-Wolfe engine fixture separately tests genuine acceptance/restoration.
    opt=t.DenseSSB(2,gtol=1e-12);base=torch.tensor([1.,2.],dtype=torch.float64)
    def quad(x):return .5*float(x@x),x.clone()
    r=opt.step(base,quad);assert r['accepted'] and r['wolfe']
    # Adversarial callback: rejected optimizer leaves BOTH parameters and H corrupted.
    reject_cfg=copy.deepcopy(cfg);reject_cfg.update(adam_steps=0,ssb_steps=1)
    saved_step=t.DenseSSB.step;seen={}
    def corrupt_reject(self,x,callback):
        seen['x']=x.clone();seen['H']=self.H.clone()
        callback(x+.001);self.H.mul_(2.)
        return dict(accepted=False,x=x,reason='test_rejected_trial')
    t.DenseSSB.step=corrupt_reject
    try:r=t.run(reject_cfg,out/'rejected')
    finally:t.DenseSSB.step=saved_step
    z=torch.load(out/'rejected/STATE.pt',weights_only=False)
    check=t.CharacteristicPINN(collision='bgk',**cfg['model']);check.load_state_dict(z['model'])
    assert torch.equal(torch.nn.utils.parameters_to_vector(check.parameters()),seen['x'])
    assert torch.equal(z['ssb']['H'],seen['H']) and z['ssb_steps']==0 and not z['history']
    assert r['status']=='STOPPED'
    result=dict(status='PASS',scope='tiny CPU restart/budget/config and exact optimizer fixture',same_process_restart=True,SIGTERM_after_optimizer_step_tested=True,rejected_trial_full_rollback_tested=True,new_process_signal_tested=False,CUDA_tested=False)
    (out/'TEST_RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)


