"""Exact-state, physics-only SSB continuation; no reference readers.

CLI plan: {parent_sha256, sources:{train_characteristic.py,characteristic.py,
ssb_optimizer.py}, additional_updates:200, train_seconds:<bounded seconds>}.
The parent must be COMPLETE. Its configuration/objective and Adam state stay
unchanged. A separate persistent continuation deadline includes downtime.
"""
import argparse, copy, hashlib, json, os, signal, sys, time
from pathlib import Path
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import train_characteristic as native
import torch

NAMES = ('train_characteristic.py','characteristic.py','ssb_optimizer.py')

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()

def run(plan, parent, out, device='cpu', resume=False, stop_after=None):
    parent=Path(parent).resolve();out=Path(out).resolve()
    assert out != parent.parent and parent != out/'STATE.pt', 'Original output is immutable'
    assert native.sha(parent)==plan['parent_sha256'], 'Parent checkpoint hash mismatch'
    binding={n:native.sha(SOURCE/n) for n in NAMES}
    assert binding==plan['sources'], 'Frozen source hash mismatch'
    assert binding['ssb_optimizer.py']==native.SSB_SHA
    assert int(plan['additional_updates'])>0 and 0<float(plan['train_seconds'])<float('inf')
    base=torch.load(parent,map_location='cpu',weights_only=False)
    assert base['status']=='COMPLETE' and base['complete_optimizer_boundary']
    assert base['sources']==binding and base['config_sha']==digest(base['config'])
    cfg=base['config']
    assert base['adam_steps']==cfg['adam_steps'] and base['ssb_steps']==cfg['ssb_steps']
    target=base['ssb_steps']+int(plan['additional_updates'])
    identity=dict(plan=plan,parent_path=str(parent),controller_sha=native.sha(__file__))
    identity_sha=digest(identity)
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'LOCK').open('a+b');old={};stopped=[False]
    try:
        if os.name=='nt':
            import msvcrt
            lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        def handler(*_):stopped[0]=True
        for sig in (signal.SIGTERM,signal.SIGINT)+((signal.SIGUSR1,) if hasattr(signal,'SIGUSR1') else ()):
            old[sig]=signal.signal(sig,handler)
        ck=out/'STATE.pt'
        if resume:
            state=torch.load(ck,map_location='cpu',weights_only=False)
            assert state['status']=='INTERRUPTED' and state['complete_optimizer_boundary']
            assert state['continuation_identity_sha']==identity_sha
            assert state['config']==cfg and state['sources']==binding
            deadline=state['continuation_deadline_epoch']
        else:
            assert not ck.exists() and not (out/'INTENT.json').exists(), 'Reconcile existing trajectory'
            state=copy.deepcopy(base);deadline=time.time()+float(plan['train_seconds'])
            with (out/'INTENT.json').open('x') as f:
                json.dump(dict(identity=identity,identity_sha=identity_sha,deadline_epoch=deadline,target_ssb_steps=target),f,indent=2)
        torch.set_num_threads(int(cfg.get('threads',2)))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=native.CharacteristicPINN(collision=cfg['collision'],**cfg['model']).to(device=device,dtype=torch.float64)
        params=list(model.parameters());adam=torch.optim.Adam(params,lr=float(cfg['lr']))
        ssb=native.DenseSSB(sum(p.numel() for p in params),device=device,**cfg.get('ssb',{}))
        def restore(s):
            model.load_state_dict(s['model']);adam.load_state_dict(s['adam']);ssb.load_state_dict(s['ssb']);native.restore_rng(s['rng'])
        restore(state)
        frozen=state['frozen'].to(device);history=copy.deepcopy(state['history']);b=state['ssb_steps']
        assert torch.equal(state['frozen'],base['frozen']) and base['ssb_steps']<=b<=target
        def save(status,reason):
            payload=copy.deepcopy(base)
            payload.update(model=model.state_dict(),adam=adam.state_dict(),ssb=ssb.state_dict(),rng=native.rng(),
                frozen=frozen.detach().cpu(),history=history,ssb_steps=b,status=status,reason=reason,
                complete_optimizer_boundary=True,continuation_identity_sha=identity_sha,
                continuation=identity,continuation_deadline_epoch=deadline,target_ssb_steps=target)
            native.atomic(ck,payload)
        def check():
            if stopped[0] or time.time()>=deadline:raise native.InterruptedStep()
        def assign(x):torch.nn.utils.vector_to_parameters(x,params)
        def fg(x):
            check();assign(x);adam.zero_grad(set_to_none=True);total=0.
            for tx in frozen.split(int(cfg.get('chunk',64))):
                check();loss,_=native.objective(model,tx);weighted=loss*(len(tx)/len(frozen));weighted.backward();total+=float(weighted.detach())
            g=torch.cat([p.grad.reshape(-1) for p in params])
            if not bool(torch.isfinite(g).all()):raise FloatingPointError('Nonfinite gradient')
            return total,g
        save('INTERRUPTED','continuation_boundary')
        status='COMPLETE';reason=None
        while b<target:
            try:
                check();result=ssb.step(torch.nn.utils.parameters_to_vector(params).detach().clone(),fg)
                # Also reject an otherwise accepted result if a stop arrived in the trial.
                check()
                if not result['accepted']:
                    restore(torch.load(ck,map_location='cpu',weights_only=False))
                    status='STOPPED';reason=result['reason'];break
                assign(result['x']);b+=1
                history.append(dict(stage='ssb',step=b,loss_after=result['loss'],alpha=result['alpha'],wolfe=result['wolfe'],evaluations=result['evaluations']))
                save('INTERRUPTED','completed_update')
                if stop_after is not None and b-base['ssb_steps']>=stop_after:raise native.InterruptedStep()
            except native.InterruptedStep:
                committed=torch.load(ck,map_location='cpu',weights_only=False);restore(committed)
                b=committed['ssb_steps'];history=committed['history'];status='INTERRUPTED';reason='signal_or_budget_or_test_boundary';break
            except Exception:
                committed=torch.load(ck,map_location='cpu',weights_only=False);restore(committed)
                b=committed['ssb_steps'];history=committed['history'];save('FAILED','uncommitted_step_failure');raise
        save(status,reason)
        result=dict(status=status,reason=reason,ssb_steps=b,additional_accepted=b-base['ssb_steps'],target_ssb_steps=target,
                    checkpoint_sha=native.sha(ck),parent_sha256=plan['parent_sha256'],no_reference_access=True,
                    continuation_deadline_epoch=deadline)
        (out/'RESULT.json').write_text(json.dumps(result,indent=2));return result
    finally:
        lock.close()
        for sig,h in old.items():signal.signal(sig,h)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--parent-sha',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    p.add_argument('--additional-steps',type=int,default=200);p.add_argument('--seconds',type=float,default=450);a=p.parse_args()
    bindings=json.loads((SOURCE/'PARENT_BINDING.json').read_text(encoding='utf-8-sig'))
    matches=[v for v in bindings.values() if v['checkpoint_sha256']==a.parent_sha]
    assert len(matches)==1, 'Parent must match the registered binding'
    plan=dict(parent_sha256=a.parent_sha,sources=matches[0]['sources'],additional_updates=a.additional_steps,train_seconds=a.seconds)
    r=run(plan,a.parent,a.out,a.device,a.resume);print(json.dumps(r),flush=True)
    raise SystemExit(0 if r['status']=='COMPLETE' else 75 if r['status']=='INTERRUPTED' else 2)
