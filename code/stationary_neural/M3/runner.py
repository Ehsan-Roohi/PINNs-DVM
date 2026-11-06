"""One bounded, deterministic SSB phase; never submits another job."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
import base_solver as base
from objective import DeterministicObjective,make_points
from ssb_optimizer import DenseSSB

ROOT=Path(__file__).resolve().parent
STOP=False
def stop(*_):
    global STOP
    STOP=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signature():return sha(ROOT/'source.sha256')
def save(path,model,opt,obj,step,identity):
    state=dict(signature=identity,model=model.state_dict(),optimizer=opt.state_dict(),
               frozen=obj.frozen_state(),frozen_hash=obj.frozen_hash,step=step,
               python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    tmp=path.with_suffix('.tmp');torch.save(state,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,path)

def resolution_check(model,grid,out):
    import physics_gate
    report=base.save_profiles(model,grid,out)
    mc=model.c;state=base.rh(mc);device=grid[0].device
    xx=torch.linspace(-mc['L'],mc['L'],mc['profile_n'],device=device)
    extra=base.quadrature(320,96,mc['vmax'],device)
    with torch.no_grad():
        first=torch.cat([base.profiles(model,z,grid) for z in xx.split(2)]).cpu().numpy()
        fine=torch.cat([base.profiles(model,z,extra) for z in xx.split(2)]).cpu().numpy()
    scales=np.array([state['rho2'],state['u1'],state['T2'],state['rho2']*state['T2']**1.5,state['rho2']*state['T2'],state['rho2']*state['T2']**2,state['rho2']*state['T2']**2,state['rho2']*state['T2']**2])
    report['independent_320x96_max']=np.max(abs(fine-first)/scales,axis=0).tolist()
    report['resolution_max']=max(max(report['normalized_refinement_max']),max(report['independent_320x96_max']))
    report['velocity_zero']=physics_gate.velocity_zero(model,mc,extra,device)
    np.savez(Path(out)/'independent_320_profiles.npz',x=xx.cpu().numpy(),fields=fine)
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='runs/M3_s101');args=ap.parse_args()
    c=json.loads((ROOT/'config.json').read_text());mc=json.loads((ROOT/'model_config.json').read_text())
    torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
    assert torch.cuda.is_available(),'Production continuation requires allocated GPU'
    device='cuda';base.seed_all(mc['seed'])
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=signature();cp=out/'checkpoint.pt';initial=ROOT/'input_checkpoint.pt'
        assert sha(initial)==c['input_checkpoint_sha256']
        model=base.Distribution(mc).to(device)
        previous=torch.load(cp if cp.exists() else initial,map_location='cpu',weights_only=False)
        if cp.exists():assert previous['signature']==identity
        else:assert previous['signature']==c['input_signature'] and previous['step']==c['input_step']
        model.load_state_dict(previous['model'])
        grid=base.quadrature(mc['nvx'],mc['nr'],mc['vmax'],device)
        x=make_points(c,device)
        obj=DeterministicObjective(model,mc,grid,x,chunk_size=c['chunk_size'],
                                   frozen_state=previous['frozen'] if cp.exists() else None)
        opt=DenseSSB(len(obj.get_flat()),device=device,c1=c['c1'],c2=c['c2'],
                     max_evals=c['max_evals'],max_step=c['max_step'],initial_step=c['initial_step'],gtol=c['gtol'])
        step=0
        if cp.exists():
            assert obj.frozen_hash==previous['frozen_hash'];opt.load_state_dict(previous['optimizer']);step=previous['step']
            random.setstate(previous['python_rng']);np.random.set_state(previous['numpy_rng']);torch.set_rng_state(previous['torch_rng'])
            if previous['cuda_rng'] is not None:torch.cuda.set_rng_state_all(previous['cuda_rng'])
        base.write_json(out/'information_ledger.json',dict(signature=identity,input_checkpoint_sha256=sha(initial),
            fixed_objective_hash=obj.frozen_hash,training_information='RH and Shakhov only; no DVM profiles or errors',
            new_optimizer_phase=True,new_ansatz_and_quadrature=True,start_step=step,source_adam_steps=20000,source_ssb_steps=2044,source_mach=2,target_mach=3,mach_continuation=True,
            objective_change='tail weights frozen once from own initial checkpoint; final scalar weights fixed',
            source_model_config_sha256=sha(ROOT/'model_config.json')))
        with (out/'hardware.jsonl').open('a') as f:f.write(json.dumps(dict(job=os.getenv('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),torch=torch.__version__,start_step=step))+'\n')
        if not cp.exists():
            baseline=base.validation(model,grid)
            baseline['refinement']=resolution_check(model,grid,out)
            assert baseline['refinement']['resolution_max']<c['resolution_limit'],'Unresolved initial model'
            base.write_json(out/'initial_resolution.json',baseline['refinement'])
            os.replace(out/'physics_profiles.npz',out/'baseline_physics_profiles.npz')
            base.write_json(out/'BASELINE.json',baseline)
            save(cp,model,opt,obj,step,identity)
        start=time.monotonic();reason='step_budget'
        while step<c['steps']:
            if STOP:reason='signal_checkpoint';break
            if time.monotonic()-start>=c['wall_seconds']:reason='wall_budget';break
            result=opt.step(obj.get_flat(),obj.value_and_grad)
            row=dict(step=step+int(result['accepted']),accepted=result['accepted'],loss=result['loss'],
                alpha=result['alpha'],reason=result['reason'],evaluations=result['evaluations'],wolfe=result['wolfe'],
                curvature=result['curvature'],seconds=time.monotonic()-start,
                raw_terms=obj.last_terms,metrics=obj.last_metrics)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row,allow_nan=False),flush=True)
            if not result['accepted']:
                reason=result['reason'];break
            step+=1
            if step%c['checkpoint_every']==0:save(cp,model,opt,obj,step,identity)
            if step%c['resolution_every']==0:
                resolution=resolution_check(model,grid,out);resolution['step']=step
                with (out/'resolution.jsonl').open('a') as f:f.write(json.dumps(resolution)+'\n')
                if resolution['resolution_max']>c['resolution_limit']:
                    reason='quadrature_failure';break
            if step%c['validation_every']==0:
                val=base.validation(model,grid);val['step']=step
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(val)+'\n')
        save(cp,model,opt,obj,step,identity)
        val=base.validation(model,grid);refinement=resolution_check(model,grid,out)
        if refinement['resolution_max']>c['resolution_limit']:reason='quadrature_failure'
        checks=base.acceptance(mc,val,refinement)
        checks['physics_checks']['quadrature']=refinement['resolution_max']<c['resolution_limit']
        checks['local_physics_gate_pass']=all(checks['physics_checks'].values())
        final_loss,gradient=obj.value_and_grad(obj.get_flat())
        status=dict(step=step,finished=step>=c['steps'] and reason!='quadrature_failure',reason=reason,signature=identity,
            frozen_objective_hash=obj.frozen_hash,loss=final_loss,gradient_norm=float(torch.linalg.vector_norm(gradient)),
            validation=val,refinement=refinement,acceptance=checks,
            checkpoint_sha256=sha(cp),peak_gpu_bytes=torch.cuda.max_memory_allocated())
        base.write_json(out/'STATUS.json',status)
        if status['finished']:base.write_json(out/'DONE.json',status)
        else:base.write_json(out/'STOPPED.json',status)
        print(json.dumps(status),flush=True)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop);main()
