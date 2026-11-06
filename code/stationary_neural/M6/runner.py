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
import validation as independent
from objective import DeterministicObjective,make_points
from ssb_optimizer import DenseSSB

ROOT=Path(__file__).resolve().parent
STOP=False
DIAGNOSTICS={}
def stop(*_):
    global STOP
    STOP=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signature():return sha(ROOT/'source.sha256')
def save(path,model,opt,obj,step,identity):
    state=dict(signature=identity,model=model.state_dict(),optimizer=opt.state_dict(),
               frozen=obj.frozen_state(),frozen_hash=obj.frozen_hash,step=step,
               python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,validation_state=DIAGNOSTICS)
    tmp=path.with_suffix('.tmp');torch.save(state,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,path)

def resolution_check(model,grid,out):
    import physics_gate
    report=base.save_profiles(model,grid,out)
    mc=model.c;state=base.rh(mc);device=grid[0].device
    xx=torch.linspace(-mc['L'],mc['L'],mc['profile_n'],device=device)
    guard_config=json.loads((ROOT/'config.json').read_text())
    guard_nvx,guard_nr=guard_config['guard_nvx'],guard_config['guard_nr']
    assert guard_nvx>mc['refine_nvx']>mc['nvx'] and guard_nr>mc['refine_nr']>mc['nr']
    extra=base.quadrature(guard_nvx,guard_nr,mc['vmax'],device)
    with torch.no_grad():
        first=torch.cat([base.profiles(model,z,grid) for z in xx.split(2)]).cpu().numpy()
        fine=torch.cat([base.profiles(model,z,extra) for z in xx.split(2)]).cpu().numpy()
    scales=np.array([state['rho2'],state['u1'],state['T2'],state['rho2']*state['T2']**1.5,state['rho2']*state['T2'],state['rho2']*state['T2']**2,state['rho2']*state['T2']**2,state['rho2']*state['T2']**2])
    report['independent_guard_max']=np.max(abs(fine-first)/scales,axis=0).tolist()
    report['independent_guard_order']=[guard_nvx,guard_nr]
    report['resolution_max']=max(max(report['normalized_refinement_max']),max(report['independent_guard_max']))
    report['velocity_zero']=physics_gate.velocity_zero(model,mc,extra,device)
    np.savez(Path(out)/'independent_guard_profiles.npz',x=xx.cpu().numpy(),fields=fine)
    return report

def restored_reason(diagnostics):
    guard=diagnostics.get('guard',{})
    return guard.get('reason') if guard.get('stop') else 'step_budget'

def completed(step,budget,reason):return step>=budget and reason=='step_budget'

def main():
    global DIAGNOSTICS
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='runs/M6_s101');args=ap.parse_args()
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
        if cp.exists():
            DIAGNOSTICS=previous['validation_state']
            for name in ('history','validation','resolution','independent_validation'):
                log=out/(name+'.jsonl')
                if log.exists():
                    rows=log.read_text().splitlines();keep=[r for r in rows if json.loads(r).get('step',0)<=step]
                    if len(keep)!=len(rows):
                        log.rename(out/(name+'.uncommitted_'+str(time.time_ns())+'.jsonl'))
                        log.write_text(''.join(r+'\n' for r in keep))
        else:
            frozen_validation=independent.make_frozen_state(model,grid)
            initial_groups=independent.evaluate(model,grid,frozen_state=frozen_validation)
            DIAGNOSTICS=dict(frozen=frozen_validation,baseline=initial_groups,guard={},last_guard_step=-1)
            base.write_json(out/'INDEPENDENT_BASELINE.json',initial_groups)
        base.write_json(out/'information_ledger.json',dict(signature=identity,input_checkpoint_sha256=sha(initial),
            fixed_objective_hash=obj.frozen_hash,training_information='RH and ordinary BGK (Pr=1, constant tau=1) only; no DVM profiles or errors',
            new_optimizer_phase=False,new_ansatz_and_quadrature=False,start_step=step,source_adam_steps=20000,source_ssb_steps=c['input_step'],source_mach=c['source_mach'],target_mach=c['target_mach'],mach_continuation=False,collision_model_change=False,spatial_repair=False,spatial_count=187,fresh_H=False,
            objective_change='NONE: exact saved frozen objective and original baseline retained',
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
        start=float(os.environ.get('BGK6_ALLOCATION_MONOTONIC_START',time.monotonic()));reason=restored_reason(DIAGNOSTICS)
        while step<c['steps'] and reason=='step_budget':
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
            if step%c['resolution_every']==0:
                resolution=resolution_check(model,grid,out);resolution['step']=step
                with (out/'resolution.jsonl').open('a') as f:f.write(json.dumps(resolution)+'\n')
                if resolution['resolution_max']>c['resolution_limit']:
                    reason='quadrature_failure';break
            if step%c['validation_every']==0:
                val=base.validation(model,grid);val['step']=step
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(val)+'\n')
            if step%c['resolution_every']==0:
                groups=independent.evaluate(model,grid,frozen_state=DIAGNOSTICS['frozen'])
                if DIAGNOSTICS['last_guard_step']!=step:
                    DIAGNOSTICS['guard']=independent.guard(groups,DIAGNOSTICS['baseline'],DIAGNOSTICS['guard'],step)
                    DIAGNOSTICS['last_guard_step']=step
                with (out/'independent_validation.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,groups=groups,guard=DIAGNOSTICS['guard']))+'\n')
                if DIAGNOSTICS['guard']['stop']:
                    reason=DIAGNOSTICS['guard']['reason'];break
            if step%c['checkpoint_every']==0:save(cp,model,opt,obj,step,identity)
        save(cp,model,opt,obj,step,identity)
        val=base.validation(model,grid);refinement=resolution_check(model,grid,out)
        if refinement['resolution_max']>c['resolution_limit']:reason='quadrature_failure'
        groups=independent.evaluate(model,grid,frozen_state=DIAGNOSTICS['frozen'])
        independent_checks=independent.acceptance(groups,refinement['resolution_max'])
        independent_checks['allow_mach_advance']=independent_checks['allow_mach_advance'] and reason=='step_budget'
        progress=independent.repair_progress(groups,DIAGNOSTICS['baseline'])
        base.write_json(out/'INDEPENDENT_FINAL.json',dict(groups=groups,acceptance=independent_checks,repair_progress=progress))
        checks=base.acceptance(mc,val,refinement)
        checks['physics_checks']['quadrature']=refinement['resolution_max']<c['resolution_limit']
        checks['local_physics_gate_pass']=all(checks['physics_checks'].values())
        final_loss,gradient=obj.value_and_grad(obj.get_flat())
        status=dict(step=step,finished=completed(step,c['steps'],reason),reason=reason,signature=identity,
            frozen_objective_hash=obj.frozen_hash,loss=final_loss,gradient_norm=float(torch.linalg.vector_norm(gradient)),
            validation=val,independent_validation=groups,independent_acceptance=independent_checks,repair_progress=progress,refinement=refinement,acceptance=checks,
            checkpoint_sha256=sha(cp),peak_gpu_bytes=torch.cuda.max_memory_allocated())
        base.write_json(out/'STATUS.json',status)
        if status['finished']:base.write_json(out/'DONE.json',status)
        else:base.write_json(out/'STOPPED.json',status)
        print(json.dumps(status),flush=True)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop);main()

