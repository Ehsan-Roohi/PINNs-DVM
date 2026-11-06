"""One blind Mach12 arm. Mathematical engine adapted verbatim from M8.
No DVM solution values, optimizer/state copying or Mach10 reference.
"""
from m12_physics_runtime import *
from scipy.interpolate import BSpline
import time,traceback,datetime
PLAN=W/'M12_SPLIT235_PLAN_20261031';protocol=json.loads((PLAN/'FROZEN_PROTOCOL.json').read_text());arm=sys.argv[1];assert arm in protocol['arms']
D=W/('M12_SPLIT235_'+arm.upper()+'_20261031');D.mkdir(exist_ok=False);start=time.monotonic()
def save(n,z):(D/n).write_text(json.dumps(z,indent=2),encoding='utf-8')
def budget():
 if time.monotonic()-start>900:raise TimeoutError('Frozen900s CPU arm budget; no extension/retry/resume')
plan={'intent':'Mach12 blind physics-only paired initializer trial','arm':arm,'protocol_SHA':sha(PLAN/'FROZEN_PROTOCOL.json'),'protocol':protocol,'new_HPC_jobs':0,'reference12_solution_values_read':False,'stopping':'lastacceptedstate only; smallstep is not convergence'}
save('INTENT.json',plan);pins=json.loads((PLAN/'INPUT_PINS.json').read_text());pins[str(Path(__file__))]=sha(Path(__file__));assert all(sha(Path(k))==v for k,v in pins.items());save('SOURCE_PINS.json',pins)
source=PLAN/(arm+'_SEED.npz');sourceSHA=sha(source);inp=np.load(source);seedX=inp['x'];seedA=inp['fields'];design=np.load(PLAN/'FIXED_GEOMETRY.npz');coarseX=design['x'];fineX=design['fine_x'];knots=design['knots'];ci=np.searchsorted(fineX,coarseX)
def bspline(x,nb=235):
 return BSpline(knots,np.eye(235),3)(np.tanh(x/J))*envelope(t.tensor(x)).numpy()[:,None]
_,_,Vh=np.linalg.svd(bspline(np.array([0.])),full_matrices=True);null=Vh[1:].T;Pnum=469;initialCoef=np.zeros(Pnum)
def geometry(x):
 B=bspline(x);Cu=np.c_[J*(B@null),np.zeros((len(x),235))];CT=np.c_[np.zeros((len(x),234)),Ts*B];E=envelope(t.tensor(x)).numpy();u0=np.interp(x,seedX,seedA[:,1]);T0=np.interp(x,seedX,seedA[:,2]);u0=E*u0+(1-E)*np.where(x<0,J,rh['u2']);T0=E*T0+(1-E)*np.where(x<0,1.,Ts);return u0,T0,Cu,CT
def profile(coef,geo):
 u0,T0,Cu,CT=geo;u=u0+Cu@coef;T=T0+CT@coef;assert np.min(u)>0 and np.min(T)>0;return np.c_[J/u,u,T]
from m8_stream_characteristic import full_sparse,reduced_fields
upstream=np.array([1.,J,1.]);downstream=np.array([rs,rh['u2'],Ts]);USE_FULL_REPORT=False
def full_transport(x,a,grid,Cu=None,CT=None):
 assert Cu is None and CT is None
 _,_,actual=full_sparse(x,a,grid,upstream,downstream,[],budget)
 return None,actual,(actual-a)/np.array([rs,J,Ts]),None

def transport(x,a,grid,Cu=None,CT=None):
 if Cu is None and USE_FULL_REPORT:return full_transport(x,a,grid)
 budget();vv,rr,ww=[z.numpy()for z in grid];v,inv=np.unique(vv,return_inverse=True);C=float(cfg['vmax']);wv=np.bincount(inv,weights=ww)/(np.pi*C*C);nx=len(x);jac=Cu is not None
 raw=np.zeros((nx,3));draw=np.zeros((nx,3,Pnum))if jac else None
 def source1(aa,mask):
  vm=v[mask];g=aa[0]/np.sqrt(2*np.pi*aa[2])*np.exp(-(vm-aa[1])**2/(2*aa[2]));return g,2*aa[2]*g
 for direction in [1,-1]:
  mask=v>0 if direction==1 else v<0;vm=v[mask];wm=wv[mask];anchor=0 if direction==1 else nx-1
  fg,fh=source1(np.array([1.,J,1.])if direction==1 else np.array([rs,rh['u2'],Ts]),mask);oldg,oldh=source1(a[anchor],mask)
  if jac:dg=np.zeros((len(vm),Pnum));dh=dg.copy();oldDg=dg.copy();oldDh=dg.copy()
  raw[anchor]+=[wm@fg,(wm*vm)@fg,wm@(vm*vm*fg+fh)]
  for i in (range(1,nx)if direction==1 else range(nx-2,-1,-1)):
   if i%40==0:budget()
   prev=i-direction;q=abs(x[i]-x[prev])/abs(vm);alpha=np.exp(-q);z=-np.expm1(-q)/q;b=z-alpha;c=1-z;g,h=source1(a[i],mask)
   fg=alpha*fg+b*oldg+c*g;fh=alpha*fh+b*oldh+c*h;raw[i]+=[wm@fg,(wm*vm)@fg,wm@(vm*vm*fg+fh)]
   if jac:
    ru=(vm-a[i,1])/a[i,2]-1/a[i,1];rt=-.5/a[i,2]+(vm-a[i,1])**2/(2*a[i,2]**2);Dg=(g*ru)[:,None]*Cu[i]+(g*rt)[:,None]*CT[i];Dh=2*a[i,2]*Dg+2*g[:,None]*CT[i]
    dg=alpha[:,None]*dg+b[:,None]*oldDg+c[:,None]*Dg;dh=alpha[:,None]*dh+b[:,None]*oldDh+c[:,None]*Dh;draw[i,0]+=wm@dg;draw[i,1]+=(wm*vm)@dg;draw[i,2]+=(wm*vm*vm)@dg+wm@dh;oldDg,oldDh=Dg,Dh
   oldg,oldh=g,h
 rho=raw[:,0];u=raw[:,1]/rho;T=(raw[:,2]/rho-u*u)/3;actual=np.c_[rho,u,T];G=(actual-a)/np.array([rs,J,Ts]);Jac=None
 if jac:
  dr=draw[:,0];du=(draw[:,1]-u[:,None]*dr)/rho[:,None];dT=(draw[:,2]-(u*u+3*T)[:,None]*dr-2*(rho*u)[:,None]*du)/(3*rho[:,None]);Jac=np.stack(((dr+J/a[:,1,None]**2*Cu)/rs,(du-Cu)/J,(dT-CT)/Ts),axis=1)
 return None,actual,G,Jac

def weights_space(x):
 h=np.diff(x);w=np.r_[h[0]/2,(h[:-1]+h[1:])/2,h[-1]/2];return w/(x[-1]-x[0])
try:
 geo=geometry(coarseX);coef=initialCoef.copy();a=profile(coef,geo);np.savez(D/'FROZEN_DESIGN.npz',x=coarseX,fine_x=fineX,seed_x=seedX,seed_macro=seedA,knots=knots,phase_null=null,u_base=geo[0],T_base=geo[1],Cu=geo[2],CT=geo[3]);tm=time.monotonic();F,actual,G,Jac=transport(coarseX,a,PRIMARY,geo[2],geo[3]);tangent_seconds=time.monotonic()-tm;sw=weights_space(coarseX);sqrtw=np.sqrt(sw)[:,None];A=(Jac*sqrtw[:,:,None]).reshape(-1,Pnum);res=(G*sqrtw).reshape(-1);loss=float(res@res);baseline_loss=loss;save('BASELINE.json',{'loss':loss,'max_macro_defect':abs(G).max(axis=0).tolist(),'full_tangent_seconds':tangent_seconds});del F
 save('INITIALIZER_ADMISSION.json',{'arm':arm,'seed_SHA':sourceSHA,'reference_solution_values_read':False,'optimizer_or_RNG_copied':False,'mass_source_exact':float(abs(a[:,0]*a[:,1]-J).max()),'phase_source_rho':float(a[np.argmin(abs(coarseX)),0]),'phase_target_rho':(1+rs)/2})
 oldDir=W/('M12_BASIS235_'+arm.upper()+'_20261031');assert (oldDir/'FAILURE.json').exists()
 oldPins=json.loads((oldDir/'SOURCE_PINS.json').read_text());assert all(sha(Path(k))==v for k,v in oldPins.items())
 for n in ['from_m8_SEED.npz','FIXED_GEOMETRY.npz','TRAIN_GEOMETRY.npz']:assert sha(PLAN/n)==sha(W/'M12_BASIS235_PLAN_20261031'/n)
 def function_AST(path,name):
  tree=ast.parse(path.read_text());return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
 for name in ['bspline','geometry','profile','transport','full_transport','weights_space']:assert function_AST(Path(__file__),name)==function_AST(W/'run_m12_basis235.py',name)
 oldBaseline=json.loads((oldDir/'BASELINE.json').read_text());assert abs(oldBaseline['loss']-baseline_loss)<1e-14
 parityReceipt=json.loads((oldDir/'ANALYTIC_RADIAL_PARITY.json').read_text());fdReceipt=json.loads((oldDir/'TANGENT_FIXTURE.json').read_text());assert parityReceipt['PASS'] and fdReceipt['PASS'];primal_seconds=parityReceipt['full_primal_seconds']
 save('INITIAL_GATES_REUSE.json',{'PASS':True,'reason':'Identical235 basis, seed, physical dependencies and complete math function ASTs; new initial loss matches; no duplicate FD/full-primal test','old_arm':str(oldDir),'parity_receipt_SHA':sha(oldDir/'ANALYTIC_RADIAL_PARITY.json'),'FD_receipt_SHA':sha(oldDir/'TANGENT_FIXTURE.json'),'reference_solution_values_read':False})
 tm=time.monotonic();_,_,sameG,_=transport(coarseX,a,PRIMARY);reduced_primal_seconds=time.monotonic()-tm;assert abs(sameG-G).max()<1e-12
 remaining=900-(time.monotonic()-start);forecast=(tangent_seconds+4*reduced_primal_seconds)*1.25+60;save('COST_ADMISSION.json',{'full_tangent_seconds':tangent_seconds,'full_primal_seconds':primal_seconds,'line_search_reduced_primal_seconds':reduced_primal_seconds,'forecast_seconds':forecast,'remaining_seconds':remaining,'pass':forecast<remaining});assert forecast<remaining,'Fixed2 refinement cannot fit; no macro update launched'
 history=[];ll=baseline_loss;stopping_reason='MAXIMUM2_STEPS'
 for stepIndex in range(1,3):
  budget()
  if stepIndex>1:
   F,actual,G,Jac=transport(coarseX,profile(coef,geo),PRIMARY,geo[2],geo[3]);del F;A=(Jac*sqrtw[:,:,None]).reshape(-1,Pnum);res=(G*sqrtw).reshape(-1);loss=float(res@res)
  change,_,rank,sing=np.linalg.lstsq(A,-res,rcond=1e-12);assert rank==Pnum,'Rank-deficient macro direction; no ridge';trust=max(abs(geo[2]@change).max()/J,abs(geo[3]@change).max()/Ts)
  if trust<1e-7:ll=loss;stopping_reason='NUMERICAL_SMALL_STEP_NOT_CONVERGENCE';break
  scale=min(1.,.005/max(trust,1e-30));change*=scale;slope=float((A.T@res)@change);accepted=False;trials=[]
  for k in range(8):
   alpha=2.**(-k);trial=coef+alpha*change
   try:ma=profile(trial,geo)
   except AssertionError:trials.append({'alpha':alpha,'positive':False});continue
   ff,aa,gg,_=transport(coarseX,ma,PRIMARY);ll=float(np.sum(sw[:,None]*gg*gg));del ff;trials.append({'alpha':alpha,'loss':ll})
   if ll<=loss+1e-4*alpha*slope:coef=trial;accepted=True;break
  if not accepted:save('ARMIJO_STALL.json',{'reason':'fixed8 Armijo limit','completed_updates':len(history),'trials':trials,'seconds':time.monotonic()-start});ll=loss;stopping_reason='ARMIJO_STALL_NOT_CONVERGENCE';break
  row={'step':stepIndex,'loss_before':loss,'loss_after':ll,'rank':int(rank),'singular_minmax_ratio':float(sing[-1]/sing[0]),'trust_scale':scale,'alpha':alpha,'max_step_du_over_J':float(abs(alpha*geo[2]@change).max()/J),'max_step_dT_over_T2':float(abs(alpha*geo[3]@change).max()/Ts),'trials':trials};history.append(row);save('HISTORY.json',history);checkpoint=D/f'STATE_STEP{stepIndex:03d}.npz';assert not checkpoint.exists();np.savez(checkpoint,step=stepIndex,coefficients=coef,x=coarseX,source_macro_input_SHA=sourceSHA);save('LATEST_POINTER.json',{'step':stepIndex,'checkpoint':checkpoint.name,'SHA':sha(checkpoint)});save('PROGRESS.json',{'step':stepIndex,'seconds':time.monotonic()-start,'loss':ll});print(json.dumps({'step':stepIndex,'loss':ll,'seconds':time.monotonic()-start}),flush=True)
 frozen={'complete_bounded_trial':True,'actual_macro_updates':len(history),'stopping_reason':stopping_reason,'plan':plan,'coefficients':coef.tolist(),'source_macro_SHA':sourceSHA,'baseline_loss':baseline_loss,'final_training_loss':ll,'neural_optimizer_updates':0,'analytic_macro_updates':len(history),'source_M8_unchanged':True,'production_admitted':False,'seconds':time.monotonic()-start};save('FROZEN_CANDIDATE.json',frozen);frozenSHA=sha(D/'FROZEN_CANDIDATE.json');save('FREEZE_ACK.json',{'SHA256':frozenSHA,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before_space_velocity_refinement':True})
 save('CORRECTION_RESULT.json',dict(frozen,candidate_SHA=frozenSHA,independent_validation_pending=True,convergence_certified=False));print(json.dumps({'correction_complete':True,'updates':len(history),'candidate_SHA':frozenSHA,'validation_pending':True}));raise SystemExit(0)
except SystemExit:raise
except BaseException as exc:save('FAILURE.json',{'type':type(exc).__name__,'message':str(exc),'seconds':time.monotonic()-start,'traceback':traceback.format_exc()});raise
