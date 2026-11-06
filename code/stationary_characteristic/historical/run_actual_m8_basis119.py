"""Physics-only joint macro/kinetic correction from frozen own NN macro seed.

Positive exact characteristic transport for piecewise-linear Maxwellian source.

Analytic tangent sweep, no DVM labels, no pseudo-label training, no NN updates.

"""

from m8_report_physics_runtime import *

import json,time,traceback,datetime

D=W/'M8_ACTUAL3975_BASIS119_20261030';D.mkdir(exist_ok=False);start=time.monotonic()

def save(n,z):(D/n).write_text(json.dumps(z,indent=2),encoding='utf-8')

def budget():

 if time.monotonic()-start>900:raise TimeoutError('Frozen900s CPU budget; no automatic extension')

source=W/'M8_ACTUAL3975_MACRO_INPUT_20261030/FROZEN_MACROS.npz';sourceSHA='e1a5273d7ab5f7cb8a2ac857a804ea13ff7ed3dd59204accad0c04813f44cd2d';assert sha(source)==sourceSHA

plan={'intent':'nested61to119 basis and2401space refinement after measured necessary kinetic/projection FAIL, exact parent field continuation; no DVM access','source':'ONLY actual originalMach8 step3975 own NN macro INPUT samples; no Mach7warmstart, characteristic output or DVM labels','source_SHA':sourceSHA,'macro_basis':119,'parameters':237,'parameterization':'rho=J/u; own phase/mass macro seed +compact physical-length cubic splines; u correction null atcenter; equilibrium RH macro endpoints','kinetic_field':'positive characteristic solution of own Maxwellian source, incoming RHexact, no NN output fitted to another solver','objective':'spatial trapezoid mean squared fixedRHnormalized self-consistency defects of rho,u,T','derivative':'analytic perpendicular Maxwellian integrals g/h for macro tangent, parity against FULL512x144 distribution plus centralFD, fullF authoritative for every objective and finalphysics','maximum_steps':4,'continuation_source_SHA':'83859c5e8c97a39087e045bdfb323ca1cccb8f3b04351f1474322483acb0dbc5','continuation_prefix_updates':'20+accepted prior enriched updates','numerical_small_step_stop':1e-7,'Armijo_stall_stop':'last actually accepted state; no convergence claim','step_limit':'max local du/J,dT/T2 <=0.005 per step; fixed8 Armijo halvings; no best endpoint selection','training_grid':'2401 fixed source macro positions,512/144,C48','postfreeze_refinement':'all8 moment profiles and fluxes at2401/4801space; fullkinetic residual at SAME original64 GL8 positions eachlevel;640/176 velocity; no refitting','original_absolute_criteria':'unchanged and checked on finalfield, never inferred from objective','threads':4,'wall_seconds':900,'cost_admission':'4 measured tangent plus2 actualanalytic primals perstep, fullfinal3levels6.5x measured fullprimal,25percentmargin+120s; same900s; fullFD2h mandatory before updates','new_HPC_jobs':0,'DVM_access':False,'neural_optimizer_updates':0,'all_original_M8_HOLDs_remain':True,'full_distribution_storage':'exact streaming; only original64+center+endpoints fullF and sourceM retained on LexarD, allnode macros and highermoments analyticradial with allquery fullquadrature parity','no_convergence_certificate_from_local_trial':True}

save('INTENT.json',plan);pins={str(W/'m8_report_physics_runtime.py'):sha(W/'m8_report_physics_runtime.py'),str(W/'m8_stream_characteristic.py'):sha(W/'m8_stream_characteristic.py'),str(Path(__file__)):sha(Path(__file__)),str(source):sha(source),str(W/'M8_ACTUAL3975_DENSE_STREAMED_COST_R2_20261030/FROZEN_CANDIDATE.json'):sha(W/'M8_ACTUAL3975_DENSE_STREAMED_COST_R2_20261030/FROZEN_CANDIDATE.json'),str(SOURCE):sha(SOURCE),str(P/'bgk_pr1_tau1_M8/base_solver.py'):sha(P/'bgk_pr1_tau1_M8/base_solver.py')};save('SOURCE_PINS.json',pins)

import sys

sys.path.insert(0,str(W/'probe-convex-runtime'))

from scipy.interpolate import BSpline

priorD=W/'M8_ACTUAL3975_DENSE_STREAMED_COST_R2_20261030'

priorResult=json.loads((priorD/'FROZEN_CANDIDATE.json').read_text());assert priorResult['complete_bounded_trial'];priorResult['candidate_SHA']=sha(priorD/'FROZEN_CANDIDATE.json');assert (priorD/'POSTFREEZE_INTERRUPTION.json').exists()

oldDesign=np.load(priorD/'FROZEN_DESIGN.npz');oldknots=oldDesign['knots'];unique=np.unique(oldknots);insertions=(unique[:-1]+unique[1:])/2;knots=np.sort(np.r_[oldknots,insertions]);assert len(knots)-4==119

def bspline(x,nb=119):

 z=np.tanh(x/J);return BSpline(knots,np.eye(119),3)(z)*envelope(t.tensor(x)).numpy()[:,None]

inp=np.load(source);seedX=inp['x'];seedA=inp['fields'];coarseX=oldDesign['fine_x'];fineX=np.sort(np.r_[coarseX,(coarseX[:-1]+coarseX[1:])/2]);ci=np.searchsorted(fineX,coarseX)

_,_,Vh=np.linalg.svd(bspline(np.array([0.])),full_matrices=True);null=Vh[1:].T;Pnum=237

oldCoef=np.array(priorResult['coefficients']);oldUc=oldDesign['phase_null']@oldCoef[:60];oldTc=oldCoef[60:]

def transfer(c):

 b=BSpline(oldknots,c,3)

 for z in insertions:b=b.insert_knot(z)

 assert np.array_equal(b.t,knots)

 return b.c

newUc=transfer(oldUc);newTc=transfer(oldTc);initialCoef=np.r_[null.T@newUc,newTc]

oldB=BSpline(oldknots,np.eye(61),3)(np.tanh(coarseX/J))*envelope(t.tensor(coarseX)).numpy()[:,None]

transferError=max(abs(bspline(coarseX)@newUc-oldB@oldUc).max(),abs(bspline(coarseX)@newTc-oldB@oldTc).max());assert transferError<1e-12

plan['continuation_source_SHA']=priorResult['candidate_SHA'];plan['continuation_prefix_updates']=25;save('INTENT.json',plan)

save('EXACT_NESTED_TRANSFER.json',{'maximum_coefficient_function_error':float(transferError),'basis61_to119_exact_knot_insertion':True,'source_macro_function_unchanged':True,'training_nodes':len(coarseX),'postfreeze_refinement_nodes':len(fineX),'reference_access':False})

def geometry(x):

 B=bspline(x);Cu=np.c_[J*(B@null),np.zeros((len(x),119))];CT=np.c_[np.zeros((len(x),118)),Ts*B];E=envelope(t.tensor(x)).numpy();u0=np.interp(x,seedX,seedA[:,1]);T0=np.interp(x,seedX,seedA[:,2]);u0=E*u0+(1-E)*np.where(x<0,J,rh['u2']);T0=E*T0+(1-E)*np.where(x<0,1.,Ts);return u0,T0,Cu,CT

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

 save('CONTINUATION_ADMISSION.json',{'source_candidate_SHA':sha(priorD/'FROZEN_CANDIDATE.json'),'prior_updates':25,'exact_nested_transfer_error':float(transferError),'spatial_mesh_refined_without_changing_source_function':True,'no_optimizer_or_rng_reset':True})

 tm=time.monotonic();fullF,fullActual,fullG,_=full_transport(coarseX,a,PRIMARY);primal_seconds=time.monotonic()-tm;del fullF;parity=float(abs(fullG-G).max());assert parity<1e-10;save('ANALYTIC_RADIAL_PARITY.json',{'PASS':True,'macro_defect_max_error_against_full512x144':parity,'analytic_g_h_all_radial_moments_for_mass_momentum_energy':True,'full_primal_seconds':primal_seconds,'finalfield_full_distribution':True})

 direction=np.sin(np.arange(1,Pnum+1));direction/=np.linalg.norm(direction);fd=[]

 for h in [1e-4,3e-5]:

  fp,_,gp,_=full_transport(coarseX,profile(coef+h*direction,geo),PRIMARY);del fp;fm,_,gm,_=full_transport(coarseX,profile(coef-h*direction,geo),PRIMARY);del fm;num=(gp-gm)/(2*h);ad=np.einsum('ijk,k->ij',Jac,direction);err=float(np.linalg.norm(num-ad)/np.linalg.norm(ad));assert err<1e-5;fd.append({'h':h,'relative_error':err})

 save('TANGENT_FIXTURE.json',{'PASS':True,'full_grid_directional_FD':fd,'Jacobian_parameter_count':Pnum})

 tm=time.monotonic();_,_,sameG,_=transport(coarseX,a,PRIMARY);reduced_primal_seconds=time.monotonic()-tm;assert abs(sameG-G).max()<1e-12

 remaining=900-(time.monotonic()-start);forecast=4*(tangent_seconds+2*reduced_primal_seconds)*1.25+6.5*primal_seconds*1.25+120;save('COST_ADMISSION.json',{'full_tangent_seconds':tangent_seconds,'full_primal_seconds':primal_seconds,'line_search_reduced_primal_seconds':reduced_primal_seconds,'forecast_seconds':forecast,'remaining_seconds':remaining,'pass':forecast<remaining});assert forecast<remaining,'Fixed4 cannot fit; no macro update launched'

 history=[];ll=baseline_loss;stopping_reason='MAXIMUM4_STEPS'

 for stepIndex in range(1,5):

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

 frozen={'complete_bounded_trial':True,'actual_macro_updates':len(history),'stopping_reason':stopping_reason,'plan':plan,'coefficients':coef.tolist(),'source_macro_SHA':sourceSHA,'baseline_loss':baseline_loss,'final_training_loss':ll,'neural_optimizer_updates':0,'analytic_macro_updates':len(history),'source_NN_unchanged':True,'production_admitted':False,'seconds':time.monotonic()-start};save('FROZEN_CANDIDATE.json',frozen);frozenSHA=sha(D/'FROZEN_CANDIDATE.json');save('FREEZE_ACK.json',{'SHA256':frozenSHA,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before_space_velocity_refinement':True})

 USE_FULL_REPORT=True

 def final_level(label,x,grid):

  ma=profile(coef,geometry(x));geom=np.load(W/'M8_ACTUAL3975_MACH7_PORT_20261030/TRAIN_GEOMETRY.npz');queries=np.r_[geom['x'],0.,-120.,120.]

  fields,qfields=reduced_fields(x,ma,grid,48.,upstream,downstream,queries,budget);Q,MQ,actual=full_sparse(x,ma,grid,upstream,downstream,queries,budget)

  parity=float(np.max(abs((fields[:,:3]-actual)/np.array([rs,J,Ts]))));assert parity<1e-10

  fullq=[]

  with t.no_grad():

   for j in range(0,len(queries),2):fullq.extend(base.integrate(t.from_numpy(Q[j:j+2]),*grid).numpy())

  fullq=np.array(fullq);scales=np.array([rs,J,Ts,rs*Ts**1.5,rs*Ts,rs*Ts**2,rs*Ts**2,rs*Ts**2]);parity8=np.max(abs(qfields-fullq)/scales,axis=0);assert max(parity8)<1e-9

  rho,u,T,q,sigma=fields[:,:5].T;flux=np.c_[rho*u/J,(rho*u*u+rho*T+sigma)/Pi,(.5*rho*u**3+2.5*rho*T*u+u*sigma+q)/energy]-1;G=(actual-ma)/np.array([rs,J,Ts]);terms=[];projected=[];full9=[]

  for j,z in enumerate(geom['x']):

   budget();diag=diagnostics(z,t.from_numpy(Q[j]),t.from_numpy((MQ[j]-Q[j])/grid[0].numpy()),grid);terms.append(diag['terms'].numpy());projected.append(diag['projected'].numpy());full9.append(float(diag['full9']@diag['full9']))

  terms=np.array(terms);projected=np.array(projected);phase=float(((fullq[-3,0]-(1+rs)/2)/((1+rs)/2))**2)

  v,r,w=grid;boundary=[]

  for j,state in [(-2,upstream),(-1,downstream)]:

   exact=base.maxwellian(v,r,t.tensor(state[0:1]),t.tensor(state[1:2]),t.tensor(state[2:3]))[0];den=exact+exact.max()*1e-7;boundary.append(float((((t.from_numpy(Q[j])-exact)/den)**2*exact*w).sum()/(exact*w).sum()))

  boundary=float(np.mean(boundary));save(f'{label}_PARITY.json',{'PASS':True,'allnode_primitive_max_error':parity,'all8_raw_central_fullquery_normalized_errors':parity8.tolist(),'query_count':len(queries),'all_original64_residual_points_unchanged':True,'no_synthetic_spatial_labels':True})

  backup=W/'full_query_output';assert backup.parent.exists();backup.mkdir(exist_ok=True);qfile=backup/f'{label}_FULL_QUERY.npz';assert not qfile.exists();np.savez_compressed(qfile,x=queries,F=Q,M_source=MQ,v=v.numpy(),r=r.numpy(),weights=w.numpy());save(f'{label}_FULL_QUERY_RECEIPT.json',{'path':str(qfile),'SHA256':sha(qfile),'bytes':qfile.stat().st_size,'queries':len(queries),'allfullqueryfields_preserved':True})

  np.savez(D/f'{label}.npz',x=x,source_macro=ma,fields=fields,terms=terms,flux=flux,projected=projected,residual_x=geom['x'],residual_weights=geom['spatial_weight'],macro_defect=G,query_x=queries,query_fields=fullq)

  return {'loss':float(np.sum(weights_space(x)[:,None]*G*G)),'max_macro_defect':abs(G).max(axis=0).tolist(),'kinetic_mean':float(geom['spatial_weight']@terms[:,0]),'tail_mean':float(geom['spatial_weight']@terms[:,1]),'raw_mean':float(geom['spatial_weight']@terms[:,2]),'full9_mean':float(geom['spatial_weight']@full9),'max_flux':float(abs(flux).max()),'max_projection':float(abs(projected).max()),'phase':phase,'boundary':boundary,'rho_min':float(fields[:,0].min()),'T_min':float(fields[:,2].min()),'negative_mass':float(np.maximum(-Q,0).max()),'actual_field_positive_by_transport':True,'fullkinetic_scope':'same original64 GL8 points and weights; allnode moments and all3fluxes with fullquery radial parity'}

 levels={};levels['PRIMARY_TRAIN_SPACE']=final_level('PRIMARY_TRAIN_SPACE',coarseX,PRIMARY);levels['PRIMARY_REFINED_SPACE']=final_level('PRIMARY_REFINED_SPACE',fineX,PRIMARY);levels['FINE_VELOCITY_REFINED_SPACE']=final_level('FINE_VELOCITY_REFINED_SPACE',fineX,base.quadrature(640,176,48.,'cpu'));co=np.load(D/'PRIMARY_TRAIN_SPACE.npz');fi=np.load(D/'PRIMARY_REFINED_SPACE.npz');vf=np.load(D/'FINE_VELOCITY_REFINED_SPACE.npz');scales=np.array([rs,J,Ts,rs*Ts**1.5,rs*Ts,rs*Ts**2,rs*Ts**2,rs*Ts**2]);space=np.max(abs(fi['fields'][ci]-co['fields'])/scales,axis=0);vel=np.max(abs(vf['fields']-fi['fields'])/scales,axis=0);assert frozenSHA==sha(D/'FROZEN_CANDIDATE.json')and all(sha(Path(k))==v for k,v in pins.items())

 out={'complete_bounded_trial':True,'actual_macro_updates':len(history),'stopping_reason':stopping_reason,'seconds':time.monotonic()-start,'plan':plan,'candidate_SHA':frozenSHA,'baseline_loss':baseline_loss,'final_training_loss':ll,'levels':levels,'normalized_space_refinement_max':space.tolist(),'normalized_velocity_refinement_max':vel.tolist(),'maximum_total_du_over_J':float(abs(geo[2]@coef).max()/J),'maximum_total_dT_over_T2':float(abs(geo[3]@coef).max()/Ts),'source_NN_unchanged':True,'production_admitted':False,'convergence_certified':False,'new_HPC_jobs':0};save('RESULT.json',out);print(json.dumps(out))

except SystemExit:raise

except BaseException as exc:save('FAILURE.json',{'type':type(exc).__name__,'message':str(exc),'seconds':time.monotonic()-start,'traceback':traceback.format_exc()});raise

