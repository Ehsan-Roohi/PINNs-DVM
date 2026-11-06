"""New explicit finite two-stage protocol, not an extension of failed attempts."""
from pathlib import Path
import ast,json,hashlib,datetime
W=Path(__file__).parent;old=W/'M12_BASIS235_COSTFIX_PLAN_20261031';new=W/'M12_SPLIT235_PLAN_20261031';new.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert (W/'M12_BASIS235_FROM_M8_20261031/FAILURE.json').exists()
p=json.loads((old/'FROZEN_PROTOCOL.json').read_text());p['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p['intent']='Two distinct capped stages: <=2 macro corrections, freeze, then read-only full3level qualification; no reference values'
p['stage_budgets']={'correction_seconds':900,'independent_three_level_validation_seconds':900,'total_new_two_stage_wall_cap':1800,'no_automatic_extension':True}
p['shared_algorithm']['cost_admission']='Correction stage: one REMAINING tangent+4reducedprimals,25percentmargin+60s must fit remaining900s. Validation independently capped900s.'
p['large_outputs']=str(W/'full_query_output');Path(p['large_outputs']).mkdir(exist_ok=False)
for n in ('FIXED_GEOMETRY.npz','TRAIN_GEOMETRY.npz','from_m8_SEED.npz'):(new/n).write_bytes((old/n).read_bytes())
(new/'FROZEN_PROTOCOL.json').write_text(json.dumps(p,indent=2));pins=json.loads((old/'INPUT_PINS.json').read_text());pins={str(new/Path(k).name) if Path(k).parent==old else k:v for k,v in pins.items()};pins[str(new/'FROZEN_PROTOCOL.json')]=sha(new/'FROZEN_PROTOCOL.json');assert all(sha(Path(k))==v for k,v in pins.items());(new/'INPUT_PINS.json').write_text(json.dumps(pins,indent=2))
s=(W/'run_m12_basis235_remaining_cost.py').read_text().replace("PLAN=W/'M12_BASIS235_COSTFIX_PLAN_20261031'","PLAN=W/'M12_SPLIT235_PLAN_20261031'").replace("'M12_BASIS235_COSTFIX_'+arm.upper()","'M12_SPLIT235_'+arm.upper()")
s=s.replace('forecast=(tangent_seconds+4*reduced_primal_seconds)*1.25+6.5*primal_seconds*1.25+120','forecast=(tangent_seconds+4*reduced_primal_seconds)*1.25+60')
start=s.index(' USE_FULL_REPORT=True\n def final_level');end=s.index('\nexcept SystemExit:raise',start)
s=s[:start]+" save('CORRECTION_RESULT.json',dict(frozen,candidate_SHA=frozenSHA,independent_validation_pending=True,convergence_certified=False));print(json.dumps({'correction_complete':True,'updates':len(history),'candidate_SHA':frozenSHA,'validation_pending':True}));raise SystemExit(0)"+s[end:]
ast.parse(s);target=W/'run_m12_split235_correction.py';assert not target.exists();target.write_text(s)
(new/'PREPARATION_RECEIPT.json').write_text(json.dumps({'driver_SHA':sha(target),'protocol_SHA':sha(new/'FROZEN_PROTOCOL.json'),'prepared_only':True,'new_two_stage1800s_cap_explicit':True,'old_attempts_preserved':True,'reference_solution_values_read':False},indent=2));print(json.dumps({'prepared':str(new),'driver_SHA':sha(target),'total_stages_cap':1800}))
