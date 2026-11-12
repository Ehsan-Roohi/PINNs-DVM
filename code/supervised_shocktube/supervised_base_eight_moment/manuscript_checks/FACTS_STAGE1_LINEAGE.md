# FACTS_STAGE1_LINEAGE — how "model 170" was obtained

Read-only audit, 2026-11-11. Nothing under `<projects>\` was modified. No `.pt` file was loaded (no torch); every statement below comes from JSON/py/txt/npz records or from JSON members read in memory from the archives.

`P` = `<projects>\JFM_1410_REVISION_20261105\SUPERVISED_S0_ALL_MOMENTS_20261107`

Model 170 = `P\HMM_FIELD_GN_FROM168_EXECUTION\NATIVE.zip` member `OUTPUT/ACCEPTED_170/STATE.pt`, SHA256 `bbee1d64…833b95` (`OUTPUT/ACCEPTED_170/CHECKPOINT_PROFILE_PAIR_ACK.json` → `checkpoint_sha256`, `accepted_updates: 170`).

The 14 driver scripts quoted below were checked byte-identical (SHA256) to the copies archived with the executed jobs (`work\lineage_driver_hash.py`).

---

## 1. State at accepted updates = 0

**It is the five-output `INPUT.pt` (SHA `a0734fcc…d09003`) with zero-initialised new heads and `q_eta = 0`; the distribution function is unchanged.** The earlier GPU training of the eight-moment model was not its starting point.

- Construction: `QAmplitudeAllMomentHMM.from_original_source(cfg, cp['model'])` — `P\HMM_ADAPTIVE_FUNCTION_TRUST_GPU_PREPARATION\train_adaptive_function_trust.py` line 62; zeroing of `higher_head` and `temporal_head` in `P\all_moment_model.py` lines 13–16; `q_eta` zero in `P\HMM_FUNCTION_TRUST_REPAIR_PREPARATION\q_amplitude_adapter.py` lines 9, 25.
- Zero-embedding check passed bitwise for f, f_t, f_x on GPU: `HMM_ADAPTIVE_FUNCTION_TRUST_GPU_EXECUTION\NATIVE.zip` → `OUTPUT/RESULT.json` key `zero_embedding_GPU_f_ft_fx` (all `"bitwise": true`), `source_checkpoint: a0734f…`.
- Lineage statement: `P\HMM_ADAPTIVE_FUNCTION_TRUST_GPU_EXECUTION\ANALYSIS\RESULT_SUMMARY.json` → `state_lineage`: "Originala073 pluszero higher/timeheads,eta0; samepositiveHMM;5newjointLMupdates."
- The first accepted update of that job (attempt 1, α = 1, radius 0.1 → 0.2, ratio 0.9558763217132981) is the first entry of the guard history carried by checkpoint 170 (compare `work\FACTS_CASE_AND_LINEAGE.json` → `guard_history_first` with `RESULT_SUMMARY.json` → `accepted_step_radius_alpha_ratio[0]`).
- Although f is unchanged, its moments and residuals at update 0 were evaluated on a different velocity grid and against a different reference than in the earlier five-output work (item 5). Baseline errors at update 0 on the chain's own grid/reference: ρ 0.0711, u 1.4925, T 0.0754, q 2.876, σ 3.089, m 105.9, R 141.1, Δ 138.6 % (399 positions; `P\HMM_FUNCTION_TRUST_CPU_EXECUTION\CPU_PREFLIGHT_REVIEW.json` → `initial_errors_percent`).

**The earlier eight-moment GPU jobs (7 Nov), both NOT_ADOPTED and not part of the 170 count:**

| | R1 | R2 |
|---|---|---|
| Folder | `P\` top level (`train_all_moments.py`, `SAVED\`, `NATIVE.zip`) | `P\TRUST_REGION_REPAIR\` |
| Slurm job | 66918125 | 66918807 |
| Optimiser | `column_scaled_LM_component_projection` (Levenberg–Marquardt; no AdamW, no learning rate) | `column_scaled_LM_damping_controlled_trust_region_component_projection` |
| Start | `INPUT.pt` + zero higher/temporal heads (no `q_eta`) | R1 endpoint (`717f02f6…`) |
| Steps | 60 accepted of 60 attempts (cap 60) | 20 accepted, 25 trials |
| Controller | μ 1e-3 → 1e-8 (÷3 per accept); parameter-step norm capped at 0.1; α = 2^-k | parameter radius 0.5 |
| Trained | all parameters (`params=list(model.parameters())`, `train_all_moments.py` line 82) | all |
| Data | 160 positions, all 8 moments, from an in-job 800-cell 32×24×24 midpoint DVM; 24 collocation points | same 160; 64 collocation points |
| Hardware / time | A100-SXM4-80GB; Slurm 518 s, internal 513.3 s | A100-PCIE-40GB; Slurm 491 s, internal 485.0 s |
| Status | `NOT_ADOPTED`, stop `MAX_UPDATES` | `NOT_ADOPTED` |

Sources: `P\SAVED\OUTPUT\RESULT.json` (`status`, `algorithm`), `P\PRESERVATION_ACK.json` (`accounting`), `P\NATIVE.zip` members `OUTPUT/HARDWARE.json`, `OUTPUT/LOSS.jsonl`, `OUTPUT/REFERENCE_REFINEMENT.json` (`selected: [800,32,24,24]`); `P\TRUST_REGION_REPAIR\NATIVE.zip` same members; `P\TRUST_REGION_REPAIR\PLAN.json`.
R1's endpoint was the start of R2 and of the `LOCAL_RBF_*` diagnostics only. The decision to restart from the original is recorded in `P\HMM_FUNCTION_TRUST_REPAIR_PREPARATION\IMPLEMENTATION_PROPOSAL.md` line 3: "restart the original `a0734fcc...` checkpoint with the existing zero higher and temporal heads … It should not load the R2 weights".
A frozen-feature ridge fit of the new heads was also tried and rejected: `P\HEADFIT_REJECTED.json` → `decision: REJECT_ALL_SHARED_HEAD_RIDGE_CANDIDATES`, `GPU_seed: "Original INPUT.pt only; zero extra heads"`.

Parameter counts: 52,753 in the chain model (`parameter_count`, e.g. `HMM_CHECKPOINTED_CONTINUATION_CPU_EXECUTION\EVIDENCE.zip` → `OUTPUT/RESULT.json` → `restored_source`); temporal head 1,032 (8×128+8); `q_eta` 1. Higher head 387 (3×128+3) is arithmetic from `all_moment_model.py` line 8, quoted as 387 in `P\CLAUDE_START_HERE.txt` line 10.

## 2. `temporal_head` and `q_eta`

- **`temporal_head`** (8 outputs, multiplied by `(1-t/tf)*(1-xi^2)^2`, so it vanishes at t = t_f): part of `AllMomentHMM` from the first version of `P\all_moment_model.py` (lines 10–11, 30–31; file dated 7 Nov, SHA `dd5c7dd2…`), zero-initialised. Present, at zero, at accepted update 0. Never set by a fit. Trained: as part of all parameters in updates 0–17; from update 17 on it is additionally the only block moved in the "restoration"/"SOC" sub-steps (item 3).
- **`q_eta`** (scalar; `qcoef *= 1 + 0.1*tanh(q_eta)`, effective q amplitude in [0.225, 0.275]): added by `q_amplitude_adapter.py` (prepared 2026-11-08 ~01:08 UTC, `P\HMM_FUNCTION_TRUST_REPAIR_PREPARATION\PLAN.json` → `representation.one_new_parameter`, `q_scalar_formula`). Introduced at accepted update 0 with value 0. Trained with all other parameters (`direction.all_parameters_including_eta: true`), not set by a one-off fit. Recorded values: η = 8.26e-4 (effective amplitude 0.25002) after the first accepted update; η = 0.0326 (0.25082) in the last trial record of the 5→17 job (`OUTPUT/RESULT.json` → `trials[*].eta` in the two GPU archives).
  The motivation was a local capacity diagnostic, not a fit carried into the model: `PLAN.json` → `counterfactual_evidence.x_minus_02.required_qcoef: 0.2580352`.
- Value of `q_eta` at update 170: **not found** in any JSON record (would require loading `STATE.pt`).

## 3. Chain of accepted updates 0 → 170

Fourteen jobs contributed (table at the end). Counter ranges: 0–5, 5–17, 17–18, 18–22, 22–24, 24–75, 75–132, 132–134, 134–161, 161–163, 163–166, 166–167, 167–168, 168–170. (The ranges 0–15, 15–75, 134–163 and 163–167 suggested in the task are not what the records show.)

Three algorithm phases:

- **0–17 (two GPU jobs).** Column-scaled Levenberg–Marquardt step on 697 residual rows (690 pointwise observation rows + 7 square-root physical aggregates), projected so that the 12 protected components do not rise to first order, scaled to a function-space trust radius, then backtracked. All 52,753 parameters. `all_parameters_trainable=True` in the checkpoint payload (`train_adaptive_function_trust.py` line 183; `train_curvature_continuation.py` line 185). The second job adds one inequality row for the selection-set relative residual and a curvature-limited first step length (`train_curvature_continuation.py` lines 209–220).
- **17–161 (seven CPU jobs).** Each accepted "joint cycle" has two parts: (a) a restoration step on the temporal head only (1,032 parameters; terminal, initial and inflow f verified bitwise unchanged), scale 10, 5 or 1; (b) the same 697-row LM step on all 52,753 parameters with α from {1, 0.5, 0.25, 0.125, 0.0625} ({1, 0.9, 0.5, …} from update 132). Recorded strings: `optimizer_kind='ColumnscaledLM/curvaturebacktracking;Jacobianrecomputedateachcurrentstate'`, `restoration_optimizer='Preconditioned1032temporalprojectedinwardgradient;nohiddenmomentumorreset'` (e.g. `train_cpu_calibration_balance_from75.py` line 164). From update 22 three inequality rows for the selection-set balance MSEs were added to the direction (`P\HMM_CALIBRATION_BALANCE_GUARD_PREPARATION\TEST_PROTOCOL.json` → `new_inequality_rows: 3`, `new_objective_rows: 0`).
- **161–170 (five CPU jobs).** `optimizer_kind='Column-scaled fixed-damping LM with group-GN fraction and Ct projection; Jacobian recomputed each accepted cycle'`. Joint step on all 52,753 parameters with cutting-plane constraints on the time derivatives of the eight coefficients at 24 gauge points and Gauss–Newton cuts on the five lower-moment groups, then up to 2 (later 6) temporal-head-only "SOC" steps. `controller_policy` strings: 161–163 `'Fixed LM_mu=1e-8 and function_radius=.025 for this max3-cycle segment; no inherited adaptive radius update'`; 167–170 `'Bounded alternative damping geometry; selected LM_mu=1e-06; function_radius=.025; canonical objective unchanged'` (`train_field_gn_production.py` line 434; `train_field_gn_from168.py` line 556).

Controller rules recorded in code: on accept μ ← max(μ/3, 1e-8), on a fully rejected direction μ ← 10 μ; radius halved if actual/predicted decrease < 0.25, doubled (max 1) if > 0.75 and the gauge exceeded 0.8 × radius (`train_adaptive_function_trust.py` lines 226–236).

Jobs in the same period that committed no update (not in the table): `HMM_FUNCTION_TRUST_CPU`, `HMM_ACTIVE_GUARD_*`, `HMM_TEMPORAL_INWARD`, `HMM_SHARED_REACHABILITY`, `HMM_COMBINED_*`, `HMM_FIXED_F_TANGENT`, `HMM_HIGHER_WEIGHT_DIRECTION`, `HMM_TWO_STEP_CONTROLLER`, `HMM_CT_*`, `HMM_CURRENT_FIELD_DIRECTION`, `HMM_FIELD_CURVATURE_REPAIR`, `HMM_FIELD_GN_FRACTION`, `HMM_FIELD_GN_FROM167`, `HMM_FIELD_GN_DAMPING_REPAIR`, `HMM_FIELD_GN_SOC_BACKTRACK`, `HMM_FIELD_GN_SOC_REPLAY_REPAIR` (each `OUTCOME.json`: `committed_updates`/`scientific_commits`/`accepted_cycles` = 0).

## 4. Acceptance rule — it changed once, at update 161

The 15 loss components are: 8 observation losses (ρ, u, T, q, σ, m, R, Δ) and 7 physical ones (`physical_raw_PDE, physical_relative_PDE, mass_balance, momentum_balance, energy_balance, initial_physical_PDE, incoming_boundary`; `ACTUAL_INPUT_BINDING.json` → `component_names`). Normalisers are the values at update 0, so Φ = Σ loss_k/normal_k = 15 at update 0 (arithmetic) and 0.8913 at update 170 (`OUTPUT/RESULT.json` of job 66948040, last accepted cycle `Phi`).

**The record does not support "all eight observation losses and all seven physical components non-increasing at every update".** What the code enforces:

**Updates 0–161** (`train_adaptive_function_trust.py` lines 147, 150, 215–222; identical text in `train_curvature_continuation.py` lines 233–240; CPU drivers lines 262–269):

```
limits=initial_losses*(1+1e-6)+1e-24
protected=[0,1,2,3,4,8,9,10,11,12,13,14]
failed += [b['component_names'][k]for k in protected if values[k]>limits[k]]
if sum(values/normal)>=sum(current_losses/normal):failed.append('joint_objective_not_decreased')
if sum(values[5:8]/normal[5:8])>=sum(current_losses[5:8]/normal[5:8]):failed.append('higher_moments_not_decreased')
```

- Strictly decreasing at each accepted update: Φ, and the normalised sum of the m, R, Δ observation losses.
- The five lower observation losses and the seven physical components were only capped at their **update-0 values × (1+1e-6)**; they were allowed to rise from one update to the next below that cap. (Example: selection-set relative residual +4.23 % from 132 to 134, `P\HMM_TRUST_INTERMEDIATE_FROM132_EXECUTION\OUTCOME.json` → `independent_review.paired_physics_change_percent_vs132.relativePDE`.)
- Function-space trust: each of five gauge blocks ≤ radius (`function_trust_proposal.py` lines 74–82).
- Field and selection guards, all relative to the **update-0 baseline** (`inherited_field_failures`, `train_adaptive_function_trust.py` lines 161–178): 399-position L2 errors of ρ, u, T, q, σ ≤ baseline × (1+1e-3) + 1e-5; of m, R, Δ strictly below baseline; S ≤ baseline × (1+1e-3); q_max, σ_min, σ_max errors ≤ baseline × (1+1e-3) + 1e-10; selection-set raw residual, relative residual and three balance RMS ≤ baseline × (1+1e-3); ρ, T positive.
- In 17–161 the joint step had to lower Φ against both the cycle start and the restored state (`'joint_objective_not_decreased_vs_restoration'`, `'…_vs_cycle_start'`).

**Updates 161–170** (`train_field_gn_from168.py` lines 366–380; same lines in `train_field_gn_production.py` 301–304 and the other `train_field_gn_*` drivers). All of the above still apply (`original`), plus comparison with the **current accepted state**, with no tolerance:

```
lower_train=[bool(loss[k]<=baseline_losses[k])for k in range(5)]
lower_field=[bool(aa['calibration399_L2_percent'][k]<=baseline_actual['calibration399_L2_percent'][k])for k in range(5)]
higher=[bool(aa['calibration399_L2_percent'][k]<baseline_actual['calibration399_L2_percent'][k])for k in (5,6,9)]
physics_flags=[bool(pp['raw_PDE']<=...),bool(pp['relative_PDE']<=...),*[bool(x<=y)for x,y in zip(pp['balance_RMS'],...)]]
full6400_lower=[...<=...for k in range(5)];full6400_higher=[...<...for k in (5,6,9)]
full=bool(not original and not trust_failed and data_ok and all(physics_flags)and phi<source_phi)
```

- Non-increasing vs the current state: the five lower observation losses; 399-position and full-6400 L2 errors of ρ, u, T, q, σ; the three q/σ peak errors; five selection-set physics quantities (raw residual, relative residual, mass/momentum/energy balance RMS on the 64 selection points).
- Strictly decreasing vs the current state: 399-position and full-6400 L2 errors of m, R, Δ individually; Φ; the m+R+Δ observation-loss group.
- The seven physical **training** components remained capped at update-0 values only; the per-update non-increase applies to the five selection-set quantities.
- Protocol wording: `P\HMM_FIELD_GN_PRODUCTION_PREPARATION\TEST_PROTOCOL.json` → `actual_guard_policy`: "Exactcurrentaccepted-cycle lower5/peaks/physics nonregression andhigher3strictdecrease at399and6400; originalcaps unchanged".

## 5. Data that entered training

- **Gradient rows:** 690 scalar observations = 3×160 (ρ, u, T) + 5×42 (q, σ, m, R, Δ), unchanged for the whole chain (`observation_counts: [160,160,160,42,42,42,42,42]` at update 0 in `train_adaptive_function_trust.py` line 83 and at update 168 in `OUTPUT/FRESH_PROCESS_ADMISSION.json`).
- **Locations** (`P\HMM_FUNCTION_TRUST_REPAIR_PREPARATION\OBSERVATION_LOCATIONS_FROZEN.npz`, SHA `c9724c94…`): macro x ∈ [−0.49875, 0.49875] (160 points); moment x ∈ [−0.10375, 0.13375] (42 points, spacing 0.005–0.0075). They are cell centres of the original 400-cell grid (`OBSERVATION_PROVENANCE.json` → `source_formula`, `actual_reference_cells: 400`).
- **Values:** linear interpolation of the **6400-cell reference** to those locations, not `TARGETS_ALL.npz`. Code: `def targets(xx):return np.column_stack([np.interp(xx,rx,ry[:,k])for k in range(8)])` with `ref=np.load(root/'REFERENCE6400_FIELDS.npz')` (`train_adaptive_function_trust.py` lines 77–83; `train_field_gn_from168.py` lines 127–137). File `P\HMM_FUNCTION_SPACE_DIAGNOSTIC\REFERENCE6400_FIELDS.npz`, SHA `c4fe0f1f…`, 6400×9 float64.
  All 14 chain drivers contain this `np.interp(xx,rx,ry…)` line and read `REFERENCE6400_FIELDS.npz` and `OBSERVATION_LOCATIONS_FROZEN.npz`; none of them mentions `TARGETS_ALL` or `TARGETS9` (text search of the 14 files). In this folder `TARGETS_ALL.npz` (400 cells) is named only by `capacity.py`, `headfit.py`, the CPU check wrappers, and the `--cpu-test` branch of `train_all_moments.py` (line 38).
- **No switch of reference values during the chain.** The same three hashes (reference, observation locations, 399 positions) are bound in the first job (`P\HMM_ADAPTIVE_FUNCTION_TRUST_GPU_EXECUTION\ACTUAL_INPUT_BINDING.json` → `inputs`) and in the last (`OUTPUT/FRESH_PROCESS_ADMISSION.json` → `reference_and_observation_member_hashes`). The switch recorded in `P\HMM_BALANCE_FROM134_PREPARATION\COMPARISON_PROVENANCE_AUDIT.json` is between the earlier five-output comparison (`historical_adopted_comparison`: job 66898929, 400 cells, 32×12×12, float32 DVM fields) and the whole eight-moment chain (`current_matched_original_comparison`: 6400 cells, 48×20×20, float64). It takes effect at accepted update 0. The audit's own conclusion: "Sameweightcheckpoint canhave differentintegratedmoments whenquadraturechanges" and "No controlledcross-grid/refmatrix computedhere".
- **Reference description** (same audit): first-order upwind then exponential BGK relaxation, finite transverse quadrature, CFL 0.4, 2240 steps, dt 8.93e-6; the audit warns not to call it a full-f snapshot.
- **m and R convention:** observation components are named `m_full`, `R_full`; model values add the discrete-Maxwellian quadrature offsets Em, ER (`function_trust_proposal.py` lines 174–190).
- **Velocity grid of the chain: 48×20×20** (19,200 nodes, endpoint `linspace(-7,7)`, w = (14/47)(14/19)²; `train_adaptive_function_trust.py` line 74). Same grid at update 168: `FRESH_PROCESS_ADMISSION.json` → `frozen_state.grid.shape [48,20,20]`, with v/w/vw SHA256 equal to `actual_grid_cpu_bytes_sha256` of the first job.
  **Trap:** the checkpoint also stores the stale `config` with `velocity: 32×12×12` and `dtype: float32` (same admission file, `frozen_state.config`). `work\FACTS_CASE_AND_LINEAGE.json` → `velocity_grid_of_checkpoint_170` reports that stale config, not the grid used.
- **Reference values that entered accept/reject decisions but not gradients:** the 399 "calibration" positions (`QUALITY_LOCATIONS_FROZEN.npz`, x ∈ [−0.4975, 0.4975]) for the whole chain; the global q_max, σ_min, σ_max of all 6400 cells for the whole chain; the full-6400 L2 errors of all eight fields from update 161 (item 4). The records call these "selection calibration, not independent test" (`P\HMM_ADAPTIVE_FUNCTION_TRUST_GPU_PREPARATION\TEST_PROTOCOL.json` → `known_limitations`).

## 6. Collocation points, guard sets, precision

Fixed for the whole chain (`train_adaptive_function_trust.py` lines 101–102; shapes re-confirmed at update 168 in `FRESH_PROCESS_ADMISSION.json` → `frozen_state`):

| Set | Size | Definition | Role |
|---|---|---|---|
| Training physics | 64 | seed 883171; t uniform in [0, t_f]; x = 32 × linspace(−0.5, 0.5) + 32 × linspace(−0.12, 0.14) | 5 components: raw residual, relative residual, mass/momentum/energy balance |
| Initial | 4 | t = 0, x = ±0.025, ±0.075 | `initial_physical_PDE` |
| Incoming boundary | 6 | t = t_f/4, t_f/2, t_f at x = ±0.5 | `incoming_boundary` |
| Selection physics | 64 | seed 970441, same x layout | guards only |
| Function gauge | 24 | `FUNCTION_CALIBRATION_POINTS.npz` | trust radius on f, C, t_f·C_t, 0.13·C_x |
| Field guards | 399 + 6400 | `QUALITY_LOCATIONS_FROZEN.npz`; reference grid | guards only |

Each physics point uses all 19,200 velocity nodes. A reserved set (128 reference cells + 64 points, seed 970442) was evaluated once after update 5 and is recorded as consumed (`ANALYSIS\RESULT_SUMMARY.json` → `consumed_evaluation`).
Precision: float64 throughout (`torch.set_default_dtype(torch.float64)`, TF32 off; `hardware: {"FP64": true, "TF32": false}`).

## 7. Attempts and compute time

- **Direction/cycle attempts along the accepted lineage: 171, against 170 accepted.** `FRESH_PROCESS_ADMISSION.json` → `parent_attempts: 169` at update 168; the 170 state adds 2 (`train_field_gn_from168.py` line 556). `work\FACTS_CASE_AND_LINEAGE.json` → `cp_attempts: 171` agrees. This counter excludes backtracking trials and all zero-commit jobs.
- **Trial evaluations in the 14 contributing jobs** (my count from `OUTPUT/RESULT.json` lists `trials`, `all_joint_trials`, `all_cycle_trials`): 392 all-parameter trial states for 170 accepted; 165 temporal-head restoration trials (145 passed) in 17–161. SOC sub-step evaluations in 161–170 not tallied.
- **Time of the 14 contributing jobs:** GPU 135 + 980 = **1,115 s** (one A100-SXM4-80GB each); CPU **21,500 s** Slurm elapsed over 11 jobs + 95 s for the non-Slurm 17→18 run = **21,595 s wall, 2 threads** (≈ 6.0 h). Worker-internal totals: GPU 1,102.8 s, CPU 21,501.5 s. Per-job sources in the table.
- **GPU ledger for the whole folder:** 2,124 s over four jobs, including R1 and R2 (`P\GPU_BUDGET_LEDGER.json` → `total_actual_seconds`).
- Zero-commit Slurm CPU jobs between updates 161 and 170: 3,104 s (383 + 626 + 76 + 299 + 261 + 134 + 292 + 226 + 502 + 45 + 260; each `LOCAL_ACK.json` → `sacct`).
- A single recorded CPU ledger for the whole chain: **not found** (only per-job `sacct` lines and partial sub-phase ledgers).

## 8. Why updates beyond 170 stalled (the upstream shoulder)

- Third cycle of the last job, `P\HMM_FIELD_GN_FROM168_EXECUTION\OUTCOME.json` → `third_attempt.arms[0]`: `"status": "GEOMETRY_REJECTED"`, `"reason": "JointCt/fieldGN/currentphysicsprojectionnotqualifiedwithin8passes"`, `"actual_trial_count": 0`. Same file → `upstream_diagnostic` (x < −0.15, "Diagnostic only; … not an acceptance criterion"): summed squared errors of m, R, Δ rose 1.63 %, 2.52 %, 2.91 % from 168 to 170 while the full-domain errors fell.
- First attempt from 170, `P\HMM_SHOULDER_PHYSICS_FROM170_EXECUTION\OUTCOME.json` → `decision`: "Retain accepted170. All three trial steps reduce shoulder PDE and global field errors, but worsen upstream m/R/Delta squared errors and energy balance."
- `P\CLAUDE_START_HERE.txt` line 13: "Saved LP with6physics+aggregate+3individual rows has optimal common margin0, zero direction; … This is only this3D estimated cone, NOT full387D impossibility."
- Later diagnosis (10 Nov, after the chain; concerns a successor state not admitted to the paper), `P\ACCEPTED_SOURCE_170T_B35\evidence\PLAN_FA.md` §1, in Persian: «شانهٔ بالادست در m، R و Delta (و خطای q و sigma در همان ناحیه) از شکل دو دروازهٔ مکانیِ دستی می‌آید، نه از وزن‌های شبکه و نه از بهینه‌ساز.» — "The upstream shoulder in m, R and Delta (and the q and σ error in the same region) comes from the shape of the two hand-set spatial gates, not from the network weights and not from the optimiser." The gates are `exp(-(x/0.13)^4)` for q, σ and `exp(-(x/0.22)^4)(1-(2x)^2)^2` for m, R, Δ (`all_moment_model.py` lines 21–24); moment observations stop at x = −0.10375. `STAGE_A_RESULT_FA.md` criterion A2 reports a 97.4 / 95.9 / 96.8 % fall in upstream squared error of m / R / Δ when only the gates outside |x| > 0.135 are replaced, with no weight changed. I did not re-run that test.

---

## Table for item 3

"Trials" = all-parameter trial states evaluated / accepted. "Rest." = temporal-head restoration trials (passed). Seconds = Slurm elapsed (`LOCAL_ACK.json`/`PRESERVATION_ACK.json` → `sacct`/`actual_elapsed_seconds`) / worker-internal (`OUTPUT/RESULT.json` → `elapsed_seconds`). μ, radius and α from `OUTPUT/RESULT.json` (`directions`, `cycle_history`, `all_joint_trials`, `all_cycle_trials`).

| Counter | Execution folder under `P\` | Job, device | Driver | Parameters moved | μ (LM damping) | Function radius | Accepted α | Trials | Rest. | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 0–5 | `HMM_ADAPTIVE_FUNCTION_TRUST_GPU_EXECUTION` | 66929060, GPU A100 80 GB | `train_adaptive_function_trust.py` | all 52,753 | 1e-3 → 1.2e-5 | 0.1 → 0.2 | 1; 0.125 ×4 | 17 / 5 | — | 135 / 129.4 |
| 5–17 | `HMM_CURVATURE_CONTINUATION_EXECUTION` | 66932634, GPU A100 80 GB | `train_curvature_continuation.py` | all | 4.1e-6 → 1e-8 | 0.2 | 0.156 falling to 0.00098 | 67 / 12 | — | 980 / 973.4 |
| 17–18 | `HMM_CHECKPOINTED_CONTINUATION_CPU_EXECUTION` | no Slurm id, CPU 2 threads | `preflight_checkpointed_continuation.py` | head 1,032, then all | 1e-8 | 0.2 | 0.125 | 3 / 1 | 1 (1) | 95.1 wall / 91.4 |
| 18–22 | `HMM_OVERNIGHT_CPU_EXECUTION` | 66934589, CPU | `train_cpu_batch_from18.py` | head, then all | 1e-8 | 0.2 | 0.5; 0.25; 0.125; 0.0625 | 19 / 4 | 5 (5) | 812 / 802.0 |
| 22–24 | `HMM_CALIBRATION_BALANCE_CPU_EXECUTION` | 66934838, CPU | `train_cpu_batch_balance_guard.py` | head, then all | 1e-8 | 0.2 | 0.5 ×2 | 4 / 2 | 2 (2) | 281 / 272.9 |
| 24–75 | `HMM_CALIBRATION_BALANCE_CONTINUATION_EXECUTION` | 66934931, CPU | `train_cpu_calibration_balance_continuation.py` | head, then all | 1e-8 | 0.2 → 0.4 → 0.05 (adaptive) | 1 ×12; 0.5 ×21; 0.25 ×10; 0.125 ×8 | 116 / 51 | 51 (51) | 6796 / 6790.2 |
| 75–132 | `HMM_CALIBRATION_BALANCE_FROM75_EXECUTION` | 66936533, CPU | `train_cpu_calibration_balance_from75.py` | head, then all | 1e-8 | 0.05, 0.1, 0.025; ends 0.025 | 1 ×15; 0.5 ×42 | 99 / 57 | 69 (57) | 6862 / 6855.8 |
| 132–134 | `HMM_TRUST_INTERMEDIATE_FROM132_EXECUTION` | 66937828, CPU | `train_cpu_trust_intermediate_from132.py` | head, then all | 1e-8 | 0.025 → 0.05 → 0.025 | 1 ×2 | 2 / 2 | 4 (2) | 226 / 217.6 |
| 134–161 | `HMM_BALANCE_FROM134_EXECUTION` | 66937948, CPU | `train_cpu_calibration_balance_from134.py` | head, then all | 1e-8 | 0.025, 0.05, 0.0125; ends 0.025 | 1 ×7; 0.9 ×18; 0.5 ×2 | 49 / 27 | 33 (27) | 3227 / 3222.4 |
| 161–163 | `HMM_FIELD_GN_PRODUCTION_EXECUTION` | 66941705, CPU | `train_field_gn_production.py` | all, then head (SOC ≤ 2) | 1e-8 fixed | 0.025 fixed | 1; 0.9 | 3 / 2 | — | 631 / 620.3 |
| 163–166 | `HMM_FIELD_GN_FROM163_EXECUTION` | 66942274, CPU | `train_field_gn_from163.py` | all, then head (SOC ≤ 2) | 1e-8 fixed | 0.025 | 0.5 ×3 | 9 / 3 | — | 1559 / 1545.7 |
| 166–167 | `HMM_FIELD_GN_WARMSTART_EXECUTION` | 66943438, CPU | `train_field_gn_warmstart.py` | all, then head (SOC ≤ 2) | 1e-8 fixed | 0.025 | 0.5 | 1 / 1 | — | 209 / 203.3 |
| 167–168 | `HMM_FIELD_GN_SOC_CX_REPAIR_EXECUTION` | 66947467, CPU | `train_field_gn_soc_cx_repair.py` | all, then head (SOC ≤ 6) | 1e-6 (selected) | 0.025 | 0.25 | 1 / 1 | — | 328 / 317.6 |
| 168–170 | `HMM_FIELD_GN_FROM168_EXECUTION` | 66948040, CPU (intel & avx512) | `train_field_gn_from168.py` | all, then head (SOC ≤ 6) | 1e-6 | 0.025 | 0.25 ×2 | 2 / 2 | — | 569 / 562.3 |

Totals: accepted 5+12+1+4+2+51+57+2+27+2+3+1+1+2 = 170; trials 392; restoration trials 165 (145 passed); GPU 1,115 s; CPU 21,595 s wall.

Recorded `optimizer_kind` / `controller_policy` strings exist only in the CPU checkpoints (17–170) and are quoted in item 3. The two GPU drivers record `all_parameters_trainable=True`, `LM_mu`, `function_radius` instead.

---

## Not verified, or not found

- No `.pt` file was opened. Checkpoint-internal values come from JSON receipts, from driver code that wrote the payload, or from `work\FACTS_CASE_AND_LINEAGE.json` (produced earlier with torch; I did not re-check it).
- `q_eta` at update 170: not found.
- What the five-output `INPUT.pt` was itself trained on (folder `SUPERVISED_S0_PEAK_PHYSICS_20261107`, job 66898929) was not opened; only what `OBSERVATION_PROVENANCE.json` and `COMPARISON_PROVENANCE_AUDIT.json` say about it is used.
- The share of the update-0 error change (e.g. q 1.18 % → 2.88 %) due to velocity grid versus reference resolution: the audit states it was not computed.
- The Stage A gate test and the 170T-B35 numbers in item 8 are quoted, not re-run.
- A later state exists: `P\ACCEPTED_SOURCE_170T_B35\` (11 Nov; 35 further updates of two output layers on a changed x-envelope). Its `START_HERE.txt` says "It is NOT admitted to the paper. The project's own handoff files still name model 170."

Scripts written for this audit (all in `work\`): `lineage_inventory.py` (+ `lineage_inventory.txt`), `lineage_outcomes.py`, `lineage_results.py`, `lineage_segments.py`, `lineage_npz.py`, `lineage_r1r2.py`, `lineage_eta.py`, `lineage_driver_hash.py`, `lineage_skim.py`.
