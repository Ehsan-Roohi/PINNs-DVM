# PINNs-DVM

Code and saved results for **Neural kinetic solvers for shock waves: sparse-moment supervision and physics-based training**.

This repository distinguishes **data-assisted** calculations (moment supervision and reference-derived scales or sampling) from **equation-trained** neural solvers and **equation-based characteristic corrections**. DVM fields are post-solve evaluation data for the latter methods; the supervised subtrees explicitly use reference information during training. Classification is case-specific, not a blanket data-free claim.

## Contents

- `code/supervised_shocktube` and `code/supervised_closure`: recovered data-assisted algorithms and case configurations, with matching saved comparisons under `data`. Read their coverage notes before attempting reproduction; native training targets and parent checkpoints are required separately.

- `code/shocktube`: independently initialised BGK and Shakhov characteristic PINNs, their self-scaled Broyden optimiser, full-state continuation and independent DVM reference solver.
- `code/independent_m2`: three independently initialised Shakhov Mach-2 calculations with seeds 141001, 141002 and 141003.
- `code/stationary_neural`: scientific kernels and configurations for the stationary continuation calculations.
- `code/stationary_characteristic`: an import-safe characteristic engine, tests, historical Mach-8/Mach-12 algorithms and input recovery requirements. See its README before attempting a historical restart.
- `data`: signed physical profiles, higher moments, matched references and stage-specific optimisation histories.
- `provenance` and `MANIFEST.json`: source identities, transformations, input bindings and release hashes.

The stationary neural results cover Shakhov Mach 2–5 and BGK Mach 6–7. Characteristic BGK corrections cover Mach 7, 8 and 12. These are own-model continuation trajectories, not separate cold starts at every Mach number. The independent Mach-2 and shock-tube experiments are identified separately.

## Run the shock-tube calculation

Install `requirements.txt` in a separate Python environment. Linux is recommended.

```sh
cd code/shocktube
python test_characteristic.py
python train_characteristic.py --config BGK_CONFIG.json --out run_bgk --device cuda
python train_characteristic.py --config SHAKHOV_CONFIG.json --out run_shakhov --device cuda
```

The supplied initial configurations perform 1000 Adam updates and 100 accepted SSB updates. The reported final states include a further 200 SSB updates using `continue_characteristic.py`, which requires the exact parent file and its SHA-256. Use its `--help` for the restart interface. A new GPU run need not reproduce floating-point results bitwise.

`reference_pair.py` is an independent evaluation solver and is not imported by training.

## Independent Mach-2 runs

```sh
cd code/independent_m2
python run_lineage.py --out YOUR_NEW_RUN_DIRECTORY --seed 141001
```

Repeat separately for seeds 141002 and 141003. Each lineage comprises 20,000 Adam updates, 500 SSB updates and 500 SSB updates with split-half quadrature. The preserved runner requires Linux file locks and its admitted V100/A100 hardware. It does not submit cluster jobs. Objectives from different quadrature stages must not be treated as one fixed objective.

## Dataset organisation

Each physical problem has one current dataset directory. Mach-number directories denote different flow cases; Adam, SSB and split-half directories denote distinct optimisation stages. Recorded-history sets A and B identify separate source records, not competing releases or necessarily consecutive stages. Original counters, phase labels and objective definitions remain in the CSV files; `provenance/loss_catalog.json` maps concise filenames to their original records. Overlapping histories are not silently merged.

Kernel copies inside case directories keep each source-bound calculation self-contained. `provenance/archive_integrity.json` records compact archive verification, while the manifest identifies every released file. Operational download logs and collector receipts are not distributed.

## Interpret the data

Shock-tube CSV columns distinguish parent, final and DVM fields. NPZ files retain saved coordinates and field names. Loss histories retain their stage conventions: Adam values are measured before stochastic minibatch updates; SSB values follow accepted updates of a fixed objective. Independent Mach-2 comparisons distinguish core and full-overlap windows. Reference centring uses the reference's own prescribed density midpoint; model coordinates and amplitudes are not fitted to it.

These finite-budget results do not establish uniform convergence. The final shock-tube continuation does not improve every field against the same reference. All three independent Mach-2 endpoints miss at least one physical acceptance tolerance. The finite-quadrature Shakhov distribution has tiny negative tails, which are retained rather than clipped. Reference-refinement and validation limitations are reported with the data.

## Verify and reproduce

```sh
python tools/verify_manifest.py
```

`MANIFEST.json` records original and release hashes, including relocated private paths and extracted portable utilities. The release contains scientific code and compact numerical evidence; it excludes large native checkpoint archives, manuscript files and reviewer correspondence. Exact historical full-state restarts require the checkpoint inputs listed in the recovery manifests. Saved profiles alone cannot reproduce an entire training trajectory.

Release checks cover syntax, saved-data plotting and small CPU algebra tests; they are not fresh GPU reruns of the published experiments. No software licence has been assigned in this repository.

## Optimizer stages and physical profiles

The [Mach-2 comparison](figures/mach2_optimizer_effect.pdf) shows recorded losses,
heat flux and stress for all three equation-trained initializations after Adam,
SSB and split-half SSB. Shared physical axes expose improvements and regressions.
These are sequential training stages, not an equal-budget optimizer benchmark.
Regenerate the figure from saved arrays without loading a neural checkpoint:

```bash
python plotting/independent_seeds/plot_optimizer_effect.py --release-data data/independent_m2 --outdir output
```
