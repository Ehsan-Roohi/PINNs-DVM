# Data-assisted BGK shock-tube continuation

The same positive macro--micro network is refined by joint fitting (45 accepted updates), separate residual and balance control (11), and extremum control (44). The scripts are byte-identical to the preserved executed sources. `legacy.py` defines the shared kinetic network and `CONFIG.json` its exact configuration.

## Training inputs

For each stage, mount a fresh working directory at `/task` containing its stage script, `legacy.py`, `CONFIG.json`, `REFERENCE.npz` and the exact preceding checkpoint named `INPUT.pt`. The reference is supplied under `data/supervised_shocktube/constrained_continuation`. Checkpoint SHA-256 identities and the native-archive hashes are in `provenance/SUPERVISED_CONTINUATION_BINDINGS.json`. The joint stage starts from the original AdamW endpoint; balance starts from the accepted joint endpoint; extrema starts from the accepted balance endpoint. Native checkpoint archives remain in the preserved research record and are not included here.

Run the corresponding script with CUDA PyTorch, NumPy and SciPy; double precision and the stored quadrature values are retained. Each script creates its own stage-specific output directory and refuses to reuse an existing directory. Training uses 160 macrofield locations, 42 locations per heat-flux/stress moment, and three reference-extremum summaries in the final block.

## Saved results and checks

The data directory contains all three recorded histories and field arrays. Each `AFTER.npz` contains coordinates, five model fields, matched reference fields, and observation indices; field order is density, velocity, temperature, heat flux and normal stress. `assessment_32.json` and `assessment_64.json` are saved independent physical evaluations, not additional training updates.

The corresponding CPU audit sources are `audit_32.py` and `audit_64.py`. They expect the preserved native directory mounted at `/native` and an output directory at `/analysis`; the fresh audit additionally requires the original checkpoint at `/analysis/ORIGINAL.pt`. Read each audit's input list before execution. Neither audit trains the model.

From the repository root, run `python tools/verify_manifest.py` to check all released file hashes. No GPU run is launched by verification.
