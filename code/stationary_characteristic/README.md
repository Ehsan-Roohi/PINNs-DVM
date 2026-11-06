# Stationary characteristic calculations

The `portable` directory contains the import-safe characteristic transport and
Jacobian engine, its exact extracted mathematics, finite-velocity diagnostic
modules, unit tests and the Mach7 own-network initializer/configuration.
It accepts a saved NPZ with `x` and `fields` shaped `(N,3)` for density, velocity,
and temperature. No reference data are loaded by this entry point.

```
cd portable
python test_tiny.py
python test_quadrature_parity.py
python engine.py --config M7_PILOT_CONFIG.json --seed M7_NN18_OWN_MACRO_GAUGE_INITIALIZER.npz --output NEW_RUN
```

The mode is deliberately named pilot: reduced objective reduction is not a
complete kinetic convergence certificate. Run the separate physical diagnostics
before interpreting a new result. Do not replace numerical tolerances to obtain
acceptance. This extracted interface is not claimed to reproduce the M8/M12
published continuation trajectories from an arbitrary new initializer.

`historical` includes the executed M8 basis119 and M12 basis235 algorithm sources
and runtime dependencies. Their top-level entry scripts execute on import;
read them before running and do not import them as utility modules. The only
portability change replaces a private absolute physics-source directory with
`physics_source` beside the scripts. Source SHA256 before/after is recorded.
Original immutable input formats, filenames, hashes and sizes are listed in
`provenance/CHARACTERISTIC_RECOVERY_INPUTS.json` at repository root. They include
frozen own-network macro samples, prior coefficients and spline geometry. Native
pin manifests must correspond to the supplied original inputs; no checksum or
scientific admission bypass is implemented. Checkpoint binaries are excluded
from this compact release; restoring these known inputs is required for exact
historical continuation. The saved published fields and moments remain available
without executing either continuation.
