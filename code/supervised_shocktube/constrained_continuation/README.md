# Data-assisted BGK shock-tube continuation

One positive macro--micro network is refined in three consecutive blocks: joint field fitting (45 accepted updates), separate kinetic-residual and balance control (11), and extremum control (44). The three scripts are the executed stage sources; they share the original network, reference and analytic initial/boundary states.

The scripts expect a container-mounted `/task` containing `legacy.py`, `CONFIG.json`, `REFERENCE.npz`, and the exact preceding full-state checkpoint as `INPUT.pt`. Run joint.py, balance.py and extrema.py consecutively with their recorded parent states. They require CUDA PyTorch and double precision. No optimizer state or counter reset is used to manufacture additional accepted steps. The saved final arrays and actual per-stage histories are under `data/supervised_shocktube/constrained_continuation`. Native restart archives remain in the preserved scientific record; checkpoint hashes and endpoint provenance are in `provenance/SUPERVISED_ADOPTED_RESULT.json`.

Training uses 160 macroscopic locations, 42 locations for each heat-flux/stress moment, and three reference-derived extremum summaries in the final block. Independent 32- and 64-point physical assessments are evaluations, not update points. These are data-assisted calculations.
