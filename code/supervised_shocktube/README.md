# Data-assisted calculations

These cases use observed moments or reference-derived information. They are not equation-trained calculations. Shock-tube anchor targets and macro-lock targets are sampled from reference fields; the configurations identify their source files and anchor counts. Closure heads are independently fitted outputs, not corrections of the underlying kinetic distribution. The R-physics ablation retains reference-informed sampling and normalisation even when direct R targets are disabled.

## Available reproduction material

The repository supplies the recovered trainer sources, recorded configurations, main saved profile arrays and reported error tables. Code imports for the stationary closure trainer are included locally. Run each trainer with `--help` to inspect its native CLI. Configurations preserve the scientific relative paths and initial-checkpoint requirements; run from a prepared case directory containing those paths.

Full historical trajectory replay additionally requires the original reference/anchor NPZ files and parent checkpoints named in each configuration. These inputs are not reconstructed from plotted curves and are not included in this compact release. In particular, the supplied configurations are not independent cold-start replicas of all reported cases. The separate scalar Delta-head trainer has not been recovered here; its retained profiles and summary data support saved-result inspection only.

## Training targets versus evaluation

`data/supervised_shocktube/evaluation` and `data/supervised_closure/evaluation` hold saved comparisons, not a substitute for complete training inputs. The adjacent `training_targets/REQUIRED_INPUTS.json` files list missing native target files. A reference may serve both roles in these data-assisted cases; unlike the equation-trained subtree, it must not be described as universally held out. No target values or missing loss histories have been fabricated.

For the R-physics ablation the recovered normaliser is S_R=2.022136926651001, Kn_eff=0.0125 and D_R approximately 161.77. Recorded configuration values remain authoritative. Failed or intermediate workload variants and submission wrappers are excluded. Single-run supervised results do not acquire independent-seed statistics from the separate equation-trained Mach-2 ensemble.
