# RADIUS-PbCWM

RADIUS is isolated under `pbcwm.methods.radius` and implements the five
method-specific components from the design contract:

* FDA: factorized reward-free dynamics atlas with a dynamic atom rank;
* REF: sequential active linear-Gaussian tracking with a pre-window active
  prior, independent non-duplicated prototype/new routing evidence, readiness,
  and confidence-gated prototype assignment;
* RNE: persistent unexplained-residual monitoring and orthogonal atom growth;
* PEC: low-rank predictive-Fisher direct local trust-region parameter steps;
* PFPA: shared-CEM candidate pair selection using the planner's elite fraction,
  uncertainty and frontier
  relevance, with coverage fallback.

The method never reads `Transition.reward` or evaluator metadata. FDA, REF and
RNE share lifetime running normalization; each observed state is counted once,
recent REF windows retain raw physical values and are re-normalized on demand,
and checkpoints retain local subsystem RNG/normalizer state without restoring
global Torch RNG. PEC remains on the ordinary optimizer until an old-prototype
Fisher sketch exists. The shared experiment protocol
and canonical lifetime runner remain in `pbcwm.protocol` and
`pbcwm.experiment`; shared CEM/reward components remain in `pbcwm.planning`
and `pbcwm.preferences`.

Run the short real-Pendulum smoke:

```text
python -m pbcwm.methods.radius.smoke --steps 64 --seed 0
```

The default YAML contains only method defaults. It intentionally does not own
stage lengths, planner horizons, preference budgets, query times, seeds, or
evaluation checkpoints.
