# RADIUS-PbCWM

RADIUS is isolated under `pbcwm.methods.radius` and implements the five
method-specific components from the design contract:

* FDA: factorized reward-free dynamics atlas with a dynamic atom rank;
* REF: sequential active linear-Gaussian tracking with independent,
  non-duplicated prototype/new window routing evidence;
* RNE: persistent unexplained-residual monitoring and orthogonal atom growth;
* PEC: low-rank predictive-Fisher direct local trust-region parameter steps;
* PFPA: shared-CEM candidate pair selection using the planner's elite fraction,
  uncertainty and frontier
  relevance, with coverage fallback.

The method never reads `Transition.reward` or evaluator metadata. FDA, REF and
RNE share lifetime running normalization; replay retains raw physical values
and checkpoints retain the normalizer state. The shared experiment protocol
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
