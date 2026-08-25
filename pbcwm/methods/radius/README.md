# RADIUS-PbCWM

RADIUS is isolated under `pbcwm.methods.radius` and implements the five
method-specific components from the design contract:

* FDA: factorized reward-free dynamics atlas with a dynamic atom rank;
* REF: soft linear-Gaussian active/memory/new context evidence;
* RNE: persistent unexplained-residual monitoring and orthogonal atom growth;
* PEC: low-rank predictive-Fisher Woodbury trust-region gradient transform;
* PFPA: shared-CEM candidate pair selection using uncertainty and frontier
  relevance, with coverage fallback.

The method never reads `Transition.reward` or evaluator metadata. The shared
experiment protocol remains in `pbcwm.protocol`, and the shared CEM/reward
components remain in `pbcwm.planning` and `pbcwm.preferences`.

Run the short real-Pendulum smoke:

```text
python -m pbcwm.methods.radius.smoke --steps 64 --seed 0
```

The default YAML contains only method defaults. It intentionally does not own
stage lengths, planner horizons, preference budgets, query times, seeds, or
evaluation checkpoints.
