# NS-Gym benchmark boundary

The Phase 1 benchmark is registered as
`nsgym/pendulum-mass-abrupt-return-v0`. It uses the external `ns-gym==1.0.12`
package and the upstream `NSClassicControlWrapper`, `DiscreteScheduler`, and
`StepWiseUpdate` APIs. NS-Gym's observation dictionary is reduced to its raw
`state` field before it reaches a PB-CWM learner.

The development schedule in
`pbcwm/configs/benchmarks/pendulum/p1_p2_p1.yaml` is `m=1.0 -> 1.5 -> 1.0`
at global steps 8 and 16. The adapter keeps this schedule across episode
resets. It returns the true Gymnasium reward for evaluator bookkeeping, while
`build_agent_transition` creates the learner transition with reward `0.0`.

Run the two smoke paths from the repository root:

```text
python -m pbcwm.benchmarks.smoke --learner random --steps 24
python -m pbcwm.benchmarks.smoke --learner static --steps 24
```

The old `pbcwm.envs.NonstationaryPendulum` remains available for regression
compatibility; no existing baseline is silently migrated to this benchmark.
