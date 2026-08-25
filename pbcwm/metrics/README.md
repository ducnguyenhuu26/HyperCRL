# PB-CWM metrics

`pbcwm.metrics` is a protocol-agnostic evaluation layer. It keeps world-model
fidelity, continual acquisition/reacquisition, reward fidelity, world--reward
coupling, planning, and oracle diagnostics separate. It does not select a
final horizon, schedule, checkpoint frequency, candidate bank, or paper
aggregation rule.

Evaluation inputs are frozen probe/candidate batches. World-model rollout is
recursive after the true initial state, and all model evaluation runs under
`torch.no_grad()` without changing learner state. Undefined metrics return
`MetricResult(valid=False, value=None, reason=...)` rather than zero.

Stable names are registered in `registry.py`; flat records can be written to
JSON or CSV with `serialization.py`. Raw returns and raw error metrics are
rejected by `aggregate_cross_environment`; only dimensionless metrics may be
averaged across environments.
