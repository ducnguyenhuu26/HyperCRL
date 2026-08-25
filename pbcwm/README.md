# PB-CWM core skeleton

This directory contains the PB-CWM model-based control core and the first
preference-based baselines. Phase 0 provides:

```text
transition stream -> static MLP dynamics -> CEM planner -> action
```

Phase 1 adds the shared preference path and `MoP-RL-Online-FT`:

```text
recent transitions -> online-fine-tuned dynamics -> CEM
imagined CEM segments -> disagreement queries -> Bradley-Terry reward ensemble
```

HyperCRL was used as an architectural reference for separating environment
interaction, transition data, learned dynamics, and model-predictive control.
The implementation is intentionally independent of HyperCRL: it has no task
IDs, task-conditioned heads, boundary detector, per-task replay, normalization,
continual-learning regularizer, RND, demonstrations, D-REX, or image-specific
components.

Phase 2 adds the GPMM dynamics baseline in `pbcwm/baselines/gpmm/`:

```text
transition -> log-space expert assignment -> selected ExactGP update -> CEM
                    \-> new expert / old expert recall / conservative merge
```

Each expert is an independent modern GPyTorch Exact GP per observation delta
dimension, with bounded memory and no stage or task identifier. The planner
sees only the current expert's mean next-state prediction. Predictive
variance is reserved for assignment and merge diagnostics. The shared
preference buffer, disagreement selector, Bradley–Terry ensemble, and CEM
planner are reused unchanged from Phase 1.

Phase 3 adds `HyperCRL-Adapt` in `pbcwm/baselines/hypercrl/`. It preserves
HyperCRL's shared hypernetwork, one embedding per discovered regime,
functional delta-state target MLP, and output-space retention. The original
task-ID assumption is replaced only by a transparent residual router:

```text
recent real residual -> persistent mismatch -> stored-embedding matching
                                         \-> new embedding fallback
```

No oracle boundary, regime ID, task ID, reward, preference score, or imagined
CEM transition enters routing. Older transition datasets are not replayed;
the retained information is the generated-weight target for each inactive
embedding. HyperCRL-Adapt is therefore a minimum necessary boundary-free
adaptation, not a new learned routing method.

Phase 4 adds `VBLRL-Adapt` in `pbcwm/baselines/vblrl/`. It uses a compact
Bayesian delta-state MLP with variational `mu/rho` parameters and an
aleatoric log-variance head, plus a bounded lifetime reservoir posterior
`q_W` and retained regime posteriors `q_k`:

```text
real delta transitions -> q_W reservoir update
                       -> posterior-predictive NLL router
                          \-> q_k reuse / q_new <- q_W snapshot
                       -> sampled posterior dynamics -> shared CEM/PbRL
```

VBLRL routing is reward-free and stage-free: the learner receives only
`Transition(obs, action, next_obs, ...)`. The environment reward and true
stage are retained only by the evaluator. CEM uses posterior-sampled
dynamics particles through the generic `sample_next` contract. This is a
minimal PB-CWM adaptation of the VBLRL mechanism, not an exact reproduction
of a legacy implementation.

Phase 5 adds `Curious Replay-Adapt` in `pbcwm/baselines/curious_replay/`.
It keeps one reward-free deterministic delta MLP and one bounded lifetime FIFO
buffer. Each entry is prioritized by the Curious Replay combination

```text
p_i = c * beta ** replay_count_i
     + (last_model_loss_i + epsilon) ** alpha
```

New transitions receive the current maximum priority. Sampled entries update
their own replay count and post-update per-sample dynamics loss. There is no
task detector, regime state, model bank, retrieval path, reward-aware replay,
or stage-aware eviction. Reacquisition is replay-assisted re-adaptation, not
retrieval of a stored regime model.

## Install and run

From the repository root:

```bash
python -m pip install -r pbcwm/requirements.txt
python train.py --config pbcwm/configs/pendulum.yaml
python train.py --config pbcwm/configs/pendulum_moprl_online_ft.yaml
python train.py --config pbcwm/configs/pendulum_gpmm_return.yaml
python train.py --config pbcwm/configs/pendulum_hypercrl_adapt_return.yaml
python train.py --config pbcwm/configs/pendulum_vblrl_adapt.yaml
python train.py --config pbcwm/configs/pendulum_vblrl_adapt_return.yaml
python train.py --config pbcwm/configs/pendulum_curious_replay_adapt.yaml
python train.py --config pbcwm/configs/pendulum_curious_replay_adapt_return.yaml
python -m pytest pbcwm/tests -q
```

The Phase-0 smoke test writes `outputs/pendulum_smoke.jsonl`; the Phase-1 smoke
test writes `outputs/pendulum_moprl_online_ft.jsonl`; the GPMM return smoke
writes `outputs/pendulum_gpmm_return.jsonl`. Logs include held-out dynamics
MSE, episode return, preference accuracy, query counts, expert diagnostics,
and the hidden true stage for evaluation only. The HyperCRL-Adapt return smoke
writes `outputs/pendulum_hypercrl_adapt_return.jsonl` and additionally logs
generated-weight drift, router decisions, embedding assignments, and
stage-to-embedding evaluator metrics. The GPMM implementation is a
standardized adaptation of the mechanism described in the Phase-2 guide, not
an exact reproduction of a legacy implementation. VBLRL logs posterior NLL,
Bayesian parameter/predictive uncertainty, q_W buffer size, acquisition and
reacquisition counts, and stage-to-posterior evaluator metrics. Curious
Replay logs priority/count/loss statistics, evaluator-only sampled replay
shares, and model/return recovery fields for returning dynamics.

## Design boundaries

`DynamicsLearner` is the only replaceable baseline seam. Stochastic learners
may additionally implement `sample_next(obs, action, num_samples)` returning
`[num_samples, batch, obs_dim]`; `CEMPlanner` consumes that optional contract
for posterior-particle rollouts. The planner still receives only a generic
batched reward callable, so `LearnedPreferenceReward` replaces
`PendulumReward` without planner changes.
`MoP-RL-Online-FT` is a standardized continual adaptation of MoP-RL's core
model-based preference-learning architecture, not an exact reproduction of the
original paper. The synthetic teacher uses ground-truth reward only to create
labels; the planner receives learned reward only.

Preference labels use `0 = trajectory A preferred` and `1 = trajectory B
preferred`; Bradley–Terry training uses `score(B) - score(A)` as the BCE logit.
