# PB-CWM experiment protocol

This package freezes the reproducibility machinery without selecting physical
P0/P1/P2/P3 parameter vectors. The canonical placeholder is
`[P0, A, B, C, B, A]`; A/B/C are permuted per root seed, stages 1--5 receive
paired deterministic jitter, and all schedule/query/checkpoint progress uses
`global_env_step`.

Run the scenario-free acceptance smoke from the repository root:

```text
python -m pbcwm.protocol.smoke --config pbcwm/configs/protocol.yaml --environment Pendulum-v1 --seed 0 --log-path outputs/protocol_smoke.jsonl
```

The future scenario adapter owns physical switching and hidden true reward.
Learners receive no protocol metadata, and evaluator checkpoints/queries are
logged separately. The existing baseline runners are not changed by this
setup.
