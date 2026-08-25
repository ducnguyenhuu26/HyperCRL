# RADIUS Hopper component validation

This directory implements the development-only validation protocol in
`RADIUS_HOPPER_COMPONENT_VALIDATION_SETUP.md`. It is Hopper-v5 only and is
not a paper benchmark.

## Protocol order

1. Generate one frozen stream with seed 0.
2. Run W0-W4 on the same learner payload.
3. Inspect fixed-stream metrics and mechanism diagnostics.
4. Run the online ablation for W1-W4.
5. Run Q0/Q1/Q2 PFPA isolation with equal timestamps and label budgets.
6. Only after these gates pass, start full RADIUS development and seeds 1/2.

The physical schedule is `[P0, A, B, C, B, A]`, with 10,000 steps per
stage. A is torso mass x1.25, B is floor friction x0.75, and C is thigh
joint damping x1.25. NS-Gym notifications are disabled. Regime metadata is
stored in the JSON sidecar and is never present in the learner transition.

## CPU setup smoke

The following creates a short synthetic fixture. It tests plumbing only and
must not be reported as Hopper evidence:

```powershell
python -m pbcwm.experiments.radius_validation.generate_fixed_stream --synthetic --steps 256 --output outputs/radius_validation/smoke_stream.npz
python -m pbcwm.experiments.radius_validation.run_fixed_stream --synthetic --stream outputs/radius_validation/smoke_stream.npz --probe-dir outputs/radius_validation/smoke_probe_banks --variant W0 --max-steps 256
```

## GPU/Hopper commands

After MuJoCo and CUDA are available, generate the real stream once:

```powershell
python -m pbcwm.experiments.radius_validation.generate_fixed_stream --seed 0 --output outputs/radius_validation/hopper_fixed_stream_seed0.npz
python -m pbcwm.experiments.radius_validation.generate_probe_bank --seed 0 --output-dir outputs/radius_validation/probe_banks_seed0
```

Then run one variant at a time on the same file:

```powershell
python -m pbcwm.experiments.radius_validation.run_fixed_stream --stream outputs/radius_validation/hopper_fixed_stream_seed0.npz --probe-dir outputs/radius_validation/probe_banks_seed0 --variant W0 --device cuda --output outputs/radius_validation/w0.json
python -m pbcwm.experiments.radius_validation.run_fixed_stream --stream outputs/radius_validation/hopper_fixed_stream_seed0.npz --probe-dir outputs/radius_validation/probe_banks_seed0 --variant W1 --device cuda --output outputs/radius_validation/w1.json
python -m pbcwm.experiments.radius_validation.run_fixed_stream --stream outputs/radius_validation/hopper_fixed_stream_seed0.npz --probe-dir outputs/radius_validation/probe_banks_seed0 --variant W2 --device cuda --output outputs/radius_validation/w2.json
python -m pbcwm.experiments.radius_validation.run_fixed_stream --stream outputs/radius_validation/hopper_fixed_stream_seed0.npz --probe-dir outputs/radius_validation/probe_banks_seed0 --variant W3 --device cuda --output outputs/radius_validation/w3.json
python -m pbcwm.experiments.radius_validation.run_fixed_stream --stream outputs/radius_validation/hopper_fixed_stream_seed0.npz --probe-dir outputs/radius_validation/probe_banks_seed0 --variant W4 --device cuda --output outputs/radius_validation/w4.json
```

The runner is intentionally not a full paper campaign launcher. Do not add
HalfCheetah, Ant, external baselines, 10 seeds, or final-paper tables to this
stage. Every result directory should retain `summary.json`, the exact config,
the stream hash, the git commit, and failure flags.
