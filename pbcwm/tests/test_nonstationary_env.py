import numpy as np

from pbcwm.envs.nonstationary_pendulum import NonstationaryPendulum


def test_hidden_parameter_schedule_and_diagnostics() -> None:
    env = NonstationaryPendulum(
        [
            {"step": 0, "mass": 1.0, "length": 1.0, "gravity": 10.0},
            {"step": 2, "mass": 1.5, "length": 1.0, "gravity": 10.0},
        ]
    )
    observation, info = env.reset(seed=0)
    assert observation.shape == (3,)
    assert info["true_dynamics_stage"] == 0
    assert env.unwrapped.m == 1.0

    first, _, _, _, first_info = env.step(np.array([0.0], dtype=np.float32))
    second, _, _, _, second_info = env.step(np.array([0.0], dtype=np.float32))
    third, _, _, _, third_info = env.step(np.array([0.0], dtype=np.float32))
    assert first.shape == second.shape == third.shape == (3,)
    assert first_info["true_dynamics_stage"] == 0
    assert second_info["true_dynamics_stage"] == 0
    assert third_info["true_dynamics_stage"] == 1
    assert third_info["true_dynamics_params"]["mass"] == 1.5
    assert observation.shape == (3,)
    env.close()
