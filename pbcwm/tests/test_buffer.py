import numpy as np
import torch

from pbcwm.core.buffer import ReplayBuffer
from pbcwm.core.types import Transition


def make_transition(index: int) -> Transition:
    obs = np.array([index, index + 1], dtype=np.float32)
    return Transition(obs, np.array([index], dtype=np.float32), obs + 1, 0.0, False, False)


def test_buffer_add_capacity_and_sample() -> None:
    buffer = ReplayBuffer(capacity=3, seed=7)
    for index in range(5):
        buffer.add(make_transition(index))

    assert len(buffer) == 3
    batch = buffer.sample(2)
    assert batch.obs.shape == (2, 2)
    assert batch.action.shape == (2, 1)
    assert batch.next_obs.shape == (2, 2)
    assert batch.obs.dtype == torch.float32
    assert set(batch.obs[:, 0].tolist()).issubset({2.0, 3.0, 4.0})
