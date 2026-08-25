"""Independent deterministic RNG streams derived from one root seed."""

from dataclasses import dataclass

import numpy as np

STREAM_NAMES = (
    "schedule_seed",
    "environment_seed",
    "learner_seed",
    "replay_seed",
    "planner_seed",
    "preference_query_seed",
    "teacher_seed",
    "evaluation_seed",
)


@dataclass(frozen=True)
class SeedStreams:
    root_seed: int
    values: dict[str, int]

    def __getitem__(self, name: str) -> int:
        return self.values[name]

    def to_dict(self) -> dict[str, int]:
        return {"root_seed": self.root_seed, **self.values}


def spawn_seed_streams(root_seed: int) -> SeedStreams:
    sequence = np.random.SeedSequence(int(root_seed))
    children = sequence.spawn(len(STREAM_NAMES))
    values = {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(STREAM_NAMES, children)
    }
    return SeedStreams(int(root_seed), values)
