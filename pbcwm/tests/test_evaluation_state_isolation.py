from pbcwm.experiment.evaluation import isolated_evaluation


class _Checkpointable:
    def __init__(self):
        self.value = 3

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


def test_evaluation_mutation_is_restored():
    learner = _Checkpointable()
    result = isolated_evaluation(learner, lambda current: (setattr(current, "value", 99), "ok")[1])
    assert result == "ok"
    assert learner.value == 3
