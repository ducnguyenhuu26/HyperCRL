import numpy as np

from pbcwm.experiment.factory import CANONICAL_METHODS, build_method
from pbcwm.methods.radius import RadiusPbCWM


def test_factory_exposes_canonical_names_and_builds_radius_and_static():
    assert "radius_pb_cwm" in CANONICAL_METHODS
    radius = build_method("radius_pb_cwm", obs_dim=2, action_dim=1, action_low=np.array([-2.0]), action_high=np.array([2.0]))
    static = build_method("static", obs_dim=2, action_dim=1)
    assert isinstance(radius, RadiusPbCWM)
    assert hasattr(static, "observe")
