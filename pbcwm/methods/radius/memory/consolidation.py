from ..types import ContextPosterior


def is_consolidation_ready(posterior: ContextPosterior, stable_steps: int, min_stable_steps: int, max_trace: float, novelty: float) -> bool:
    return stable_steps >= min_stable_steps and float(posterior.covariance.trace()) <= max_trace and novelty < 1.0
