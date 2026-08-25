"""Config parsing and validation for benchmark schedules."""

from collections.abc import Mapping

from .base import BenchmarkSpec, Regime


def benchmark_spec_from_mapping(config: Mapping[str, object]) -> BenchmarkSpec:
    """Parse the small YAML benchmark schema without hard-coding a schedule."""

    raw = config.get("benchmark", config)
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark config must be a mapping")
    raw_regimes = raw.get("regimes")
    if not isinstance(raw_regimes, list):
        raise ValueError("benchmark.regimes must be a list")
    regimes: list[Regime] = []
    for item in raw_regimes:
        if not isinstance(item, Mapping):
            raise ValueError("each regime must be a mapping")
        parameters = item.get("parameters")
        if not isinstance(parameters, Mapping):
            parameters = {str(raw["parameter"]): float(item["value"])}
        regimes.append(
            Regime(
                start_step=int(item["start_step"]),
                parameters={str(key): float(value) for key, value in parameters.items()},
            )
        )
    fixed = raw.get("fixed_parameters", {})
    if not isinstance(fixed, Mapping):
        raise ValueError("fixed_parameters must be a mapping")
    return BenchmarkSpec(
        name=str(raw["name"]),
        provider=str(raw.get("provider", "nsgym")),
        base_env=str(raw["base_env"]),
        parameter=str(raw["parameter"]),
        regimes=tuple(regimes),
        total_steps=int(raw.get("total_steps", 1)),
        change_notification=bool(raw.get("notifications", {}).get("change", False)),
        delta_change_notification=bool(raw.get("notifications", {}).get("delta_change", False)),
        fixed_parameters={str(key): float(value) for key, value in fixed.items()},
    )
