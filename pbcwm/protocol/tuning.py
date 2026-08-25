"""Fail-closed development tuning budget ledger."""


class TuningBudget:
    def __init__(self, max_configs: int, development_seeds: tuple[int, ...]) -> None:
        self.max_configs = int(max_configs)
        self.development_seeds = tuple(development_seeds)
        self._seen: dict[tuple[str, str], set[str]] = {}

    def register(self, method: str, environment: str, config_id: str) -> None:
        key = (method, environment)
        configs = self._seen.setdefault(key, set())
        if config_id not in configs and len(configs) >= self.max_configs:
            raise RuntimeError(f"tuning budget exceeded for {method} x {environment}")
        configs.add(config_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "max_configs_per_method_env": self.max_configs,
            "development_seeds": list(self.development_seeds),
            "registered": {f"{method}::{environment}": sorted(configs) for (method, environment), configs in self._seen.items()},
        }
