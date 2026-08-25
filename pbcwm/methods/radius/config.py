"""RADIUS method defaults; shared protocol values stay outside this config."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class AtlasConfig:
    hidden_size: int = 256
    initial_rank: int = 2
    max_rank: int = 16
    context_l2: float = 1e-4
    atom_orthogonality: float = 1e-3


@dataclass(frozen=True)
class REFConfig:
    context_window: int = 128
    min_context_samples: int = 32
    residual_sigma: float = 1.0
    active_prior_bonus: float = 2.0
    memory_prior_mass: float = 1.0
    new_prior_mass: float = 0.25
    new_prior_std: float = 2.0
    context_process_noise: float = 1e-3
    numerical_jitter: float = 1e-6


@dataclass(frozen=True)
class MemoryConfig:
    max_prototypes: int = 32
    min_stable_steps: int = 256
    max_trace_for_consolidation: float = 0.5
    prototype_merge_mahalanobis: float = 2.5
    fusion_weight: float = 0.25


@dataclass(frozen=True)
class RNEConfig:
    residual_threshold: float = 3.0
    new_hypothesis_threshold: float = 0.6
    persistence_steps: int = 64
    cooldown_steps: int = 1000
    orthogonalization_ridge: float = 1e-4
    initialization_updates: int = 200
    initialization_lr: float = 1e-3
    new_context_variance: float = 1.0
    min_model_updates_before_expansion: int = 500


@dataclass(frozen=True)
class PECConfig:
    enabled: bool = True
    mode: str = "trust_region"
    forgetting_budget: float = 1e-3
    fisher_damping: float = 1e-3
    predictive_sigma: float = 1.0
    fisher_sketch_rank: int = 32
    fisher_refresh_interval: int = 2000
    anchors_per_prototype: int = 128
    optimizer_integration: str = "direct_parameter_step"
    max_step_norm: float = 1.0


@dataclass(frozen=True)
class PFPAConfig:
    enabled: bool = True
    context_samples: int = 4
    frontier_fraction: float = 0.8
    max_pair_action_similarity: float = 0.98


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    replay_capacity: int = 50000


@dataclass(frozen=True)
class AblationConfig:
    disable_recurrent_memory: bool = False
    disable_rne: bool = False
    disable_pec: bool = False
    disable_pfpa: bool = False
    hard_context_routing: bool = False
    fixed_atlas_rank: bool = False


@dataclass(frozen=True)
class RadiusConfig:
    name: str = "radius_pb_cwm"
    atlas: AtlasConfig = AtlasConfig()
    ref: REFConfig = REFConfig()
    memory: MemoryConfig = MemoryConfig()
    rne: RNEConfig = RNEConfig()
    pec: PECConfig = PECConfig()
    pfpa: PFPAConfig = PFPAConfig()
    training: TrainingConfig = TrainingConfig()
    ablations: AblationConfig = AblationConfig()


def _section(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = root.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"method.{name} must be a mapping")
    return value


def radius_config_from_mapping(data: Mapping[str, Any]) -> RadiusConfig:
    root = data.get("method", data)
    if not isinstance(root, Mapping):
        raise ValueError("radius config must contain a method mapping")
    def build(cls, name):
        values = {key: value for key, value in _section(root, name).items() if key in cls.__dataclass_fields__}
        return cls(**values)
    config = RadiusConfig(
        name=str(root.get("name", "radius_pb_cwm")),
        atlas=build(AtlasConfig, "atlas"),
        ref=build(REFConfig, "ref"),
        memory=build(MemoryConfig, "memory"),
        rne=build(RNEConfig, "rne"),
        pec=build(PECConfig, "pec"),
        pfpa=build(PFPAConfig, "pfpa"),
        training=build(TrainingConfig, "training"),
        ablations=build(AblationConfig, "ablations"),
    )
    if config.atlas.initial_rank <= 0 or config.atlas.initial_rank > config.atlas.max_rank:
        raise ValueError("atlas rank bounds are invalid")
    if config.ref.context_window < config.ref.min_context_samples:
        raise ValueError("context_window must cover min_context_samples")
    if config.ref.residual_sigma <= 0.0 or config.ref.new_prior_std <= 0.0:
        raise ValueError("REF scales must be positive")
    if config.ref.context_process_noise < 0.0:
        raise ValueError("context_process_noise must be non-negative")
    if min(config.ref.active_prior_bonus, config.ref.memory_prior_mass, config.ref.new_prior_mass) <= 0.0:
        raise ValueError("REF prior masses must be positive")
    if not 0.0 <= config.pfpa.frontier_fraction <= 1.0:
        raise ValueError("frontier_fraction must be in [0, 1]")
    if config.rne.min_model_updates_before_expansion < 0:
        raise ValueError("min_model_updates_before_expansion must be non-negative")
    if config.pec.max_step_norm <= 0.0:
        raise ValueError("max_step_norm must be positive")
    if config.pec.optimizer_integration not in {"direct_parameter_step", "transformed_gradient"}:
        raise ValueError("unsupported PEC optimizer integration")
    return config


def load_radius_config(path: str | Path) -> RadiusConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return radius_config_from_mapping(yaml.safe_load(handle))
