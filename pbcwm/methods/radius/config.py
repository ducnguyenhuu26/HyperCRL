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
    min_model_updates_before_tracking: int = 32
    prototype_assignment_probability: float = 0.8


@dataclass(frozen=True)
class MemoryConfig:
    max_prototypes: int = 32
    min_stable_steps: int = 256
    max_trace_for_consolidation: float = 0.5
    prototype_merge_mahalanobis: float = 2.5
    fusion_weight: float = 0.25
    min_model_updates_before_consolidation: int = 256
    stable_residual_threshold: float = 1.25
    max_mean_variance_for_consolidation: float = 0.5


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
    min_fisher_rank: int = 1


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
    replay_context_mode: str = "prototype_if_available"


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
    allowed_root = {"name", "atlas", "ref", "memory", "rne", "pec", "pfpa", "training", "ablations"}
    unknown_root = set(root) - allowed_root
    if unknown_root:
        raise ValueError(f"method contains unknown keys: {sorted(unknown_root)}")
    def build(cls, name):
        section = _section(root, name)
        unknown = set(section) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"method.{name} contains unknown keys: {sorted(unknown)}")
        values = dict(section)
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
    if config.atlas.hidden_size <= 0 or config.atlas.initial_rank <= 0 or config.atlas.initial_rank > config.atlas.max_rank:
        raise ValueError("atlas rank bounds are invalid")
    if config.ref.min_context_samples <= 0 or config.ref.context_window < config.ref.min_context_samples:
        raise ValueError("context_window must cover positive min_context_samples")
    if config.ref.residual_sigma <= 0.0 or config.ref.new_prior_std <= 0.0:
        raise ValueError("REF scales must be positive")
    if config.ref.context_process_noise < 0.0 or config.ref.numerical_jitter < 0.0:
        raise ValueError("REF process noise and numerical jitter must be non-negative")
    if min(config.ref.active_prior_bonus, config.ref.memory_prior_mass, config.ref.new_prior_mass) <= 0.0:
        raise ValueError("REF prior masses must be positive")
    if config.ref.min_model_updates_before_tracking < 0 or not 0.0 < config.ref.prototype_assignment_probability <= 1.0:
        raise ValueError("REF readiness and assignment probability are invalid")
    if config.memory.max_prototypes <= 0 or config.memory.min_stable_steps <= 0:
        raise ValueError("memory capacity and stable steps must be positive")
    if config.memory.max_mean_variance_for_consolidation <= 0.0 or config.memory.stable_residual_threshold <= 0.0:
        raise ValueError("memory uncertainty and residual thresholds must be positive")
    if config.memory.prototype_merge_mahalanobis < 0.0 or not 0.0 < config.memory.fusion_weight <= 1.0:
        raise ValueError("memory merge settings are invalid")
    if config.rne.persistence_steps <= 0 or config.rne.cooldown_steps < 0:
        raise ValueError("RNE persistence/cooldown settings are invalid")
    if min(config.rne.residual_threshold, config.rne.new_hypothesis_threshold, config.rne.new_context_variance) <= 0.0:
        raise ValueError("RNE thresholds and context variance must be positive")
    if config.rne.initialization_updates <= 0 or config.rne.initialization_lr <= 0.0 or config.rne.orthogonalization_ridge <= 0.0:
        raise ValueError("RNE initialization settings are invalid")
    if not 0.0 <= config.pfpa.frontier_fraction <= 1.0:
        raise ValueError("frontier_fraction must be in [0, 1]")
    if config.pfpa.context_samples <= 0 or not 0.0 <= config.pfpa.max_pair_action_similarity <= 1.0:
        raise ValueError("PFPA settings are invalid")
    if config.rne.min_model_updates_before_expansion < 0:
        raise ValueError("min_model_updates_before_expansion must be non-negative")
    if config.pec.forgetting_budget <= 0.0 or config.pec.fisher_damping <= 0.0 or config.pec.predictive_sigma <= 0.0 or config.pec.max_step_norm <= 0.0:
        raise ValueError("PEC budget, damping, and step norm must be positive")
    if config.pec.fisher_sketch_rank < 0 or config.pec.fisher_refresh_interval <= 0 or config.pec.anchors_per_prototype <= 0 or config.pec.min_fisher_rank < 0:
        raise ValueError("PEC rank/refresh settings are invalid")
    if config.pec.optimizer_integration not in {"direct_parameter_step", "transformed_gradient"}:
        raise ValueError("unsupported PEC optimizer integration")
    if config.training.batch_size <= 0 or config.training.replay_capacity < config.training.batch_size or config.training.learning_rate <= 0.0:
        raise ValueError("training batch/replay/learning-rate settings are invalid")
    if config.training.replay_context_mode not in {"historical", "prototype_if_available"}:
        raise ValueError("unsupported replay_context_mode")
    return config


def load_radius_config(path: str | Path) -> RadiusConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return radius_config_from_mapping(yaml.safe_load(handle))
