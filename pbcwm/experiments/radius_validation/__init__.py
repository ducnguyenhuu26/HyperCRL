"""Hopper-only RADIUS component validation infrastructure."""

__all__ = ["generate_fixed_stream", "load_fixed_stream", "build_variant", "variant_config"]


def __getattr__(name):
    if name in {"generate_fixed_stream", "load_fixed_stream"}:
        from .generate_fixed_stream import generate_fixed_stream, load_fixed_stream

        return {"generate_fixed_stream": generate_fixed_stream, "load_fixed_stream": load_fixed_stream}[name]
    if name in {"build_variant", "variant_config"}:
        from .variants import build_variant, variant_config

        return {"build_variant": build_variant, "variant_config": variant_config}[name]
    raise AttributeError(name)
