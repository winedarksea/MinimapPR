"""Classifier-backend availability probing and resolution.

Stdlib-only (``importlib.util.find_spec``) so it can be imported from
``config.py`` without dragging in numpy/tensorflow or creating a
``config`` <-> ``classifiers`` import cycle.

A backend is a named registry entry with a set of Python packages it needs.
``probe_backends()`` reports which are importable. Adding a future backend is a
single registry entry here plus its routing/factory wiring.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache

# name -> required importable module specs. Empty tuple = always available.
_BACKEND_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "heuristic": (),
    "yamnet": ("tensorflow",),
    "birdnet": ("birdnet",),
    "moonshine_stt": ("huggingface_hub", "transformers", "onnxruntime"),
    "drone_head": ("onnxruntime",),
}


@dataclass(frozen=True, slots=True)
class BackendAvailability:
    name: str
    available: bool
    reason: str


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        # A broken/partial install can raise from find_spec; treat as absent.
        return False


def _probe_one(name: str, requirements: tuple[str, ...]) -> BackendAvailability:
    missing = [module for module in requirements if not _module_available(module)]
    if not missing:
        return BackendAvailability(name=name, available=True, reason="installed")
    return BackendAvailability(
        name=name,
        available=False,
        reason=f"missing: {', '.join(missing)}",
    )


@lru_cache(maxsize=1)
def probe_backends() -> tuple[BackendAvailability, ...]:
    """Probe every registered backend once per process (lru_cached)."""
    return tuple(
        _probe_one(name, requirements)
        for name, requirements in _BACKEND_REQUIREMENTS.items()
    )


def backend_available(name: str) -> bool:
    for entry in probe_backends():
        if entry.name == name:
            return entry.available
    return False


__all__ = [
    "BackendAvailability",
    "backend_available",
    "probe_backends",
]
