"""Classifier-backend availability probing and resolution.

Stdlib-only (``importlib.util.find_spec``) so it can be imported from
``config.py`` without dragging in numpy/tensorflow or creating a
``config`` <-> ``classifiers`` import cycle.

A backend is a named registry entry with a set of Python packages it needs.
``probe_backends()`` reports which are importable; ``resolve_backend()`` maps a
configured value (possibly ``"auto"``) onto a concrete backend, preferring the
most capable installed one. Adding a future backend (e.g. speech-to-text) is a
single registry entry here plus its factory wiring.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache

# Preference order for ``"auto"`` resolution: most capable first. Heuristic is
# always available and is the guaranteed terminal fallback.
_AUTO_PREFERENCE: tuple[str, ...] = ("birdnet", "yamnet", "heuristic")

# name -> required importable module specs. Empty tuple = always available.
_BACKEND_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "heuristic": (),
    "yamnet": ("tensorflow", "tensorflow_hub"),
    "birdnet": ("birdnet",),
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
        _probe_one(name, _BACKEND_REQUIREMENTS[name]) for name in _AUTO_PREFERENCE
    )


def backend_available(name: str) -> bool:
    for entry in probe_backends():
        if entry.name == name:
            return entry.available
    return False


@lru_cache(maxsize=8)
def resolve_backend(configured: str) -> str:
    """Resolve a configured backend value onto a concrete backend name.

    An explicit backend name passes through unchanged (even if not installed —
    the factory keeps its own per-backend try/except fallback). ``"auto"``
    selects the first available backend in preference order (birdnet > yamnet >
    heuristic), always terminating at heuristic.
    """
    value = (configured or "").strip().lower()
    if value and value != "auto":
        return value
    for entry in probe_backends():
        if entry.available:
            return entry.name
    return "heuristic"


__all__ = [
    "BackendAvailability",
    "backend_available",
    "probe_backends",
    "resolve_backend",
]
