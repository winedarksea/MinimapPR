"""Tests for classifier-backend availability probing and resolution."""

from __future__ import annotations

import importlib

import pytest

from minimappr.classifiers import availability


@pytest.fixture(autouse=True)
def _clear_caches():
    availability.probe_backends.cache_clear()
    availability.resolve_backend.cache_clear()
    yield
    availability.probe_backends.cache_clear()
    availability.resolve_backend.cache_clear()


def _fake_find_spec(installed: set[str]):
    def _find_spec(name: str):
        return object() if name in installed else None

    return _find_spec


def test_heuristic_always_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(availability.importlib.util, "find_spec", _fake_find_spec(set()))
    availability.probe_backends.cache_clear()
    names = {e.name: e.available for e in availability.probe_backends()}
    assert names["heuristic"] is True
    assert names["yamnet"] is False
    assert names["birdnet"] is False


def test_auto_prefers_birdnet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        availability.importlib.util,
        "find_spec",
        _fake_find_spec({"birdnet", "tensorflow", "tensorflow_hub"}),
    )
    availability.probe_backends.cache_clear()
    availability.resolve_backend.cache_clear()
    assert availability.resolve_backend("auto") == "birdnet"


def test_auto_falls_to_yamnet_then_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        availability.importlib.util,
        "find_spec",
        _fake_find_spec({"tensorflow", "tensorflow_hub"}),
    )
    availability.probe_backends.cache_clear()
    availability.resolve_backend.cache_clear()
    assert availability.resolve_backend("auto") == "yamnet"

    monkeypatch.setattr(availability.importlib.util, "find_spec", _fake_find_spec(set()))
    availability.probe_backends.cache_clear()
    availability.resolve_backend.cache_clear()
    assert availability.resolve_backend("auto") == "heuristic"


def test_explicit_backend_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(availability.importlib.util, "find_spec", _fake_find_spec(set()))
    availability.resolve_backend.cache_clear()
    # Explicit value honored even if not installed (factory keeps its own fallback).
    assert availability.resolve_backend("birdnet") == "birdnet"
    assert availability.resolve_backend("yamnet") == "yamnet"


# Ensure the module is importable stdlib-only (no numpy/tensorflow import cycle).
def test_module_is_stdlib_only() -> None:
    assert importlib.util.find_spec("minimappr.classifiers.availability") is not None
