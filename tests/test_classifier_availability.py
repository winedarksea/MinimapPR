"""Tests for classifier-backend availability probing."""

from __future__ import annotations

import importlib

import pytest

from minimappr.classifiers import availability


@pytest.fixture(autouse=True)
def _clear_caches():
    availability.probe_backends.cache_clear()
    yield
    availability.probe_backends.cache_clear()


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
    assert names["moonshine_stt"] is False
    assert names["drone_head"] is False


def test_backend_available_reflects_installed_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        availability.importlib.util,
        "find_spec",
        _fake_find_spec(
            {"birdnet", "tensorflow", "onnxruntime", "huggingface_hub", "transformers"}
        ),
    )
    availability.probe_backends.cache_clear()
    assert availability.backend_available("birdnet") is True
    assert availability.backend_available("yamnet") is True
    assert availability.backend_available("drone_head") is True
    assert availability.backend_available("moonshine_stt") is True
    assert availability.backend_available("nonexistent") is False


def test_yamnet_without_tensorflow_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        availability.importlib.util, "find_spec", _fake_find_spec(set())
    )
    availability.probe_backends.cache_clear()
    assert availability.backend_available("yamnet") is False
    reasons = {e.name: e.reason for e in availability.probe_backends()}
    assert "tensorflow" in reasons["yamnet"]


# Ensure the module is importable stdlib-only (no numpy/tensorflow import cycle).
def test_module_is_stdlib_only() -> None:
    assert importlib.util.find_spec("minimappr.classifiers.availability") is not None
