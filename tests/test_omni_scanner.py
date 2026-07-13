"""ContinuousOmniScanner unit tests with a stub classifier + registry + buffer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from minimappr.core.omni_scanner import ContinuousOmniScanner, OmniScanResult
from minimappr.models import ClassificationResult, NodeSpec, NodeType


@dataclass
class _Runtime:
    spec: NodeSpec
    sensor_ids: list[str]


class _StubRegistry:
    def __init__(self, runtimes):
        self._runtimes = runtimes

    async def list_nodes(self):
        return self._runtimes


class _StubBuffer:
    """Returns a fixed window for any sensor set."""

    def __init__(self, windows, sample_rate_hz=16000, end_ns=1_000):
        self._windows = windows
        self._sr = sample_rate_hz
        self._end_ns = end_ns

    async def get_recent_window_for_sensors(self, sensor_ids, window_seconds):
        available = {sid: self._windows[sid] for sid in sensor_ids if sid in self._windows}
        if not available:
            return None
        return available, self._sr, self._end_ns


class _StubClassifier:
    def __init__(self, label="Bird", confidence=0.7):
        self.calls = 0
        self.close_calls = 0
        self._label = label
        self._conf = confidence

    def classify(self, samples, sample_rate_hz):
        self.calls += 1
        return ClassificationResult(
            label=self._label, confidence=self._conf, scores={self._label: self._conf}, features={}
        )

    def close(self):
        self.close_calls += 1

    def cancel_pending(self): ...


class _Settings:
    omni_scan_enabled = True
    omni_scan_interval_seconds = 30.0
    omni_scan_window_seconds = 15.0
    omni_scan_min_rms = 0.01


def _node(node_id="node-a"):
    return NodeSpec(
        id=node_id,
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 2.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
    )


@pytest.mark.asyncio
async def test_scan_once_produces_result_for_loudest_sensor():
    loud = np.full(16000, 0.2, dtype=np.float32)
    quiet = np.full(16000, 0.02, dtype=np.float32)
    windows = {"node-a:ch0": quiet, "node-a:ch1": loud}
    results: list[OmniScanResult] = []

    scanner = ContinuousOmniScanner(
        settings=_Settings(),
        classifier=_StubClassifier(),
        registry=_StubRegistry([_Runtime(_node(), ["node-a:ch0", "node-a:ch1"])]),
        buffer=_StubBuffer(windows),
        sink=lambda r: results.append(r) or _noop(),
    )
    await scanner.scan_once()

    assert len(results) == 1
    assert results[0].sensor_id == "node-a:ch1"  # loudest
    assert results[0].classification.label == "Bird"
    assert scanner.stats()["scans_completed"] == 1


@pytest.mark.asyncio
async def test_rms_gate_skips_quiet_node():
    quiet = np.full(16000, 0.001, dtype=np.float32)
    results = []
    scanner = ContinuousOmniScanner(
        settings=_Settings(),
        classifier=_StubClassifier(),
        registry=_StubRegistry([_Runtime(_node(), ["node-a:ch0"])]),
        buffer=_StubBuffer({"node-a:ch0": quiet}),
        sink=lambda r: results.append(r) or _noop(),
    )
    await scanner.scan_once()
    assert results == []
    assert scanner.stats()["scans_skipped_rms"] == 1


@pytest.mark.asyncio
async def test_no_audio_counts_skip():
    scanner = ContinuousOmniScanner(
        settings=_Settings(),
        classifier=_StubClassifier(),
        registry=_StubRegistry([_Runtime(_node(), ["node-a:ch0"])]),
        buffer=_StubBuffer({}),  # no windows -> None
        sink=lambda r: _noop(),
    )
    await scanner.scan_once()
    assert scanner.stats()["scans_skipped_no_audio"] == 1


@pytest.mark.asyncio
async def test_disabled_scanner_does_not_start():
    class _Off(_Settings):
        omni_scan_enabled = False

    scanner = ContinuousOmniScanner(
        settings=_Off(),
        classifier=_StubClassifier(),
        registry=_StubRegistry([]),
        buffer=_StubBuffer({}),
        sink=lambda r: _noop(),
    )
    await scanner.start()
    assert scanner.stats()["running"] is False
    await scanner.stop()


@pytest.mark.asyncio
async def test_stop_closes_classifier_exactly_once():
    classifier = _StubClassifier()
    scanner = ContinuousOmniScanner(
        settings=_Settings(),
        classifier=classifier,
        registry=_StubRegistry([]),
        buffer=_StubBuffer({}),
        sink=lambda r: _noop(),
    )
    await scanner.start()
    await scanner.stop()
    assert classifier.close_calls == 1

    # Double-stop must be safe and not double-close.
    await scanner.stop()
    assert classifier.close_calls == 2


async def _noop():
    return None
