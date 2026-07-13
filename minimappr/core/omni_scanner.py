"""Continuous omni scanner — periodic site-wide classification of ambient audio.

The scanner is the "BirdNET always runs on continuous omni batches (~30s)" half of
the classifier overhaul. It runs one asyncio task, started/stopped with the
FusionNode. Each tick it pulls a recent window per registered node from the ring
buffers, RMS-gates it, runs the ``omni_continuous`` routing-context classifier in
a worker thread, and hands the result to a sink callback (the FusionNode's
``ingest_continuous_scan_result``) which routes it through the normal omni
assembly + site-wide dedupe path.

Scheduling is deadline-based with skip-if-behind backpressure: if a scan cycle
overruns the interval the next deadline is reset to "now" (a scan is skipped) and
an overrun is counted, so a slow classifier can never make the scanner spiral.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np

from minimappr.classifiers.base import AudioClassifier
from minimappr.core.node_registry import NodeRegistry
from minimappr.models import ClassificationResult, NodeSpec

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OmniScanResult:
    """One node's continuous-scan classification, handed to the sink."""

    node: NodeSpec
    sensor_id: str
    audio: np.ndarray
    sample_rate_hz: int
    end_time_ns: int
    classification: ClassificationResult
    rms: float


ScanSink = Callable[[OmniScanResult], Awaitable[None]]


def _rms(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


class ContinuousOmniScanner:
    def __init__(
        self,
        *,
        settings,
        classifier: AudioClassifier,
        registry: NodeRegistry,
        buffer,
        sink: ScanSink,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._classifier = classifier
        self._registry = registry
        self._buffer = buffer
        self._sink = sink
        self._clock = clock

        self._interval_seconds = max(1.0, float(settings.omni_scan_interval_seconds))
        self._window_seconds = max(0.5, float(settings.omni_scan_window_seconds))
        self._min_rms = max(0.0, float(settings.omni_scan_min_rms))

        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._overrun_log_last_s = 0.0

        # Telemetry surfaced in FusionNode.status().
        self._scans_completed = 0
        self._scans_skipped_rms = 0
        self._scans_skipped_no_audio = 0
        self._overruns = 0
        self._last_scan_monotonic: float | None = None
        self._errors = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if self._task is not None:
            return
        if not self._settings.omni_scan_enabled:
            logger.info("Continuous omni scanner disabled (omni_scan_enabled=False); idle.")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="omni-scanner")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        try:
            await asyncio.wait_for(asyncio.to_thread(self._classifier.close), timeout=10.0)
        except Exception:  # noqa: BLE001
            logger.warning("Omni scanner classifier close failed or timed out", exc_info=True)

    # ------------------------------------------------------------------ #
    # Loop
    # ------------------------------------------------------------------ #
    async def _run_loop(self) -> None:
        deadline = self._clock() + self._interval_seconds
        while not self._stop_event.is_set():
            now = self._clock()
            wait = deadline - now
            if wait > 0:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=wait)
                    return  # stop requested
                except asyncio.TimeoutError:
                    pass
            start = self._clock()
            try:
                await self.scan_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — one bad tick must not kill the loop
                self._errors += 1
                logger.exception("Continuous omni scan tick failed")
            elapsed = self._clock() - start
            if elapsed > self._interval_seconds:
                # Skip-if-behind: reset deadline to now so we don't spiral.
                self._overruns += 1
                self._maybe_log_overrun(elapsed)
                deadline = self._clock()
            else:
                deadline += self._interval_seconds

    def _maybe_log_overrun(self, elapsed: float) -> None:
        now = self._clock()
        if now - self._overrun_log_last_s >= 30.0:
            self._overrun_log_last_s = now
            logger.warning(
                "Omni scan overran interval (%.1fs > %.1fs); skipping ahead.",
                elapsed,
                self._interval_seconds,
            )

    # ------------------------------------------------------------------ #
    # One scan
    # ------------------------------------------------------------------ #
    async def scan_once(self) -> None:
        nodes = await self._registry.list_nodes()
        for runtime in nodes:
            if self._stop_event.is_set():
                return
            await self._scan_node(runtime.spec, runtime.sensor_ids)

    async def _scan_node(self, node: NodeSpec, sensor_ids: list[str]) -> None:
        if not sensor_ids or node.position_m is None:
            return
        window = await self._buffer.get_recent_window_for_sensors(
            sensor_ids, self._window_seconds
        )
        if window is None:
            self._scans_skipped_no_audio += 1
            return
        aligned, sample_rate_hz, end_time_ns = window
        # Reference/omni sensor = loudest available channel for this node.
        sensor_id, audio = max(aligned.items(), key=lambda kv: _rms(kv[1]))
        rms = _rms(audio)
        if rms < self._min_rms:
            self._scans_skipped_rms += 1
            return

        classification = await asyncio.to_thread(
            self._classifier.classify, audio, sample_rate_hz
        )
        self._scans_completed += 1
        self._last_scan_monotonic = self._clock()
        await self._sink(
            OmniScanResult(
                node=node,
                sensor_id=sensor_id,
                audio=audio,
                sample_rate_hz=sample_rate_hz,
                end_time_ns=end_time_ns,
                classification=classification,
                rms=rms,
            )
        )

    # ------------------------------------------------------------------ #
    # Telemetry
    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, object]:
        return {
            "enabled": bool(self._settings.omni_scan_enabled),
            "running": self._task is not None and not self._task.done(),
            "interval_seconds": self._interval_seconds,
            "window_seconds": self._window_seconds,
            "min_rms": self._min_rms,
            "scans_completed": self._scans_completed,
            "scans_skipped_rms": self._scans_skipped_rms,
            "scans_skipped_no_audio": self._scans_skipped_no_audio,
            "overruns": self._overruns,
            "errors": self._errors,
        }


__all__ = ["ContinuousOmniScanner", "OmniScanResult", "ScanSink"]
