"""In-memory ingest consumer for sidecar-backed ingest via SSE.

Replaces the filesystem-polling IngestSpoolConsumer with an async SSE stream
that receives DSP result manifests (localization_result, classifier_render)
directly from the Rust sidecar over HTTP.  No audio or metadata touches disk.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from minimappr.api.rust_dsp_manifests import (
    LocalizedClassifierRenderRequest,
    load_localized_render_manifest_bundle,
)
from minimappr.interfaces import IngestTransport
from minimappr.models import EnvironmentSampleIn, NodeSpec
from minimappr.utils.audio import decode_pcm16le_b64, rms

if TYPE_CHECKING:
    from minimappr.core.audio_buffer import MultiSensorBuffer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StreamConsumerConfig:
    """Configuration for the SSE-based ingest consumer."""

    sidecar_base_url: str
    """Base URL of the Rust ingest sidecar (e.g. http://127.0.0.1:8081)."""

    reconnect_delay_seconds: float = 1.0
    """Delay between SSE reconnect attempts after a disconnect."""

    read_timeout_seconds: float = 30.0
    """HTTP read timeout for the SSE connection."""

    max_reconnect_backoff_seconds: float = 30.0
    """Cap for exponential reconnect backoff."""


@dataclass(frozen=True, slots=True)
class SidecarNodeSnapshot:
    """Latest in-memory node status extracted from sidecar localization results."""

    node_payload: dict[str, Any]
    last_seen_ns: int
    last_sample_time_ns: int | None = None
    sample_rate_hz: int | None = None
    active_sensor_count: int | None = None
    rms: float | None = None
    latest_environment: dict[str, Any] = field(default_factory=dict)


def _enrich_node_payload_from_context(
    node_payload: dict[str, Any],
    node_context: dict[str, Any],
) -> dict[str, Any]:
    """Merge frame-level fields from node_context into the node payload dict.

    The binary node header carries static fields (position, sensors, capabilities).
    Frame-level fields like ``time_quality`` live at the top level of
    ``node_context`` and are not present in the node header.  Injecting them
    into ``node.metadata`` ensures every ``upsert_node`` call carries a
    complete picture regardless of which manifest type triggered it.
    """
    time_quality = node_context.get("time_quality")
    if not isinstance(time_quality, str) or not time_quality:
        return node_payload
    node_payload = dict(node_payload)
    metadata: dict[str, Any] = dict(node_payload.get("metadata") or {})
    metadata["time_quality"] = time_quality
    node_payload["metadata"] = metadata
    return node_payload


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _environment_sample_from_context(
    node_context: dict[str, Any],
    *,
    fallback_timestamp_ns: int,
) -> EnvironmentSampleIn | None:
    environment_payload = node_context.get("environment")
    if not isinstance(environment_payload, dict):
        return None
    sample_payload = dict(environment_payload)
    sample_payload.setdefault("timestamp_ns", fallback_timestamp_ns)
    try:
        sample = EnvironmentSampleIn.model_validate(sample_payload)
    except Exception:  # noqa: BLE001
        return None
    return sample if sample.has_any_measurement() else None


def _audio_debug_from_context(node_context: dict[str, Any]) -> tuple[int | None, int | None, float | None]:
    audio_debug_payload = node_context.get("audio_debug")
    if not isinstance(audio_debug_payload, dict):
        return (None, None, None)
    return (
        _coerce_int(audio_debug_payload.get("sample_rate_hz")),
        _coerce_int(audio_debug_payload.get("active_sensor_count")),
        _coerce_float(audio_debug_payload.get("rms")),
    )


def _environment_summary(sample: EnvironmentSampleIn | None) -> dict[str, Any]:
    if sample is None:
        return {}
    payload = sample.model_dump(mode="json")
    return {
        key: value
        for key, value in payload.items()
        if value is not None or key == "metadata"
    }


class IngestStreamConsumer:
    """Async consumer that receives DSP manifests from the sidecar via SSE.

    The consumer maintains a persistent HTTP connection to
    ``/api/v1/dsp/stream`` and delivers each manifest to the
    ``IngestTransport`` as soon as it arrives.  No filesystem polling,
    no claim files, no SQLite cursor state.
    """

    def __init__(
        self,
        *,
        config: StreamConsumerConfig,
        ingest_transport: IngestTransport,
        audio_buffer: "MultiSensorBuffer | None" = None,
    ) -> None:
        self._config = config
        self._ingest_transport = ingest_transport
        self._audio_buffer = audio_buffer
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_event_id: str | None = None
        self._latest_node_snapshots: dict[str, SidecarNodeSnapshot] = {}

    def snapshot_nodes(self) -> dict[str, SidecarNodeSnapshot]:
        """Return the latest node state derived from sidecar heartbeat manifests."""
        return dict(self._latest_node_snapshots)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Idempotent start — spawns the background SSE listener."""
        if self.is_running:
            return
        if self._task is not None and self._task.done():
            self._task = None
        self._running = True
        self._task = asyncio.create_task(
            self._run_loop(),
            name="ingest-stream-consumer",
        )
        logger.info("IngestStreamConsumer started (sidecar=%s)", self._config.sidecar_base_url)

    async def stop(self) -> None:
        """Graceful stop — cancels the background task and waits for exit."""
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("IngestStreamConsumer stopped")

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Maintain SSE connection with automatic reconnect."""
        reconnect_delay = self._config.reconnect_delay_seconds
        while self._running:
            try:
                await self._connect_and_consume()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "IngestStreamConsumer connection lost: %s. Reconnecting in %.1fs...",
                    exc,
                    reconnect_delay,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(
                    reconnect_delay * 2,
                    self._config.max_reconnect_backoff_seconds,
                )
            else:
                # Clean disconnect (sidecar shut down gracefully).
                if self._running:
                    logger.info(
                        "IngestStreamConsumer disconnected cleanly. Reconnecting in %.1fs...",
                        reconnect_delay,
                    )
                    await asyncio.sleep(reconnect_delay)

    async def _connect_and_consume(self) -> None:
        """Open SSE connection and process events until disconnect."""
        url = f"{self._config.sidecar_base_url.rstrip('/')}/api/v1/dsp/stream"
        timeout = self._client_timeout()
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", url, headers=self._stream_request_headers()) as response:
                response.raise_for_status()
                logger.debug("IngestStreamConsumer SSE connected")
                current_data_lines: list[str] = []
                current_event_type = "message"
                current_event_id: str | None = None
                async for line in response.aiter_lines():
                    if not self._running:
                        break
                    line = line.rstrip("\r")
                    if line == "":
                        await self._dispatch_sse_event(
                            event_type=current_event_type,
                            event_id=current_event_id,
                            data_lines=current_data_lines,
                        )
                        current_data_lines = []
                        current_event_type = "message"
                        current_event_id = None
                        continue
                    if line.startswith(":"):
                        continue
                    field_name, separator, field_value = line.partition(":")
                    if separator == "":
                        continue
                    field_value = field_value.lstrip()
                    if field_name == "data":
                        current_data_lines.append(field_value)
                    elif field_name == "event":
                        current_event_type = field_value or "message"
                    elif field_name == "id":
                        current_event_id = field_value or None
                await self._dispatch_sse_event(
                    event_type=current_event_type,
                    event_id=current_event_id,
                    data_lines=current_data_lines,
                )

    def _stream_request_headers(self) -> dict[str, str]:
        headers = {"Accept": "text/event-stream"}
        if self._last_event_id:
            headers["Last-Event-ID"] = self._last_event_id
        return headers

    def _client_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=10.0,
            read=self._config.read_timeout_seconds,
            write=10.0,
            pool=5.0,
        )

    async def _dispatch_sse_event(
        self,
        *,
        event_type: str,
        event_id: str | None,
        data_lines: list[str],
    ) -> None:
        if not data_lines:
            return
        payload = "\n".join(data_lines).strip()
        if not payload:
            return
        if event_type == "replay_gap":
            self._handle_replay_gap(payload)
            return
        if event_type not in {"", "message"}:
            logger.debug("IngestStreamConsumer ignoring SSE event=%s", event_type)
            return
        await self._handle_event(payload)
        if event_id:
            self._last_event_id = event_id

    def _handle_replay_gap(self, payload: str) -> None:
        try:
            gap_payload = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("IngestStreamConsumer replay gap signaled with malformed payload")
            return
        logger.warning(
            (
                "IngestStreamConsumer replay gap detected: requested_after=%s "
                "oldest_available=%s skipped_messages=%s"
            ),
            gap_payload.get("requested_after"),
            gap_payload.get("oldest_available"),
            gap_payload.get("skipped_messages"),
        )

    async def _handle_event(self, payload: str) -> None:
        """Parse a single SSE data line and route to the transport."""
        try:
            manifest: dict[str, Any] = json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.warning("IngestStreamConsumer dropped malformed SSE payload: %s", exc)
            return

        manifest_type = str(manifest.get("manifest_type", ""))
        if manifest_type == "localization_result":
            await self._handle_localization_result(manifest)
        elif manifest_type == "classifier_render":
            await self._handle_classifier_render(manifest)
        elif manifest_type == "env_sample_append":
            await self._handle_env_sample_append(manifest)
        elif manifest_type == "raw_audio_frame":
            await self._handle_raw_audio_frame(manifest)
        else:
            logger.debug("IngestStreamConsumer ignoring manifest_type=%s", manifest_type)

    async def _handle_raw_audio_frame(self, manifest: dict[str, Any]) -> None:
        """Mirror sidecar raw ingest frames into Python's bounded live buffer."""
        node_context = manifest.get("node_context")
        if not isinstance(node_context, dict):
            logger.warning("raw_audio_frame missing node_context; skipping")
            return
        node_payload = node_context.get("node")
        if not isinstance(node_payload, dict):
            logger.warning("raw_audio_frame missing node payload; skipping")
            return
        frame_payload = manifest.get("raw_audio_frame")
        if not isinstance(frame_payload, dict):
            logger.warning("raw_audio_frame missing frame payload; skipping")
            return
        raw_audio_bytes = manifest.get("raw_audio_bytes")
        if not isinstance(raw_audio_bytes, str) or not raw_audio_bytes:
            logger.warning("raw_audio_frame missing inline PCM bytes; skipping")
            return

        node_payload = _enrich_node_payload_from_context(node_payload, node_context)
        try:
            node = NodeSpec.model_validate(node_payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("raw_audio_frame node validation failed: %s", exc)
            return

        sample_rate_hz = _coerce_int(frame_payload.get("sample_rate_hz"))
        channel_count = _coerce_int(frame_payload.get("channel_count"))
        if channel_count is None:
            channel_count = _coerce_int(frame_payload.get("channels"))
        start_time_ns = _coerce_int(frame_payload.get("start_time_ns"))
        end_time_ns = _coerce_int(frame_payload.get("end_time_ns"))
        if sample_rate_hz is None or sample_rate_hz <= 0:
            logger.warning("raw_audio_frame has invalid sample_rate_hz; skipping")
            return
        if channel_count is None or channel_count <= 0:
            logger.warning("raw_audio_frame has invalid channel_count; skipping")
            return
        if start_time_ns is None:
            start_time_ns = _coerce_int(node_context.get("toa_ns"))
        if start_time_ns is None:
            start_time_ns = _coerce_int(manifest.get("created_ns")) or time.time_ns()

        try:
            channels_first = decode_pcm16le_b64(raw_audio_bytes, channel_count)
        except ValueError as exc:
            logger.warning("raw_audio_frame PCM decode failed: %s", exc)
            return
        if channels_first.shape[0] != channel_count or channels_first.shape[1] == 0:
            logger.warning("raw_audio_frame decoded empty or mismatched channel payload")
            return

        if end_time_ns is None:
            frame_duration_ns = int(
                round(channels_first.shape[1] * 1_000_000_000 / sample_rate_hz)
            )
            end_time_ns = start_time_ns + frame_duration_ns

        audio_debug_context = dict(node_context)
        audio_debug_context["audio_debug"] = {
            "sample_rate_hz": sample_rate_hz,
            "active_sensor_count": channel_count,
            "rms": max(rms(channel) for channel in channels_first),
        }
        self._record_node_snapshot(
            node=node,
            node_context=audio_debug_context,
            node_audio_time_ns=end_time_ns,
        )

        if self._audio_buffer is None:
            return
        start_sample_index = _coerce_int(frame_payload.get("start_sample_index"))
        end_sample_index = _coerce_int(frame_payload.get("end_sample_index"))
        for channel_index, samples in enumerate(channels_first):
            await self._audio_buffer.append(
                sensor_id=f"{node.id}:ch{channel_index}",
                sample_rate_hz=sample_rate_hz,
                start_time_ns=start_time_ns,
                samples=samples,
                start_sample_index=start_sample_index,
                end_sample_index=end_sample_index,
                end_time_ns=end_time_ns,
            )

    async def _handle_localization_result(self, manifest: dict[str, Any]) -> None:
        """Deliver a localization result (with optional embedded classifier_render)."""
        node_context = manifest.get("node_context")
        if not isinstance(node_context, dict):
            logger.warning("localization_result missing node_context; skipping")
            return

        node_payload = node_context.get("node")
        if not isinstance(node_payload, dict):
            logger.warning("localization_result missing node payload; skipping")
            return

        # Merge frame-level fields from node_context into node.metadata before
        # validating.  The binary node header carries static fields; time_quality
        # lives at the context level and must travel with every heartbeat upsert
        # so the DB record stays current.  Without this, cadence heartbeats
        # (every ~250ms) overwrite the metadata with no time_quality, undoing
        # what the 30s classifier-render path correctly sets.
        node_payload = _enrich_node_payload_from_context(node_payload, node_context)

        try:
            node = NodeSpec.model_validate(node_payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("localization_result node validation failed: %s", exc)
            return

        # Always refresh node heartbeat from localization_result so online/offline
        # status does not depend on classifier embedding details.
        node_audio_time_ns = node_context.get("toa_ns")
        if not isinstance(node_audio_time_ns, int):
            node_audio_time_ns = int(manifest.get("created_ns") or time.time_ns())
        self._record_node_snapshot(
            node=node,
            node_context=node_context,
            node_audio_time_ns=node_audio_time_ns,
        )

        # If the manifest carries an embedded classifier_render, deliver it as a
        # localized render so Python gets the full audio + classification bundle.
        classifier_render = manifest.get("classifier_render")
        if isinstance(classifier_render, dict):
            try:
                bundle = load_localized_render_manifest_bundle(
                    manifest_payload=manifest,
                    journal_streams_dir=None,  # Memory-path: no journal dir needed.
                )
                await self._ingest_transport.deliver_localized_render(
                    bundle.request,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to deliver localized render from localization_result: %s", exc)

    async def _handle_env_sample_append(self, manifest: dict[str, Any]) -> None:
        env_samples_payload = manifest.get("env_samples")
        if not isinstance(env_samples_payload, dict):
            logger.warning("env_sample_append missing env_samples payload; skipping")
            return

        samples_payload = env_samples_payload.get("samples")
        if not isinstance(samples_payload, list):
            logger.warning("env_sample_append missing samples list; skipping")
            return

        for sample_payload in samples_payload:
            if not isinstance(sample_payload, dict):
                continue
            node_id = str(sample_payload.get("node_id") or "").strip()
            if not node_id:
                continue
            sample_mapping = sample_payload.get("sample")
            if not isinstance(sample_mapping, dict):
                continue
            try:
                sample = EnvironmentSampleIn.model_validate(sample_mapping)
            except Exception as exc:  # noqa: BLE001
                logger.warning("env_sample_append sample validation failed: %s", exc)
                continue
            await self._ingest_transport.deliver_environment_sample(
                node_id=node_id,
                sample=sample,
            )

    async def _handle_classifier_render(self, manifest: dict[str, Any]) -> None:
        """Deliver a standalone classifier_render manifest."""
        try:
            bundle = load_localized_render_manifest_bundle(
                manifest_payload=manifest,
                journal_streams_dir=None,  # Memory-path: no journal dir needed.
            )
            await self._ingest_transport.deliver_localized_render(bundle.request)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to deliver classifier_render: %s", exc)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    def _record_node_snapshot(
        self,
        *,
        node: NodeSpec,
        node_context: dict[str, Any],
        node_audio_time_ns: int,
    ) -> None:
        sample_rate_hz, active_sensor_count, rms = _audio_debug_from_context(node_context)
        environment_sample = _environment_sample_from_context(
            node_context,
            fallback_timestamp_ns=node_audio_time_ns,
        )
        existing = self._latest_node_snapshots.get(node.id)
        self._latest_node_snapshots[node.id] = SidecarNodeSnapshot(
            node_payload=node.model_dump(mode="json"),
            last_seen_ns=time.time_ns(),
            last_sample_time_ns=node_audio_time_ns,
            sample_rate_hz=(
                sample_rate_hz
                if sample_rate_hz is not None
                else existing.sample_rate_hz if existing is not None else None
            ),
            active_sensor_count=(
                active_sensor_count
                if active_sensor_count is not None
                else existing.active_sensor_count if existing is not None else None
            ),
            rms=rms if rms is not None else existing.rms if existing is not None else None,
            latest_environment=(
                _environment_summary(environment_sample)
                if environment_sample is not None
                else existing.latest_environment if existing is not None else {}
            ),
        )
