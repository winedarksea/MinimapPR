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
from dataclasses import dataclass
from typing import Any

import httpx

from minimappr.api.rust_dsp_manifests import (
    LocalizedClassifierRenderRequest,
    load_localized_render_manifest_bundle,
)
from minimappr.interfaces import IngestTransport
from minimappr.models import EnvironmentSampleIn, NodeSpec

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
    ) -> None:
        self._config = config
        self._ingest_transport = ingest_transport
        self._running = False
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Idempotent start — spawns the background SSE listener."""
        if self._running:
            return
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
        timeout = httpx.Timeout(
            connect=10.0,
            read=self._config.read_timeout_seconds,
            pool=5.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", url, headers={"Accept": "text/event-stream"}) as response:
                response.raise_for_status()
                logger.debug("IngestStreamConsumer SSE connected")
                async for line in response.aiter_lines():
                    if not self._running:
                        break
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if not payload:
                        continue
                    await self._handle_event(payload)

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
        else:
            logger.debug("IngestStreamConsumer ignoring manifest_type=%s", manifest_type)

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

        try:
            node = NodeSpec.model_validate(node_payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("localization_result node validation failed: %s", exc)
            return

        # Always refresh node heartbeat from localization_result so online/offline
        # status does not depend on classifier embedding details.
        await self._ingest_transport.deliver_node_heartbeat(node)

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
