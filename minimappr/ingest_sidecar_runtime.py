"""Managed ingest sidecar runtime helpers.

This module isolates process supervision, readiness probing, and launch
environment assembly so the FastAPI entrypoint does not need to own the full
sidecar runtime lifecycle inline.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Awaitable, Callable

from minimappr.config import IngestSidecarProcessConfig, IngestSidecarStartupConfig


class IngestSidecarRuntimeState:
    """Mutable runtime health for the supervised ingest sidecar process."""

    __slots__ = ("status", "pid", "restart_count", "last_exit_code", "_current_process")

    def __init__(self) -> None:
        self.status: str = "disabled"
        self.pid: int | None = None
        self.restart_count: int = 0
        self.last_exit_code: int | None = None
        self._current_process: "asyncio.subprocess.Process | None" = None


def ingest_sidecar_is_running(state) -> bool:
    sidecar_state: IngestSidecarRuntimeState | None = getattr(state, "sidecar_state", None)
    return bool(
        state.settings.ingest_sidecar_enabled
        and sidecar_state is not None
        and sidecar_state.status == "running"
    )


def should_autostart_ingest_sidecar(settings) -> bool:
    """Return True when the managed sidecar process should be started."""
    return (
        getattr(settings, "process_role", "combined") != "ingest"
        and getattr(settings, "ingest_backend", "rust") == "rust"
        and settings.ingest_sidecar_enabled
        and not settings.direct_ingest_enabled
    )


def _sidecar_healthcheck_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/healthz"


def ingest_sidecar_startup_config(settings) -> IngestSidecarStartupConfig:
    startup_config_factory = getattr(settings, "ingest_sidecar_startup_config", None)
    if callable(startup_config_factory):
        return startup_config_factory()
    return IngestSidecarStartupConfig(
        ready_timeout_seconds=float(getattr(settings, "ingest_sidecar_ready_timeout_seconds", 5.0)),
        ready_poll_interval_seconds=float(getattr(settings, "ingest_sidecar_ready_poll_interval_seconds", 0.1)),
        healthcheck_timeout_seconds=float(getattr(settings, "ingest_sidecar_healthcheck_timeout_seconds", 0.5)),
    )


def ingest_sidecar_process_config(settings) -> IngestSidecarProcessConfig:
    process_config_factory = getattr(settings, "ingest_sidecar_process_config", None)
    if callable(process_config_factory):
        return process_config_factory()
    sidecar_port = int(getattr(settings, "ingest_sidecar_port", getattr(settings, "ingest_port", 8081)))
    return IngestSidecarProcessConfig(
        binary_path=Path(getattr(settings, "ingest_sidecar_binary_path", "dist/minimappr-ingest-sidecar")),
        spool_dir=Path(getattr(settings, "ingest_spool_dir")),
        consumer_name=str(getattr(settings, "ingest_consumer_name", "python-ingest")),
        ingest_port=int(getattr(settings, "ingest_port", sidecar_port)),
        sidecar_port=sidecar_port,
        storage_mode=str(getattr(settings, "ingest_storage_mode", "spool")),
        total_journal_budget_bytes=int(getattr(settings, "ingest_sidecar_total_journal_budget_bytes", 268_435_456)),
        admission_reserve_bytes=int(getattr(settings, "ingest_sidecar_admission_reserve_bytes", 16_777_216)),
        allow_non_tmpfs_journal=bool(getattr(settings, "ingest_sidecar_allow_non_tmpfs_journal", False)),
        memory_only_live_path=bool(getattr(settings, "ingest_sidecar_memory_only_live_path", True)),
    )


def sidecar_classification_window_seconds(settings) -> float:
    classification_window_seconds = float(getattr(settings, "classification_window_seconds", 0.0))
    if classification_window_seconds > 0.0:
        return classification_window_seconds
    return float(getattr(settings, "localization_window_seconds", 0.08))


def sidecar_classifier_render_min_interval_seconds(
    settings,
    *,
    classification_window_seconds: float,
) -> float:
    classifier_render_min_interval_seconds = float(
        getattr(settings, "classifier_render_min_interval_seconds", 0.0)
    )
    if classifier_render_min_interval_seconds > 0.0:
        return classifier_render_min_interval_seconds

    overlap_seconds = float(getattr(settings, "birdnet_chunk_overlap_seconds", 0.0))
    if classification_window_seconds > 0.0 and overlap_seconds >= 0.0:
        return max(0.0, classification_window_seconds - overlap_seconds)
    return 0.0


def _sidecar_classifier_command_json(
    settings,
    *,
    default_classifier_command_json_builder: Callable[[object], str | None],
) -> str | None:
    classifier_command_json = os.environ.get("MINIMAPPR_SIDECAR_CLASSIFIER_COMMAND_JSON")
    if classifier_command_json is not None:
        return classifier_command_json
    return default_classifier_command_json_builder(settings)


def build_ingest_sidecar_environment(
    settings,
    *,
    default_classifier_command_json_builder: Callable[[object], str | None],
) -> dict[str, str]:
    process_config = ingest_sidecar_process_config(settings)
    classification_window_seconds = sidecar_classification_window_seconds(settings)
    classifier_render_min_interval_seconds = sidecar_classifier_render_min_interval_seconds(
        settings,
        classification_window_seconds=classification_window_seconds,
    )
    env = {
        **os.environ,
        # Keep spool dir in sync with the Python consumer regardless of what
        # env vars the operator may have set for the Rust binary directly.
        "MINIMAPPR_INGEST_SPOOL_DIR": str(process_config.spool_dir),
        "MINIMAPPR_INGEST_CONSUMER_NAME": process_config.consumer_name,
        "MINIMAPPR_INGEST_PORT": str(process_config.ingest_port),
        "MINIMAPPR_SIDECAR_PORT": str(process_config.sidecar_port),
        "MINIMAPPR_SIDECAR_STORAGE_MODE": process_config.storage_mode,
        "MINIMAPPR_SIDECAR_TOTAL_JOURNAL_BUDGET_BYTES": str(
            process_config.total_journal_budget_bytes
        ),
        "MINIMAPPR_SIDECAR_ADMISSION_RESERVE_BYTES": str(
            process_config.admission_reserve_bytes
        ),
        "MINIMAPPR_SIDECAR_ALLOW_NON_TMPFS_JOURNAL": str(
            process_config.allow_non_tmpfs_journal
        ).lower(),
        "MINIMAPPR_SIDECAR_MEMORY_ONLY_LIVE_PATH": str(
            process_config.memory_only_live_path
        ).lower(),
        "MINIMAPPR_RUNTIME_PROFILE": str(getattr(settings, "runtime_profile", "default")),
        "MINIMAPPR_LOCALIZATION_WINDOW_SECONDS": str(
            getattr(settings, "localization_window_seconds", 0.08)
        ),
        "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS": str(classification_window_seconds),
        "MINIMAPPR_CLASSIFIER_RENDER_MIN_INTERVAL_SECONDS": str(classifier_render_min_interval_seconds),
        "MINIMAPPR_MAX_SENSOR_BUFFER_SECONDS": str(
            getattr(settings, "max_sensor_buffer_seconds", 32.0)
        ),
        "MINIMAPPR_DSP_LOCALIZATION_RMS_GATE": str(
            getattr(settings, "trigger_rms", 0.015)
        ),
        "MINIMAPPR_TRIGGER_COOLDOWN_SECONDS": str(
            getattr(settings, "trigger_cooldown_seconds", 0.8)
        ),
        "MINIMAPPR_LOCALIZATION_BAND_MIN_HZ": str(
            getattr(settings, "localization_band_min_hz", 300.0)
        ),
        "MINIMAPPR_LOCALIZATION_BAND_MAX_HZ": str(
            getattr(settings, "localization_band_max_hz", 3500.0)
        ),
        "MINIMAPPR_DEFAULT_TEMPERATURE_C": str(getattr(settings, "default_temperature_c", 20.0)),
        "MINIMAPPR_DEFAULT_HUMIDITY": str(getattr(settings, "default_humidity", 0.5)),
        "MINIMAPPR_SITE_ORIGIN_LAT": str(getattr(settings, "site_origin_lat", 0.0)),
        "MINIMAPPR_SITE_ORIGIN_LON": str(getattr(settings, "site_origin_lon", 0.0)),
        "MINIMAPPR_SITE_ORIGIN_ALT_M": str(getattr(settings, "site_origin_alt_m", 0.0)),
        "MINIMAPPR_BIRDNET_TRIGGER_MIN_CONFIDENCE": str(
            getattr(settings, "birdnet_trigger_min_confidence", 0.4)
        ),
        "MINIMAPPR_BIRDNET_GEO_MIN_CONFIDENCE": str(
            getattr(settings, "birdnet_geo_min_confidence", 0.03)
        ),
    }
    classifier_command_json = _sidecar_classifier_command_json(
        settings,
        default_classifier_command_json_builder=default_classifier_command_json_builder,
    )
    if classifier_command_json is not None:
        env["MINIMAPPR_SIDECAR_CLASSIFIER_COMMAND_JSON"] = classifier_command_json
    return env


def ingest_stream_consumer_runtime(
    state,
    *,
    ingest_runtime_base_url_builder: Callable[[object], str],
    stream_consumer_config_class,
):
    settings = getattr(state, "settings", None)
    sidecar_state = getattr(state, "sidecar_state", None)
    ingest_transport = getattr(state, "ingest_transport", None)
    if settings is None or ingest_transport is None:
        return None
    if getattr(settings, "ingest_backend", "python") != "rust":
        return None
    if sidecar_state is None or sidecar_state.status != "running":
        return None
    return {
        "config": stream_consumer_config_class(
            sidecar_base_url=ingest_runtime_base_url_builder(settings),
        ),
        "ingest_transport": ingest_transport,
        "audio_buffer": getattr(state, "audio_buffer", None),
    }


def build_ingest_stream_consumer(
    state,
    *,
    ingest_stream_consumer_class,
    ingest_runtime_base_url_builder: Callable[[object], str],
    stream_consumer_config_class,
):
    runtime = ingest_stream_consumer_runtime(
        state,
        ingest_runtime_base_url_builder=ingest_runtime_base_url_builder,
        stream_consumer_config_class=stream_consumer_config_class,
    )
    if runtime is None:
        return None
    return ingest_stream_consumer_class(
        config=runtime["config"],
        ingest_transport=runtime["ingest_transport"],
        audio_buffer=runtime["audio_buffer"],
    )


async def _stop_ingest_stream_consumer(
    state,
    consumer,
    *,
    clear_state_attrs: Callable[..., None],
    logger,
    reason: str,
) -> bool:
    try:
        await consumer.stop()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to stop ingest stream consumer while %s: %s", reason, exc)
        return False
    if getattr(state, "ingest_stream_consumer", None) is consumer:
        clear_state_attrs(state, "ingest_stream_consumer")
    return True


async def ensure_ingest_stream_consumer_running(
    state,
    *,
    clear_state_attrs: Callable[..., None],
    ingest_stream_consumer_class,
    ingest_runtime_base_url_builder: Callable[[object], str],
    logger,
    stream_consumer_config_class,
) -> bool:
    runtime = ingest_stream_consumer_runtime(
        state,
        ingest_runtime_base_url_builder=ingest_runtime_base_url_builder,
        stream_consumer_config_class=stream_consumer_config_class,
    )
    consumer = getattr(state, "ingest_stream_consumer", None)
    if runtime is None:
        if consumer is not None:
            await _stop_ingest_stream_consumer(
                state,
                consumer,
                clear_state_attrs=clear_state_attrs,
                logger=logger,
                reason="disabling sidecar SSE consumption",
            )
        return False

    desired_config = runtime["config"]
    desired_transport = runtime["ingest_transport"]
    desired_audio_buffer = runtime["audio_buffer"]
    consumer_already_present = consumer is not None
    if consumer is not None:
        consumer_config = getattr(consumer, "_config", None)
        consumer_transport = getattr(consumer, "_ingest_transport", None)
        consumer_audio_buffer = getattr(consumer, "_audio_buffer", None)
        if (
            getattr(consumer_config, "sidecar_base_url", None) != desired_config.sidecar_base_url
            or consumer_transport is not desired_transport
            or consumer_audio_buffer is not desired_audio_buffer
        ):
            if not await _stop_ingest_stream_consumer(
                state,
                consumer,
                clear_state_attrs=clear_state_attrs,
                logger=logger,
                reason="replacing sidecar SSE configuration",
            ):
                return consumer.is_running
            consumer = None
            consumer_already_present = False

    if consumer is None:
        consumer = ingest_stream_consumer_class(
            config=desired_config,
            ingest_transport=desired_transport,
            audio_buffer=desired_audio_buffer,
        )
        state.ingest_stream_consumer = consumer
    if consumer.is_running:
        return True
    logger.warning(
        "Ingest stream consumer is %s; starting sidecar SSE consumer",
        "stopped" if consumer_already_present else "not initialized",
    )
    consumer.start()
    return True


async def launch_managed_ingest_sidecar(
    settings,
    *,
    logger,
    sidecar_state_class,
    start_ingest_sidecar: Callable[[object], Awaitable["asyncio.subprocess.Process | None"]],
    supervise_ingest_sidecar: Callable[[object, "asyncio.subprocess.Process", IngestSidecarRuntimeState], Awaitable[None]],
    startup_failure_log_message: str,
) -> tuple[IngestSidecarRuntimeState, asyncio.Task | None]:
    sidecar_state = sidecar_state_class()
    sidecar_supervision_task: asyncio.Task | None = None
    if should_autostart_ingest_sidecar(settings):
        try:
            sidecar_process = await start_ingest_sidecar(settings)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: %s", startup_failure_log_message, exc)
            sidecar_state.status = "startup_failed"
        else:
            if sidecar_process is not None:
                sidecar_state._current_process = sidecar_process
                sidecar_state.status = "running"
                sidecar_state.pid = sidecar_process.pid
                sidecar_supervision_task = asyncio.create_task(
                    supervise_ingest_sidecar(settings, sidecar_process, sidecar_state)
                )
            else:
                sidecar_state.status = "binary_not_found"
    else:
        sidecar_state.status = "disabled"
    return sidecar_state, sidecar_supervision_task


async def shutdown_managed_ingest_sidecar(
    sidecar_state: IngestSidecarRuntimeState,
    sidecar_supervision_task: asyncio.Task | None,
    *,
    force_kill_on_timeout: bool,
    logger,
    shutdown_timeout_seconds: float,
) -> None:
    if sidecar_supervision_task is not None:
        sidecar_supervision_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(sidecar_supervision_task, timeout=shutdown_timeout_seconds)

    current_sidecar = sidecar_state._current_process
    if current_sidecar is None or current_sidecar.returncode is not None:
        return

    logger.info("Stopping ingest sidecar (pid %d)", current_sidecar.pid)
    current_sidecar.terminate()
    if not force_kill_on_timeout:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(current_sidecar.wait(), timeout=shutdown_timeout_seconds)
        return

    try:
        await asyncio.wait_for(current_sidecar.wait(), timeout=shutdown_timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning("Ingest sidecar did not exit cleanly; sending SIGKILL")
        current_sidecar.kill()
        await current_sidecar.wait()


def fetch_ingest_sidecar_health(port: int, timeout_seconds: float = 0.5) -> dict[str, object] | None:
    request = urllib.request.Request(_sidecar_healthcheck_url(port), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if getattr(response, "status", None) != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, ValueError, urllib.error.HTTPError, urllib.error.URLError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def probe_ingest_sidecar_ready(port: int, timeout_seconds: float = 0.5) -> bool:
    payload = fetch_ingest_sidecar_health(port, timeout_seconds)
    return isinstance(payload, dict) and payload.get("status") == "ok"


async def wait_for_ingest_sidecar_ready(
    process: "asyncio.subprocess.Process",
    *,
    port: int,
    timeout_seconds: float = 5.0,
    poll_interval_seconds: float = 0.1,
    probe_ready: Callable[[int], bool] | None = None,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    if probe_ready is None:
        probe_ready = probe_ingest_sidecar_ready
    while True:
        if process.returncode is not None:
            if await asyncio.to_thread(probe_ready, port):
                raise RuntimeError(
                    "Ingest sidecar child process exited, but readiness endpoint is already healthy on "
                    f"port {port}. Another ingest worker is likely already running."
                )
            raise RuntimeError(
                f"Ingest sidecar exited before readiness check completed (code {process.returncode})"
            )
        if await asyncio.to_thread(probe_ready, port):
            return
        if loop.time() >= deadline:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=1.0)
            raise RuntimeError(
                f"Ingest sidecar did not become ready on port {port} within {timeout_seconds:.1f}s"
            )
        await asyncio.sleep(poll_interval_seconds)


async def start_ingest_sidecar(
    settings,
    *,
    default_classifier_command_json_builder: Callable[[object], str | None],
    probe_ready: Callable[..., bool],
    wait_for_ready: Callable[..., Awaitable[None]],
    create_subprocess_exec: Callable[..., Awaitable[asyncio.subprocess.Process]],
    logger,
) -> "asyncio.subprocess.Process | None":
    """Launch the Rust ingest sidecar as a managed child process."""
    process_config = ingest_sidecar_process_config(settings)
    sidecar_startup = ingest_sidecar_startup_config(settings)
    binary = process_config.binary_path
    if not binary.exists():
        logger.warning(
            "Ingest sidecar binary not found at %s; sidecar will not start. "
            "Build it with: scripts/build_rust.sh --sidecar",
            binary,
        )
        return None
    probe_ready_for_configured_timeout = functools.partial(
        probe_ready,
        timeout_seconds=sidecar_startup.healthcheck_timeout_seconds,
    )
    if await asyncio.to_thread(probe_ready_for_configured_timeout, process_config.sidecar_port):
        raise RuntimeError(
            "Ingest sidecar readiness endpoint is already healthy on "
            f"port {process_config.sidecar_port}. Another ingest worker is likely already running."
        )
    env = build_ingest_sidecar_environment(
        settings,
        default_classifier_command_json_builder=default_classifier_command_json_builder,
    )
    logger.info(
        "Starting ingest sidecar: %s (port %d, storage %s, spool %s, journal budget %d, reserve %d, allow non-tmpfs %s)",
        binary,
        process_config.sidecar_port,
        process_config.storage_mode,
        process_config.spool_dir,
        process_config.total_journal_budget_bytes,
        process_config.admission_reserve_bytes,
        process_config.allow_non_tmpfs_journal,
    )
    process = await create_subprocess_exec(str(binary), env=env)
    await wait_for_ready(
        process,
        port=process_config.sidecar_port,
        timeout_seconds=sidecar_startup.ready_timeout_seconds,
        poll_interval_seconds=sidecar_startup.ready_poll_interval_seconds,
        probe_ready=probe_ready_for_configured_timeout,
    )
    logger.info("Ingest sidecar ready on port %d", process_config.sidecar_port)
    return process


async def supervise_ingest_sidecar(
    settings,
    initial_process: "asyncio.subprocess.Process",
    state: IngestSidecarRuntimeState,
    *,
    start_ingest_sidecar: Callable[[object], Awaitable[asyncio.subprocess.Process | None]],
    logger,
) -> None:
    """Watch the ingest sidecar and restart it on unexpected exit."""
    process = initial_process
    state._current_process = process
    state.status = "running"
    state.pid = process.pid
    while True:
        returncode = await process.wait()
        state.last_exit_code = returncode
        if returncode in (0, -15):
            state.status = "stopped"
            state._current_process = None
            return
        state.restart_count += 1
        state.status = "restarting"
        logger.warning(
            "Ingest sidecar exited unexpectedly (code %d); restarting (attempt %d)",
            returncode,
            state.restart_count,
        )
        await asyncio.sleep(2.0)
        try:
            new_process = await start_ingest_sidecar(settings)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to restart ingest sidecar: %s", exc)
            state.status = "crashed"
            state._current_process = None
            return
        if new_process is None:
            state.status = "binary_not_found"
            state._current_process = None
            return
        process = new_process
        state._current_process = process
        state.status = "running"
        state.pid = process.pid