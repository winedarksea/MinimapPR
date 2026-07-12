from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from minimappr import main
from minimappr.config import Settings
from minimappr.ingest_sidecar_runtime import (
    IngestSidecarRuntimeState,
    ensure_ingest_stream_consumer_running,
    shutdown_managed_ingest_sidecar,
)


class _FakeResponse:
    def __init__(self, *, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


class _FakeProcess:
    def __init__(self, *, pid: int = 4321, returncode: int | None = None) -> None:
        self.pid = pid
        self.returncode = returncode
        self.terminate_calls = 0
        self.wait_calls = 0

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.terminate_calls += 1

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class _FakeIngestStreamConsumer:
    def __init__(
        self,
        *,
        config,
        ingest_transport,
        audio_buffer=None,
        coordinate_frame=None,
    ) -> None:
        self._config = config
        self._ingest_transport = ingest_transport
        self._audio_buffer = audio_buffer
        self._coordinate_frame = coordinate_frame
        self._running = False
        self.start_calls = 0
        self.stop_calls = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self.start_calls += 1
        self._running = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self._running = False


def _clear_state_attrs(state, *names: str) -> None:
    for name in names:
        if hasattr(state, name):
            delattr(state, name)


def test_probe_ingest_sidecar_ready_accepts_ok_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(status=200, body=b'{"status":"ok"}'),
    )

    assert main._probe_ingest_sidecar_ready(18081) is True


def test_probe_ingest_sidecar_ready_rejects_non_ok_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(status=200, body=b'{"status":"degraded"}'),
    )

    assert main._probe_ingest_sidecar_ready(18081) is False


def test_fetch_ingest_sidecar_health_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(
            status=200,
            body=b'{"status":"ok","backend":{"storage_mode":"journal"}}',
        ),
    )

    assert main._fetch_ingest_sidecar_health(18081) == {
        "status": "ok",
        "backend": {"storage_mode": "journal"},
    }


def test_should_autostart_ingest_sidecar_when_sidecar_enabled_and_direct_ingest_disabled() -> None:
    settings = SimpleNamespace(ingest_sidecar_enabled=True, direct_ingest_enabled=False)
    assert main._should_autostart_ingest_sidecar(settings) is True


def test_should_not_autostart_ingest_sidecar_when_direct_ingest_enabled() -> None:
    settings = SimpleNamespace(ingest_sidecar_enabled=True, direct_ingest_enabled=True)
    assert main._should_autostart_ingest_sidecar(settings) is False


def test_build_ingest_sidecar_environment_falls_back_to_localization_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("MINIMAPPR_SIDECAR_CLASSIFIER_COMMAND_JSON", raising=False)
    settings = SimpleNamespace(
        ingest_spool_dir=tmp_path / "spool",
        ingest_consumer_name="python-ingest",
        ingest_sidecar_port=18081,
        ingest_storage_mode="journal",
        ingest_sidecar_total_journal_budget_bytes=1024,
        ingest_sidecar_admission_reserve_bytes=128,
        ingest_sidecar_allow_non_tmpfs_journal=True,
        localization_window_seconds=0.08,
        classification_window_seconds=0.0,
        birdnet_chunk_overlap_seconds=0.02,
    )

    env = main._build_ingest_sidecar_environment(settings)

    assert env["MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS"] == "0.08"
    assert env["MINIMAPPR_CLASSIFIER_RENDER_MIN_INTERVAL_SECONDS"] == "0.06"


@pytest.mark.asyncio
async def test_wait_for_ingest_sidecar_ready_returns_when_probe_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(main, "_probe_ingest_sidecar_ready", lambda *args, **kwargs: True)

    await main._wait_for_ingest_sidecar_ready(
        process,
        port=18081,
        timeout_seconds=0.0,
        poll_interval_seconds=0.0,
    )

    assert process.terminate_calls == 0
    assert process.wait_calls == 0


@pytest.mark.asyncio
async def test_wait_for_ingest_sidecar_ready_terminates_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(main, "_probe_ingest_sidecar_ready", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="did not become ready"):
        await main._wait_for_ingest_sidecar_ready(
            process,
            port=18081,
            timeout_seconds=0.0,
            poll_interval_seconds=0.0,
        )

    assert process.terminate_calls == 1
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_wait_for_ingest_sidecar_ready_reports_existing_worker_when_port_already_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(returncode=-9)
    monkeypatch.setattr(main, "_probe_ingest_sidecar_ready", lambda *args, **kwargs: True)

    with pytest.raises(RuntimeError, match="already running"):
        await main._wait_for_ingest_sidecar_ready(
            process,
            port=18081,
            timeout_seconds=0.0,
            poll_interval_seconds=0.0,
        )


@pytest.mark.asyncio
async def test_start_ingest_sidecar_waits_for_healthcheck(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    binary_path = tmp_path / "minimappr-ingest-sidecar"
    binary_path.write_text("stub", encoding="utf-8")
    process = _FakeProcess()
    observed: dict[str, object] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        observed["argv"] = args
        observed["env"] = kwargs["env"]
        observed["start_new_session"] = kwargs.get("start_new_session")
        return process

    async def fake_wait_for_ingest_sidecar_ready(process_arg, *, port: int, **kwargs) -> None:
        observed["process"] = process_arg
        observed["port"] = port
        observed["timeout_seconds"] = kwargs.get("timeout_seconds")
        observed["poll_interval_seconds"] = kwargs.get("poll_interval_seconds")
        observed["probe_ready"] = kwargs.get("probe_ready")

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(main, "_wait_for_ingest_sidecar_ready", fake_wait_for_ingest_sidecar_ready)

    settings = SimpleNamespace(
        ingest_sidecar_binary_path=binary_path,
        ingest_spool_dir=tmp_path / "spool",
        ingest_consumer_name="python-ingest",
        ingest_sidecar_port=18081,
        ingest_storage_mode="journal",
        ingest_sidecar_total_journal_budget_bytes=1024,
        ingest_sidecar_admission_reserve_bytes=128,
        ingest_sidecar_allow_non_tmpfs_journal=True,
    )

    started_process = await main._start_ingest_sidecar(settings)

    assert started_process is process
    assert observed["process"] is process
    assert observed["port"] == 18081
    assert observed["timeout_seconds"] == pytest.approx(5.0)
    assert observed["poll_interval_seconds"] == pytest.approx(0.1)
    assert callable(observed["probe_ready"])
    assert observed["argv"] == (str(binary_path),)
    if sys.platform != "win32":
        assert observed["start_new_session"] is True
    assert observed["env"]["MINIMAPPR_SIDECAR_ALLOW_NON_TMPFS_JOURNAL"] == "true"
    assert observed["env"]["MINIMAPPR_SIDECAR_MEMORY_ONLY_LIVE_PATH"] == "true"
    assert observed["env"]["MINIMAPPR_INGEST_PORT"] == "18081"
    assert observed["env"]["MINIMAPPR_SIDECAR_PORT"] == "18081"


@pytest.mark.asyncio
async def test_managed_sidecar_shutdown_signals_leader_for_graceful_helper_cleanup(
) -> None:
    process = _FakeProcess(pid=5432)
    state = IngestSidecarRuntimeState()
    state._current_process = process
    await shutdown_managed_ingest_sidecar(
        state,
        None,
        force_kill_on_timeout=False,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        shutdown_timeout_seconds=1.0,
    )

    assert process.terminate_calls == 1


@pytest.mark.asyncio
async def test_start_ingest_sidecar_reports_existing_worker_if_port_already_healthy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    binary_path = tmp_path / "minimappr-ingest-sidecar"
    binary_path.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(main, "_probe_ingest_sidecar_ready", lambda *args, **kwargs: True)

    settings = SimpleNamespace(
        ingest_sidecar_binary_path=binary_path,
        ingest_spool_dir=tmp_path / "spool",
        ingest_consumer_name="python-ingest",
        ingest_sidecar_port=18081,
        ingest_storage_mode="journal",
        ingest_sidecar_total_journal_budget_bytes=1024,
        ingest_sidecar_admission_reserve_bytes=128,
        ingest_sidecar_allow_non_tmpfs_journal=True,
    )

    with pytest.raises(RuntimeError, match="already running"):
        await main._start_ingest_sidecar(settings)


@pytest.mark.asyncio
async def test_ensure_ingest_stream_consumer_running_replaces_running_mismatched_consumer() -> None:
    state = SimpleNamespace(
        settings=SimpleNamespace(ingest_backend="rust"),
        sidecar_state=SimpleNamespace(status="running"),
        ingest_transport=object(),
        audio_buffer=object(),
    )
    old_consumer = _FakeIngestStreamConsumer(
        config=SimpleNamespace(sidecar_base_url="http://127.0.0.1:8081"),
        ingest_transport=state.ingest_transport,
        audio_buffer=state.audio_buffer,
    )
    old_consumer.start()
    state.ingest_stream_consumer = old_consumer

    result = await ensure_ingest_stream_consumer_running(
        state,
        clear_state_attrs=_clear_state_attrs,
        ingest_stream_consumer_class=_FakeIngestStreamConsumer,
        ingest_runtime_base_url_builder=lambda settings: "http://127.0.0.1:9999",
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
        stream_consumer_config_class=lambda sidecar_base_url: SimpleNamespace(
            sidecar_base_url=sidecar_base_url
        ),
    )

    assert result is True
    assert old_consumer.stop_calls == 1
    assert state.ingest_stream_consumer is not old_consumer
    assert state.ingest_stream_consumer.is_running is True
    assert state.ingest_stream_consumer._config.sidecar_base_url == "http://127.0.0.1:9999"


@pytest.mark.asyncio
async def test_ensure_ingest_stream_consumer_running_stops_consumer_when_runtime_disabled() -> None:
    state = SimpleNamespace(
        settings=SimpleNamespace(ingest_backend="python"),
        sidecar_state=SimpleNamespace(status="running"),
        ingest_transport=object(),
        audio_buffer=object(),
    )
    consumer = _FakeIngestStreamConsumer(
        config=SimpleNamespace(sidecar_base_url="http://127.0.0.1:8081"),
        ingest_transport=state.ingest_transport,
        audio_buffer=state.audio_buffer,
    )
    consumer.start()
    state.ingest_stream_consumer = consumer

    result = await ensure_ingest_stream_consumer_running(
        state,
        clear_state_attrs=_clear_state_attrs,
        ingest_stream_consumer_class=_FakeIngestStreamConsumer,
        ingest_runtime_base_url_builder=lambda settings: "http://127.0.0.1:8081",
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
        stream_consumer_config_class=lambda sidecar_base_url: SimpleNamespace(
            sidecar_base_url=sidecar_base_url
        ),
    )

    assert result is False
    assert consumer.stop_calls == 1
    assert not hasattr(state, "ingest_stream_consumer")
