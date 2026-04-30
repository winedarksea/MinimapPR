from __future__ import annotations

from types import SimpleNamespace

import pytest

from minimappr import main


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

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


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


@pytest.mark.asyncio
async def test_wait_for_ingest_sidecar_ready_returns_when_probe_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(main, "_probe_ingest_sidecar_ready", lambda port: True)

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
    monkeypatch.setattr(main, "_probe_ingest_sidecar_ready", lambda port: False)

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
        return process

    async def fake_wait_for_ingest_sidecar_ready(process_arg, *, port: int, **kwargs) -> None:
        observed["process"] = process_arg
        observed["port"] = port

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
    )

    started_process = await main._start_ingest_sidecar(settings)

    assert started_process is process
    assert observed["process"] is process
    assert observed["port"] == 18081
    assert observed["argv"] == (str(binary_path),)