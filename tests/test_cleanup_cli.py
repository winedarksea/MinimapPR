from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from minimappr.__main__ import main
from minimappr.models import DetectionEvent, NodeSpec, NodeType
from minimappr.storage.db import Storage


def test_plain_minimappr_routes_to_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(app: str, *, host: str, port: int, **kwargs: object) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr("minimappr.__main__.uvicorn.run", fake_run)
    monkeypatch.setenv("MINIMAPPR_HOST", "127.0.0.1")
    monkeypatch.setenv("MINIMAPPR_PORT", "9090")

    main([])

    assert captured == {
        "app": "minimappr.main:app",
        "host": "127.0.0.1",
        "port": 9090,
    }


def test_cleanup_full_requires_yes_without_dry_run() -> None:
    with pytest.raises(SystemExit):
        main(["cleanup", "full"])


def test_cleanup_full_yes_removes_db_and_managed_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "minimappr.db"
    db_path.write_bytes(b"sqlite")
    snippet_dir = tmp_path / "snippets"
    snippet_dir.mkdir()
    (snippet_dir / "a.wav").write_bytes(b"wav")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "a.bin").write_bytes(b"artifact")

    monkeypatch.setattr("minimappr.__main__.uvicorn.run", lambda *args, **kwargs: None)
    main(
        [
            "cleanup",
            "full",
            "--yes",
            "--db-path",
            str(db_path),
            "--snippet-dir",
            str(snippet_dir),
            "--artifact-dir",
            str(artifact_dir),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "full"
    assert output["db_removed"] is True
    assert db_path.exists() is False
    assert snippet_dir.exists() is False
    assert artifact_dir.exists() is False


def test_cleanup_full_removes_the_wal_and_shm_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A WAL-mode database is three files; leaving two behind is not a full wipe."""
    db_path = tmp_path / "minimappr.db"
    db_path.write_bytes(b"sqlite")
    wal_path = tmp_path / "minimappr.db-wal"
    wal_path.write_bytes(b"wal")
    shm_path = tmp_path / "minimappr.db-shm"
    shm_path.write_bytes(b"shm")

    monkeypatch.setattr("minimappr.__main__.uvicorn.run", lambda *args, **kwargs: None)
    main(["cleanup", "full", "--yes", "--db-path", str(db_path)])

    output = json.loads(capsys.readouterr().out)
    assert output["db_removed"] is True
    assert db_path.exists() is False
    assert wal_path.exists() is False
    assert shm_path.exists() is False
    removed_by_path = {row["path"]: row["removed"] for row in output["db_files"]}
    assert removed_by_path[str(wal_path)] is True
    assert removed_by_path[str(shm_path)] is True


def test_cleanup_full_warns_when_no_database_exists_at_the_resolved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default db_path is relative, so a wrong CWD silently wipes nothing."""
    monkeypatch.setattr("minimappr.__main__.uvicorn.run", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    main(["cleanup", "full", "--yes"])

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["db_removed"] is False
    assert output["db_path"] == str((tmp_path / "data" / "minimappr.db").resolve())
    assert "no database found at" in captured.err
    assert output["db_path"] in captured.err


def test_cleanup_full_yes_removes_known_runtime_cache_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_file = tmp_path / "data" / "yamnet_class_map.csv"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("display_name,index\\nBird,1\\n", encoding="utf-8")

    monkeypatch.setattr("minimappr.__main__.uvicorn.run", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    main(["cleanup", "full", "--yes"])

    output = json.loads(capsys.readouterr().out)
    cache_paths = output.get("cache_paths", [])
    yamnet_cache_rows = [row for row in cache_paths if row.get("path") == "data/yamnet_class_map.csv"]

    assert cache_file.exists() is False
    assert yamnet_cache_rows
    assert yamnet_cache_rows[0]["removed"] is True


def test_cleanup_partial_dry_run_reports_candidates_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def _prepare() -> None:
        db_path = tmp_path / "partial.db"
        storage = Storage(db_path)
        await storage.initialize()

        now_ns = 7_000_000_000_000_000_000
        old_ns = now_ns - 120_000_000_000

        await storage.upsert_node(
            NodeSpec(
                id="cli-node",
                node_type=NodeType.POINT,
                position_m=(0.0, 0.0, 0.0),
                sensor_offsets_m=[(0.0, 0.0, 0.0)],
                capabilities=["audio"],
            ),
            last_seen_ns=old_ns,
        )

        snippet_file = tmp_path / "cli.wav"
        snippet_file.write_bytes(b"wav")
        await storage.insert_detection(
            detection=DetectionEvent(
                id="det-cli",
                timestamp_ns=old_ns,
                position_m=(0.0, 0.0, 0.0),
                confidence=0.55,
                gdop=1.3,
                label="common_label",
                label_category="unknown",
                label_confidence=0.55,
                reference_sensor="cli-node:ch0",
            ),
            snippet_path=str(snippet_file),
            snippet_expires_ns=None,
            retention_tier="short",
        )
        await storage.close()

    async def _verify() -> str:
        reopened = Storage(tmp_path / "partial.db")
        await reopened.initialize()
        try:
            detections = {row["id"]: row for row in await reopened.list_detections(limit=10)}
            return str(detections["det-cli"]["snippet_path"])
        finally:
            await reopened.close()

    db_path = tmp_path / "partial.db"
    now_ns = 7_000_000_000_000_000_000
    snippet_file = tmp_path / "cli.wav"
    policy_path = tmp_path / "retention_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "version": 1,
                "defaults": {
                    "snippet_max_age_seconds": 60,
                    "artifact_max_age_seconds": 60,
                },
            }
        ),
        encoding="utf-8",
    )
    asyncio.run(_prepare())

    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(db_path))
    monkeypatch.setenv("MINIMAPPR_RETENTION_POLICY_PATH", str(policy_path))
    monkeypatch.setattr("minimappr.__main__.uvicorn.run", lambda *args, **kwargs: None)

    main(["cleanup", "partial", "--dry-run", "--now-ns", str(now_ns), "--policy", str(policy_path)])

    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "partial"
    assert output["dry_run"] is True
    assert output["summary"]["snippet_files_deleted"] == 1
    assert snippet_file.exists() is True

    assert asyncio.run(_verify()) == str(snippet_file)
