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

    def fake_run(app: str, *, host: str, port: int) -> None:
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
