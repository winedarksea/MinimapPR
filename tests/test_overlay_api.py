from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from minimappr.main import app
from minimappr.storage.db import Storage


def _configure_env(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    db_path = tmp_path / "test.db"
    overlay_dir = tmp_path / "overlays"
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(db_path))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_MAP_OVERLAY_DIR", str(overlay_dir))
    monkeypatch.setenv("MINIMAPPR_EFFECTOR_SNAPSHOT_DIR", str(tmp_path / "effector_snapshots"))
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    return db_path, overlay_dir


def _upload_png(client: TestClient, name: str = "Barn floorplan"):
    return client.post(
        "/api/v1/overlays",
        data={
            "name": name,
            "kind": "image",
            "opacity": "0.65",
            "bounds": "[[44.0,-93.0],[44.0,-92.9],[43.9,-92.9],[43.9,-93.0]]",
        },
        files={"file": ("floorplan.png", b"\x89PNG\r\n\x1a\nfake", "image/png")},
    )


def test_overlay_upload_list_patch_content_and_delete(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)

    with TestClient(app) as client:
        upload = _upload_png(client)
        assert upload.status_code == 201
        body = upload.json()
        overlay_id = body["id"]
        assert body["name"] == "Barn floorplan"
        assert body["kind"] == "image"
        assert body["opacity"] == 0.65
        assert body["content_url"] == f"/api/v1/overlays/{overlay_id}/content"

        listed = client.get("/api/v1/overlays")
        assert listed.status_code == 200
        assert listed.json()[0]["id"] == overlay_id

        content = client.get(body["content_url"])
        assert content.status_code == 200
        assert content.headers["content-type"].startswith("image/png")
        assert content.content.startswith(b"\x89PNG")

        patch = client.patch(
            f"/api/v1/overlays/{overlay_id}",
            json={
                "name": "Barn west",
                "opacity": 0.35,
                "enabled": False,
                "storey": "L1",
                "metadata": {"source": "operator"},
            },
        )
        assert patch.status_code == 200
        patched = patch.json()
        assert patched["name"] == "Barn west"
        assert patched["opacity"] == 0.35
        assert patched["enabled"] is False
        assert patched["metadata"]["source"] == "operator"

        delete = client.delete(f"/api/v1/overlays/{overlay_id}")
        assert delete.status_code == 200
        assert client.get("/api/v1/overlays").json() == []
        assert client.get(body["content_url"]).status_code == 404


def test_overlay_upload_rejects_bad_extension_mime_and_size(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)

    with TestClient(app) as client:
        bad_ext = client.post(
            "/api/v1/overlays",
            data={"name": "bad", "kind": "image"},
            files={"file": ("bad.txt", b"not-image", "image/png")},
        )
        assert bad_ext.status_code == 415

        bad_mime = client.post(
            "/api/v1/overlays",
            data={"name": "bad", "kind": "image"},
            files={"file": ("bad.png", b"not-image", "text/plain")},
        )
        assert bad_mime.status_code == 415

        too_large = client.post(
            "/api/v1/overlays",
            data={"name": "large", "kind": "geojson"},
            files={"file": ("large.geojson", b"{}" + (b" " * (20 * 1024 * 1024)), "application/geo+json")},
        )
        assert too_large.status_code == 413


def test_overlay_content_rejects_paths_outside_overlay_dir(monkeypatch, tmp_path: Path) -> None:
    db_path, _overlay_dir = _configure_env(monkeypatch, tmp_path)
    outside_file = tmp_path / "outside.png"
    outside_file.write_bytes(b"\x89PNG\r\n\x1a\noutside")

    async def _seed() -> None:
        storage = Storage(db_path)
        await storage.initialize()
        await storage.upsert_map_overlay(
            overlay_id="ovl-escape",
            name="escape",
            kind="image",
            file_path=str(outside_file),
            mime="image/png",
            bounds=[],
            opacity=0.75,
            storey=None,
            enabled=True,
            created_ns=1,
            metadata={},
        )
        await storage.close()

    asyncio.run(_seed())

    with TestClient(app) as client:
        response = client.get("/api/v1/overlays/ovl-escape/content")
        assert response.status_code == 403
