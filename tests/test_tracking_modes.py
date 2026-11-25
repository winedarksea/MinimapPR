"""Per-category tracking modes (localization overhaul P4): config and TrackManager."""

from __future__ import annotations

import pytest

from minimappr.config import Settings
from minimappr.core.tracking import TrackManager
from minimappr.core.tracking_mode_config import (
    TrackingMode,
    TrackingModeConfig,
    tracking_mode_by_category_with,
)

T0 = 1_700_000_000_000_000_000
SECOND = 1_000_000_000


def test_shipped_defaults_hold_wildlife_and_move_everything_else() -> None:
    modes = TrackingModeConfig.from_mapping(Settings().tracking_mode_by_category)
    assert modes.mode_for_category("wildlife") is TrackingMode.STATIONARY_HOLD
    assert modes.mode_for_category(" Wildlife ") is TrackingMode.STATIONARY_HOLD
    for category in ("vehicle", "human", "security", "something_new"):
        assert modes.mode_for_category(category) is TrackingMode.CONTINUOUS_MOTION
    assert modes.mode_for_label("american robin bird") is TrackingMode.STATIONARY_HOLD
    assert modes.mode_for_label("drone") is TrackingMode.CONTINUOUS_MOTION


def test_unlabeled_tracks_always_use_continuous_motion() -> None:
    nothing_tracked = TrackingModeConfig.from_mapping({"default": "none"})
    for category in ("unknown", "", None):
        assert nothing_tracked.mode_for_category(category) is TrackingMode.CONTINUOUS_MOTION
    assert nothing_tracked.mode_for_label("") is TrackingMode.CONTINUOUS_MOTION
    assert not nothing_tracked.tracks_category("vehicle")


@pytest.mark.parametrize(
    ("mapping", "fragment"),
    [
        ({"wildlife": "stationary_hold"}, "'default' entry"),
        ({"default": "hover"}, "hover"),
        ({"unknown": "none", "default": "continuous_motion"}, "may not configure 'unknown'"),
        ({" ": "none", "default": "continuous_motion"}, "empty"),
    ],
)
def test_invalid_mappings_are_rejected(mapping, fragment) -> None:
    with pytest.raises(ValueError, match=fragment):
        TrackingModeConfig.from_mapping(mapping)
    with pytest.raises(ValueError, match=fragment):
        Settings(tracking_mode_by_category=mapping)


def test_entity_reporting_defaults_on() -> None:
    assert Settings().entity_reporting_enabled is True


async def _update(manager: TrackManager, step: int, label: str, category: str, position=(5.0, 5.0, 2.0)):
    return await manager.update(
        timestamp_ns=T0 + step * SECOND, position_m=position, label=label, label_category=category, confidence=0.7
    )


async def test_none_category_never_starts_or_feeds_a_track() -> None:
    manager = TrackManager(Settings(tracking_mode_by_category=tracking_mode_by_category_with("human", TrackingMode.NONE)))
    assert await _update(manager, 0, "speech", "human") is None
    assert await manager.snapshot(include_hidden=True) == []
    assert (await _update(manager, 1, "drone", "vehicle")) is not None


async def test_label_set_routes_an_unlabeled_track_to_its_category_mode() -> None:
    manager = TrackManager(Settings())
    track = await _update(manager, 0, "unknown", "unknown")
    assert not manager._filter.is_perched(track.id)
    labelled = await _update(manager, 4, "american robin", "wildlife")
    assert labelled.id == track.id and manager._filter.is_perched(track.id)
    # One abstention on a labelled bird does not flip it back to constant velocity.
    await _update(manager, 8, "unknown", "unknown")
    assert manager._filter.is_perched(track.id)


async def test_label_change_to_an_untracked_category_ends_the_track() -> None:
    manager = TrackManager(Settings(tracking_mode_by_category=tracking_mode_by_category_with("human", TrackingMode.NONE)))
    track = await _update(manager, 0, "unknown", "unknown")
    async with manager._lock:
        assert manager.apply_label_category_tracking_mode(track.id, "human") is False
    assert await manager.snapshot(include_hidden=True) == []


async def test_wildlife_in_continuous_motion_reads_a_hop_as_velocity() -> None:
    manager = TrackManager(
        Settings(tracking_mode_by_category=tracking_mode_by_category_with("wildlife", TrackingMode.CONTINUOUS_MOTION))
    )
    for step, x in enumerate((0.0, 1.0, 2.0, 3.0)):
        track = await _update(manager, step, "american robin", "wildlife", position=(x, 0.0, 2.0))
    assert track.velocity_mps[0] > 0.3


def _configure_api_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")


def test_tracking_mode_by_category_is_patchable_and_validated_whole(monkeypatch, tmp_path) -> None:
    from fastapi.testclient import TestClient

    from minimappr.main import app

    _configure_api_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        assert client.patch("/api/v1/config", json={"tracking_mode_by_category": "none"}).status_code == 422
        missing_default = client.patch("/api/v1/config", json={"tracking_mode_by_category": {"wildlife": "none"}})
        assert missing_default.status_code == 422
        mapping = tracking_mode_by_category_with("vehicle", TrackingMode.NONE)
        accepted = client.patch("/api/v1/config", json={"tracking_mode_by_category": mapping})
        assert accepted.status_code == 200
        # TrackManager is built once at startup.
        assert "tracking_mode_by_category" in accepted.json()["restart_required"]
        assert client.get("/api/v1/config").json()["tracking_mode_by_category"] == mapping
