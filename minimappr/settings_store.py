"""Persistent sparse settings overrides (YAML ``config.yml``).

Settings changed through the UI (``PATCH /api/v1/config``) must survive a
restart. This module owns the persisted-overrides file: a sparse YAML mapping
of allowlisted keys to values that overlays the dataclass defaults + env vars
at ``Settings.from_env()`` time.

Precedence (lowest to highest): dataclass defaults -> env vars -> persisted
overrides. A UI-saved value therefore wins over an env var, which is the
approved behavior — operators are warned via a startup log of each applied
override key and via ``persisted_override_keys`` in ``GET /api/v1/config``.

Resilience mirrors ``rules.py``: a missing file yields ``{}``; malformed YAML
or a non-dict document warns and yields ``{}`` rather than crashing startup.
Writes are atomic (temp file + ``os.replace``), mirroring the rules PUT path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_OVERRIDES_PATH = Path("data/config.yml")

# The single source of truth for which settings may be patched at runtime and
# persisted. ``main.py`` imports this so the HTTP allowlist and the persisted
# overrides can never drift apart. Values are the coercion target type.
CONFIG_PATCH_ALLOWLIST: dict[str, type] = {
    "trigger_rms": float,
    "trigger_cooldown_seconds": float,
    "localization_window_seconds": float,
    "preprocess_enabled": bool,
    "audio_highpass_hz": float,
    "audio_lowpass_hz": float,
    "localization_algorithm": str,
    "localization_strategy": str,
    "localization_band_min_hz": float,
    "localization_band_max_hz": float,
    "classification_audio_source": str,
    "birdnet_enabled": bool,
    "drone_head_enabled": bool,
    "drone_head_min_confidence": float,
    "stt_enabled": bool,
    "stt_trigger_min_confidence": float,
    "transcript_retention_seconds": float,
    "retention_yamnet_audio_seconds": int,
    "retention_birdnet_audio_seconds": int,
    "retention_drone_audio_seconds": int,
    "retention_alert_audio_seconds": int,
    "retention_detection_metadata_seconds": int,
    "omni_scan_enabled": bool,
    "omni_scan_interval_seconds": float,
    "omni_scan_window_seconds": float,
    "omni_scan_min_rms": float,
    "min_localization_confidence": float,
    "skip_localization_for_classification": bool,
    "yamnet_min_confidence": float,
    "detection_min_confidence": float,
    "cop_detections_max_items": int,
    "cop_tracks_max_items": int,
    "cop_detections_max_age_seconds": float,
    "cop_tracks_max_age_seconds": float,
    "beamformer_type": str,
    "birdnet_chunked_dispatch_enabled": bool,
    "birdnet_trigger_min_confidence": float,
    "birdnet_geo_min_confidence": float,
    "tracking_filter": str,
    "fusion_worker_count": int,
    "coordinate_mode": str,
    "hass_enabled": bool,
    "hass_base_url": str,
    "hass_token": str,
    "hass_mqtt_host": str,
    "hass_mqtt_port": int,
}


def config_overrides_path(configured: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the overrides file path.

    Precedence: explicit ``configured`` argument -> ``MINIMAPPR_CONFIG_PATH``
    env var -> ``DEFAULT_CONFIG_OVERRIDES_PATH``.
    """
    if configured is not None and str(configured).strip():
        return Path(configured)
    env = os.getenv("MINIMAPPR_CONFIG_PATH")
    if env is not None and env.strip():
        return Path(env)
    return DEFAULT_CONFIG_OVERRIDES_PATH


def load_overrides(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the persisted overrides mapping, or ``{}`` on any problem.

    Missing file -> ``{}``. Malformed YAML or a non-dict document -> a warning
    and ``{}`` (matches ``rules.py`` resilience; never crash startup).
    """
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Unable to read config overrides %s: %s", file_path, exc)
        return {}
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "Config overrides %s is not a mapping (got %s); ignoring.",
            file_path,
            type(raw).__name__,
        )
        return {}
    return {str(key): value for key, value in raw.items()}


def allowlisted_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Restrict a raw overrides mapping to allowlisted keys.

    Unknown keys are dropped with a warning rather than raising: an operator's
    stale hand-edited file should not brick startup.
    """
    kept: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in CONFIG_PATCH_ALLOWLIST:
            kept[key] = value
        else:
            logger.warning("Ignoring non-allowlisted config override key: %s", key)
    return kept


def save_overrides(path: str | os.PathLike[str], overrides: dict[str, Any]) -> None:
    """Atomically persist the overrides mapping to ``path`` as YAML.

    Restricted to allowlisted keys. Written to a temp file then ``os.replace``d
    into place so a concurrent reader never sees a partial document.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = allowlisted_overrides(overrides)
    tmp_path = file_path.with_name(f".{file_path.name}.tmp")
    tmp_path.write_text(
        yaml.safe_dump(payload, default_flow_style=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, file_path)


__all__ = [
    "CONFIG_PATCH_ALLOWLIST",
    "DEFAULT_CONFIG_OVERRIDES_PATH",
    "allowlisted_overrides",
    "config_overrides_path",
    "load_overrides",
    "save_overrides",
]
