"""Versioned ambisonics encoder profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AmbisonicsProfile:
    name: str
    frame_duration_ms: float = 64.0
    hop_fraction: float = 0.25
    min_parametric_hz: float = 100.0
    max_parametric_fraction_of_nyquist: float = 0.90
    intensity_smoothing_ms: float = 60.0
    diffuseness_smoothing_ms: float = 120.0
    max_parametric_blend: float = 0.85
    min_confidence_for_blend: float = 0.12
    output_peak_target: float = 0.98


PROFILE_JSON_PATH = Path(__file__).with_name("ambisonics_profiles.json")


def _load_profiles_from_json(path: Path = PROFILE_JSON_PATH) -> dict[str, AmbisonicsProfile]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles: dict[str, AmbisonicsProfile] = {}
    for name, values in payload.items():
        if not isinstance(values, dict):
            raise ValueError(f"profile {name!r} must be a JSON object")
        profiles[str(name)] = AmbisonicsProfile(name=str(name), **values)
    return profiles


PROFILES: dict[str, AmbisonicsProfile] = _load_profiles_from_json()
LINEAR_V1 = PROFILES["linear_v1"]
PARAMETRIC_V2 = PROFILES["parametric_v2"]


def get_profile(profile: str | AmbisonicsProfile) -> AmbisonicsProfile:
    if isinstance(profile, AmbisonicsProfile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown ambisonics profile {profile!r}") from exc
