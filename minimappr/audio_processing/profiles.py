"""Immutable, versioned audio-processing profile specifications."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

YAMNET_PROFILE_NAME = "yamnet"
LISTENING_PROFILE_NAME = "listening"


@dataclass(frozen=True, slots=True)
class AudioProcessingProfile:
    name: str
    stages: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class AudioProcessingConfiguration:
    version: int
    profiles: Mapping[str, AudioProcessingProfile]

    def profile(self, name: str) -> AudioProcessingProfile:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise ValueError(f"Unknown audio processing profile {name!r}") from exc


def _immutable_stage(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(raw))


def _configuration(raw: Mapping[str, Any], source: str) -> AudioProcessingConfiguration:
    version = int(raw.get("version", 1))
    if version != 1:
        raise ValueError(f"{source}: audio processing version must be 1")
    profiles: dict[str, AudioProcessingProfile] = {}
    for name, profile_raw in dict(raw.get("profiles") or {}).items():
        if not isinstance(profile_raw, Mapping):
            raise ValueError(f"{source}: profile {name!r} must be an object")
        stages_raw = profile_raw.get("stages", [])
        if not isinstance(stages_raw, list):
            raise ValueError(f"{source}: profile {name!r}.stages must be a list")
        stages = tuple(_immutable_stage(stage) for stage in stages_raw)
        profiles[str(name)] = AudioProcessingProfile(str(name), stages)
    return AudioProcessingConfiguration(version, MappingProxyType(profiles))


_DEFAULT_DOCUMENT = {
    "version": 1,
    "profiles": {
        "yamnet": {
            "stages": [
                {"type": "mean_center"},
                {
                    "type": "bounded_rms_gain",
                    "target_rms_dbfs": -20.0,
                    "max_gain_db": 20.0 * math.log10(64.0),
                    "peak_ceiling_dbfs": 20.0 * math.log10(0.98),
                    "boost_only": True,
                },
            ]
        },
        "listening": {
            "stages": [
                {"type": "mean_center"},
                {
                    "type": "bounded_rms_gain",
                    "target_rms_dbfs": -24.0,
                    "max_gain_db": 24.0,
                    "peak_ceiling_dbfs": -1.0,
                    "boost_only": True,
                },
            ]
        },
    },
}

DEFAULT_AUDIO_PROCESSING_CONFIGURATION = _configuration(_DEFAULT_DOCUMENT, "defaults")


def load_audio_processing_configuration(path: Path | str | None) -> AudioProcessingConfiguration:
    if path is None or not Path(path).exists():
        return DEFAULT_AUDIO_PROCESSING_CONFIGURATION
    source_path = Path(path)
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    return _configuration(raw, str(source_path))


def profile_fingerprint(profile: AudioProcessingProfile) -> str:
    canonical = json.dumps(
        {"name": profile.name, "stages": [dict(stage) for stage in profile.stages]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
