"""Loaders for field-mode audio fixtures.

Two tiers:
  * committed clips in ``tests/fixtures/field_audio/`` (manifest.toml + ATTRIBUTION), always
    available — ``load_field_audio_manifest`` / ``load_field_audio_clip``;
  * larger remote/cached bundles listed in ``tests/fixtures/field_bundles.toml`` and pulled by
    ``scripts/fetch_field_fixtures.py`` into a cache outside git — ``require_field_bundle``
    returns the verified local path, or skips loudly (fails under CI, where a silent skip
    would hide regressions; see docs/field_mode_test_strategy.md Layer 3 "Storage").
"""

from __future__ import annotations

import importlib.util
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIELD_AUDIO_DIR = FIXTURES_DIR / "field_audio"
FIELD_AUDIO_MANIFEST_PATH = FIELD_AUDIO_DIR / "manifest.toml"
FIELD_AUDIO_ATTRIBUTION_PATH = FIELD_AUDIO_DIR / "ATTRIBUTION"
FIELD_BUNDLES_MANIFEST_PATH = FIXTURES_DIR / "field_bundles.toml"
FETCH_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_field_fixtures.py"
FETCH_COMMAND_HINT = "python scripts/fetch_field_fixtures.py"
CI_FALSE_VALUES = ("", "0", "false", "no")


@dataclass(frozen=True, slots=True)
class FieldAudioClip:
    file: str
    species: str
    scientific_name: str
    call_type: str
    band_hz: tuple[float, float]
    xc_id: int
    recordist: str
    licence: str
    licence_url: str
    duration_s: float
    role: str
    same_band_group: str | None
    needs_listen_check: bool

    @property
    def path(self) -> Path:
        return FIELD_AUDIO_DIR / self.file


def load_field_audio_manifest() -> dict:
    return tomllib.loads(FIELD_AUDIO_MANIFEST_PATH.read_text(encoding="utf-8"))


def field_audio_clips(role: str | None = None) -> list[FieldAudioClip]:
    """Manifest entries as typed records; ``role`` filters to "bird" or "ambience"."""
    clips = []
    for entry in load_field_audio_manifest()["clip"]:
        clip = FieldAudioClip(
            file=entry["file"],
            species=entry["species"],
            scientific_name=entry["scientific_name"],
            call_type=entry["call_type"],
            band_hz=(float(entry["band_hz"][0]), float(entry["band_hz"][1])),
            xc_id=int(entry["xc_id"]),
            recordist=entry["recordist"],
            licence=entry["licence"],
            licence_url=entry["licence_url"],
            duration_s=float(entry["duration_s"]),
            role=entry["role"],
            same_band_group=entry.get("same_band_group"),
            needs_listen_check=bool(entry.get("needs_listen_check", False)),
        )
        if role is None or clip.role == role:
            clips.append(clip)
    return clips


def field_audio_clip(file_or_call_type: str) -> FieldAudioClip:
    """Look up a clip by file name or call type (first match), e.g. ``"tonal_whistle"``."""
    for clip in field_audio_clips():
        if file_or_call_type in (clip.file, clip.call_type):
            return clip
    raise KeyError(f"no field audio clip matches {file_or_call_type!r}")


def load_field_audio_clip(clip: FieldAudioClip | str) -> tuple[np.ndarray, int]:
    """Return (float32 mono samples in [-1, 1], sample_rate_hz)."""
    if isinstance(clip, str):
        clip = field_audio_clip(clip)
    samples, sample_rate_hz = sf.read(str(clip.path), dtype="float32", always_2d=False)
    return samples, int(sample_rate_hz)


# -- remote / cached bundles ---------------------------------------------------------------


def load_fetch_script_module():
    """scripts/ is not a package; load the fetch script so manifest validation (cache_path
    traversal, sha256 format, archive suffix) and verification have one implementation."""
    spec = importlib.util.spec_from_file_location("fetch_field_fixtures", FETCH_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fetch_script = load_fetch_script_module()
FIELD_FIXTURE_CACHE_ENV = _fetch_script.CACHE_ENV
DEFAULT_FIELD_FIXTURE_CACHE = _fetch_script.DEFAULT_CACHE
sha256_of_file = _fetch_script.sha256_of_file


def field_fixture_cache_dir() -> Path:
    return _fetch_script.resolve_cache_dir()


def load_field_bundles_manifest() -> list[dict]:
    """Validated ``[[bundle]]`` entries from field_bundles.toml."""
    return _fetch_script.load_bundles(FIELD_BUNDLES_MANIFEST_PATH)


def _running_under_ci() -> bool:
    return os.environ.get("CI", "").strip().lower() not in CI_FALSE_VALUES


def require_field_bundle(name: str) -> Path:
    """Verified cached bundle path (the extracted directory for ``extract = true`` bundles),
    else skip loudly — ``pytest.fail`` under CI, where a silent skip would hide regressions."""
    import pytest

    bundles = {bundle["name"]: bundle for bundle in load_field_bundles_manifest()}
    if name not in bundles:
        raise KeyError(f"{name!r} is not listed in {FIELD_BUNDLES_MANIFEST_PATH}")
    entry = bundles[name]
    local_path = field_fixture_cache_dir() / entry["cache_path"]
    result_path = local_path
    problem = None
    if not local_path.exists():
        problem = "is not in the local cache"
    else:
        mismatch = _fetch_script.verify(local_path, entry)
        if mismatch:
            problem = f"is in the cache but does not match field_bundles.toml ({mismatch})"
        elif entry.get("extract"):
            result_path = _fetch_script.extracted_dir_for(local_path)
            if not result_path.is_dir():
                problem = f"is cached but not extracted to {result_path}"
    if problem is None:
        return result_path
    message = (
        f"\n\n*** FIELD FIXTURE MISSING: bundle {name!r} {problem} ({local_path}).\n"
        f"*** Fetch it with:  {FETCH_COMMAND_HINT} --only {name}\n"
        f"*** (cache dir: ${FIELD_FIXTURE_CACHE_ENV} or {DEFAULT_FIELD_FIXTURE_CACHE})\n"
    )
    if _running_under_ci():
        pytest.fail(message)
    pytest.skip(message)
