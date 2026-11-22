"""Field-mode audio fixtures: committed clips load cleanly and are fully attributed; the
remote-bundle manifest/fetch path verifies hashes and skips loudly when the cache is absent."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from tests import field_audio_fixtures as faf

FIELD_AUDIO_MAX_TOTAL_BYTES = 10 * 1024 * 1024
MIN_BIRD_IN_BAND_POWER_FRACTION = 0.75
LEVEL_TOLERANCE_DB = 0.1
REPO_ROOT = Path(__file__).resolve().parents[1]

CLIPS = faf.field_audio_clips()


def _load_fetch_script():
    spec = importlib.util.spec_from_file_location(
        "fetch_field_fixtures", REPO_ROOT / "scripts" / "fetch_field_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_lists_expected_call_types():
    call_types = Counter(clip.call_type for clip in CLIPS)
    for required in ("tonal_whistle", "dry_trill", "broadband_call", "harsh_call", "ambience"):
        assert call_types[required] >= 1, f"missing a {required} fixture"
    groups = Counter(clip.same_band_group for clip in CLIPS if clip.same_band_group)
    assert groups and all(count >= 2 for count in groups.values())
    assert len({clip.file for clip in CLIPS}) == len(CLIPS)


@pytest.mark.parametrize("clip", CLIPS, ids=lambda c: c.file)
def test_clip_is_48k_mono_pcm16_with_manifest_duration(clip):
    info = sf.info(str(clip.path))
    assert info.samplerate == 48_000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert info.frames == round(clip.duration_s * 48_000)

    samples, sample_rate_hz = faf.load_field_audio_clip(clip)
    assert sample_rate_hz == 48_000 and samples.dtype == np.float32
    assert np.all(np.isfinite(samples))
    manifest = faf.load_field_audio_manifest()
    if clip.role == "ambience":
        rms_dbfs = 20 * np.log10(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        assert abs(rms_dbfs - manifest["ambience_rms_dbfs"]) < LEVEL_TOLERANCE_DB * 5
    else:
        peak_dbfs = 20 * np.log10(np.max(np.abs(samples)))
        assert abs(peak_dbfs - manifest["bird_peak_dbfs"]) < LEVEL_TOLERANCE_DB


@pytest.mark.parametrize("clip", faf.field_audio_clips(role="bird"), ids=lambda c: c.file)
def test_bird_energy_sits_in_declared_band(clip):
    samples, sample_rate_hz = faf.load_field_audio_clip(clip)
    lo_hz, hi_hz = clip.band_hz
    assert 0 < lo_hz < hi_hz < sample_rate_hz / 2
    freqs, power = signal.welch(samples, sample_rate_hz, nperseg=4096)
    in_band = power[(freqs >= lo_hz) & (freqs <= hi_hz)].sum() / power.sum()
    assert in_band >= MIN_BIRD_IN_BAND_POWER_FRACTION


@pytest.mark.parametrize("clip", CLIPS, ids=lambda c: c.file)
def test_every_clip_is_attributed_and_redistributable(clip):
    attribution = faf.FIELD_AUDIO_ATTRIBUTION_PATH.read_text(encoding="utf-8")
    row = next((line for line in attribution.splitlines() if line.startswith(clip.file)), None)
    assert row is not None, f"{clip.file} has no ATTRIBUTION row"
    assert f"XC{clip.xc_id}" in row
    assert clip.recordist in row
    assert clip.licence in row
    assert clip.licence.startswith("CC BY") and "ND" not in clip.licence
    assert clip.licence_url.startswith("https://creativecommons.org/licenses/")
    assert clip.licence_url in attribution


def test_committed_field_audio_fits_size_budget():
    on_disk = {p.name for p in faf.FIELD_AUDIO_DIR.glob("*.wav")}
    assert on_disk == {clip.file for clip in CLIPS}, "unlisted or missing WAVs"
    total = sum(p.stat().st_size for p in faf.FIELD_AUDIO_DIR.iterdir())
    assert total <= FIELD_AUDIO_MAX_TOTAL_BYTES


# -- remote bundles ----------------------------------------------------------------------


def _write_bundle_manifest(path: Path, source_file: Path, sha256: str) -> None:
    path.write_text(
        "[[bundle]]\n"
        'name = "tiny"\n'
        f'sha256 = "{sha256}"\n'
        f"size_bytes = {source_file.stat().st_size}\n"
        f'source = "file://{source_file}"\n'
        'cache_path = "nested/tiny.bin"\n'
        'description = "test"\n'
        'licence = "test"\n',
        encoding="utf-8",
    )


def test_committed_bundle_manifest_parses():
    fetch_script = _load_fetch_script()
    bundles = fetch_script.load_bundles()
    assert isinstance(bundles, list)
    assert isinstance(faf.load_field_bundles_manifest(), dict)


def test_fetch_verifies_and_require_returns_cached_bundle(tmp_path, monkeypatch):
    fetch_script = _load_fetch_script()
    source_file = tmp_path / "src.bin"
    source_file.write_bytes(b"field fixture payload")
    manifest = tmp_path / "bundles.toml"
    _write_bundle_manifest(manifest, source_file, fetch_script.sha256_of_file(source_file))
    cache = tmp_path / "cache"

    bundles = fetch_script.load_bundles(manifest)
    assert fetch_script.fetch(bundles, cache, None) == 0
    assert (cache / "nested" / "tiny.bin").read_bytes() == b"field fixture payload"

    monkeypatch.setattr(faf, "FIELD_BUNDLES_MANIFEST_PATH", manifest)
    monkeypatch.setenv(faf.FIELD_FIXTURE_CACHE_ENV, str(cache))
    assert faf.require_field_bundle("tiny") == cache / "nested" / "tiny.bin"


def test_fetch_rejects_hash_mismatch(tmp_path):
    fetch_script = _load_fetch_script()
    source_file = tmp_path / "src.bin"
    source_file.write_bytes(b"field fixture payload")
    manifest = tmp_path / "bundles.toml"
    _write_bundle_manifest(manifest, source_file, "0" * 64)
    cache = tmp_path / "cache"
    assert fetch_script.fetch(fetch_script.load_bundles(manifest), cache, None) == 1
    assert not (cache / "nested" / "tiny.bin").exists()


def test_require_bundle_skips_loudly_or_fails_under_ci(tmp_path, monkeypatch):
    source_file = tmp_path / "src.bin"
    source_file.write_bytes(b"x")
    manifest = tmp_path / "bundles.toml"
    _write_bundle_manifest(manifest, source_file, "0" * 64)
    monkeypatch.setattr(faf, "FIELD_BUNDLES_MANIFEST_PATH", manifest)
    monkeypatch.setenv(faf.FIELD_FIXTURE_CACHE_ENV, str(tmp_path / "empty_cache"))

    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(pytest.skip.Exception, match=r"(?s)FIELD FIXTURE MISSING.*fetch_field_fixtures"):
        faf.require_field_bundle("tiny")
    monkeypatch.setenv("CI", "1")
    with pytest.raises(pytest.fail.Exception, match="FIELD FIXTURE MISSING"):
        faf.require_field_bundle("tiny")
