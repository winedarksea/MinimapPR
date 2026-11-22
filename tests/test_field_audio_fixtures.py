"""Field-mode audio fixtures: committed clips load cleanly and are fully attributed; the
remote-bundle manifest/fetch path verifies hashes and skips loudly when the cache is absent."""

from __future__ import annotations

import io
import tarfile
import tomllib
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from tests import field_audio_fixtures as faf

FIELD_AUDIO_MAX_TOTAL_BYTES = 10 * 1024 * 1024
MIN_BIRD_IN_BAND_POWER_FRACTION = 0.75
BIRD_PEAK_TOLERANCE_DB = 0.1
# RMS of a 30 s bed drifts more than a sample peak under 16-bit rounding and edge fades.
AMBIENCE_RMS_TOLERANCE_DB = 0.5

CLIPS = faf.field_audio_clips()


def test_manifest_lists_expected_call_types():
    call_types = Counter(clip.call_type for clip in CLIPS)
    for required in ("tonal_whistle", "dry_trill", "broadband_call", "harsh_call", "ambience"):
        assert call_types[required] >= 1, f"missing a {required} fixture"
    groups = Counter(clip.same_band_group for clip in CLIPS if clip.same_band_group)
    assert groups and all(count >= 2 for count in groups.values())
    assert len({clip.file for clip in CLIPS}) == len(CLIPS)


@pytest.mark.parametrize("clip", CLIPS, ids=lambda c: c.file)
def test_clip_is_48k_mono_pcm16_with_manifest_duration(clip):
    manifest = faf.load_field_audio_manifest()
    expected_rate_hz = int(manifest["sample_rate_hz"])
    info = sf.info(str(clip.path))
    assert info.samplerate == expected_rate_hz
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert info.frames == round(clip.duration_s * expected_rate_hz)

    samples, sample_rate_hz = faf.load_field_audio_clip(clip)
    assert sample_rate_hz == expected_rate_hz and samples.dtype == np.float32
    assert np.all(np.isfinite(samples))
    if clip.role == "ambience":
        rms_dbfs = 20 * np.log10(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        assert abs(rms_dbfs - manifest["ambience_rms_dbfs"]) < AMBIENCE_RMS_TOLERANCE_DB
    else:
        peak_dbfs = 20 * np.log10(np.max(np.abs(samples)))
        assert abs(peak_dbfs - manifest["bird_peak_dbfs"]) < BIRD_PEAK_TOLERANCE_DB


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
    cells = [cell.strip() for cell in row.split("|")]
    assert cells[1] == f"XC{clip.xc_id}"
    assert cells[3] == clip.recordist
    assert cells[4] == clip.licence
    assert float(cells[6]) == clip.duration_s
    assert clip.licence.startswith("CC BY") and "ND" not in clip.licence
    assert clip.licence_url.startswith("https://creativecommons.org/licenses/")
    assert clip.licence_url in attribution
    if "NC" in clip.licence:
        assert clip.file in attribution.split("Licence note:")[1], "NC clip not called out"


def test_committed_field_audio_fits_size_budget():
    on_disk = {p.name for p in faf.FIELD_AUDIO_DIR.glob("*.wav")}
    assert on_disk == {clip.file for clip in CLIPS}, "unlisted or missing WAVs"
    total = sum(p.stat().st_size for p in faf.FIELD_AUDIO_DIR.iterdir())
    assert total <= FIELD_AUDIO_MAX_TOTAL_BYTES


# -- remote bundles ----------------------------------------------------------------------

BUNDLE_PAYLOAD = b"field fixture payload"


def _write_bundle_manifest(path: Path, source_file: Path, sha256: str,
                           cache_path: str = "nested/tiny.bin", extract: bool = False) -> None:
    path.write_text(
        "[[bundle]]\n"
        'name = "tiny"\n'
        f'sha256 = "{sha256}"\n'
        f"size_bytes = {source_file.stat().st_size}\n"
        f'source = "{source_file.as_uri()}"\n'
        f'cache_path = "{cache_path}"\n'
        + ("extract = true\n" if extract else "")
        + 'description = "test"\n'
        'licence = "test"\n',
        encoding="utf-8",
    )


@pytest.fixture()
def fetch_script():
    return faf.load_fetch_script_module()


@pytest.fixture()
def payload_file(tmp_path):
    source_file = tmp_path / "src.bin"
    source_file.write_bytes(BUNDLE_PAYLOAD)
    return source_file


@pytest.fixture()
def bundle_env(tmp_path, monkeypatch):
    """Points the loader at a tmp manifest + cache; returns (manifest_path, cache_dir)."""
    manifest = tmp_path / "bundles.toml"
    cache = tmp_path / "cache"
    monkeypatch.setattr(faf, "FIELD_BUNDLES_MANIFEST_PATH", manifest)
    monkeypatch.setenv(faf.FIELD_FIXTURE_CACHE_ENV, str(cache))
    monkeypatch.delenv("CI", raising=False)
    return manifest, cache


def test_committed_bundle_manifest_parses(fetch_script):
    assert isinstance(fetch_script.load_bundles(), list)
    assert isinstance(faf.load_field_bundles_manifest(), list)


def test_fetch_verifies_and_require_returns_cached_bundle(fetch_script, payload_file, bundle_env):
    manifest, cache = bundle_env
    _write_bundle_manifest(manifest, payload_file, fetch_script.sha256_of_file(payload_file))

    assert fetch_script.fetch(fetch_script.load_bundles(manifest), cache, None) == 0
    assert (cache / "nested" / "tiny.bin").read_bytes() == BUNDLE_PAYLOAD
    assert faf.require_field_bundle("tiny") == cache / "nested" / "tiny.bin"


def test_fetch_extracts_archive_and_require_returns_extracted_dir(tmp_path, fetch_script,
                                                                 payload_file, bundle_env):
    manifest, cache = bundle_env
    archive = tmp_path / "set.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload_file, arcname="set/clip.bin")
    _write_bundle_manifest(manifest, archive, fetch_script.sha256_of_file(archive),
                           cache_path="set.tar.gz", extract=True)

    assert fetch_script.fetch(fetch_script.load_bundles(manifest), cache, None) == 0
    extracted = faf.require_field_bundle("tiny")
    assert extracted == cache / "set"
    assert (extracted / "set" / "clip.bin").read_bytes() == BUNDLE_PAYLOAD
    # Re-running replaces the extracted copy instead of failing on the existing directory.
    assert fetch_script.fetch(fetch_script.load_bundles(manifest), cache, None) == 0


def test_fetch_rejects_hash_mismatch(fetch_script, payload_file, bundle_env):
    manifest, cache = bundle_env
    _write_bundle_manifest(manifest, payload_file, "0" * 64)
    assert fetch_script.fetch(fetch_script.load_bundles(manifest), cache, None) == 1
    assert not (cache / "nested" / "tiny.bin").exists()


@pytest.mark.parametrize("cache_path", ["../escape.bin", "/abs/escape.bin", "a/../../b.bin"])
def test_manifest_rejects_cache_path_traversal(tmp_path, fetch_script, payload_file, cache_path):
    manifest = tmp_path / "bundles.toml"
    _write_bundle_manifest(manifest, payload_file, "0" * 64, cache_path=cache_path)
    with pytest.raises(ValueError, match="inside the cache"):
        fetch_script.load_bundles(manifest)


def test_manifest_rejects_extract_on_non_archive(tmp_path, fetch_script, payload_file):
    manifest = tmp_path / "bundles.toml"
    _write_bundle_manifest(manifest, payload_file, "0" * 64, extract=True)
    with pytest.raises(ValueError, match="not a tar archive"):
        fetch_script.load_bundles(manifest)


def test_ssh_source_accepts_plain_spec(fetch_script):
    remote, port = fetch_script.rsync_remote_spec("ssh://user@127.0.0.1:2222/home/u/x.tar.gz")
    assert (remote, port) == ("user@127.0.0.1:/home/u/x.tar.gz", 2222)


@pytest.mark.parametrize("source", [
    "ssh://127.0.0.1/abs/x",                      # no user
    "ssh://-oProxyCommand=evil@host/abs/x",       # option injection via user
    "ssh://user@-oProxyCommand=evil/abs/x",       # option injection via host
    "ssh://user@host/abs/$(touch%20pwned)",       # remote shell expansion
    "ssh://user@host/abs/a;rm",
    "ssh://user@host/abs/a%20b",
])
def test_ssh_source_rejects_injection(fetch_script, source):
    with pytest.raises(ValueError):
        fetch_script.rsync_remote_spec(source)


@pytest.mark.parametrize("argv", [
    ["fetch", "--only", "a", "--cache-dir", "/c"],
    ["--only", "a", "--cache-dir", "/c", "fetch"],
    ["--cache-dir", "/c", "--only", "a"],
])
def test_cli_accepts_options_before_or_after_subcommand(fetch_script, argv):
    args = fetch_script.build_parser().parse_args(argv)
    assert args.only == ["a"] and args.cache_dir == Path("/c")


def test_register_emits_valid_toml_for_awkward_text(fetch_script, payload_file):
    awkward_description = 'has "quotes" \\ and é'
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        fetch_script.main(["register", str(payload_file), "--name", "n",
                           "--source", payload_file.as_uri(),
                           "--description", awkward_description, "--licence", "private"])
    entry = tomllib.loads(buffer.getvalue())["bundle"][0]
    assert entry["description"] == awkward_description
    assert entry["sha256"] == fetch_script.sha256_of_file(payload_file)


def test_require_bundle_skips_loudly_or_fails_under_ci(monkeypatch, payload_file, bundle_env):
    manifest, _ = bundle_env
    _write_bundle_manifest(manifest, payload_file, "0" * 64)

    for not_ci_value in ("", "false", "0"):
        monkeypatch.setenv("CI", not_ci_value)
        with pytest.raises(pytest.skip.Exception,
                           match=r"(?s)FIELD FIXTURE MISSING.*fetch_field_fixtures"):
            faf.require_field_bundle("tiny")
    monkeypatch.setenv("CI", "true")
    with pytest.raises(pytest.fail.Exception, match="FIELD FIXTURE MISSING"):
        faf.require_field_bundle("tiny")


def test_require_bundle_rejects_corrupted_cache(fetch_script, payload_file, bundle_env):
    manifest, cache = bundle_env
    _write_bundle_manifest(manifest, payload_file, fetch_script.sha256_of_file(payload_file))
    cached = cache / "nested" / "tiny.bin"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(BUNDLE_PAYLOAD[::-1])  # same size, different content
    with pytest.raises(pytest.skip.Exception, match="sha256"):
        faf.require_field_bundle("tiny")
