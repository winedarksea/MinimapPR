#!/usr/bin/env python3
"""Rebuild tests/fixtures/field_audio/*.wav from xeno-canto sources listed in manifest.toml.

Reproducibility tool, not a test dependency: the rendered WAVs are committed. It uses the
keyless public ``https://xeno-canto.org/<id>/download`` route (API v3 needs a personal key)
with a generic User-Agent, so no personal information leaves the machine.

Processing per clip (documented in ATTRIBUTION too):
  1. slice [source_start_s, source_start_s + duration_s) at the source rate;
  2. take ``source_channel`` (default 0) — channels are never summed, so spaced-pair stereo
     cannot comb-filter the clip;
  3. remove DC, resample to 48 kHz with a polyphase filter (scipy ``resample_poly``);
  4. 10 ms raised-cosine fade in/out so clip edges do not click;
  5. birds: scale to a -3 dBFS sample peak; ambience: scale to -30 dBFS RMS;
  6. write 16-bit PCM WAV (round-to-nearest, no dither).

It also prints a speech-band screen per clip (fraction of 20 ms frames where 100-1000 Hz
energy is both well above its own floor and above the clip's bird band). Screening is a
heuristic; clips flagged ``needs_listen_check`` still need a human listen.

Downloaded audio is untrusted: run as ``python -I scripts/build_field_audio_fixtures.py
--cache-dir <fresh empty dir outside the repo>``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import tomllib
import urllib.request
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

REPO_ROOT = Path(__file__).resolve().parents[1]
FIELD_AUDIO_DIR = REPO_ROOT / "tests" / "fixtures" / "field_audio"
USER_AGENT = "minimappr-fixture-fetch/0.1"
EDGE_FADE_SECONDS = 0.010
SPEECH_BAND_HZ = (100.0, 1000.0)
SPEECH_FRAME_SECONDS = 0.020
SPEECH_ACTIVE_ABOVE_FLOOR_DB = 10.0
DOWNLOAD_PAUSE_SECONDS = 1.0


def download_source(xc_id: int, cache_dir: Path) -> Path:
    destination = cache_dir / f"XC{xc_id}.audio"
    if destination.exists():
        return destination
    request = urllib.request.Request(
        f"https://xeno-canto.org/{xc_id}/download", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())
    time.sleep(DOWNLOAD_PAUSE_SECONDS)
    return destination


def render_clip(source_path: Path, clip: dict, manifest: dict) -> np.ndarray:
    target_rate = int(manifest["sample_rate_hz"])
    info = sf.info(str(source_path))
    start_frame = int(round(float(clip["source_start_s"]) * info.samplerate))
    frame_count = int(round(float(clip["duration_s"]) * info.samplerate))
    audio, source_rate = sf.read(
        str(source_path), start=start_frame, frames=frame_count, always_2d=True, dtype="float64"
    )
    mono = audio[:, int(clip.get("source_channel", 0))]
    mono = mono - mono.mean()
    ratio = Fraction(target_rate, source_rate)
    mono = signal.resample_poly(mono, ratio.numerator, ratio.denominator)
    mono = mono[: int(round(float(clip["duration_s"]) * target_rate))]

    fade_len = int(EDGE_FADE_SECONDS * target_rate)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, fade_len))
    mono[:fade_len] *= ramp
    mono[-fade_len:] *= ramp[::-1]

    if clip["role"] == "ambience":
        rms = float(np.sqrt(np.mean(mono**2)))
        mono *= 10 ** (float(manifest["ambience_rms_dbfs"]) / 20) / rms
        if np.max(np.abs(mono)) >= 1.0:
            raise ValueError(f"{clip['file']}: ambience clips at the RMS target")
    else:
        mono *= 10 ** (float(manifest["bird_peak_dbfs"]) / 20) / float(np.max(np.abs(mono)))
    return mono


def speech_band_active_fraction(mono: np.ndarray, sample_rate: int, bird_band_hz) -> float:
    frame = int(SPEECH_FRAME_SECONDS * sample_rate)
    freqs, _, spec = signal.stft(mono, sample_rate, nperseg=frame * 2, noverlap=frame)
    power = np.abs(spec) ** 2
    speech = power[(freqs >= SPEECH_BAND_HZ[0]) & (freqs <= SPEECH_BAND_HZ[1])].sum(0) + 1e-20
    bird = power[(freqs >= bird_band_hz[0]) & (freqs <= bird_band_hz[1])].sum(0) + 1e-20
    floor = np.percentile(speech, 10)
    active = (10 * np.log10(speech / floor) > SPEECH_ACTIVE_ABOVE_FLOOR_DB) & (speech > bird)
    return float(active.mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="fresh directory for raw downloads (keep it outside the repo)")
    parser.add_argument("--only", nargs="*", default=None, help="subset of clip file names")
    args = parser.parse_args(argv)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = tomllib.loads((FIELD_AUDIO_DIR / "manifest.toml").read_text(encoding="utf-8"))
    for clip in manifest["clip"]:
        if args.only and clip["file"] not in args.only:
            continue
        source = download_source(int(clip["xc_id"]), args.cache_dir)
        mono = render_clip(source, clip, manifest)
        out_path = FIELD_AUDIO_DIR / clip["file"]
        sf.write(str(out_path), mono, int(manifest["sample_rate_hz"]), subtype="PCM_16")
        speech_fraction = speech_band_active_fraction(
            mono, int(manifest["sample_rate_hz"]), clip["band_hz"]
        )
        print(f"{clip['file']}: {len(mono) / manifest['sample_rate_hz']:.2f} s, "
              f"{out_path.stat().st_size / 1e6:.2f} MB, speech-band active {speech_fraction:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
