#!/usr/bin/env python3
"""Score WAV clips against the classification noise-floor texture gate.

Run this over a directory of recent detection snippets before flipping the gate
from annotate-only into demotion, to confirm no genuine detection would be
flagged:

  python scripts/validate_texture_gate.py --dir data/snippets

A clip is FLAGged as noise-floor texture when its per-frame energy contrast is
below ``--contrast-db`` *and* its median spectral flatness is above
``--flatness-min`` — the same AND condition the orchestrator applies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from minimappr.utils.audio import (
    energy_contrast_db,
    framed_spectral_flatness_median,
    read_wav_mono,
)


def score_file(
    path: Path,
    *,
    contrast_threshold_db: float,
    flatness_min: float,
    frame_ms: float,
) -> tuple[float | None, float | None, str]:
    samples, sample_rate_hz = read_wav_mono(path)
    contrast = energy_contrast_db(samples, sample_rate_hz, frame_ms=frame_ms)
    flatness = framed_spectral_flatness_median(samples, sample_rate_hz, frame_ms=frame_ms)
    if contrast is None or flatness is None:
        return contrast, flatness, "SKIP-short"
    gated = contrast < contrast_threshold_db and flatness > flatness_min
    return contrast, flatness, "FLAG" if gated else "PASS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True, help="Directory of WAV clips to score")
    parser.add_argument("--contrast-db", type=float, default=8.0)
    parser.add_argument("--flatness-min", type=float, default=0.2)
    parser.add_argument("--frame-ms", type=float, default=100.0)
    args = parser.parse_args()

    paths = sorted(args.dir.glob("*.wav"))
    if not paths:
        print(f"no .wav files under {args.dir}")
        return 1

    counts = {"FLAG": 0, "PASS": 0, "SKIP-short": 0, "ERROR": 0}
    width = max(len(p.name) for p in paths)
    for path in paths:
        try:
            contrast, flatness, verdict = score_file(
                path,
                contrast_threshold_db=args.contrast_db,
                flatness_min=args.flatness_min,
                frame_ms=args.frame_ms,
            )
        except (OSError, ValueError) as exc:
            counts["ERROR"] += 1
            print(f"{path.name:<{width}}  {'-':>10}  {'-':>8}  ERROR ({exc})")
            continue
        counts[verdict] += 1
        contrast_text = "-" if contrast is None else f"{contrast:.2f}"
        flatness_text = "-" if flatness is None else f"{flatness:.4f}"
        print(f"{path.name:<{width}}  {contrast_text:>10}  {flatness_text:>8}  {verdict}")

    total = len(paths)
    print(
        f"\n{total} file(s): {counts['FLAG']} FLAG, {counts['PASS']} PASS, "
        f"{counts['SKIP-short']} SKIP-short, {counts['ERROR']} ERROR "
        f"(contrast < {args.contrast_db} dB AND flatness > {args.flatness_min})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
