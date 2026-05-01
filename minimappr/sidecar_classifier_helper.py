"""Persistent stdin/stdout helper that classifies sidecar render audio with BirdNET."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from minimappr.classifiers.birdnet import BirdNETClassifier
from minimappr.config import Settings


def _decode_pcm16le_mono(path: Path) -> np.ndarray:
    raw_bytes = path.read_bytes()
    if len(raw_bytes) % 2 != 0:
        raise ValueError(f"PCM16LE render at {path} has an odd byte length")
    return np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0


def _emit_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    settings = Settings.from_env()
    classifier = BirdNETClassifier(
        min_confidence=settings.birdnet_trigger_min_confidence,
        latitude=settings.site_origin_lat,
        longitude=settings.site_origin_lon,
        geo_min_confidence=settings.birdnet_geo_min_confidence,
    )
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            request_id = -1
            try:
                payload = json.loads(line)
                request_id = int(payload["request_id"])
                pcm16le_path = Path(str(payload["pcm16le_path"]))
                sample_rate_hz = int(payload["sample_rate_hz"])
                classification = classifier.classify(
                    _decode_pcm16le_mono(pcm16le_path),
                    sample_rate_hz,
                )
                _emit_response(
                    {
                        "request_id": request_id,
                        "label": classification.label,
                        "label_confidence": classification.confidence,
                        "scores": classification.scores,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                _emit_response(
                    {
                        "request_id": request_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        classifier.close()


if __name__ == "__main__":
    main()