"""Persistent stdin/stdout helper classifying sidecar render audio.

Runs the ``localized_render`` routing composite (YAMNet + BirdNET, plus any
chains such as the drone head) on renders handed over by the Rust sidecar.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from minimappr.classifiers.factory import create_context_classifier
from minimappr.classifiers.routing import CONTEXT_LOCALIZED_RENDER
from minimappr.config import Settings


def _decode_pcm16le_mono(path: Path) -> np.ndarray:
    raw_bytes = path.read_bytes()
    return _decode_pcm16le_mono_bytes(raw_bytes)


def _decode_pcm16le_mono_bytes(raw_bytes: bytes) -> np.ndarray:
    if len(raw_bytes) % 2 != 0:
        raise ValueError("PCM16LE render has an odd byte length")
    return np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0


def _json_safe_features(features: dict[str, Any]) -> dict[str, Any]:
    """Drop anything json.dumps can't serialize (defensive: composite/chained
    classifiers already strip embedding ndarrays)."""
    safe: dict[str, Any] = {}
    for key, value in features.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        safe[key] = value
    return safe


def _emit_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    settings = Settings.from_env()
    classifier = create_context_classifier(settings, CONTEXT_LOCALIZED_RENDER)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            request_id = -1
            try:
                payload = json.loads(line)
                request_id = int(payload["request_id"])
                sample_rate_hz = int(payload["sample_rate_hz"])
                if "pcm16le_b64" in payload:
                    pcm16le_bytes = base64.b64decode(str(payload["pcm16le_b64"]))
                    audio = _decode_pcm16le_mono_bytes(pcm16le_bytes)
                else:
                    pcm16le_path = Path(str(payload["pcm16le_path"]))
                    audio = _decode_pcm16le_mono(pcm16le_path)
                classification = classifier.classify(
                    audio,
                    sample_rate_hz,
                )
                _emit_response(
                    {
                        "request_id": request_id,
                        "label": classification.label,
                        "label_confidence": classification.confidence,
                        "scores": classification.scores,
                        "features": _json_safe_features(classification.features),
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