"""Optional YAMNet classifier wrapper."""

from __future__ import annotations

import csv
import io
import logging
import urllib.request
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from minimappr.classifiers.base import AudioClassifier
from minimappr.models import ClassificationResult


logger = logging.getLogger(__name__)

# Local fallback path for the class map so the server starts offline after the
# first successful fetch.  Relative to CWD (the project root when run normally).
_CLASS_MAP_CACHE_PATH = Path("data/yamnet_class_map.csv")
_YAMNET_TARGET_RMS = 0.10
_YAMNET_MAX_INPUT_GAIN = 32.0
_EPSILON = 1e-12

# Bumped whenever `_prepare_waveform_for_yamnet` changes so embedding caches and
# model metadata can invalidate on preprocessing drift (train/serve consistency).
YAMNET_PREPROCESS_VERSION = "yamnet-prep-v1"

# Public aliases of the conditioning constants for training-time metadata.
YAMNET_TARGET_RMS = _YAMNET_TARGET_RMS
YAMNET_MAX_INPUT_GAIN = _YAMNET_MAX_INPUT_GAIN


def _prepare_waveform_for_yamnet(
    waveform: np.ndarray,
    *,
    target_rms: float = _YAMNET_TARGET_RMS,
    max_input_gain: float = _YAMNET_MAX_INPUT_GAIN,
) -> np.ndarray:
    """Normalize low-level waveforms so YAMNet sees stable input energy.

    YAMNet can over-predict "Silence" when valid events are well below typical
    training loudness. We keep this conditioning conservative by only boosting
    when RMS is low, capping the gain, and preserving peak headroom.
    """
    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)

    centered = waveform.astype(np.float32, copy=False) - np.mean(waveform, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(centered), dtype=np.float64) + _EPSILON))
    peak = float(np.max(np.abs(centered)) + _EPSILON)

    # Gain floor is 1.0 so we avoid attenuating normal signals unless headroom
    # would otherwise clip after prior preprocessing.
    gain_from_rms = max(1.0, float(target_rms) / max(rms, _EPSILON))
    gain_from_headroom = 0.98 / peak
    gain = min(float(max_input_gain), gain_from_rms, gain_from_headroom)
    gain = max(gain, min(1.0, gain_from_headroom))

    return (centered * gain).astype(np.float32)


# Public alias so training code (embedding_cache) applies identical conditioning
# without reaching into a private symbol. Same function, no behavior change.
prepare_waveform_for_yamnet = _prepare_waveform_for_yamnet


class YAMNetClassifier(AudioClassifier):
    CLASS_MAP_URL = (
        "https://raw.githubusercontent.com/tensorflow/models/master/"
        "research/audioset/yamnet/yamnet_class_map.csv"
    )

    def __init__(
        self,
        min_confidence: float = 0.25,
        *,
        target_rms: float = _YAMNET_TARGET_RMS,
        max_input_gain: float = _YAMNET_MAX_INPUT_GAIN,
        keep_embeddings: bool = False,
    ) -> None:
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("YAMNet backend requires tensorflow and tensorflow-hub") from exc

        self._tf = tf
        self._hub = hub
        self._model = hub.load("https://tfhub.dev/google/yamnet/1")
        self._class_names = self._load_class_names()
        self._min_confidence = min_confidence
        self._target_rms = float(target_rms)
        self._max_input_gain = float(max_input_gain)
        # Off on the hot path: 1024-d embeddings would bloat feature_summary_json
        # and the sidecar classifier bridge. Enabled only for offline extraction
        # at training-promotion time (minimappr/calibration/embeddings.py).
        self._keep_embeddings = bool(keep_embeddings)

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        waveform = samples.astype(np.float32)
        if sample_rate_hz != 16000:
            waveform = resample_poly(waveform, up=16000, down=sample_rate_hz).astype(np.float32)
        waveform = _prepare_waveform_for_yamnet(
            waveform,
            target_rms=self._target_rms,
            max_input_gain=self._max_input_gain,
        )

        scores, embeddings, _ = self._model(waveform)
        mean_scores = np.mean(scores.numpy(), axis=0)
        top_idx = int(np.argmax(mean_scores))
        top_conf = float(mean_scores[top_idx])

        label = self._class_name(top_idx)
        if top_conf < self._min_confidence:
            label = "unknown"

        top_k = np.argsort(mean_scores)[-5:][::-1]
        scores_map = {self._class_name(int(index)): float(mean_scores[int(index)]) for index in top_k}

        features: dict = {"model": "yamnet"}
        if self._keep_embeddings:
            embedding_frames = embeddings.numpy().astype(np.float32)
            features["embedding"] = np.mean(embedding_frames, axis=0).astype(np.float32)
            # Per-frame [N, 1024] embeddings for embedding-input chain stages
            # (e.g. the drone head). Stripped before feature_summary_json by
            # ChainedClassifier/CompositeClassifier.
            features["embedding_frames"] = embedding_frames
            features["embedding_model"] = "yamnet/1"
            features["embedding_dim"] = int(features["embedding"].shape[0])

        return ClassificationResult(
            label=label,
            confidence=max(0.0, min(1.0, top_conf)),
            scores=scores_map,
            features=features,
        )

    def _class_name(self, index: int) -> str:
        if index < len(self._class_names):
            return self._class_names[index]
        return f"class_{index}"

    def _load_class_names(self) -> list[str]:
        # 1. Try the local disk cache written by a previous successful fetch.
        if _CLASS_MAP_CACHE_PATH.exists():
            try:
                text = _CLASS_MAP_CACHE_PATH.read_text(encoding="utf-8")
                names = _parse_class_map_csv(text)
                if names:
                    return names
            except Exception:
                pass  # Fall through to network fetch.

        # 2. Fetch from upstream and persist to disk for future offline starts.
        try:
            with urllib.request.urlopen(self.CLASS_MAP_URL, timeout=10) as response:  # nosec B310
                text = response.read().decode("utf-8")
            names = _parse_class_map_csv(text)
            if names:
                try:
                    _CLASS_MAP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    _CLASS_MAP_CACHE_PATH.write_text(text, encoding="utf-8")
                except Exception:
                    pass  # Cache write failure is non-fatal.
            return names
        except Exception as exc:
            logger.warning(
                "YAMNet class map unavailable (network: %s; cache: %s). "
                "Class names will fall back to 'class_N' indices.",
                exc,
                _CLASS_MAP_CACHE_PATH,
            )
            return []


def _parse_class_map_csv(text: str) -> list[str]:
    rows = csv.DictReader(io.StringIO(text))
    return [row["display_name"] for row in rows if "display_name" in row]

