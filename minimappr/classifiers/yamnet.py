"""Optional YAMNet classifier wrapper."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging

import numpy as np
from scipy.signal import resample_poly

from minimappr.classifiers.base import AudioClassifier
from minimappr.models import ClassificationResult
from minimappr.audio_processing.levels import apply_bounded_rms_gain, apply_level_profile
from minimappr.audio_processing.profiles import (
    AudioProcessingProfile,
)
from minimappr.classifiers.yamnet_model import (
    load_bundled_yamnet_model,
    yamnet_model_asset_directory,
)


logger = logging.getLogger(__name__)

_YAMNET_TARGET_RMS = 0.10
_YAMNET_MAX_INPUT_GAIN = 64.0
_EPSILON = 1e-12

# Bumped whenever `_prepare_waveform_for_yamnet` changes so embedding caches and
# model metadata can invalidate on preprocessing drift (train/serve consistency).
YAMNET_PREPROCESS_VERSION = "yamnet-prep-v2"
def yamnet_preprocess_fingerprint(
    target_rms: float = _YAMNET_TARGET_RMS,
    max_input_gain: float = _YAMNET_MAX_INPUT_GAIN,
) -> str:
    document = {
        "name": "yamnet",
        "stages": [
            {"type": "mean_center"},
            {
                "type": "bounded_rms_gain",
                "target_rms_dbfs": 20.0 * float(np.log10(target_rms)),
                "max_gain_db": 20.0 * float(np.log10(max_input_gain)),
                "peak_ceiling_dbfs": 20.0 * float(np.log10(0.98)),
                "boost_only": True,
            },
        ],
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


YAMNET_PREPROCESS_FINGERPRINT = yamnet_preprocess_fingerprint()

# Public aliases of the conditioning constants for training-time metadata.
YAMNET_TARGET_RMS = _YAMNET_TARGET_RMS
YAMNET_MAX_INPUT_GAIN = _YAMNET_MAX_INPUT_GAIN


def _prepare_waveform_for_yamnet(
    waveform: np.ndarray,
    *,
    target_rms: float = _YAMNET_TARGET_RMS,
    max_input_gain: float = _YAMNET_MAX_INPUT_GAIN,
    preprocess_profile: AudioProcessingProfile | None = None,
) -> np.ndarray:
    """Normalize low-level waveforms so YAMNet sees stable input energy.

    YAMNet can over-predict "Silence" when valid events are well below typical
    training loudness. We keep this conditioning conservative by only boosting
    when RMS is low, capping the gain, and preserving peak headroom.
    """
    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)

    if preprocess_profile is not None:
        conditioned, _ = apply_level_profile(waveform, preprocess_profile)
    else:
        conditioned, _ = apply_bounded_rms_gain(
            waveform,
            target_rms=float(target_rms),
            max_gain=float(max_input_gain),
            peak_ceiling=0.98,
            center=True,
            boost_only=True,
        )
    return conditioned


# Public alias so training code (embedding_cache) applies identical conditioning
# without reaching into a private symbol. Same function, no behavior change.
prepare_waveform_for_yamnet = _prepare_waveform_for_yamnet


class YAMNetClassifier(AudioClassifier):
    def __init__(
        self,
        min_confidence: float = 0.25,
        *,
        target_rms: float = _YAMNET_TARGET_RMS,
        max_input_gain: float = _YAMNET_MAX_INPUT_GAIN,
        keep_embeddings: bool = False,
        preprocess_profile: AudioProcessingProfile | None = None,
    ) -> None:
        try:
            import tensorflow as tf
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("YAMNet backend requires tensorflow") from exc

        self._tf = tf
        self._model = load_bundled_yamnet_model()
        self._class_names = self._load_class_names()
        self._min_confidence = min_confidence
        self._target_rms = float(target_rms)
        self._max_input_gain = float(max_input_gain)
        self._preprocess_profile = preprocess_profile
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
            preprocess_profile=self._preprocess_profile,
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
        try:
            class_map_path = yamnet_model_asset_directory() / "assets" / "yamnet_class_map.csv"
            text = class_map_path.read_text(encoding="utf-8")
            names = _parse_class_map_csv(text)
            if not names:
                raise ValueError("class map contains no display names")
            return names
        except Exception as exc:
            logger.warning(
                "Bundled YAMNet class map unavailable (%s; path: %s). "
                "Class names will fall back to 'class_N' indices.",
                exc,
                yamnet_model_asset_directory() / "assets" / "yamnet_class_map.csv",
            )
            return []


def _parse_class_map_csv(text: str) -> list[str]:
    rows = csv.DictReader(io.StringIO(text))
    return [row["display_name"] for row in rows if "display_name" in row]
