"""Optional BirdNET V2.4 species classifier wrapper (birdnet >= 0.2.12)."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import resample_poly

from minimappr.classifiers.base import AudioClassifier
from minimappr.models import ClassificationResult


logger = logging.getLogger(__name__)

# BirdNET V2.4 operates at 48 kHz; all audio is resampled to this rate.
_BIRDNET_SAMPLE_RATE_HZ = 48_000
# Number of top species to include in the returned scores map.
_SCORES_MAP_TOP_K = 5


class BirdNETClassifier(AudioClassifier):
    """Wraps the BirdNET V2.4 Protobuf model for bird species identification.

    Intended as a downstream ChainStage triggered when a base classifier
    (e.g. YAMNet) returns a bird-related label, providing species-level
    resolution beyond the coarse "bird_like" category.

    Requires the ``birdnet`` package (``pip install birdnet``).
    The model files are downloaded on first instantiation (~125 MB).
    """

    def __init__(self, min_confidence: float = 0.1) -> None:
        try:
            from birdnet import model_loader
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "BirdNET backend requires the birdnet package: pip install birdnet"
            ) from exc

        self._min_confidence = min_confidence
        # Model weights are downloaded here on first run if not cached.
        self._model = model_loader.load("acoustic", "2.4", "pb")

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        audio = samples.astype(np.float32)
        if sample_rate_hz != _BIRDNET_SAMPLE_RATE_HZ:
            audio = resample_poly(
                audio,
                up=_BIRDNET_SAMPLE_RATE_HZ,
                down=sample_rate_hz,
            ).astype(np.float32)

        # predict_arrays handles audio chunking internally; no manual batching needed.
        result = self._model.predict_arrays(
            (audio, _BIRDNET_SAMPLE_RATE_HZ),
            top_k=_SCORES_MAP_TOP_K,
            default_confidence_threshold=self._min_confidence,
            apply_sigmoid=True,
        )

        # to_structured_array() yields rows with fields: species_name, confidence,
        # start_time, end_time — sorted by confidence descending, above-threshold only.
        detections = result.to_structured_array()

        if len(detections) == 0:
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                scores={},
                features={"model": "birdnet_v2m4"},
            )

        # Deduplicate across segments: keep max confidence per species.
        per_species: dict[str, float] = {}
        for row in detections:
            name = str(row["species_name"])
            conf = float(row["confidence"])
            if name not in per_species or conf > per_species[name]:
                per_species[name] = conf

        sorted_species = sorted(per_species.items(), key=lambda item: item[1], reverse=True)
        top_raw_label, top_conf = sorted_species[0]
        label = _extract_common_name(top_raw_label)

        scores_map: dict[str, float] = {
            _extract_common_name(sp): conf for sp, conf in sorted_species[:_SCORES_MAP_TOP_K]
        }

        return ClassificationResult(
            label=label,
            confidence=max(0.0, min(1.0, top_conf)),
            scores=scores_map,
            features={"model": "birdnet_v2m4", "raw_species": top_raw_label},
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_common_name(species_label: str) -> str:
    """Extract the common name from BirdNET's 'ScientificName_Common Name' format.

    BirdNET V2.4 uses underscore-separated binomial + common name, e.g.
    ``'Turdus migratorius_American Robin'``.  Returns the common name lowercased
    for consistency with the pipeline label convention.  Falls back to the full
    lowercased label when no separator is found.
    """
    if "_" in species_label:
        return species_label.split("_", 1)[1].strip().lower()
    return species_label.strip().lower()
