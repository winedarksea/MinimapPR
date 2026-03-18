"""Optional BirdNET V2M4 species classifier wrapper."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import resample_poly

from minimappr.classifiers.base import AudioClassifier
from minimappr.models import ClassificationResult


logger = logging.getLogger(__name__)

# BirdNET V2M4 model constants — must match birdnet package assumptions.
_BIRDNET_SAMPLE_RATE_HZ = 48_000
_BIRDNET_CHUNK_SIZE_S = 3.0
_BIRDNET_CHUNK_SAMPLES = int(_BIRDNET_SAMPLE_RATE_HZ * _BIRDNET_CHUNK_SIZE_S)  # 144 000


class BirdNETClassifier(AudioClassifier):
    """Wraps the BirdNET V2M4 Protobuf model for bird species identification.

    Intended as a downstream ChainStage triggered when a base classifier
    (e.g. YAMNet) returns a bird-related label, providing species-level
    resolution beyond the coarse "bird_like" category.

    Requires the ``birdnet`` package (``pip install birdnet``).
    The model files are downloaded on first instantiation (~40 MB).
    """

    def __init__(self, min_confidence: float = 0.1) -> None:
        try:
            from birdnet.models.v2m4.model_v2m4_protobuf import AudioModelV2M4Protobuf
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "BirdNET backend requires the birdnet package: pip install birdnet"
            ) from exc

        self._min_confidence = min_confidence
        # Model weights are downloaded here on first run if not cached.
        self._model = AudioModelV2M4Protobuf()
        # Cache species as an indexed list — OrderedSet supports indexing but
        # list conversion avoids repeated type coercions inside classify().
        self._species: list[str] = list(self._model.species)

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        audio = samples.astype(np.float32)
        if sample_rate_hz != _BIRDNET_SAMPLE_RATE_HZ:
            audio = resample_poly(
                audio,
                up=_BIRDNET_SAMPLE_RATE_HZ,
                down=sample_rate_hz,
            ).astype(np.float32)

        batch = _build_chunk_batch(audio, _BIRDNET_CHUNK_SAMPLES)

        # Returns (n_chunks, n_species) float32 activation scores.
        raw_scores = self._model.predict_species(batch)
        mean_scores = np.mean(raw_scores, axis=0)

        top_idx = int(np.argmax(mean_scores))
        top_conf = float(mean_scores[top_idx])

        raw_species_label = (
            self._species[top_idx] if top_idx < len(self._species) else f"species_{top_idx}"
        )
        label = _extract_common_name(raw_species_label)
        if top_conf < self._min_confidence:
            label = "unknown"

        top_k_indices = np.argsort(mean_scores)[-5:][::-1]
        scores_map: dict[str, float] = {
            _extract_common_name(
                self._species[int(i)] if int(i) < len(self._species) else f"species_{i}"
            ): float(mean_scores[int(i)])
            for i in top_k_indices
        }

        return ClassificationResult(
            label=label,
            confidence=max(0.0, min(1.0, top_conf)),
            scores=scores_map,
            features={"model": "birdnet_v2m4", "raw_species": raw_species_label},
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_chunk_batch(audio: np.ndarray, chunk_samples: int) -> np.ndarray:
    """Split *audio* into fixed-length chunks, zero-padding the last one."""
    if len(audio) == 0:
        return np.zeros((1, chunk_samples), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk.astype(np.float32))
    return np.stack(chunks)


def _extract_common_name(species_label: str) -> str:
    """Extract the common name from BirdNET's 'ScientificName_Common Name' format.

    BirdNET V2M4 uses underscore-separated binomial + common name, e.g.
    ``'Turdus migratorius_American Robin'``.  Returns the common name lowercased
    for consistency with the pipeline label convention.  Falls back to the full
    lowercased label when no separator is found.
    """
    if "_" in species_label:
        return species_label.split("_", 1)[1].strip().lower()
    return species_label.strip().lower()
