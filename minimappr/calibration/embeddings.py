"""Offline YAMNet embedding extraction for promoted training examples.

Runs at training-promotion time (not on the ingest hot path — 1024-d vectors
would bloat feature_summary_json and the sidecar classifier bridge). The
classifier is a lazy singleton; when tensorflow is unavailable extraction
degrades to None and promotion still succeeds.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "yamnet/1"
EMBEDDING_DIM = 1024
EMBEDDING_POOLING = "mean"

_classifier: Any = None
_classifier_failed = False


def _get_classifier() -> Any:
    """Lazily construct YAMNetClassifier(keep_embeddings=True); None if TF missing."""
    global _classifier, _classifier_failed
    if _classifier is not None or _classifier_failed:
        return _classifier
    try:
        from minimappr.classifiers.yamnet import YAMNetClassifier

        _classifier = YAMNetClassifier(keep_embeddings=True)
    except Exception as exc:  # tensorflow/hub unavailable or model load failure
        _classifier_failed = True
        logger.warning("YAMNet embedding extraction unavailable: %s", exc)
    return _classifier


def _atomic_save_npy(destination: Path, array: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as output:
            np.save(output, array)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _extract(wav_path: Path, out_path: Path) -> Optional[dict]:
    classifier = _get_classifier()
    if classifier is None:
        return None
    from minimappr.utils.audio import read_wav_mono

    samples, sample_rate_hz = read_wav_mono(wav_path)
    result = classifier.classify(samples, sample_rate_hz)
    embedding = result.features.get("embedding")
    if embedding is None:
        return None
    embedding = np.asarray(embedding, dtype=np.float32)
    _atomic_save_npy(out_path, embedding)
    return {
        "path": str(out_path),
        "model": result.features.get("embedding_model", EMBEDDING_MODEL),
        "dim": int(embedding.shape[0]),
        "pooling": EMBEDDING_POOLING,
    }


async def extract_embedding_npy(wav_path: Path, out_path: Path) -> Optional[dict]:
    """Extract a mean-pooled YAMNet embedding from a WAV into a .npy file.

    Returns an embedding-info dict for the training manifest, or None when
    extraction is unavailable/failed (promotion must not be blocked).
    """
    try:
        return await asyncio.to_thread(_extract, Path(wav_path), Path(out_path))
    except Exception as exc:  # noqa: BLE001 — never fail the promotion path
        logger.warning("embedding extraction failed for %s: %s", wav_path, exc)
        return None
