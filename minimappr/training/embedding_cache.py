"""YAMNet embedding computation + on-disk cache for the drone-head trainer.

:class:`YamnetEmbedder` loads tfhub YAMNet once and turns waveforms into
``[n_frames, 1024]`` embeddings, applying the *identical* runtime conditioning
(:func:`prepare_waveform_for_yamnet`) so train and serve see the same input.

:class:`EmbeddingCache` memoizes embeddings to ``.npy`` files keyed by content
hash + preprocessing version + segment + variant, so re-running the trainer
(warm cache) skips the expensive YAMNet forward pass. Augmented variants are
cached too — they are deterministic per seed, and the forward pass is the cost
being avoided.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import resample_poly

from minimappr.classifiers.yamnet import (
    YAMNET_PREPROCESS_VERSION,
    prepare_waveform_for_yamnet,
)
from minimappr.audio_processing.profiles import AudioProcessingProfile

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
EMBEDDING_DIM = 1024
# Chunk long waveforms so a single YAMNet forward never holds an unbounded
# spectrogram in memory. ~60 s at 16 kHz.
_CHUNK_SAMPLES = 60 * SAMPLE_RATE


class YamnetEmbedder:
    """Wraps tfhub YAMNet embedding extraction with runtime-parity preprocessing."""

    def __init__(self, preprocess_profile: AudioProcessingProfile | None = None) -> None:
        try:
            import tensorflow_hub as hub  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("YAMNet embedder requires tensorflow and tensorflow-hub") from exc
        self._model = hub.load("https://tfhub.dev/google/yamnet/1")
        self._preprocess_profile = preprocess_profile

    def embed(self, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Return ``[n_frames, 1024]`` float32 embeddings for ``waveform``."""
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE and waveform.size:
            waveform = resample_poly(waveform, up=SAMPLE_RATE, down=sample_rate).astype(np.float32)
        waveform = prepare_waveform_for_yamnet(
            waveform, preprocess_profile=self._preprocess_profile
        )
        if waveform.size == 0:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        chunks: list[np.ndarray] = []
        for start in range(0, waveform.size, _CHUNK_SAMPLES):
            chunk = waveform[start : start + _CHUNK_SAMPLES]
            _, embeddings, _ = self._model(chunk)
            chunks.append(np.asarray(embeddings.numpy(), dtype=np.float32))
        if not chunks:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return np.concatenate(chunks, axis=0)


def file_content_hash(path: Path) -> str:
    """Return the first 16 hex chars of the sha256 of file bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


class EmbeddingCache:
    """Content-addressed ``.npy`` embedding cache with atomic writes."""

    def __init__(self, cache_dir: Path, prep_version: str = YAMNET_PREPROCESS_VERSION) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._prep_version = prep_version
        self.hits = 0
        self.misses = 0

    def make_key(
        self,
        content_hash: str,
        *,
        offset_ms: int = 0,
        dur_ms: int = 0,
        variant: str = "orig",
    ) -> str:
        """Build the cache key for a segment/variant of a content-hashed source."""
        return f"{content_hash}-{self._prep_version}-{offset_ms}-{dur_ms}-{variant}"

    def get_or_compute(self, key: str, fn: Callable[[], np.ndarray]) -> np.ndarray:
        """Return cached embeddings for ``key`` or compute + persist them."""
        path = self._cache_dir / f"{key}.npy"
        if path.exists():
            try:
                arr = np.load(path)
                self.hits += 1
                return np.asarray(arr, dtype=np.float32)
            except (OSError, ValueError) as exc:
                logger.warning("Corrupt cache entry %s (%s); recomputing", path, exc)
        arr = np.asarray(fn(), dtype=np.float32)
        self.misses += 1
        self._atomic_save(path, arr)
        return arr

    def _atomic_save(self, path: Path, arr: np.ndarray) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp.open("wb") as f:
                np.save(f, arr)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


__all__ = ["YamnetEmbedder", "EmbeddingCache", "file_content_hash", "EMBEDDING_DIM"]
