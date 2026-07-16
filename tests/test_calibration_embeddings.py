"""YAMNet embedding extraction tests (stubbed model + optional TF integration)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

import minimappr.calibration.embeddings as embeddings_module
from minimappr.calibration.embeddings import extract_embedding_npy
from minimappr.models import ClassificationResult
from minimappr.utils.audio import write_wav_mono


class _StubEmbeddingClassifier:
    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        return ClassificationResult(
            label="drone",
            confidence=0.8,
            scores={},
            features={
                "model": "yamnet",
                "embedding": np.linspace(0.0, 1.0, 1024, dtype=np.float32),
                "embedding_model": "yamnet/1",
                "embedding_dim": 1024,
            },
        )


@pytest.fixture()
def _wav(tmp_path: Path) -> Path:
    path = tmp_path / "snippet.wav"
    samples = np.sin(np.linspace(0, 200 * np.pi, 16_000)).astype(np.float32) * 0.3
    write_wav_mono(path, samples, 16_000)
    return path


def test_extract_embedding_with_stub(monkeypatch, tmp_path: Path, _wav: Path) -> None:
    monkeypatch.setattr(embeddings_module, "_get_classifier", lambda: _StubEmbeddingClassifier())
    out_path = tmp_path / "det-1.npy"
    info = asyncio.run(extract_embedding_npy(_wav, out_path))
    assert info == {
        "path": str(out_path),
        "model": "yamnet/1",
        "dim": 1024,
        "pooling": "mean",
    }
    loaded = np.load(out_path)
    assert loaded.shape == (1024,)
    assert loaded.dtype == np.float32


def test_extract_embedding_graceful_without_tf(monkeypatch, tmp_path: Path, _wav: Path) -> None:
    monkeypatch.setattr(embeddings_module, "_get_classifier", lambda: None)
    info = asyncio.run(extract_embedding_npy(_wav, tmp_path / "det-2.npy"))
    assert info is None
    assert not (tmp_path / "det-2.npy").exists()


def test_extract_embedding_real_yamnet(tmp_path: Path, _wav: Path) -> None:
    pytest.importorskip("tensorflow")
    from minimappr.classifiers.yamnet import YAMNetClassifier

    try:
        classifier = YAMNetClassifier(keep_embeddings=True)
    except RuntimeError as exc:
        pytest.skip(f"YAMNet unavailable: {exc}")
    result = classifier.classify(
        np.sin(np.linspace(0, 400 * np.pi, 32_000)).astype(np.float32) * 0.3, 16_000
    )
    embedding = result.features["embedding"]
    assert embedding.shape == (1024,)
    assert result.features["embedding_dim"] == 1024
