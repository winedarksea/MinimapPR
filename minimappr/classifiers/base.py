"""Classifier interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from minimappr.models import ClassificationResult


class AudioClassifier(ABC):
    @abstractmethod
    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources held by this classifier (e.g. subprocess pools).

        The default implementation is a no-op.  Subclasses that own background
        processes or threads must override this to terminate them cleanly.
        """
        return

    def cancel_pending(self) -> None:
        """Best-effort cancellation for in-flight classify() work.

        Classification orchestration may timeout a classify() call while the
        backend worker is still blocked on model inference or IPC. Backends
        that can interrupt in-flight work should override this hook.
        """
        return


class EmbeddingClassifier(AudioClassifier):
    """Classifier consuming embedding frames instead of raw audio.

    Used by chain stages declared with ``input: "embedding"`` in the routing
    config — they read the parent classifier's in-memory embedding frames
    (e.g. YAMNet ``[N, 1024]``) rather than re-running a frontend on audio.
    """

    @abstractmethod
    def classify_embedding(self, frames: np.ndarray) -> ClassificationResult:
        """Classify ``[embedding_dim]`` or ``[n_frames, embedding_dim]`` frames."""
        raise NotImplementedError

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        raise NotImplementedError(
            f"{type(self).__name__} is embedding-only; route it as a chain stage "
            'with input: "embedding"'
        )
