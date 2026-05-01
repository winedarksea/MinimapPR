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
