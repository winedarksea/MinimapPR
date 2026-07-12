"""Moonshine (ONNX) speech-to-text transcriber.

Not an :class:`AudioClassifier` — this produces text, not a label/score
distribution, so it is driven directly by :class:`SpeechCaptureManager`
rather than through the routing/chaining machinery.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

_TARGET_SAMPLE_RATE_HZ = 16_000


class MoonshineUnavailableError(RuntimeError):
    pass


class MoonshineTranscriber:
    def __init__(self, model_name: str = "moonshine/base") -> None:
        try:
            import moonshine_onnx
        except ImportError as exc:  # pragma: no cover - exercised via availability tests
            raise MoonshineUnavailableError(
                "moonshine_onnx is not installed; install the 'stt' extra "
                "(pip install useful-moonshine-onnx) to enable speech transcription"
            ) from exc
        self._moonshine_onnx = moonshine_onnx
        self._model_name = model_name

    def transcribe(self, samples: np.ndarray, sample_rate_hz: int) -> str:
        waveform = samples.astype(np.float32, copy=False)
        if sample_rate_hz != _TARGET_SAMPLE_RATE_HZ and waveform.size > 0:
            waveform = resample_poly(
                waveform, up=_TARGET_SAMPLE_RATE_HZ, down=sample_rate_hz
            ).astype(np.float32)
        if waveform.size == 0:
            return ""
        tokens = self._moonshine_onnx.transcribe(waveform, self._model_name)
        if isinstance(tokens, (list, tuple)):
            text = " ".join(str(t) for t in tokens).strip()
        else:
            text = str(tokens).strip()
        return text


__all__ = ["MoonshineTranscriber", "MoonshineUnavailableError"]
