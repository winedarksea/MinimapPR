"""BirdNET+ V3.0 species classifier on ONNX Runtime (``birdnet_model_version="3.0"``).

Model: BirdNET+ V3.0.1 Global 12K ONNX (Zenodo record 24362980, Apache 2.0),
fetched by ``scripts/fetch_birdnet_v3.py``. One graph from raw 32 kHz float32
samples (``input [batch, samples]``, variable length) to per-class
probabilities (``predictions [batch, 11725]``, sigmoid already applied in the
graph) plus ``embeddings [batch, 1280]``.

Results use the same shape as the v2.4 wrapper so the existing BirdNET result
path (composite routing, taxonomy, rules, retention) needs no changes: the
label is the lower-cased English common name, ``scores`` holds the top-k, and
``features["raw_species"]`` is ``Scientific_Common``. ``features["model"]`` is
``birdnet_v3``; ``features["taxon_group"]`` lets the taxonomy fallback treat
only animal classes as wildlife (V3 also has Gun, Siren, Engine, ...).

Threading: ``classify`` is synchronous and CPU-bound; the fusion stage already
calls classifiers from worker threads (never on the asyncio loop). One ORT
session is shared - ``InferenceSession.run`` is thread-safe - and
``num_threads`` bounds its intra-op pool so concurrent workers keep cores.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from minimappr.classifiers.base import AudioClassifier
from minimappr.classifiers.birdnet_v3_audio import (
    BirdNETV3WindowPolicy,
    prepare_birdnet_v3_windows,
)
from minimappr.classifiers.birdnet_v3_labels import (
    BirdNETV3Label,
    load_birdnet_v3_labels,
    load_regional_species_names,
    resolve_allowed_label_indices,
)
from minimappr.models import ClassificationResult

logger = logging.getLogger(__name__)

BIRDNET_V3_MODEL_NAME = "birdnet_v3"
BIRDNET_V3_MODEL_RELEASE = "3.0.1"
_PREDICTIONS_OUTPUT_NAME = "predictions"
_DEFAULT_SCORES_TOP_K = 5

# (model_path, num_threads) -> object with ``run(output_names, feeds)`` and
# ``get_inputs()``. Injected by tests so no model file is needed.
SessionFactory = Callable[[str, int], Any]


def create_onnxruntime_session(model_path: str, num_threads: int) -> Any:
    import onnxruntime as ort  # noqa: PLC0415 - heavy import, only when V3 is selected

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, int(num_threads))
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])


class BirdNETV3Classifier(AudioClassifier):
    """BirdNET+ V3.0 wrapper with variable windows, silence guard and regional filter."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        labels_path: str | Path,
        min_confidence: float = 0.1,
        window_policy: BirdNETV3WindowPolicy | None = None,
        species_list_path: str | Path | None = None,
        num_threads: int = 4,
        top_k: int = _DEFAULT_SCORES_TOP_K,
        session_factory: SessionFactory = create_onnxruntime_session,
    ) -> None:
        model_path = Path(model_path)
        labels_path = Path(labels_path)
        for required in (model_path, labels_path):
            if not required.is_file():
                raise FileNotFoundError(
                    f"BirdNET V3 file missing: {required} (run scripts/fetch_birdnet_v3.py)"
                )
        self._labels: list[BirdNETV3Label] = load_birdnet_v3_labels(labels_path)
        self._min_confidence = float(min_confidence)
        self._top_k = max(1, int(top_k))
        self._window_policy = window_policy or BirdNETV3WindowPolicy()
        allowed_names = (
            load_regional_species_names(species_list_path) if species_list_path else None
        )
        self._allowed_indices = resolve_allowed_label_indices(self._labels, allowed_names)
        self._model_precision = _precision_from_filename(model_path)
        self._session = session_factory(str(model_path), int(num_threads))
        self._input_name = self._session.get_inputs()[0].name
        self._closed = False
        self.windows_scored = 0
        self.windows_with_nonfinite_output = 0
        self.clips_rejected_by_silence_guard = 0
        logger.info(
            "BirdNET V3 loaded: %s (%d labels, %s species filter, precision %s)",
            model_path.name,
            len(self._labels),
            "regional" if self._allowed_indices is not None else "no",
            self._model_precision,
        )

    @property
    def labels(self) -> list[BirdNETV3Label]:
        return self._labels

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        if self._closed:
            raise RuntimeError("BirdNETV3Classifier has been closed")
        windows, verdict = prepare_birdnet_v3_windows(samples, sample_rate_hz, self._window_policy)
        if not verdict.accepted or not windows:
            self.clips_rejected_by_silence_guard += 1
            return self._unknown_result(silence_guard=verdict.reason or "no_scorable_window")
        probabilities = self.predict_window_probabilities([audio for _, _, audio in windows])
        finite_rows = np.all(np.isfinite(probabilities), axis=1)
        self.windows_scored += len(windows)
        self.windows_with_nonfinite_output += int(np.count_nonzero(~finite_rows))
        if not np.any(finite_rows):
            return self._unknown_result(silence_guard="nonfinite_model_output")
        # Max over windows: a call scored strongly in one window should not be
        # averaged away by the quiet windows around it (same as the v2.4 path).
        clip_probabilities = np.max(probabilities[finite_rows], axis=0)
        return self._build_result(clip_probabilities, window_count=int(np.count_nonzero(finite_rows)))

    def predict_window_probabilities(self, window_audio: list[np.ndarray]) -> np.ndarray:
        """``[n_windows, n_labels]`` probabilities; equal-length windows share one run."""
        rows: list[np.ndarray] = []
        lengths = {audio.size for audio in window_audio}
        groups = (
            [np.stack(window_audio)]
            if len(lengths) == 1
            else [audio[np.newaxis, :] for audio in window_audio]
        )
        for batch in groups:
            output = self._session.run(
                [_PREDICTIONS_OUTPUT_NAME], {self._input_name: batch.astype(np.float32, copy=False)}
            )[0]
            rows.append(np.asarray(output, dtype=np.float64).reshape(batch.shape[0], -1))
        probabilities = np.concatenate(rows, axis=0)
        if probabilities.shape[1] != len(self._labels):
            raise RuntimeError(
                f"BirdNET V3 model emits {probabilities.shape[1]} classes but labels list {len(self._labels)}"
            )
        return probabilities

    def _build_result(self, clip_probabilities: np.ndarray, *, window_count: int) -> ClassificationResult:
        candidate_indices = (
            self._allowed_indices
            if self._allowed_indices is not None
            else np.arange(clip_probabilities.size)
        )
        candidate_scores = clip_probabilities[candidate_indices]
        top_count = min(self._top_k, candidate_scores.size)
        top_positions = np.argpartition(candidate_scores, -top_count)[-top_count:]
        top_positions = top_positions[np.argsort(candidate_scores[top_positions])[::-1]]
        ranked = [
            (self._labels[int(candidate_indices[position])], float(candidate_scores[position]))
            for position in top_positions
            if float(candidate_scores[position]) >= self._min_confidence
        ]
        if not ranked:
            return self._unknown_result(window_count=window_count)
        top_label, top_confidence = ranked[0]
        scores: dict[str, float] = {}
        for label, confidence in ranked:
            scores[label.pipeline_label] = max(scores.get(label.pipeline_label, 0.0), confidence)
        return ClassificationResult(
            label=top_label.pipeline_label,
            confidence=max(0.0, min(1.0, top_confidence)),
            scores=scores,
            features={
                **self._provenance_features(),
                "raw_species": top_label.raw_species,
                "scientific_name": top_label.scientific_name,
                "birdnet_id": top_label.birdnet_id,
                "taxon_group": top_label.taxon_group,
                "window_count": window_count,
            },
        )

    def _unknown_result(self, **extra_features: Any) -> ClassificationResult:
        return ClassificationResult(
            label="unknown",
            confidence=0.0,
            scores={},
            features={**self._provenance_features(), **extra_features},
        )

    def _provenance_features(self) -> dict[str, Any]:
        return {
            "model": BIRDNET_V3_MODEL_NAME,
            "model_release": BIRDNET_V3_MODEL_RELEASE,
            "model_precision": self._model_precision,
        }

    def close(self) -> None:
        self._closed = True
        self._session = None


def _precision_from_filename(model_path: Path) -> str:
    name = model_path.name.lower()
    for marker in ("int8", "fp16", "fp32"):
        if marker in name:
            return marker + ("_pruned" if "pruned" in name else "")
    return "unknown"
