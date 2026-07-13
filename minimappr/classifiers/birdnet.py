"""Optional BirdNET V2.4 species classifier wrapper (birdnet >= 0.2.12)."""

from __future__ import annotations

import logging
import multiprocessing
import queue
import threading
import time
from typing import Any

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

    def __init__(
        self,
        min_confidence: float = 0.1,
        *,
        pool_size: int = 1,
        latitude: float | None = None,
        longitude: float | None = None,
        geo_min_confidence: float = 0.03,
    ) -> None:
        try:
            from birdnet import model_loader
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "BirdNET backend requires the birdnet package: pip install birdnet"
            ) from exc

        self._min_confidence = min_confidence
        self._custom_species_list = _predict_site_species_list(
            model_loader=model_loader,
            latitude=latitude,
            longitude=longitude,
            geo_min_confidence=geo_min_confidence,
        )
        self._model = model_loader.load("acoustic", "2.4", "tf", library="tflite")
        # Tracks in-flight prediction sessions so close() can cancel them from
        # another thread during server shutdown without waiting for subprocess I/O.
        self._session_lock: threading.Lock = threading.Lock()
        self._inflight_sessions: set[Any] = set()
        self._closed: bool = False

        # Pre-create a pool of predict_sessions so the TFLite model worker
        # subprocess is only spawned once, amortizing ~125 MB load overhead
        # across all classify() calls.
        self._session_ctxs: list[Any] = []
        self._all_sessions: list[Any] = []
        self._session_pool: queue.Queue[Any] = queue.Queue()
        children_before = set(multiprocessing.active_children())
        for _ in range(pool_size):
            ctx = self._model.predict_session(
                top_k=_SCORES_MAP_TOP_K,
                default_confidence_threshold=self._min_confidence,
                apply_sigmoid=True,
                n_workers=1,
                custom_species_list=self._custom_species_list,
            )
            self._session_ctxs.append(ctx)
            session = ctx.__enter__()
            self._all_sessions.append(session)
            self._session_pool.put(session)
        # Only this instance's worker processes — never touch siblings' children.
        self._child_procs: list[Any] = [
            proc for proc in multiprocessing.active_children() if proc not in children_before
        ]

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        if self._closed:
            raise RuntimeError("BirdNETClassifier has been closed")

        audio = samples.astype(np.float32)
        if sample_rate_hz != _BIRDNET_SAMPLE_RATE_HZ:
            audio = resample_poly(
                audio,
                up=_BIRDNET_SAMPLE_RATE_HZ,
                down=sample_rate_hz,
            ).astype(np.float32)

        # Acquire a pooled session with a short-poll loop (bounded by an overall
        # 120s deadline) so close() can unblock us promptly via a None sentinel
        # instead of us waiting out a single long queue.get().
        deadline = time.monotonic() + 120.0
        session = None
        while True:
            if self._closed:
                raise RuntimeError("BirdNETClassifier has been closed")
            try:
                session = self._session_pool.get(timeout=0.5)
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise RuntimeError("BirdNETClassifier: no session available in pool")
                continue
            if session is None:
                raise RuntimeError("BirdNETClassifier has been closed")
            break

        with self._session_lock:
            self._inflight_sessions.add(session)
        try:
            result = session.run_arrays([(audio, _BIRDNET_SAMPLE_RATE_HZ)])
        finally:
            with self._session_lock:
                self._inflight_sessions.discard(session)
            if not self._closed:
                self._session_pool.put(session)

        # to_structured_array() yields rows with fields: species_name, confidence,
        # start_time, end_time — sorted by confidence descending, above-threshold only.
        detections = result.to_structured_array()  # type: ignore[union-attr]

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

        scores_map: dict[str, float] = {}
        for sp, conf in sorted_species[:_SCORES_MAP_TOP_K]:
            common = _extract_common_name(sp)
            scores_map[common] = max(scores_map.get(common, 0.0), conf)

        return ClassificationResult(
            label=label,
            confidence=max(0.0, min(1.0, top_conf)),
            scores=scores_map,
            features={"model": "birdnet_v2m4", "raw_species": top_raw_label},
        )

    def close(self) -> None:
        """Cancel any in-flight prediction and terminate BirdNET worker subprocesses.

        Idempotent. BirdNET spawns multiprocessing workers that load TensorFlow
        models. On SIGINT those workers can die without signalling the result
        queue, leaving a Consumer thread blocked forever in ``run_arrays`` or a
        thread parked in the session-pool ``get()``. This sequence wakes every
        such waiter before touching subprocesses: pool sentinels unblock
        ``classify()``'s pool-get loop, then ``cancel()`` unblocks any
        ``run_arrays`` already in flight (~1s per birdnet's Consumer poll), then
        only *this instance's* worker processes (``_child_procs``) are
        terminated/joined — never the process-global ``active_children()``,
        which would kill siblings (e.g. a second BirdNETClassifier instance).
        """
        if self._closed:
            return
        self._closed = True

        # Wake any thread parked in classify()'s session_pool.get() immediately.
        for _ in self._session_ctxs:
            self._session_pool.put(None)

        # Cancel every session — pooled (idle) or in-flight — to unblock any
        # run_arrays() call within ~1s, and to guard against a thread that
        # grabs a real idle session from the queue in the brief race window
        # right after the None sentinels above are enqueued.
        for session in self._all_sessions:
            try:
                session.cancel()
            except Exception:  # noqa: BLE001
                pass

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self._session_lock:
                if not self._inflight_sessions:
                    break
            time.sleep(0.05)

        # Exit all pooled session context managers to cleanly shut down their
        # underlying predict worker subprocesses. Safe now that workers are
        # dead/cancelled: ctx.__exit__ joins with no timeout then unlinks shm.
        for ctx in self._session_ctxs:
            try:
                ctx.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

        # Terminate-then-join only this instance's own worker processes.
        for proc in self._child_procs:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        for proc in self._child_procs:
            try:
                proc.join(timeout=1.0)
            except Exception:  # noqa: BLE001
                pass

    def cancel_pending(self) -> None:
        """Best-effort cancellation of all currently running BirdNET sessions."""
        with self._session_lock:
            sessions = set(self._inflight_sessions)
        for session in sessions:
            try:
                session.cancel()
            except Exception:  # noqa: BLE001
                pass


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


def _predict_site_species_list(
    *,
    model_loader: Any,
    latitude: float | None,
    longitude: float | None,
    geo_min_confidence: float,
) -> list[str] | None:
    if latitude is None or longitude is None:
        return None

    try:
        geo_model = model_loader.load("geo", "2.4", "tf", library="tflite")
        geo_result = geo_model.predict(
            float(latitude),
            float(longitude),
            week=None,
            min_confidence=float(geo_min_confidence),
        )
        species_list = [str(species) for species in geo_result.to_set()]
    except Exception as exc:  # pragma: no cover - optional refinement
        logger.warning(
            "BirdNET geo filter unavailable for lat=%s lon=%s (%s); using unconstrained species list.",
            latitude,
            longitude,
            exc,
        )
        return None

    if not species_list:
        logger.warning(
            "BirdNET geo filter returned no species for lat=%s lon=%s threshold=%.3f; using unconstrained species list.",
            latitude,
            longitude,
            geo_min_confidence,
        )
        return None
    return species_list
