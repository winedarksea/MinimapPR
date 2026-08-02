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

# A `run_arrays()` call costs ~1.0 s of fixed producer/worker/analyzer barrier
# synchronization regardless of payload — measured 1048 ms for 0.25 s of audio
# and 1349 ms for 30 s. The marginal cost is only ~10 ms per second of audio, so
# the number of *calls* is what dominates, not their size. Coalescing concurrent
# callers into one call is worth 3-4x:
#
#   clips/call (30 s each)   1      2      4      8     16     32     64
#   ms per clip           1344    824    561    442    383    351    333
#   payload MiB              6     12     23     46     92    184    369
#
# Returns flatten past ~16 while the payload keeps growing linearly, and every
# byte is serialized to a worker subprocess — 64x30 s is 369 MB in flight, which
# a Pi-class box cannot absorb. 16 captures 3.5x of the available 4.0x.
_DEFAULT_MAX_BATCH_SIZE = 16
# The `input` column that attributes a result row back to its clip is uint8.
_MAX_ADDRESSABLE_BATCH = 256


class _BatchRequest:
    """One caller's clip waiting to ride along in a coalesced BirdNET call."""

    __slots__ = ("audio", "done", "result", "error")

    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio
        self.done = threading.Event()
        self.result: ClassificationResult | None = None
        self.error: BaseException | None = None

    def settle(
        self,
        *,
        result: ClassificationResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.done.set()


class BirdNETClassifier(AudioClassifier):
    """Wraps the BirdNET V2.4 TFLite model for bird species identification.

    Intended as a downstream ChainStage triggered when a base classifier
    (e.g. YAMNet) returns a bird-related label, providing species-level
    resolution beyond the coarse "bird_like" category.

    Requires the ``birdnet`` package (``pip install birdnet``).
    The model files are downloaded on first instantiation (~125 MB).
    """

    # Class-level defaults so instances built via ``__new__`` with hand-set
    # attributes — the established pattern for testing this class without the
    # optional birdnet package — behave as un-batched rather than raising
    # AttributeError from classify()/close().
    _batching_enabled: bool = False
    _batch_max_size: int = 1
    _batch_max_wait_seconds: float = 0.0
    _batch_arriving: int = 0
    _batch_thread: threading.Thread | None = None
    batches_dispatched: int = 0
    clips_batched: int = 0

    def __init__(
        self,
        min_confidence: float = 0.1,
        *,
        pool_size: int = 1,
        latitude: float | None = None,
        longitude: float | None = None,
        geo_min_confidence: float = 0.03,
        batch_max_size: int = _DEFAULT_MAX_BATCH_SIZE,
        batch_max_wait_seconds: float = 0.0,
        overlap_duration_s: float = 0.0,
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
                overlap_duration_s=overlap_duration_s,
            )
            self._session_ctxs.append(ctx)
            session = ctx.__enter__()
            self._all_sessions.append(session)
            self._session_pool.put(session)
        # Only this instance's worker processes — never touch siblings' children.
        self._child_procs: list[Any] = [
            proc for proc in multiprocessing.active_children() if proc not in children_before
        ]

        # Request coalescing. `batch_max_wait_seconds <= 0` disables it entirely
        # and every call goes straight through, preserving the original path.
        self._batch_max_size = max(1, min(int(batch_max_size), _MAX_ADDRESSABLE_BATCH))
        self._batch_max_wait_seconds = max(0.0, float(batch_max_wait_seconds))
        self._batching_enabled = self._batch_max_wait_seconds > 0.0
        self._batch_lock = threading.Lock()
        self._batch_pending: list[_BatchRequest] = []
        # Callers that have entered classify() but not yet appended their
        # request — see _batch_loop for why the window keys off this.
        self._batch_arriving = 0
        self._batch_ready = threading.Condition(self._batch_lock)
        self._batch_thread: threading.Thread | None = None
        self.batches_dispatched = 0
        self.clips_batched = 0
        if self._batching_enabled:
            self._batch_thread = threading.Thread(
                target=self._batch_loop,
                name="birdnet-batcher",
                daemon=True,
            )
            self._batch_thread.start()

    @property
    def mean_batch_size(self) -> float:
        """Clips per dispatched call — the whole point of the coalescer."""
        if self.batches_dispatched == 0:
            return 0.0
        return self.clips_batched / self.batches_dispatched

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        if self._closed:
            raise RuntimeError("BirdNETClassifier has been closed")

        if not self._batching_enabled:
            return self._run_single(self._to_model_rate(samples, sample_rate_hz))

        # Register as an arriving caller *before* resampling so the batcher
        # holds its window open for us; _classify_batched clears the
        # registration as it appends the request.
        with self._batch_ready:
            self._batch_arriving += 1
        try:
            audio = self._to_model_rate(samples, sample_rate_hz)
        except BaseException:
            with self._batch_ready:
                self._batch_arriving -= 1
                self._batch_ready.notify()
            raise
        return self._classify_batched(audio)

    def _to_model_rate(self, samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
        audio = samples.astype(np.float32)
        if sample_rate_hz != _BIRDNET_SAMPLE_RATE_HZ:
            audio = resample_poly(
                audio,
                up=_BIRDNET_SAMPLE_RATE_HZ,
                down=sample_rate_hz,
            ).astype(np.float32)
        return audio

    def _classify_batched(self, audio: np.ndarray) -> ClassificationResult:
        """Hand the clip to the batcher and block until its result comes back.

        Resampling already happened on the caller's thread, so the batch worker
        only pays for the shared inference call.
        """
        request = _BatchRequest(audio)
        with self._batch_ready:
            self._batch_pending.append(request)
            self._batch_arriving -= 1
            self._batch_ready.notify()
        # No timeout here on purpose: the classification stage already bounds
        # this via asyncio.wait_for, and close() settles every pending request.
        request.done.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:  # pragma: no cover - settle() always sets one
            raise RuntimeError("BirdNET batch produced no result")
        return request.result

    def _batch_loop(self) -> None:
        """Collect concurrent callers into one `run_arrays()` call.

        A batch closes as soon as it is full, *or* no further caller is still on
        its way in, *or* the collection window expires — whichever comes first.

        That middle condition is what keeps the window honest. Concurrency here
        is capped by the number of fusion classification workers, not by the
        depth of the classification queue: a worker calls classify() for one
        item at a time, so at most `workers` requests can ever be pending at
        once. With the deployed 2 workers against a `batch_max_size` of 16 the
        batch can never fill, and keying the close purely off `batch_max_size`
        made every dispatch sit out the entire window for a batch of 1-2 —
        pure added latency on the pipeline's tightest stage, which is the
        opposite of the point. Waiting only pays when a caller has entered
        classify() and has not appended yet; once `_batch_arriving` hits zero
        nothing more can arrive until this dispatch frees a worker, so we go.
        """
        while not self._closed:
            with self._batch_ready:
                while not self._batch_pending and not self._closed:
                    self._batch_ready.wait(timeout=0.5)
                if self._closed:
                    break
                # First request opens the window; keep collecting until the batch
                # is full or the window closes.
                window_closes_at = time.monotonic() + self._batch_max_wait_seconds
                while (
                    len(self._batch_pending) < self._batch_max_size
                    and self._batch_arriving > 0
                    and not self._closed
                ):
                    remaining = window_closes_at - time.monotonic()
                    if remaining <= 0:
                        break
                    self._batch_ready.wait(timeout=remaining)
                batch = self._batch_pending[: self._batch_max_size]
                del self._batch_pending[: len(batch)]
            if not batch:
                continue
            self._dispatch_batch(batch)

        # Drain on shutdown so no caller is left blocked forever.
        with self._batch_ready:
            pending, self._batch_pending = self._batch_pending, []
        for request in pending:
            request.settle(error=RuntimeError("BirdNETClassifier has been closed"))

    def _dispatch_batch(self, batch: list[_BatchRequest]) -> None:
        try:
            session = self._acquire_session()
        except BaseException as exc:  # noqa: BLE001 - propagate to every waiter
            for request in batch:
                request.settle(error=exc)
            return
        try:
            with self._session_lock:
                self._inflight_sessions.add(session)
            try:
                result = session.run_arrays(
                    [(request.audio, _BIRDNET_SAMPLE_RATE_HZ) for request in batch]
                )
            finally:
                with self._session_lock:
                    self._inflight_sessions.discard(session)
                if not self._closed:
                    self._session_pool.put(session)
            self.batches_dispatched += 1
            self.clips_batched += len(batch)
            rows_by_input = self._rows_by_input(result, len(batch))
            for index, request in enumerate(batch):
                request.settle(result=self._build_result(rows_by_input[index]))
        except BaseException as exc:  # noqa: BLE001 - one bad batch must not hang callers
            logger.warning("BirdNET batch of %d failed: %s", len(batch), exc)
            for request in batch:
                if not request.done.is_set():
                    request.settle(error=exc)

    @staticmethod
    def _rows_by_input(result: Any, batch_size: int) -> list[list[Any]]:
        """Split a batched structured array back out per input clip.

        ``to_structured_array()`` returns one flat table for the whole call; the
        ``input`` column is the index of the clip each row came from.
        """
        grouped: list[list[Any]] = [[] for _ in range(batch_size)]
        detections = result.to_structured_array()
        for row in detections:
            index = int(row["input"])
            if 0 <= index < batch_size:
                grouped[index].append(row)
        return grouped

    def _run_single(self, audio: np.ndarray) -> ClassificationResult:
        session = self._acquire_session()
        with self._session_lock:
            self._inflight_sessions.add(session)
        try:
            result = session.run_arrays([(audio, _BIRDNET_SAMPLE_RATE_HZ)])
        finally:
            with self._session_lock:
                self._inflight_sessions.discard(session)
            if not self._closed:
                self._session_pool.put(session)
        return self._build_result(list(result.to_structured_array()))

    def _acquire_session(self) -> Any:
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
            return session

    def _build_result(self, detections: list[Any]) -> ClassificationResult:
        """Turn one clip's result rows into a ClassificationResult.

        Rows carry fields: input, start_time, end_time, species_name, confidence
        — above-threshold only. Shared by the single and batched paths so both
        produce byte-identical results.
        """
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

        # Wake the batch collector and settle everything already queued, or those
        # callers block forever on their _BatchRequest event. Absent entirely on
        # __new__-built instances that never enabled batching.
        batch_ready = getattr(self, "_batch_ready", None)
        if batch_ready is not None:
            with batch_ready:
                pending, self._batch_pending = self._batch_pending, []
                batch_ready.notify_all()
            for request in pending:
                request.settle(error=RuntimeError("BirdNETClassifier has been closed"))

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

        # BirdNET's session __exit__ calls Process.join() with no timeout. Its
        # idle workers wait on a start signal and do not observe session.end(),
        # so calling __exit__ first can strand the shutdown thread forever.
        # Kill our workers before that unbounded join; they are no longer
        # usable after cancellation and the join then returns immediately.
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

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self._session_lock:
                if not self._inflight_sessions:
                    break
            time.sleep(0.05)

        # Context exit releases shared memory and BirdNET's daemon helper
        # threads. It is safe only after the owned worker processes above have
        # exited because its internal joins have no timeout.
        for ctx in self._session_ctxs:
            try:
                ctx.__exit__(None, None, None)
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
