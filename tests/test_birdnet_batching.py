"""BirdNET request coalescing.

A `run_arrays()` call costs ~1.0 s of fixed barrier synchronization regardless
of payload (measured: 1048 ms for 0.25 s of audio, 1349 ms for 30 s), so the
number of calls dominates classification cost. These tests use a fake session so
the batching machinery is verified without the ~125 MB model.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from minimappr.classifiers.birdnet import BirdNETClassifier
from minimappr.config import Settings


class _FakeResult:
    """Mimics AcousticDataPredictionResult.to_structured_array()."""

    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size

    def to_structured_array(self):
        rows = []
        for index in range(self._batch_size):
            rows.append(
                {
                    "input": index,
                    "start_time": 0.0,
                    "end_time": 3.0,
                    "species_name": f"Genus species_Species {index}",
                    "confidence": 0.5 + index / 100.0,
                }
            )
            rows.append(
                {
                    "input": index,
                    "start_time": 0.0,
                    "end_time": 3.0,
                    "species_name": f"Other genus_Runner Up {index}",
                    "confidence": 0.1,
                }
            )
        return rows


class _FakeSession:
    """Records the batch sizes it was asked to run."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.calls: list[int] = []
        self._delay_s = delay_s
        self._lock = threading.Lock()

    def run_arrays(self, inputs):
        with self._lock:
            self.calls.append(len(inputs))
        if self._delay_s:
            time.sleep(self._delay_s)
        return _FakeResult(len(inputs))

    def cancel(self) -> None:
        pass


def _classifier(
    session: _FakeSession,
    *,
    batch_max_wait_seconds: float,
    batch_max_size: int = 16,
) -> BirdNETClassifier:
    """Build a BirdNETClassifier around a fake session, bypassing __init__."""
    c = BirdNETClassifier.__new__(BirdNETClassifier)
    import queue as _queue

    c._min_confidence = 0.05
    c._closed = False
    c._session_lock = threading.Lock()
    c._inflight_sessions = set()
    c._session_ctxs = []
    c._all_sessions = [session]
    c._session_pool = _queue.Queue()
    c._session_pool.put(session)
    c._child_procs = []
    c._batch_max_size = max(1, int(batch_max_size))
    c._batch_max_wait_seconds = max(0.0, float(batch_max_wait_seconds))
    c._batching_enabled = c._batch_max_wait_seconds > 0.0
    c._batch_lock = threading.Lock()
    c._batch_pending = []
    c._batch_ready = threading.Condition(c._batch_lock)
    c._batch_thread = None
    c.batches_dispatched = 0
    c.clips_batched = 0
    if c._batching_enabled:
        c._batch_thread = threading.Thread(target=c._batch_loop, daemon=True)
        c._batch_thread.start()
    return c


def _clip(seconds: float = 1.0, sr: int = 48_000) -> np.ndarray:
    return np.zeros(int(sr * seconds), dtype=np.float32)


def _run_concurrently(c: BirdNETClassifier, count: int, sr: int = 48_000) -> list:
    results: list = [None] * count
    clip = _clip()

    def call(i: int) -> None:
        results[i] = c.classify(clip, sr)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    return results


class TestBatching:
    def test_callers_arriving_during_a_dispatch_share_the_next_call(self) -> None:
        """The coalescing that actually happens in production.

        Concurrency into classify() is capped by the fusion classification
        worker count, so a batch is never assembled from a cold burst — it is
        assembled from the callers that pile up while the previous inference
        call is in flight. Those must ride together in one call.
        """
        session = _FakeSession(delay_s=0.5)
        c = _classifier(session, batch_max_wait_seconds=1.0, batch_max_size=8)
        try:
            first: list = [None]
            lead = threading.Thread(target=lambda: first.__setitem__(0, c.classify(_clip(), 48_000)))
            lead.start()
            # Let the lead caller's dispatch get under way, then pile on.
            time.sleep(0.15)
            results = _run_concurrently(c, 5)
            lead.join(timeout=10.0)
        finally:
            c.close()

        assert first[0] is not None
        assert all(r is not None for r in results)
        # The lead clip went alone; the five that queued behind it shared a call.
        assert sum(session.calls) == 6
        assert session.calls == [1, 5]

    def test_lone_caller_does_not_wait_out_the_window(self) -> None:
        """A caller with nobody behind it must dispatch immediately.

        Waiting only pays while another caller is mid-arrival. Sitting out the
        full window for a batch of one is pure added latency on the pipeline's
        tightest stage — the regression this guard exists to catch.
        """
        session = _FakeSession()
        c = _classifier(session, batch_max_wait_seconds=5.0, batch_max_size=16)
        try:
            started = time.monotonic()
            result = c.classify(_clip(), 48_000)
            elapsed = time.monotonic() - started
        finally:
            c.close()

        assert result.label == "species 0"
        assert elapsed < 1.0, f"lone caller sat in the window for {elapsed:.2f}s"

    def test_each_caller_gets_its_own_clips_result(self) -> None:
        """Rows are attributed back per clip via the `input` column."""
        session = _FakeSession(delay_s=0.5)
        c = _classifier(session, batch_max_wait_seconds=1.0, batch_max_size=8)
        try:
            # Park a lead caller in run_arrays so the four below it are
            # guaranteed to be assembled into a single batch.
            lead = threading.Thread(target=lambda: c.classify(_clip(), 48_000))
            lead.start()
            time.sleep(0.15)
            results = _run_concurrently(c, 4)
            lead.join(timeout=10.0)
        finally:
            c.close()

        assert session.calls == [1, 4]
        labels = sorted(r.label for r in results)
        assert labels == ["species 0", "species 1", "species 2", "species 3"]
        by_label = {r.label: r for r in results}
        assert by_label["species 0"].confidence == pytest.approx(0.50)
        assert by_label["species 3"].confidence == pytest.approx(0.53)
        # Each caller keeps its own runner-up too, not the batch's.
        assert "runner up 3" in by_label["species 3"].scores

    def test_batch_size_cap_splits_into_multiple_calls(self) -> None:
        session = _FakeSession(delay_s=0.1)
        c = _classifier(session, batch_max_wait_seconds=1.0, batch_max_size=2)
        try:
            results = _run_concurrently(c, 6)
        finally:
            c.close()

        assert all(r is not None for r in results)
        assert sum(session.calls) == 6
        assert max(session.calls) <= 2
        assert len(session.calls) >= 3

    def test_batching_disabled_runs_straight_through(self) -> None:
        session = _FakeSession()
        c = _classifier(session, batch_max_wait_seconds=0.0)
        try:
            result = c.classify(_clip(), 48_000)
        finally:
            c.close()

        assert result.label == "species 0"
        assert session.calls == [1]
        assert c.batches_dispatched == 0

    def test_a_single_caller_still_returns_within_the_window(self) -> None:
        session = _FakeSession()
        c = _classifier(session, batch_max_wait_seconds=0.3, batch_max_size=16)
        try:
            started = time.monotonic()
            result = c.classify(_clip(), 48_000)
            elapsed = time.monotonic() - started
        finally:
            c.close()

        assert result.label == "species 0"
        # Waits out the window (nothing else arrives) but must not hang.
        assert elapsed < 5.0

    def test_session_failure_propagates_to_every_caller(self) -> None:
        """One bad batch must raise for all its callers, never hang them."""

        class _Boom(_FakeSession):
            def run_arrays(self, inputs):
                raise RuntimeError("worker died")

        session = _Boom()
        c = _classifier(session, batch_max_wait_seconds=0.3, batch_max_size=8)
        errors: list = [None, None]

        def call(i: int) -> None:
            try:
                c.classify(_clip(), 48_000)
            except Exception as exc:  # noqa: BLE001
                errors[i] = exc

        try:
            threads = [threading.Thread(target=call, args=(i,)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)
            assert all(not t.is_alive() for t in threads)
        finally:
            c.close()

        assert all(isinstance(e, RuntimeError) for e in errors)

    def test_close_releases_callers_waiting_on_a_batch(self) -> None:
        """close() must settle pending requests or shutdown deadlocks."""
        session = _FakeSession(delay_s=1.0)
        c = _classifier(session, batch_max_wait_seconds=30.0, batch_max_size=16)
        failures: list = [None]

        def call() -> None:
            try:
                c.classify(_clip(), 48_000)
            except Exception as exc:  # noqa: BLE001
                failures[0] = exc

        # The lead caller occupies the batcher inside run_arrays; the caller we
        # care about then parks in _batch_pending with no dispatch to settle it.
        lead = threading.Thread(target=lambda: c.classify(_clip(), 48_000))
        lead.start()
        time.sleep(0.2)
        thread = threading.Thread(target=call)
        thread.start()
        time.sleep(0.2)  # let it park in the pending queue
        c.close()
        thread.join(timeout=10.0)
        lead.join(timeout=10.0)

        assert not thread.is_alive(), "close() left a caller blocked in the batcher"
        assert isinstance(failures[0], RuntimeError)


class TestBatchSettings:
    def test_defaults(self) -> None:
        settings = Settings()
        assert settings.birdnet_batch_max_wait_seconds == pytest.approx(0.5)
        assert settings.birdnet_batch_max_size == 16

    def test_window_is_clamped_under_the_stage_timeout(self) -> None:
        """Collection happens inside the stage timeout, so a long window would
        starve the inference that follows and time out every batched item.

        Clamped rather than rejected: the default window must stay valid under
        any stage timeout, including the very short ones tests and low-latency
        deployments use.
        """
        clamped = Settings(
            birdnet_batch_max_wait_seconds=30.0,
            classifier_stage_timeout_seconds=30.0,
        )
        assert clamped.birdnet_batch_max_wait_seconds == pytest.approx(15.0)

        # A short stage timeout must not make the *default* window invalid.
        short = Settings(classifier_stage_timeout_seconds=0.01)
        assert short.birdnet_batch_max_wait_seconds == pytest.approx(0.005)

        untouched = Settings(
            birdnet_batch_max_wait_seconds=2.0,
            classifier_stage_timeout_seconds=30.0,
        )
        assert untouched.birdnet_batch_max_wait_seconds == pytest.approx(2.0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("birdnet_batch_max_wait_seconds", -1.0),
            ("birdnet_batch_max_size", 0),
        ],
    )
    def test_invalid_values_rejected(self, field: str, value: float) -> None:
        with pytest.raises(ValueError):
            Settings(**{field: value})

    def test_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("MINIMAPPR_BIRDNET_BATCH_MAX_WAIT_SECONDS", "2.5")
        monkeypatch.setenv("MINIMAPPR_BIRDNET_BATCH_MAX_SIZE", "4")
        settings = Settings.from_env()
        assert settings.birdnet_batch_max_wait_seconds == pytest.approx(2.5)
        assert settings.birdnet_batch_max_size == 4
