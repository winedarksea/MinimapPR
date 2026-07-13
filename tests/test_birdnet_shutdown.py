"""BirdNETClassifier shutdown behavior: close() must unblock classify() promptly.

Constructs BirdNETClassifier via __new__ with hand-set attributes and fake
session/process objects so these tests do not require the optional `birdnet`
package to be installed.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

from minimappr.classifiers.birdnet import BirdNETClassifier


class _FakeResult:
    def to_structured_array(self):
        return []


class _FakeSession:
    def __init__(self, name: str) -> None:
        self.name = name
        self.cancel_calls = 0
        self.run_calls = 0
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_calls += 1
        self._cancel_event.set()

    def run_arrays(self, batch):
        self.run_calls += 1
        self._cancel_event.wait(timeout=5.0)
        return _FakeResult()


class _FakeCtx:
    def __init__(self, process: "_FakeProc | None" = None) -> None:
        self.exit_calls = 0
        self._process = process

    def __exit__(self, *exc_info) -> None:
        if self._process is not None:
            assert not self._process.is_alive()
        self.exit_calls += 1


class _FakeProc:
    def __init__(self, name: str) -> None:
        self.name = name
        self.terminate_calls = 0
        self.join_calls = 0
        self._alive = True

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        self.join_calls += 1

    def is_alive(self) -> bool:
        return self._alive


def _make_classifier() -> BirdNETClassifier:
    clf = BirdNETClassifier.__new__(BirdNETClassifier)
    clf._min_confidence = 0.1
    clf._session_lock = threading.Lock()
    clf._inflight_sessions = set()
    clf._closed = False
    clf._session_ctxs = []
    clf._all_sessions = []
    clf._session_pool = queue.Queue()
    clf._child_procs = []
    return clf


def test_close_unblocks_thread_parked_in_pool_get():
    clf = _make_classifier()
    clf._session_ctxs = [_FakeCtx()]
    clf._all_sessions = [_FakeSession("s0")]
    # Pool intentionally left empty: classify() will park in the get-loop.

    outcome: dict[str, BaseException] = {}

    def worker() -> None:
        try:
            clf.classify(np.zeros(10, dtype=np.float32), 48_000)
        except Exception as exc:  # noqa: BLE001
            outcome["exc"] = exc

    t = threading.Thread(target=worker)
    start = time.monotonic()
    t.start()
    time.sleep(0.2)  # let it settle into the get() loop
    clf.close()
    t.join(timeout=5.0)
    elapsed = time.monotonic() - start

    assert not t.is_alive()
    assert elapsed < 2.0
    assert isinstance(outcome.get("exc"), RuntimeError)


def test_close_cancels_sessions_and_unblocks_blocking_run_arrays():
    clf = _make_classifier()
    session = _FakeSession("s0")
    ctx = _FakeCtx()
    clf._session_ctxs = [ctx]
    clf._all_sessions = [session]
    clf._session_pool.put(session)

    outcome: dict[str, object] = {}

    def worker() -> None:
        try:
            outcome["result"] = clf.classify(np.zeros(10, dtype=np.float32), 48_000)
        except Exception as exc:  # noqa: BLE001
            outcome["exc"] = exc

    t = threading.Thread(target=worker)
    t.start()

    deadline = time.monotonic() + 2.0
    while session.run_calls == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert session.run_calls == 1

    start = time.monotonic()
    clf.close()
    t.join(timeout=5.0)
    elapsed = time.monotonic() - start

    assert not t.is_alive()
    assert elapsed < 2.0
    assert session.cancel_calls >= 1
    assert ctx.exit_calls == 1


def test_close_never_terminates_processes_outside_child_procs(monkeypatch):
    clf = _make_classifier()
    own_proc = _FakeProc("own")
    sibling_proc = _FakeProc("sibling")
    clf._child_procs = [own_proc]
    clf._session_ctxs = [_FakeCtx()]
    clf._all_sessions = [_FakeSession("s0")]

    monkeypatch.setattr(
        "minimappr.classifiers.birdnet.multiprocessing.active_children",
        lambda: [sibling_proc],
    )

    clf.close()

    assert own_proc.terminate_calls == 1
    assert sibling_proc.terminate_calls == 0


def test_close_terminates_owned_workers_before_context_exit():
    clf = _make_classifier()
    own_proc = _FakeProc("own")
    clf._child_procs = [own_proc]
    clf._session_ctxs = [_FakeCtx(process=own_proc)]
    clf._all_sessions = [_FakeSession("s0")]

    clf.close()

    assert own_proc.terminate_calls == 1
    assert own_proc.join_calls == 1


def test_pool_size_greater_than_one_tracks_and_cancels_every_session_idempotent():
    clf = _make_classifier()
    s1, s2 = _FakeSession("s1"), _FakeSession("s2")
    ctx1, ctx2 = _FakeCtx(), _FakeCtx()
    clf._session_ctxs = [ctx1, ctx2]
    clf._all_sessions = [s1, s2]
    clf._session_pool.put(s1)
    clf._session_pool.put(s2)

    results: list[object] = []

    def worker() -> None:
        try:
            results.append(clf.classify(np.zeros(10, dtype=np.float32), 48_000))
        except Exception as exc:  # noqa: BLE001
            results.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()

    deadline = time.monotonic() + 2.0
    while (s1.run_calls == 0 or s2.run_calls == 0) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert s1.run_calls == 1 and s2.run_calls == 1
    with clf._session_lock:
        assert clf._inflight_sessions == {s1, s2}

    clf.close()
    for t in threads:
        t.join(timeout=5.0)

    assert s1.cancel_calls == 1 and s2.cancel_calls == 1
    assert ctx1.exit_calls == 1 and ctx2.exit_calls == 1

    # Idempotent: a second close() must not re-cancel or re-exit.
    clf.close()
    assert s1.cancel_calls == 1 and s2.cancel_calls == 1
    assert ctx1.exit_calls == 1 and ctx2.exit_calls == 1
