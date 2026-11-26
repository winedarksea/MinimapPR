"""Collect every sidecar DSP manifest of a long streaming run.

``replay_dsp_event_manifests`` reads the SSE replay ring once at the end, but
the ring holds 512 events: a run of minutes overflows it and loses its oldest
solves. This collector follows ``/api/v1/dsp/stream`` live on a thread and
reconnects with ``Last-Event-ID`` after every read timeout, so a partial event
cut by the timeout is replayed rather than lost (manifests dedupe by id).
"""

from __future__ import annotations

import http.client
import json
import socket
import threading

# Read timeout before reconnecting; also bounds how long ``stop`` waits.
STREAM_READ_TIMEOUT_SECONDS = 2.0
# After stop, drain what is still in the ring until it has been idle this long.
FINAL_DRAIN_IDLE_SECONDS = 3.0


class LiveSidecarManifestCollector:
    def __init__(self, sidecar_port: int) -> None:
        self._sidecar_port = sidecar_port
        self._manifests_by_id: dict[str, dict] = {}
        self._last_event_id = 0
        self._replay_gap_count = 0
        self._stop_requested = threading.Event()
        self._thread = threading.Thread(target=self._follow_stream, name="sidecar-manifest-collector", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[dict]:
        """Stop following, drain the ring and return every manifest, in arrival order."""
        self._stop_requested.set()
        self._thread.join(timeout=STREAM_READ_TIMEOUT_SECONDS * 3)
        self._read_stream_until_idle(FINAL_DRAIN_IDLE_SECONDS)
        return list(self._manifests_by_id.values())

    @property
    def replay_gap_count(self) -> int:
        """Gaps the sidecar reported (events lost before this collector read them)."""
        return self._replay_gap_count

    def _follow_stream(self) -> None:
        while not self._stop_requested.is_set():
            self._read_stream_until_idle(STREAM_READ_TIMEOUT_SECONDS)

    def _read_stream_until_idle(self, idle_seconds: float) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self._sidecar_port, timeout=idle_seconds)
        try:
            connection.request("GET", "/api/v1/dsp/stream", headers={"Last-Event-ID": str(self._last_event_id)})
            response = connection.getresponse()
            if response.status != 200:
                return
            event_name, event_id, data = "message", None, None
            while True:
                try:
                    line = response.readline()
                except (socket.timeout, TimeoutError):
                    return
                if not line:
                    return
                text = line.decode("utf-8").rstrip("\r\n")
                if text.startswith("event:"):
                    event_name = text.partition(":")[2].strip()
                elif text.startswith("id:"):
                    event_id = int(text.partition(":")[2].strip())
                elif text.startswith("data:"):
                    data = text.partition(":")[2]
                elif not text:
                    self._dispatch(event_name, event_id, data)
                    event_name, event_id, data = "message", None, None
        except (ConnectionError, OSError, http.client.HTTPException):
            return
        finally:
            connection.close()

    def _dispatch(self, event_name: str, event_id: int | None, data: str | None) -> None:
        if event_name == "replay_gap":
            self._replay_gap_count += 1
            return
        if data is None:
            return
        manifest = json.loads(data)
        self._manifests_by_id[str(manifest.get("manifest_id"))] = manifest
        if event_id is not None:
            self._last_event_id = max(self._last_event_id, event_id)
