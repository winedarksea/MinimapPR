"""E10 hot-path release gate: replay one deterministic input through a checkout's real server.

The run launches the checkout's FastAPI app (``hot_path_release_gate_launcher.py``,
slow classifier injected), which launches its own Rust sidecar in the
production memory-only live path. Pre-rendered posts from the scenario are
sent in real time, one sender thread per node, while nodes join and leave.
Metrics written to the result JSON:

* packet gaps: sidecar firmware-sequence gaps + app frame-sequence gaps +
  holes in the raw audio the sidecar forwards to consumers;
* ingest queue drops: sidecar overload rejections and dropped frames, app
  trigger-queue drops; HTTP-rejected posts are counted separately;
* buffer discontinuities: sidecar re-anchors / window underruns, app buffer
  re-anchors / rate wipes;
* solve latency: trigger packet accepted -> pair-bearing localization result on
  the DSP stream (p50 / p95);
* peak RSS of the sidecar and the app (sampled);
* known calls reaching stored detections: distinct scenario calls a stored
  detection (``/api/v1/detections``) is attributed to by event time, and
  classification backlog drops (app classification drops + stage backpressure);
* raw-audio disk writes: bytes / files created or grown under the data, spool
  and sidecar storage dirs, excluding the SQLite DB and detection snippets
  (which are reported on their own: kept detections may write snippets).

Only HTTP, the process table and the file system are used to observe the
server, so the harness in a newer checkout can measure an older one.
"""

from __future__ import annotations

import http.client
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from minimappr.sim.replay_gate_config import HotPathGateConfig
from tests.hot_path_release_gate_metrics import (
    NANOSECONDS_PER_MILLISECOND,
    GateRunCapture,
    assemble_gate_metrics,
    aggregate_repeated_run_metrics,
    observer_was_complete,
)
from tests.hot_path_release_gate_probes import (
    DspStreamListener,
    PeakRssSampler,
    http_get_json,
    snapshot_file_sizes,
    written_files,
)
from tests.hot_path_release_gate_scenario import HotPathScenario, ScheduledPost, build_post_schedule
from tests.rust_sidecar_binary import build_rust_sidecar_binary_in

RESULT_SCHEMA_VERSION = 1
SIDECAR_BUILD_PROFILE = "debug"
APP_READY_TIMEOUT_SECONDS = 120.0
APP_READY_POLL_SECONDS = 0.25
APP_SHUTDOWN_TIMEOUT_SECONDS = 30.0
DSP_STREAM_CONNECT_TIMEOUT_SECONDS = 10.0
RSS_SAMPLE_INTERVAL_SECONDS = 0.1
POST_TIMEOUT_SECONDS = 10.0
DETECTION_LIST_LIMIT = 1_000
# Short classification window so classifier renders arrive about once a second:
# with the production 30 s window a slow classifier would never be loaded.
GATE_CLASSIFICATION_WINDOW_SECONDS = 3.0
# The app's DSP stream consumer logs these when its SSE connection to the
# sidecar drops (e.g. it lagged the broadcast); events in between are lost to it.
APP_STREAM_RECONNECT_LOG_MARKERS = (
    "IngestStreamConsumer connection lost",
    "IngestStreamConsumer disconnected cleanly",
)
HEURISTIC_ROUTING = {
    "version": 1,
    "classifiers": {"heuristic": {"backend": "heuristic"}},
    "contexts": {
        "detection_trigger": {"run": ["heuristic"]},
        "localized_render": {"run": ["heuristic"]},
        "omni_continuous": {"run": []},
    },
    "chains": [],
    "triggers": [],
}
LAUNCHER_PATH = Path(__file__).resolve().with_name("hot_path_release_gate_launcher.py")
RUNTIME_SOURCE_DIRECTORIES = ("minimappr", "minimappr-ingest-sidecar", "minimappr-core")


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _git_output(checkout_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout_root), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def checkout_identity(checkout_root: Path) -> dict:
    # Listed rather than reduced to a flag: uncommitted scoring-only files
    # (e.g. minimappr/sim) do not change what the server runs, and a reader
    # needs the paths to judge that.
    dirty_lines = _git_output(checkout_root, "status", "--porcelain", "--", *RUNTIME_SOURCE_DIRECTORIES).splitlines()
    return {
        "commit": _git_output(checkout_root, "rev-parse", "HEAD"),
        "branch": _git_output(checkout_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted_paths_under_runtime_directories": [line[3:] for line in dirty_lines],
    }


def machine_identity() -> dict:
    cpu_model = platform.processor()
    if sys.platform == "darwin":
        cpu_model = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
        ).stdout.strip() or cpu_model
    import psutil

    # Only hardware identity: an OS point update or Python patch release must not
    # make the baseline "another machine" and silently un-gate latency/RSS.
    return {
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "memory_gib": round(psutil.virtual_memory().total / 1024**3, 1),
    }


class _GateServer:
    """The checkout's app + sidecar as subprocesses, with contained data directories."""

    def __init__(self, checkout_root: Path, work_dir: Path, scenario: HotPathScenario, sidecar_binary: Path) -> None:
        self.checkout_root = checkout_root
        self.work_dir = work_dir
        self.data_dir = work_dir / "data"
        self.app_port = _free_tcp_port()
        self.sidecar_port = _free_tcp_port()
        self.scenario = scenario
        self.sidecar_binary = sidecar_binary
        self.process: subprocess.Popen | None = None
        self.sidecar_pid: int | None = None
        self.diagnostics_at_start: dict = {}

    @property
    def snippet_dir(self) -> Path:
        return self.data_dir / "snippets"

    def watched_directories(self) -> list[Path]:
        # checkout/data/{ingest,spool} are the relative defaults a server
        # that ignored the contained-dir env would fall back to.
        return [self.data_dir, self.checkout_root / "data" / "ingest", self.checkout_root / "data" / "spool"]

    def environment(self) -> dict[str, str]:
        routing_path = self.work_dir / "classifier_routing.json"
        routing_path.write_text(json.dumps(HEURISTIC_ROUTING), encoding="utf-8")
        return {
            **os.environ,
            "PYTHONPATH": str(self.checkout_root),
            "MINIMAPPR_CONFIG_PATH": str(self.work_dir / "config.yml"),
            "MINIMAPPR_CLASSIFIER_ROUTING_CONFIG_PATH": str(routing_path),
            "MINIMAPPR_DB_PATH": str(self.data_dir / "minimappr.db"),
            "MINIMAPPR_SNIPPET_DIR": str(self.snippet_dir),
            "MINIMAPPR_LARGE_ARTIFACT_DIR": str(self.data_dir / "artifacts"),
            "MINIMAPPR_INGEST_SPOOL_DIR": str(self.data_dir / "spool"),
            "MINIMAPPR_INGEST_STORAGE_DIR": str(self.data_dir / "ingest"),
            "MINIMAPPR_DIRECT_INGEST_ENABLED": "false",
            "MINIMAPPR_INGEST_SIDECAR_ENABLED": "true",
            "MINIMAPPR_INGEST_SIDECAR_BINARY_PATH": str(self.sidecar_binary),
            "MINIMAPPR_SIDECAR_PORT": str(self.sidecar_port),
            "MINIMAPPR_SIDECAR_ALLOW_NON_TMPFS_JOURNAL": "true",
            # Python classifies renders (the slow consumer); no Rust classifier helper.
            "MINIMAPPR_SIDECAR_CLASSIFIER_COMMAND_JSON": "",
            "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS": str(GATE_CLASSIFICATION_WINDOW_SECONDS),
        }

    def start(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(LAUNCHER_PATH),
                str(self.checkout_root),
                str(self.app_port),
                str(self.scenario.slow_classifier_delay_seconds),
                str(self.work_dir / "served_identity.json"),
            ],
            cwd=str(self.checkout_root),
            env=self.environment(),
            stdout=(self.work_dir / "app.log").open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + APP_READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"app exited during startup; see {self.work_dir / 'app.log'}")
            try:
                diagnostics = http_get_json(self.app_port, "/api/v1/system/diagnostics")
            except OSError:
                diagnostics = {}
            sidecar = diagnostics.get("sidecar") or {}
            if sidecar.get("status") == "running" and (sidecar.get("health") or {}).get("status") == "ok":
                self.sidecar_pid = int(sidecar["pid"])
                self.diagnostics_at_start = diagnostics
                return
            time.sleep(APP_READY_POLL_SECONDS)
        raise RuntimeError(f"app + sidecar not ready within {APP_READY_TIMEOUT_SECONDS} s; see {self.work_dir / 'app.log'}")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout=APP_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=APP_SHUTDOWN_TIMEOUT_SECONDS)
        if self.sidecar_pid is not None:
            # The app normally reaps its sidecar; a wedged shutdown must not
            # leave one holding the port for the next run.
            try:
                os.kill(self.sidecar_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class _NodeSender(threading.Thread):
    """Posts one node's uploads on schedule over a keep-alive connection, recording outcomes."""

    def __init__(self, sidecar_port: int, posts: list[ScheduledPost], run_start_monotonic: float) -> None:
        super().__init__(name=f"hot-path-gate-sender-{posts[0].node_id}", daemon=True)
        self.sidecar_port = sidecar_port
        self.posts = posts
        self.run_start_monotonic = run_start_monotonic
        self.accepted_monotonic_ns_by_post: dict[tuple[str, int], int] = {}
        self.round_trip_ms: list[float] = []
        self.rejected_post_count = 0

    def run(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self.sidecar_port, timeout=POST_TIMEOUT_SECONDS)
        try:
            for post in self.posts:
                delay_seconds = self.run_start_monotonic + post.send_offset_seconds - time.monotonic()
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
                sent_ns = time.monotonic_ns()
                try:
                    connection.request(
                        "POST", "/api/v1/ingest/store-forward", body=post.body, headers={"Content-Type": "application/json"}
                    )
                    response = connection.getresponse()
                    response.read()
                    accepted = 200 <= response.status < 300
                except (OSError, http.client.HTTPException):
                    connection.close()
                    accepted = False
                completed_ns = time.monotonic_ns()
                self.round_trip_ms.append((completed_ns - sent_ns) / NANOSECONDS_PER_MILLISECOND)
                if accepted:
                    self.accepted_monotonic_ns_by_post[(post.node_id, post.first_sample_index)] = completed_ns
                else:
                    self.rejected_post_count += 1
        finally:
            connection.close()


def _count_app_stream_reconnects(app_log_path: Path) -> int:
    if not app_log_path.exists():
        return 0
    log_text = app_log_path.read_text(encoding="utf-8", errors="replace")
    return sum(log_text.count(marker) for marker in APP_STREAM_RECONNECT_LOG_MARKERS)


def _run_once(checkout_root: Path, work_dir: Path, scenario: HotPathScenario, sidecar_binary: Path) -> GateRunCapture:
    server = _GateServer(checkout_root, work_dir, scenario, sidecar_binary)
    rss_sampler = PeakRssSampler(RSS_SAMPLE_INTERVAL_SECONDS)
    try:
        server.start()
        listener = DspStreamListener(server.sidecar_port)
        listener.start(DSP_STREAM_CONNECT_TIMEOUT_SECONDS)
        rss_sampler.watch("app", server.process.pid)
        rss_sampler.watch("sidecar", server.sidecar_pid)
        rss_sampler.start()
        files_before = snapshot_file_sizes(server.watched_directories())
        scenario_start_time_ns = time.time_ns()
        posts = build_post_schedule(scenario, start_time_ns=scenario_start_time_ns)
        run_start = time.monotonic()
        senders = [
            _NodeSender(server.sidecar_port, [p for p in posts if p.node_id == node.node_id], run_start)
            for node in scenario.nodes
        ]
        for sender in senders:
            sender.start()
        for sender in senders:
            sender.join()
        time.sleep(scenario.settle_seconds)
        endpoint_snapshots = {
            "sidecar_health": http_get_json(server.sidecar_port, "/healthz"),
            "sidecar_dsp_status": http_get_json(server.sidecar_port, "/api/v1/dsp/status"),
            "sidecar_summary": http_get_json(server.sidecar_port, "/api/v1/diagnostics/summary"),
            "fusion_status": http_get_json(server.app_port, "/api/v1/fusion/status"),
            "app_diagnostics": http_get_json(server.app_port, "/api/v1/system/diagnostics"),
        }
        detections = http_get_json(server.app_port, f"/api/v1/detections?limit={DETECTION_LIST_LIMIT}")
        observations = listener.freeze()
        peak_rss_mb = rss_sampler.stop()
        # Read before shutdown so teardown disconnects are not counted.
        app_stream_reconnects = _count_app_stream_reconnects(work_dir / "app.log")
    finally:
        rss_sampler.stop()
        server.stop()
    # Full endpoint payloads stay next to the run for diagnosis; the result keeps a summary.
    (work_dir / "endpoint_snapshots.json").write_text(json.dumps(endpoint_snapshots, indent=1, default=str), encoding="utf-8")
    (work_dir / "detections.json").write_text(json.dumps(detections, indent=1, default=str), encoding="utf-8")
    accepted_ns_by_post: dict[tuple[str, int], int] = {}
    for sender in senders:
        accepted_ns_by_post.update(sender.accepted_monotonic_ns_by_post)
    return GateRunCapture(
        scenario=scenario,
        scenario_start_time_ns=scenario_start_time_ns,
        posts=posts,
        accepted_monotonic_ns_by_post=accepted_ns_by_post,
        post_round_trip_ms=[ms for sender in senders for ms in sender.round_trip_ms],
        rejected_post_count=sum(sender.rejected_post_count for sender in senders),
        observations=observations,
        detections=detections if isinstance(detections, list) else [],
        peak_rss_mb=peak_rss_mb,
        app_stream_reconnect_count=app_stream_reconnects,
        written_bytes_by_path=written_files(files_before, snapshot_file_sizes(server.watched_directories())),
        snippet_root=str(server.snippet_dir.resolve()),
        **endpoint_snapshots,
    )


def run_hot_path_release_gate(
    checkout_root: Path, work_dir: Path, scenario: HotPathScenario, *, repeat_count: int = 1
) -> dict:
    """Run the scenario ``repeat_count`` times against ``checkout_root``; metrics are per-metric medians."""
    checkout_root = checkout_root.resolve()
    sidecar_binary = build_rust_sidecar_binary_in(checkout_root)
    gate_config = HotPathGateConfig()
    runs = []
    for run_index in range(repeat_count):
        run_dir = work_dir / f"run-{run_index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Latency and RSS on a shared dev machine track its load; record it so
        # a noisy baseline or candidate run can be recognised as such.
        load_average_at_start = os.getloadavg()
        metrics, components = assemble_gate_metrics(_run_once(checkout_root, run_dir, scenario, sidecar_binary))
        components["load_average_1_5_15_min_at_start"] = [round(value, 2) for value in load_average_at_start]
        served_identity = json.loads((run_dir / "served_identity.json").read_text(encoding="utf-8"))
        runs.append({"metrics": metrics, "components": components, "served_minimappr_file": served_identity["minimappr_file"]})
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "checkout": checkout_identity(checkout_root),
        "machine": machine_identity(),
        "sidecar_build_profile": SIDECAR_BUILD_PROFILE,
        "scenario": scenario.to_json_dict(),
        "repeat_count": repeat_count,
        "observer_complete": all(observer_was_complete(run["components"]) for run in runs),
        "metrics": aggregate_repeated_run_metrics([run["metrics"] for run in runs], gate_config.metric_tolerances),
        "runs": runs,
    }


def write_result(result: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
