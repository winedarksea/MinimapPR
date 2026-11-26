"""Faint-call admission sweep (review item F): identical calls through both
ingest backends and each default-off trigger candidate, stage by stage.

Per scheduled call (``tests/faint_admission_sweep_fixtures.py``) one row:

    received    every upload covering the call was accepted by ingest
    admitted    an event trigger fired on the call (sidecar: an
                ``event_trigger_*`` solve manifest; direct: a frame answered
                ``triggered``) within one localization window of it
    solved      a usable localization from that trigger: bearing within the
                tier scorer's tolerance (sidecar: the solve manifest; direct:
                the stored localized detection, the only solve it exposes)
    classified  a trigger-driven detection stamped within the call carries the
                constant classifier's label
    stored      that detection is stored localized (its modality is kept)

The sidecar also renders the classification window periodically with no
energy gate (``birdnet_omni_fallback``); with a constant classifier those say
nothing about admission, so they are left out of the stages and reported as
``periodic omni coverage`` (the call lies inside such a render).

Triggers away from every call are false triggers, per hour of that exposure.
"""

from __future__ import annotations

import os
import resource
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import psutil
import pytest

from minimappr.models import TimeQuality
from minimappr.sim.replay_entity_metrics import angular_difference_deg, bearing_deg
from tests.faint_admission_funnel import FUNNEL_APP_ENVIRONMENT, LOCALIZATION_WINDOW_SECONDS, NODE_ID, funnel_scene
from tests.faint_admission_sweep_fixtures import (
    FRAME_SECONDS,
    NOISE_ONLY_TAIL_SECONDS,
    SWEEP_SNR_STEPS_DB,
    UPLOAD_SECONDS,
    Background,
    ScheduledCall,
    admission_schedule,
    render_admission_capture,
    schedule_capture_seconds,
)
from tests.range_matrix_cells import TIER_SCORING_CONFIG
from tests.realistic_scene_full_app_run import (
    INGEST_BACKEND_PYTHON_DIRECT,
    INGEST_BACKEND_RUST_SIDECAR,
    FullAppRunResult,
    StreamingNode,
    run_streaming_scene_through_full_app,
)
from tests.rust_sidecar_e2e_helpers import FAKE_CLASSIFIER_LABEL

NANOSECONDS_PER_SECOND = 1_000_000_000
SECONDS_PER_HOUR = 3600.0
EVENT_TRIGGER_ATTEMPT_PREFIX = "event_trigger_"
USABLE_SOLVE_ALGORITHM = "srp_phat"
MODALITY_LOCALIZED = "localized"
MODALITY_NONE = "none"
SWEEP_DETECTION_WAIT_SECONDS = 6.0
# A detection stamped within this of a call's arrival span belongs to it.
DETECTION_OVERLAP_MARGIN_SECONDS = 0.25
PERIODIC_OMNI_RENDER_KIND = "birdnet_omni_fallback"
FLOOR_STARTUP_SECONDS = 30.0

SWEEP_APP_ENVIRONMENT = {
    **FUNNEL_APP_ENVIRONMENT,
    # Wall-clock scheduled; replayed offline instead (faint_admission_offline_replays.py).
    "MINIMAPPR_OMNI_SCAN_ENABLED": "false",
    # Room for every call's detection plus false triggers in one listing.
    "MINIMAPPR_COP_DETECTIONS_MAX_ITEMS": "1000",
    "MINIMAPPR_DETECTION_MIN_CONFIDENCE": "0",
}


@dataclass(frozen=True)
class AdmissionCandidate:
    name: str
    ingest_backend: str
    environment: dict[str, str] = field(default_factory=dict)
    relative_trigger_enabled: bool = False
    short_block_enabled: bool = False


ADMISSION_CANDIDATES = (
    # Shipped defaults: the sidecar's relative trigger is on, direct ingest's is off.
    AdmissionCandidate("sidecar_default", INGEST_BACKEND_RUST_SIDECAR, relative_trigger_enabled=True),
    AdmissionCandidate(
        "sidecar_short_block",
        INGEST_BACKEND_RUST_SIDECAR,
        {"MINIMAPPR_LOCALIZATION_TRIGGER_SHORT_BLOCK_EXCESS_ENABLED": "true"},
        relative_trigger_enabled=True,
        short_block_enabled=True,
    ),
    AdmissionCandidate("direct_default", INGEST_BACKEND_PYTHON_DIRECT),
    AdmissionCandidate(
        "direct_relative",
        INGEST_BACKEND_PYTHON_DIRECT,
        {"MINIMAPPR_TRIGGER_NOISE_FLOOR_RELATIVE_ENABLED": "true"},
        relative_trigger_enabled=True,
    ),
    AdmissionCandidate(
        "direct_relative_short_block",
        INGEST_BACKEND_PYTHON_DIRECT,
        {"MINIMAPPR_TRIGGER_NOISE_FLOOR_RELATIVE_ENABLED": "true", "MINIMAPPR_TRIGGER_SHORT_BLOCK_EXCESS_ENABLED": "true"},
        relative_trigger_enabled=True,
        short_block_enabled=True,
    ),
)


@dataclass(frozen=True)
class CallAdmissionRow:
    candidate: str
    background: str
    call_index: int
    kind: str
    alignment: str
    in_band_snr_db: float
    received: bool
    admitted: bool
    solved: bool
    best_bearing_error_deg: float | None
    classified: bool
    stored: bool
    stored_modality: str
    periodic_omni_coverage: bool


@dataclass(frozen=True)
class AdmissionCellResult:
    candidate: str
    ingest_backend: str
    background: str
    rows: tuple[CallAdmissionRow, ...]
    false_trigger_seconds: tuple[float, ...]
    false_trigger_exposure_seconds: float
    capture_seconds: float
    wall_seconds: float
    test_process_cpu_seconds: float
    sidecar_cpu_seconds: float | None
    # Seconds the harness waited after streaming for the sidecar to process
    # the whole capture, and whether it did (False: late calls were scored
    # before the sidecar reached them).
    sidecar_drain_wait_seconds: float | None
    sidecar_caught_up: bool | None
    load_average_1m_start: float
    load_average_1m_end: float
    manifest_replay_gaps: int

    @property
    def false_triggers(self) -> int:
        return len(self.false_trigger_seconds)

    @property
    def startup_false_triggers(self) -> int:
        """Those in the floor's first convergence (same span as the offline replay's startup)."""
        return sum(seconds < FLOOR_STARTUP_SECONDS for seconds in self.false_trigger_seconds)

    @property
    def false_triggers_per_hour(self) -> float:
        return self.false_triggers * SECONDS_PER_HOUR / max(self.false_trigger_exposure_seconds, 1e-9)


def _seconds_ns(run: FullAppRunResult, seconds: float) -> int:
    return run.capture_start_packet_time_ns + int(round(seconds * NANOSECONDS_PER_SECOND))


def _trigger_times_seconds(run: FullAppRunResult) -> list[tuple[float, list[dict]]]:
    """Every event trigger as (capture seconds, its solve manifests). One
    sidecar trigger yields several manifests (one per render/output) that share
    its window centre, so they are grouped into one trigger."""
    capture_start_ns = run.capture_start_packet_time_ns
    if run.ingest_backend == INGEST_BACKEND_RUST_SIDECAR:
        manifests_by_centre_ns: dict[int, list[dict]] = {}
        for manifest in run.manifests_by_id.values():
            provenance = (manifest.get("localization") or {}).get("solve_provenance") or {}
            if str(provenance.get("attempt_kind") or "").startswith(EVENT_TRIGGER_ATTEMPT_PREFIX):
                manifests_by_centre_ns.setdefault(int(provenance.get("window_center_time_ns") or 0), []).append(manifest)
        return [
            ((centre_ns - capture_start_ns) / NANOSECONDS_PER_SECOND, manifests)
            for centre_ns, manifests in sorted(manifests_by_centre_ns.items())
        ]
    return [
        ((int(frame["start_time_ns"]) - capture_start_ns) / NANOSECONDS_PER_SECOND, [])
        for receipt in run.upload_receipts
        for frame in receipt.frame_results
        if frame.get("triggered")
    ]


def _capture_seconds(run: FullAppRunResult, timestamp_ns) -> float:
    return (int(timestamp_ns) - run.capture_start_packet_time_ns) / NANOSECONDS_PER_SECOND


def _is_periodic_omni_render(detection: dict) -> bool:
    return (detection.get("feature_summary") or {}).get("rust_render_kind") == PERIODIC_OMNI_RENDER_KIND


def _periodic_omni_covers(run: FullAppRunResult, call: ScheduledCall) -> bool:
    for detection in run.detections:
        summary = detection.get("feature_summary") or {}
        if _is_periodic_omni_render(detection) and summary.get("rust_render_start_ns") and summary.get("rust_render_end_ns"):
            start_s = _capture_seconds(run, summary["rust_render_start_ns"])
            end_s = _capture_seconds(run, summary["rust_render_end_ns"])
            if start_s <= call.arrival_start_seconds and call.arrival_end_seconds <= end_s:
                return True
    return False


def _bearing_error_deg(scene, position_m) -> float:
    node_m = scene.node_position_enu_m(NODE_ID)
    return angular_difference_deg(bearing_deg(node_m, np.asarray(position_m)), bearing_deg(node_m, np.asarray(scene.source_position_enu_m)))


def _call_received(run: FullAppRunResult, call: ScheduledCall) -> bool:
    first = int(call.arrival_start_seconds // UPLOAD_SECONDS)
    last = int(call.arrival_end_seconds // UPLOAD_SECONDS)
    receipts = {receipt.upload_index: receipt for receipt in run.upload_receipts}
    return all(index in receipts and receipts[index].fully_received for index in range(first, last + 1))


def score_call(scene, run: FullAppRunResult, call: ScheduledCall, candidate: str, background: str, triggers) -> CallAdmissionRow:
    # The sidecar's trigger window ends at an upload and the direct trigger is
    # stamped at its frame start, so allow one window (or frame) either side.
    slack_seconds = max(LOCALIZATION_WINDOW_SECONDS, FRAME_SECONDS)
    call_triggers = [
        manifests
        for trigger_seconds, manifests in triggers
        if call.arrival_start_seconds - slack_seconds <= trigger_seconds <= call.arrival_end_seconds + slack_seconds
    ]
    detections = [
        detection
        for detection in run.detections
        if not _is_periodic_omni_render(detection)
        and call.arrival_start_seconds - DETECTION_OVERLAP_MARGIN_SECONDS
        <= _capture_seconds(run, detection["timestamp_ns"])
        <= call.arrival_end_seconds + DETECTION_OVERLAP_MARGIN_SECONDS
    ]
    if run.ingest_backend == INGEST_BACKEND_RUST_SIDECAR:
        positions = [
            manifest["localization"]["position_m"]
            for manifests in call_triggers
            for manifest in manifests
            if manifest["localization"].get("resolved_algorithm") == USABLE_SOLVE_ALGORITHM
            and manifest["localization"].get("position_m") is not None
        ]
    else:
        positions = [d["position_m"] for d in detections if d.get("reporting_modality") == MODALITY_LOCALIZED] if call_triggers else []
    errors = [_bearing_error_deg(scene, position) for position in positions]
    best_error = min(errors) if errors else None
    modalities = {str(detection.get("reporting_modality")) for detection in detections}
    return CallAdmissionRow(
        candidate=candidate,
        background=background,
        call_index=call.call_index,
        kind=call.kind_name,
        alignment=call.alignment,
        in_band_snr_db=call.in_band_snr_db,
        received=_call_received(run, call),
        admitted=bool(call_triggers),
        solved=best_error is not None and best_error <= TIER_SCORING_CONFIG.bearing_correct_max_error_deg,
        best_bearing_error_deg=best_error,
        classified=any(detection.get("label") == FAKE_CLASSIFIER_LABEL for detection in detections),
        stored=MODALITY_LOCALIZED in modalities,
        stored_modality=MODALITY_LOCALIZED if MODALITY_LOCALIZED in modalities else (sorted(modalities)[0] if modalities else MODALITY_NONE),
        periodic_omni_coverage=_periodic_omni_covers(run, call),
    )


def count_false_triggers(schedule: list[ScheduledCall], triggers, capture_seconds: float) -> tuple[list[float], float]:
    """False trigger times (capture seconds) and the noise-only exposure they came from."""
    zones = sorted(call.exclusion_zone_seconds() for call in schedule)
    false_triggers = [
        trigger_seconds for trigger_seconds, _ in triggers if not any(start <= trigger_seconds <= end for start, end in zones)
    ]
    covered = 0.0
    cursor = 0.0
    for start, end in zones:
        start, end = max(start, cursor), min(end, capture_seconds)
        if end > start:
            covered += end - start
            cursor = end
    return false_triggers, capture_seconds - covered


def _sidecar_cpu_seconds() -> float | None:
    for child in psutil.Process().children(recursive=True):
        try:
            if "minimappr-ingest-sidecar" in " ".join(child.cmdline()):
                times = child.cpu_times()
                return times.user + times.system
        except psutil.Error:
            continue
    return None


def run_admission_cell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    candidate: AdmissionCandidate,
    background: Background,
    *,
    snr_steps_db: Sequence[float] = SWEEP_SNR_STEPS_DB,
    noise_only_tail_seconds: float = NOISE_ONLY_TAIL_SECONDS,
) -> AdmissionCellResult:
    scene = funnel_scene()
    schedule = admission_schedule(snr_steps_db)
    captures, _reference_in_band_rms = render_admission_capture(scene, background, schedule, noise_only_tail_seconds)
    node = StreamingNode(scene_node=scene.nodes[0], time_quality=TimeQuality.GPS_LOCKED, packet_clock_error_ns=0)
    sidecar_cpu: list[float] = []
    load_start = os.getloadavg()[0]
    cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    wall_start = time.monotonic()

    def _complete(run: FullAppRunResult) -> bool:
        # Never early: sample the sidecar's CPU while it is still alive.
        cpu = _sidecar_cpu_seconds()
        if cpu is not None:
            sidecar_cpu.append(cpu)
        return False

    run = run_streaming_scene_through_full_app(
        monkeypatch,
        tmp_path,
        scene=scene,
        nodes=[node],
        captures_by_node=captures,
        app_environment={**SWEEP_APP_ENVIRONMENT, **candidate.environment},
        detection_wait_seconds=SWEEP_DETECTION_WAIT_SECONDS,
        detections_complete=_complete,
        ingest_backend=candidate.ingest_backend,
        collect_manifests_live=candidate.ingest_backend == INGEST_BACKEND_RUST_SIDECAR,
    )
    wall_seconds = time.monotonic() - wall_start
    cpu_end = resource.getrusage(resource.RUSAGE_SELF)
    capture_seconds = schedule_capture_seconds(schedule, noise_only_tail_seconds)
    triggers = _trigger_times_seconds(run)
    false_triggers, exposure_seconds = count_false_triggers(schedule, triggers, capture_seconds)
    return AdmissionCellResult(
        candidate=candidate.name,
        ingest_backend=candidate.ingest_backend,
        background=background.name,
        rows=tuple(score_call(scene, run, call, candidate.name, background.name, triggers) for call in schedule),
        false_trigger_seconds=tuple(false_triggers),
        false_trigger_exposure_seconds=exposure_seconds,
        capture_seconds=capture_seconds,
        wall_seconds=wall_seconds,
        test_process_cpu_seconds=(cpu_end.ru_utime + cpu_end.ru_stime) - (cpu_start.ru_utime + cpu_start.ru_stime),
        sidecar_cpu_seconds=max(sidecar_cpu) if sidecar_cpu else None,
        sidecar_drain_wait_seconds=run.sidecar_drain_wait_seconds,
        sidecar_caught_up=run.sidecar_caught_up,
        load_average_1m_start=load_start,
        load_average_1m_end=os.getloadavg()[0],
        manifest_replay_gaps=run.manifest_replay_gap_count,
    )


def cell_result_to_json(cell: AdmissionCellResult) -> dict:
    payload = asdict(cell)
    payload["false_triggers_per_hour"] = cell.false_triggers_per_hour
    payload["startup_false_triggers"] = cell.startup_false_triggers
    return payload
