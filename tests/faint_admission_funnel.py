"""Faint admission funnel (plan item E7): where does a faint event get lost?

One emission per run from a single tetra node's view, streamed through the
real sidecar and the full app with the app's own trigger defaults. Each run
yields one funnel row:

    trigger      an event-triggered sidecar solve whose window covers the emission
    solve        that solve returned a usable SRP fix (not degraded) whose bearing
                 from the node is within the tier scorer's bearing tolerance
    storage      the stored detection covering the emission and its modality
                 (``localized`` / ``omni`` / ``none``)

``losing_stage`` is the first stage that failed. A noise-only run of the same
configuration measures the false-trigger rate the rows are taken at.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest

from minimappr.core.geo import LocalCoordinateFrame
from minimappr.models import NodeType, TimeQuality
from minimappr.sim.replay_entity_metrics import angular_difference_deg, bearing_deg
from tests.acoustic_scene_synthesis import AcousticScene, scene_node_near_enu
from tests.range_matrix_cells import (
    AIR_ABSORPTION_DB_PER_M,
    COORDINATE_MODE,
    LAYOUT_ANCHOR_ENU_M,
    SAMPLE_RATE_HZ,
    SCENE_HUMIDITY_FRACTION,
    SCENE_TEMPERATURE_C,
    SITE_ORIGIN,
    SOUND_SPEED_MPS,
    SOURCE_BAND_HZ,
    TETRA_SENSOR_OFFSETS_BODY_M,
    TIER_SCORING_CONFIG,
)
from tests.range_matrix_scoring import detections_overlapping_span
from tests.realistic_capture_synthesis import (
    Emission,
    SourceBurstSpec,
    render_calibrated_captures,
    spreading_gains_relative_to_nearest,
)
from tests.realistic_scene_full_app_run import StreamingNode, run_streaming_scene_through_full_app

NODE_ID = "tetra-funnel"
SOURCE_OFFSET_FROM_NODE_M = (24.0, 17.0, 5.0)
SOURCE_BURST = SourceBurstSpec(duration_seconds=0.5, band_hz=SOURCE_BAND_HZ, edge_ramp_seconds=0.005, seed=7117)
# Long enough for the relative trigger's 1 s warm-up and its floor to settle.
PRE_ROLL_SECONDS = 2.0
TAIL_SECONDS = 1.0
NOISE_ONLY_CAPTURE_SECONDS = 12.0
DETECTION_WAIT_SECONDS = 12.0
LOCALIZATION_WINDOW_SECONDS = 0.08
NANOSECONDS_PER_SECOND = 1_000_000_000
SECONDS_PER_MINUTE = 60.0

# The app's own trigger defaults (``Settings.from_env``): E7 measures what a
# site gets out of the box, so the gate is not tuned per site.
APP_DEFAULT_TRIGGER_RMS = 0.001
APP_DEFAULT_TRIGGER_COOLDOWN_SECONDS = 0.8
# The solver's absolute reference-channel floor before R4 (``SrpPhatConfig.min_channel_rms``).
OLD_ABSOLUTE_SOLVER_FLOOR_RMS = 1.0e-4

EVENT_TRIGGER_ATTEMPT_PREFIX = "event_trigger_"
ATTEMPT_KIND_NOISE_FLOOR_RELATIVE = "event_trigger_noise_floor_relative"
USABLE_SOLVE_ALGORITHM = "srp_phat"
MODALITY_NONE = "none"
STAGE_TRIGGER = "trigger"
STAGE_SOLVE = "solve"
STAGE_STORAGE = "storage"


@dataclass(frozen=True)
class FunnelSite:
    name: str
    # Full-band white sensor noise RMS (the site's floor).
    sensor_noise_rms: float
    noise_seed: int


# Suburban ambient (-80 dBFS) and a quiet rural site at MEMS self-noise (-92 dBFS).
NORMAL_SITE = FunnelSite(name="normal", sensor_noise_rms=1.0e-4, noise_seed=31)
QUIET_SITE = FunnelSite(name="quiet", sensor_noise_rms=2.5e-5, noise_seed=37)

FUNNEL_APP_ENVIRONMENT = {
    "MINIMAPPR_SITE_ORIGIN_SOURCE": "manual",
    "MINIMAPPR_SITE_ORIGIN_LAT": repr(SITE_ORIGIN.lat),
    "MINIMAPPR_SITE_ORIGIN_LON": repr(SITE_ORIGIN.lon),
    "MINIMAPPR_SITE_ORIGIN_ALT_M": repr(SITE_ORIGIN.alt_m),
    "MINIMAPPR_COORDINATE_MODE": COORDINATE_MODE,
    "MINIMAPPR_DEFAULT_TEMPERATURE_C": repr(SCENE_TEMPERATURE_C),
    "MINIMAPPR_DEFAULT_HUMIDITY": repr(SCENE_HUMIDITY_FRACTION),
    "MINIMAPPR_TRIGGER_RMS": repr(APP_DEFAULT_TRIGGER_RMS),
    "MINIMAPPR_TRIGGER_COOLDOWN_SECONDS": repr(APP_DEFAULT_TRIGGER_COOLDOWN_SECONDS),
    "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS": "3.0",
    "MINIMAPPR_REPORTING_WINDOW_SECONDS": "0.25",
    "MINIMAPPR_LOCALIZATION_WINDOW_SECONDS": repr(LOCALIZATION_WINDOW_SECONDS),
    "MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M": "0.5",
    "MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M": "2.0",
}


@dataclass(frozen=True)
class FunnelRow:
    site: str
    in_band_snr_db: float | None
    window_rms_over_floor_db: float | None
    triggered: bool
    trigger_kinds: tuple[str, ...]
    event_solve_algorithms: tuple[str, ...]
    usable_localization: bool
    best_bearing_error_deg: float | None
    stored_modality: str
    losing_stage: str | None
    false_triggers_per_minute: float | None = None


def funnel_scene() -> AcousticScene:
    frame = LocalCoordinateFrame(origin=SITE_ORIGIN, mode=COORDINATE_MODE)
    node = scene_node_near_enu(
        frame,
        node_id=NODE_ID,
        node_type=NodeType.SIRITH_TETRA,
        approximate_position_enu_m=LAYOUT_ANCHOR_ENU_M,
        sensor_offsets_body_m=TETRA_SENSOR_OFFSETS_BODY_M,
    )
    node_position_m = frame.geo_to_local(node.position_geo)
    return AcousticScene(
        site_origin=SITE_ORIGIN,
        coordinate_mode=COORDINATE_MODE,
        nodes=(node,),
        source_position_enu_m=tuple(float(p + d) for p, d in zip(node_position_m, SOURCE_OFFSET_FROM_NODE_M)),
        sample_rate_hz=SAMPLE_RATE_HZ,
        sound_speed_mps=SOUND_SPEED_MPS,
    )


def funnel_captures(scene: AcousticScene, site: FunnelSite, in_band_snr_db: float | None) -> dict[str, np.ndarray]:
    """``None`` SNR renders the noise-only capture."""
    if in_band_snr_db is None:
        emissions, capture_seconds = [], NOISE_ONLY_CAPTURE_SECONDS
    else:
        emissions = [Emission(PRE_ROLL_SECONDS, in_band_snr_db, SOURCE_BURST)]
        _first, last_end_s = scene.emission_to_arrival_span_seconds(SOURCE_BURST.duration_seconds)
        capture_seconds = PRE_ROLL_SECONDS + last_end_s + TAIL_SECONDS
    return render_calibrated_captures(
        scene,
        emissions,
        sensor_gain_relative_to_reference=spreading_gains_relative_to_nearest(scene, AIR_ABSORPTION_DB_PER_M),
        capture_seconds=capture_seconds,
        sensor_noise_rms=site.sensor_noise_rms,
        noise_seed=site.noise_seed,
    )


def loudest_event_window_rms(scene: AcousticScene, capture: np.ndarray) -> float:
    """Largest per-channel RMS over localization windows inside the emission:
    what both the absolute trigger and the solver's reference channel see."""
    window_samples = int(round(LOCALIZATION_WINDOW_SECONDS * SAMPLE_RATE_HZ))
    first_s, last_end_s = scene.emission_to_arrival_span_seconds(SOURCE_BURST.duration_seconds)
    start = int((PRE_ROLL_SECONDS + first_s) * SAMPLE_RATE_HZ)
    end = int((PRE_ROLL_SECONDS + last_end_s) * SAMPLE_RATE_HZ) - window_samples
    return max(
        float(np.sqrt(np.mean(capture[:, offset : offset + window_samples].astype(np.float64) ** 2, axis=1)).max())
        for offset in range(start, end, window_samples // 2)
    )


def _event_trigger_manifests(manifests: list[dict], span_ns: tuple[int, int]) -> list[dict]:
    # The trigger examines the window ending at an upload, so a covering
    # window's centre lies within one window of the arrival span.
    window_ns = int(LOCALIZATION_WINDOW_SECONDS * NANOSECONDS_PER_SECOND)
    selected = []
    for manifest in manifests:
        provenance = (manifest.get("localization") or {}).get("solve_provenance") or {}
        if not str(provenance.get("attempt_kind") or "").startswith(EVENT_TRIGGER_ATTEMPT_PREFIX):
            continue
        centre_ns = int(provenance.get("window_center_time_ns") or 0)
        if span_ns[0] - window_ns <= centre_ns <= span_ns[1] + window_ns:
            selected.append(manifest)
    return selected


def _attempt_kinds(manifests: list[dict]) -> tuple[str, ...]:
    return tuple(sorted({str(manifest["localization"]["solve_provenance"]["attempt_kind"]) for manifest in manifests}))


def _bearing_error_deg(scene: AcousticScene, position_m) -> float:
    node_m = scene.node_position_enu_m(NODE_ID)
    truth_m = np.asarray(scene.source_position_enu_m)
    return angular_difference_deg(bearing_deg(node_m, np.asarray(position_m)), bearing_deg(node_m, truth_m))


def _stored_modality(detections: list[dict]) -> str:
    modalities = {str(detection.get("reporting_modality")) for detection in detections}
    for modality in ("localized", "omni"):
        if modality in modalities:
            return modality
    return sorted(modalities)[0] if modalities else MODALITY_NONE


def run_funnel_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, site: FunnelSite, in_band_snr_db: float | None
) -> FunnelRow:
    scene = funnel_scene()
    captures = funnel_captures(scene, site, in_band_snr_db)
    node = StreamingNode(scene_node=scene.nodes[0], time_quality=TimeQuality.GPS_LOCKED, packet_clock_error_ns=0)
    emission_offset_ns = int(round(PRE_ROLL_SECONDS * NANOSECONDS_PER_SECOND))
    first_s, last_end_s = scene.emission_to_arrival_span_seconds(SOURCE_BURST.duration_seconds)

    def span_for(run) -> tuple[int, int]:
        emission_ns = run.capture_start_packet_time_ns + emission_offset_ns
        return emission_ns + int(first_s * NANOSECONDS_PER_SECOND), emission_ns + int(last_end_s * NANOSECONDS_PER_SECOND)

    run = run_streaming_scene_through_full_app(
        monkeypatch,
        tmp_path,
        scene=scene,
        nodes=[node],
        captures_by_node=captures,
        app_environment=FUNNEL_APP_ENVIRONMENT,
        detection_wait_seconds=DETECTION_WAIT_SECONDS if in_band_snr_db is not None else 0.0,
        detections_complete=lambda run: bool(detections_overlapping_span(run.detections, span_for(run))),
        replay_manifests_over_sse=True,
    )
    manifests = list(run.manifests_by_id.values())
    if in_band_snr_db is None:
        capture_end_ns = run.capture_start_packet_time_ns + int(NOISE_ONLY_CAPTURE_SECONDS * NANOSECONDS_PER_SECOND)
        false_triggers = _event_trigger_manifests(manifests, (run.capture_start_packet_time_ns, capture_end_ns))
        return FunnelRow(
            site=site.name,
            in_band_snr_db=None,
            window_rms_over_floor_db=None,
            triggered=bool(false_triggers),
            trigger_kinds=_attempt_kinds(false_triggers),
            event_solve_algorithms=(),
            usable_localization=False,
            best_bearing_error_deg=None,
            stored_modality=MODALITY_NONE,
            losing_stage=None,
            false_triggers_per_minute=len(false_triggers) * SECONDS_PER_MINUTE / NOISE_ONLY_CAPTURE_SECONDS,
        )
    span_ns = span_for(run)
    trigger_manifests = _event_trigger_manifests(manifests, span_ns)
    usable_errors = [
        _bearing_error_deg(scene, manifest["localization"]["position_m"])
        for manifest in trigger_manifests
        if manifest["localization"].get("resolved_algorithm") == USABLE_SOLVE_ALGORITHM
        and manifest["localization"].get("position_m") is not None
    ]
    best_error = min(usable_errors) if usable_errors else None
    usable = best_error is not None and best_error <= TIER_SCORING_CONFIG.bearing_correct_max_error_deg
    modality = _stored_modality(detections_overlapping_span(run.detections, span_ns))
    if not trigger_manifests:
        losing_stage = STAGE_TRIGGER
    elif not usable:
        losing_stage = STAGE_SOLVE
    elif modality != "localized":
        losing_stage = STAGE_STORAGE
    else:
        losing_stage = None
    return FunnelRow(
        site=site.name,
        in_band_snr_db=in_band_snr_db,
        window_rms_over_floor_db=20.0 * np.log10(loudest_event_window_rms(scene, captures[NODE_ID]) / site.sensor_noise_rms),
        triggered=bool(trigger_manifests),
        trigger_kinds=_attempt_kinds(trigger_manifests),
        event_solve_algorithms=tuple(sorted({str(m["localization"].get("resolved_algorithm")) for m in trigger_manifests})),
        usable_localization=usable,
        best_bearing_error_deg=best_error,
        stored_modality=modality,
        losing_stage=losing_stage,
    )


def format_funnel_table(rows: list[FunnelRow]) -> str:
    header = "site    snr_db  win/floor_dB  trig  kinds                               solve                         usable  bearing_err  stored     lost_at   false_trig/min"
    lines = [header]
    for row in rows:
        lines.append(
            f"{row.site:7s} {'-' if row.in_band_snr_db is None else f'{row.in_band_snr_db:6.1f}':>6s}  "
            f"{'-' if row.window_rms_over_floor_db is None else f'{row.window_rms_over_floor_db:6.1f}':>12s}  "
            f"{str(row.triggered):5s} {','.join(row.trigger_kinds) or '-':35s} {','.join(row.event_solve_algorithms) or '-':29s} "
            f"{str(row.usable_localization):6s}  {'-' if row.best_bearing_error_deg is None else f'{row.best_bearing_error_deg:.1f}':>11s}  "
            f"{row.stored_modality:9s}  {row.losing_stage or '-':8s}  "
            f"{'-' if row.false_triggers_per_minute is None else f'{row.false_triggers_per_minute:.2f}'}"
        )
    return "\n".join(lines)


def write_funnel_report(rows: list[FunnelRow], path: Path) -> None:
    path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
