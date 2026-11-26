"""Faint-call admission across both ingest backends (review item F).

Default suite: fixture calibration, the offline omni replay, and one short
direct-ingest run that proves the backend switch admits and stores identical
uploads. E7's 0.5 s event and 12 s noise control stay the sidecar smoke tests
(``tests/test_faint_admission_funnel.py``).

Full sweep (``MINIMAPPR_ENABLE_FAINT_ADMISSION_SWEEP=1``, ~25-30 min): every
candidate x background through the full app, plus the offline omni and
noise-only trigger replays. Tables print and are written (with JSON) to
``MINIMAPPR_FAINT_FUNNEL_REPORT_PATH`` when set; they are what
``docs/faint_call_admission_results.md`` reports. Report-only: it asserts no
recall, only that the harness saw every upload accepted.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from minimappr.config import Settings
from tests.faint_admission_funnel import funnel_scene
from tests.faint_admission_offline_replays import (
    clearest_over_mean_gain_db,
    replay_omni_scans,
    replay_python_trigger_on_noise,
)
from tests.faint_admission_sweep import ADMISSION_CANDIDATES, run_admission_cell
from tests.faint_admission_sweep_fixtures import (
    ALIGNMENT_FRAME_ALIGNED,
    ALIGNMENT_FRAME_STRADDLING,
    ALIGNMENT_STARTUP,
    BACKGROUNDS,
    CALL_KINDS,
    FRAME_SECONDS,
    QUIET_BACKGROUND,
    SWEEP_SNR_STEPS_DB,
    UPLOAD_SECONDS,
    admission_schedule,
    in_band_rms,
    render_admission_capture,
)
from tests.faint_admission_sweep_report import (
    admission_by_kind_table,
    funnel_summary_table,
    omni_replay_table,
    schedule_note,
    write_sweep_report,
)
from tests.range_matrix_cells import SAMPLE_RATE_HZ, SOURCE_BAND_HZ

SWEEP_ENV = "MINIMAPPR_ENABLE_FAINT_ADMISSION_SWEEP"
REPORT_PATH_ENV = "MINIMAPPR_FAINT_FUNNEL_REPORT_PATH"
CALIBRATION_TOLERANCE_DB = 1.0
SMOKE_SNR_STEPS_DB = (30.0,)
SMOKE_NOISE_ONLY_TAIL_SECONDS = 8.0
# 20 min of noise per background: one false trigger is 3/h.
NOISE_REPLAY_SECONDS = 1200.0
DECIBELS_PER_AMPLITUDE_DECADE = 20.0


def _call_level_db(capture: np.ndarray, start_s: float, end_s: float, reference_rms: float) -> float:
    segment = capture[:, int(start_s * SAMPLE_RATE_HZ) : int(end_s * SAMPLE_RATE_HZ)].astype(np.float64)
    return DECIBELS_PER_AMPLITUDE_DECADE * np.log10(in_band_rms(segment, SOURCE_BAND_HZ) / reference_rms)


def test_schedule_places_calls_on_and_across_frame_boundaries_without_overlap():
    schedule = admission_schedule()
    # Startup calls sit inside the warm-up on purpose and may share a zone.
    zones = sorted(call.exclusion_zone_seconds() for call in schedule if call.alignment != ALIGNMENT_STARTUP)
    assert all(earlier[1] <= later[0] for earlier, later in zip(zones, zones[1:]))
    for call in schedule:
        frame_position = (call.arrival_start_seconds % FRAME_SECONDS) / FRAME_SECONDS
        if call.alignment == ALIGNMENT_FRAME_ALIGNED and call.kind_name == "chip":
            # Inside one frame: starts and ends within the same 20 ms frame.
            assert int(call.arrival_start_seconds / FRAME_SECONDS) == int(call.arrival_end_seconds / FRAME_SECONDS)
        if call.alignment == ALIGNMENT_FRAME_STRADDLING:
            boundary = round(call.arrival_start_seconds / UPLOAD_SECONDS + 0.5) * UPLOAD_SECONDS
            assert call.arrival_start_seconds < boundary < call.arrival_end_seconds, (call, frame_position)


def test_calls_are_rendered_at_their_in_band_snr_over_each_background():
    scene = funnel_scene()
    loudest_step = max(SWEEP_SNR_STEPS_DB)
    for background in BACKGROUNDS:
        schedule = admission_schedule((loudest_step,))
        captures, reference_rms = render_admission_capture(scene, background, schedule, noise_only_tail_seconds=2.0)
        capture = next(iter(captures.values()))
        sustained = next(call for call in schedule if call.kind_name == "sustained")
        # Mean over the microphones, so slightly under the nearest microphone's level.
        level_db = _call_level_db(capture, sustained.arrival_start_seconds + 0.1, sustained.arrival_end_seconds - 0.1, reference_rms)
        assert level_db == pytest.approx(loudest_step, abs=CALIBRATION_TOLERANCE_DB), background.name


def test_omni_replay_scores_the_mean_and_the_clearest_channel_on_identical_audio():
    scene = funnel_scene()
    schedule = admission_schedule((max(SWEEP_SNR_STEPS_DB),))
    captures, _ = render_admission_capture(scene, QUIET_BACKGROUND, schedule, noise_only_tail_seconds=30.0)
    capture = next(iter(captures.values()))
    sensor_ids = scene.nodes[0].sensor_ids
    settings = Settings()
    for clearest in (False, True):
        results = replay_omni_scans(capture, sensor_ids, schedule, clearest_channel_gate=clearest, settings=settings)
        assert {result.phase for result in results} >= {"startup"}
        # The default absolute gate (0.001 RMS over a 21 s window) is far above a quiet site's calls.
        assert all(not result.classified_call_indices for result in results)
    chip = next(call for call in schedule if call.kind_name == "chip")
    assert clearest_over_mean_gain_db(capture, chip) >= 0.0


def test_direct_ingest_backend_admits_and_stores_identical_uploads(monkeypatch, tmp_path: Path):
    """The backend switch end to end: loud calls through Python direct ingest
    with its relative trigger on, and no false trigger in the quiet gaps."""
    candidate = next(candidate for candidate in ADMISSION_CANDIDATES if candidate.name == "direct_relative")
    cell = run_admission_cell(
        monkeypatch,
        tmp_path,
        candidate,
        QUIET_BACKGROUND,
        snr_steps_db=SMOKE_SNR_STEPS_DB,
        noise_only_tail_seconds=SMOKE_NOISE_ONLY_TAIL_SECONDS,
    )
    assert all(row.received for row in cell.rows)
    syllable_and_sustained = [row for row in cell.rows if row.kind in {"syllable", "sustained"} and row.alignment == ALIGNMENT_FRAME_ALIGNED]
    assert syllable_and_sustained and all(row.admitted and row.stored for row in syllable_and_sustained), cell.rows
    assert cell.false_triggers == 0


@pytest.mark.skipif(os.environ.get(SWEEP_ENV) != "1", reason=f"faint-call admission sweep is opt-in ({SWEEP_ENV}=1)")
def test_faint_call_admission_sweep(monkeypatch, tmp_path: Path):
    settings = Settings()
    cells = []
    for background in BACKGROUNDS:
        for candidate in ADMISSION_CANDIDATES:
            cell_dir = tmp_path / f"{candidate.name}_{background.name}"
            cell_dir.mkdir()
            with monkeypatch.context() as cell_monkeypatch:
                cells.append(run_admission_cell(cell_monkeypatch, cell_dir, candidate, background))
    noise_replays = []
    for background in BACKGROUNDS:
        for candidate in ADMISSION_CANDIDATES:
            noise_replays.append(
                replay_python_trigger_on_noise(
                    candidate.name,
                    background,
                    settings,
                    relative_enabled=candidate.relative_trigger_enabled,
                    short_block_enabled=candidate.short_block_enabled,
                    replay_seconds=NOISE_REPLAY_SECONDS,
                )
            )
    scene = funnel_scene()
    schedule = admission_schedule()
    omni_results = []
    clearest_gain_db_by_kind: dict[str, float] = {}
    for background in BACKGROUNDS:
        captures, _ = render_admission_capture(scene, background, schedule)
        capture = next(iter(captures.values()))
        for clearest in (False, True):
            for result in replay_omni_scans(capture, scene.nodes[0].sensor_ids, schedule, clearest_channel_gate=clearest, settings=settings):
                omni_results.append((background.name, result))
        if background is QUIET_BACKGROUND:
            loudest = max(SWEEP_SNR_STEPS_DB)
            for kind in CALL_KINDS:
                gains = [
                    clearest_over_mean_gain_db(capture, call)
                    for call in schedule
                    if call.kind_name == kind.name and call.in_band_snr_db == loudest
                ]
                clearest_gain_db_by_kind[kind.name] = float(np.median(gains))
    report = "\n\n".join(
        [
            schedule_note(),
            funnel_summary_table(cells, noise_replays),
            admission_by_kind_table(cells),
            omni_replay_table(omni_results, len(schedule), clearest_gain_db_by_kind),
        ]
    )
    print(f"\n[faint-admission-sweep]\n{report}")
    report_dir = Path(os.environ.get(REPORT_PATH_ENV) or tmp_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "faint_call_admission_sweep.md").write_text(report, encoding="utf-8")
    write_sweep_report(report_dir / "faint_call_admission_sweep.json", cells, noise_replays, omni_results)
    assert all(row.received for cell in cells for row in cell.rows), "an upload was rejected: the funnel is not comparable"
