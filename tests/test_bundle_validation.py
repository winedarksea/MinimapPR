"""Calibration bundle validation against the field capture protocol (plan item E9).

The good bundle is the E4 replay-realism bundle (``tests/replay_realism_bundles.py``)
with the survey fields the protocol asks for; every bad bundle is that bundle
with one targeted edit, so each test isolates one check.
"""

from __future__ import annotations

import dataclasses
import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from minimappr.calibration.bundle_ground_truth_checks import range_ladder_rung_for
from minimappr.calibration.bundle_validation import (
    main,
    validate_calibration_artifact_directory,
    validate_calibration_bundle,
)
from minimappr.calibration.bundle_validation_findings import (
    SEVERITY_BY_FINDING_CODE,
    BundleFindingCode as Code,
    BundleValidationConfig,
    FindingSeverity,
)
from tests.replay_realism_bundles import (
    PLANAR_NODE_ID,
    POINT_NODE_ID,
    SAMPLE_RATE_44_1_KHZ,
    SAMPLE_RATE_48_KHZ,
    TETRA_NODE_ID,
    build_replay_scene,
    planar_plan,
    point_plan,
    render_captures_at_node_rates,
    tetra_plan,
    write_replay_bundle,
)

# The replay bundle is 2 s with one call; the protocol defaults ask for minutes
# and three calls per rung, which their own tests exercise.
SHORT_BUNDLE_CONFIG = dataclasses.replace(
    BundleValidationConfig(), minimum_capture_duration_seconds=1.0, minimum_events_per_range_rung=1
)
HAND_MEASURED_POSITION_STD_M = 0.05
TRUTH_POSITION_STD_M = 0.05
TIMESTAMP_STEP_NS = 21_000_000  # the E4 step: over the 10 ms residual limit
SMALL_TIMESTAMP_STEP_NS = 5_000_000  # over the 2 ms step threshold, under the residual limit
STEPPED_FRAME_INDEX = 22
CLOCK_DRIFT_PPM = 100.0
DRIFT_TOLERANCE_PPM = 1.0
DROPPED_FRAME_INDICES = range(10, 20)  # 200 ms of a 2 s capture: 10 % lost
IDENTITY_ORIENTATION = {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}


@dataclass
class EditableBundle:
    manifest: dict
    ground_truth: dict
    frame_timing_by_node_id: dict[str, dict]
    removed_members: set[str] = field(default_factory=set)

    def node(self, node_id: str) -> dict:
        return next(node for node in self.manifest["nodes"] if node["node_id"] == node_id)

    def frames(self, node_id: str) -> list[dict]:
        return self.frame_timing_by_node_id[node_id]["frames"]

    @property
    def event(self) -> dict:
        return self.ground_truth["events"][0]


BundleEdit = Callable[[EditableBundle], None]


def rewrite_bundle(source_path: Path, out_path: Path, edit: BundleEdit) -> Path:
    """Copy a bundle zip with its JSON members edited in place by ``edit``."""
    with zipfile.ZipFile(source_path) as source:
        members = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(members["manifest.json"])
    timing_member_by_node_id = {node["node_id"]: node["frame_timing_file"] for node in manifest["nodes"]}
    editable = EditableBundle(
        manifest=manifest,
        ground_truth=json.loads(members["ground_truth.json"]),
        frame_timing_by_node_id={node_id: json.loads(members[name]) for node_id, name in timing_member_by_node_id.items()},
    )
    edit(editable)
    members["manifest.json"] = json.dumps(editable.manifest).encode()
    members["ground_truth.json"] = json.dumps(editable.ground_truth).encode()
    for node_id, name in timing_member_by_node_id.items():
        members[name] = json.dumps(editable.frame_timing_by_node_id[node_id]).encode()
    with zipfile.ZipFile(out_path, "w") as out:
        for name, data in members.items():
            if name not in editable.removed_members:
                out.writestr(name, data)
    return out_path


def add_protocol_survey_fields(bundle: EditableBundle) -> None:
    for node in bundle.manifest["nodes"]:
        node["position_source"] = "operator_override"
        node["position_std_m"] = HAND_MEASURED_POSITION_STD_M
    bundle.node(PLANAR_NODE_ID)["orientation"] = dict(IDENTITY_ORIENTATION)
    bundle.node(TETRA_NODE_ID)["body_to_world_quaternion_wxyz"] = [1.0, 0.0, 0.0, 0.0]
    bundle.event["geometry"]["position_std_m"] = TRUTH_POSITION_STD_M


def _write_surveyed_bundle(tmp_path: Path, node_plans) -> Path:
    scene = build_replay_scene(node_plans)
    captures = render_captures_at_node_rates(scene, node_plans)
    raw_path = write_replay_bundle(tmp_path, scene=scene, node_plans=node_plans, captures=captures)
    return rewrite_bundle(raw_path, tmp_path / "surveyed.zip", add_protocol_survey_fields)


@pytest.fixture(scope="module")
def good_bundle_path(tmp_path_factory) -> Path:
    plans = [planar_plan(SAMPLE_RATE_48_KHZ), tetra_plan(SAMPLE_RATE_48_KHZ), point_plan(SAMPLE_RATE_48_KHZ)]
    return _write_surveyed_bundle(tmp_path_factory.mktemp("good_bundle"), plans)


@pytest.fixture(scope="module")
def mixed_rate_bundle_path(tmp_path_factory) -> Path:
    plans = [planar_plan(SAMPLE_RATE_48_KHZ), tetra_plan(SAMPLE_RATE_44_1_KHZ), point_plan(SAMPLE_RATE_48_KHZ)]
    return _write_surveyed_bundle(tmp_path_factory.mktemp("mixed_rate_bundle"), plans)


def _validate_edited(source_path: Path, tmp_path: Path, edit: BundleEdit, config: BundleValidationConfig = SHORT_BUNDLE_CONFIG):
    return validate_calibration_bundle(rewrite_bundle(source_path, tmp_path / "edited.zip", edit), config)


def _finding(report, code: Code):
    return next(finding for finding in report.findings if finding.code is code)


# --- the good bundle -----------------------------------------------------------


def test_good_bundle_has_no_errors_or_warnings(good_bundle_path: Path) -> None:
    report = validate_calibration_bundle(good_bundle_path, SHORT_BUNDLE_CONFIG)
    assert not report.has_errors and not report.has_warnings, "\n".join(f.one_line() for f in report.findings)
    assert {Code.CAPTURE_SUMMARY, Code.RANGE_LADDER_COVERAGE, Code.NODE_CLOCK_SUMMARY, Code.NODE_GAP_STATISTICS} <= report.codes()
    assert {f.node_id for f in report.findings if f.code is Code.NODE_CLOCK_SUMMARY} == {PLANAR_NODE_ID, TETRA_NODE_ID, POINT_NODE_ID}
    assert _finding(report, Code.RANGE_LADDER_COVERAGE).details["event_count_by_rung"]["1m"] == 1


def test_every_code_has_one_severity_and_the_report_round_trips_to_json(good_bundle_path: Path) -> None:
    assert set(SEVERITY_BY_FINDING_CODE) == set(Code)
    payload = validate_calibration_bundle(good_bundle_path, SHORT_BUNDLE_CONFIG).to_json_payload()
    assert payload["error_count"] == 0 and payload["warning_count"] == 0
    assert all(entry["severity"] == SEVERITY_BY_FINDING_CODE[Code(entry["code"])].value for entry in payload["findings"])


def test_protocol_defaults_flag_a_short_capture_and_thin_rungs(good_bundle_path: Path) -> None:
    report = validate_calibration_bundle(good_bundle_path)
    assert report.codes(FindingSeverity.WARNING) == {Code.CAPTURE_TOO_SHORT, Code.RANGE_LADDER_RUNG_UNDERCOVERED}
    assert {f.details["rung_m"] for f in report.findings if f.code is Code.RANGE_LADDER_RUNG_EMPTY} == {10.0, 100.0, 1000.0}


def _extracted_artifact_directory(bundle_path: Path, tmp_path: Path) -> tuple[Path, dict]:
    artifact_dir = tmp_path / "artifact"
    with zipfile.ZipFile(bundle_path) as bundle_zip:
        bundle_zip.extractall(artifact_dir)
    ground_truth_payload = json.loads((artifact_dir / "ground_truth.json").read_text())
    (artifact_dir / "ground_truth.json").unlink()  # supplied from storage, never read from the directory
    return artifact_dir, ground_truth_payload


def test_cli_exit_status_and_json(good_bundle_path: Path, tmp_path: Path, capsys) -> None:
    short_arguments = ["--minimum-capture-duration-seconds", "1", "--minimum-events-per-range-rung", "1"]
    assert main([str(good_bundle_path), *short_arguments]) == 0
    assert main([str(good_bundle_path)]) == 0  # warnings alone never fail
    assert main([str(good_bundle_path), "--strict"]) == 1
    capsys.readouterr()
    assert main([str(good_bundle_path), "--json", *short_arguments]) == 0
    assert json.loads(capsys.readouterr().out)["error_count"] == 0
    broken = rewrite_bundle(good_bundle_path, tmp_path / "broken.zip", lambda b: b.node(PLANAR_NODE_ID).pop("position_source"))
    assert main([str(broken), *short_arguments]) == 1


# --- one targeted defect per check ---------------------------------------------


def _set_node(node_id: str, **fields) -> BundleEdit:
    return lambda bundle: bundle.node(node_id).update(fields)


def _drop_node_keys(node_id: str, *keys: str) -> BundleEdit:
    def edit(bundle: EditableBundle) -> None:
        for key in keys:
            bundle.node(node_id).pop(key, None)

    return edit


def _shift_position_m(bundle: EditableBundle) -> None:
    node = bundle.node(PLANAR_NODE_ID)
    node["position_m"] = [node["position_m"][0] + 5.0, *node["position_m"][1:]]


def _overlap_frames(bundle: EditableBundle) -> None:
    bundle.frames(TETRA_NODE_ID)[5]["sample_offset"] -= 10


def _swap_frames(bundle: EditableBundle) -> None:
    frames = bundle.frames(TETRA_NODE_ID)
    frames[5], frames[6] = frames[6], frames[5]


def _overrun_last_frame(bundle: EditableBundle) -> None:
    bundle.frames(TETRA_NODE_ID)[-1]["sample_count"] += 1000


def _set_time_quality(value: str | None, first_frame_index: int = 0) -> BundleEdit:
    def edit(bundle: EditableBundle) -> None:
        for frame in bundle.frames(TETRA_NODE_ID)[first_frame_index:]:
            frame["time_quality"] = value

    return edit


def _drop_frame_timing_member(bundle: EditableBundle) -> None:
    bundle.removed_members.add(bundle.node(TETRA_NODE_ID)["frame_timing_file"])


def _remove_audio(bundle: EditableBundle) -> None:
    bundle.removed_members.add(bundle.node(TETRA_NODE_ID)["audio_file"])


def _add_channel(bundle: EditableBundle) -> None:
    node = bundle.node(POINT_NODE_ID)
    node["channel_sensor_ids"].append(f"{POINT_NODE_ID}:extra")
    node["sensor_offsets_m"].append([0.1, 0.0, 0.0])
    node["orientation"] = dict(IDENTITY_ORIENTATION)


def _extend_window(bundle: EditableBundle) -> None:
    bundle.manifest["time_window"]["end_ns"] += 10_000_000_000


TARGETED_DEFECTS: list[tuple[str, BundleEdit, Code, str | None]] = [
    ("site_mode", lambda b: b.manifest["site"].update(coordinate_mode="utm"), Code.SITE_COORDINATE_MODE_UNKNOWN, None),
    ("site_origin", lambda b: b.manifest["site"]["origin"].pop("lat"), Code.SITE_ORIGIN_INVALID, None),
    ("window", lambda b: b.manifest["time_window"].update(end_ns=b.manifest["time_window"]["start_ns"]), Code.CAPTURE_WINDOW_INVALID, None),
    ("node_type", _drop_node_keys(TETRA_NODE_ID, "node_type"), Code.NODE_TYPE_MISSING, TETRA_NODE_ID),
    ("no_position", _drop_node_keys(POINT_NODE_ID, "position_m", "position_geo"), Code.NODE_POSITION_MISSING, POINT_NODE_ID),
    ("frame_mismatch", _shift_position_m, Code.NODE_POSITION_FRAME_MISMATCH, PLANAR_NODE_ID),
    (
        "far_from_origin",
        lambda b: (b.node(POINT_NODE_ID).pop("position_geo"), b.node(POINT_NODE_ID).update(position_m=[50_000.0, 0.0, 0.0])),
        Code.NODE_POSITION_FAR_FROM_SITE_ORIGIN,
        POINT_NODE_ID,
    ),
    ("no_source", _drop_node_keys(PLANAR_NODE_ID, "position_source"), Code.NODE_POSITION_SOURCE_MISSING, PLANAR_NODE_ID),
    ("unknown_source", _set_node(PLANAR_NODE_ID, position_source="tape"), Code.NODE_POSITION_SOURCE_UNKNOWN, PLANAR_NODE_ID),
    ("hand_no_std", _drop_node_keys(PLANAR_NODE_ID, "position_std_m"), Code.HAND_MEASURED_POSITION_STD_MISSING, PLANAR_NODE_ID),
    (
        "gps_no_std",
        lambda b: (b.node(POINT_NODE_ID).update(position_source="reported_gps"), b.node(POINT_NODE_ID).pop("position_std_m")),
        Code.NODE_POSITION_STD_MISSING,
        POINT_NODE_ID,
    ),
    ("negative_std", _set_node(PLANAR_NODE_ID, position_std_m=-1.0), Code.NODE_POSITION_STD_INVALID, PLANAR_NODE_ID),
    ("fallback", _set_node(POINT_NODE_ID, position_source="fallback_static"), Code.NODE_POSITION_UNTRUSTED, POINT_NODE_ID),
    ("no_orientation", _set_node(PLANAR_NODE_ID, orientation={}), Code.NODE_ORIENTATION_MISSING, PLANAR_NODE_ID),
    (
        "bad_quaternion",
        _set_node(TETRA_NODE_ID, body_to_world_quaternion_wxyz=[1.0, 1.0, 0.0, 0.0]),
        Code.NODE_ORIENTATION_INVALID,
        TETRA_NODE_ID,
    ),
    (
        "orientation_conflict",
        _set_node(TETRA_NODE_ID, orientation={"yaw_deg": 30.0, "pitch_deg": 0.0, "roll_deg": 0.0}),
        Code.NODE_ORIENTATION_CONFLICT,
        TETRA_NODE_ID,
    ),
    ("offsets", lambda b: b.node(TETRA_NODE_ID)["sensor_offsets_m"].pop(), Code.NODE_SENSOR_OFFSETS_MISMATCH, TETRA_NODE_ID),
    ("audio_missing", _remove_audio, Code.NODE_AUDIO_MISSING, TETRA_NODE_ID),
    ("channel_count", _add_channel, Code.NODE_AUDIO_CHANNEL_MISMATCH, POINT_NODE_ID),
    ("rate", _set_node(PLANAR_NODE_ID, sample_rate_hz=SAMPLE_RATE_44_1_KHZ), Code.NODE_SAMPLE_RATE_INCONSISTENT, PLANAR_NODE_ID),
    ("coverage", _extend_window, Code.NODE_WINDOW_COVERAGE_PARTIAL, PLANAR_NODE_ID),
    ("timing_missing", _drop_node_keys(TETRA_NODE_ID, "frame_timing_file"), Code.FRAME_TIMING_MISSING, TETRA_NODE_ID),
    ("timing_unreadable", _drop_frame_timing_member, Code.FRAME_TIMING_UNREADABLE, TETRA_NODE_ID),
    (
        "timing_node",
        lambda b: b.frame_timing_by_node_id[TETRA_NODE_ID].update(node_id=PLANAR_NODE_ID),
        Code.FRAME_TIMING_NODE_MISMATCH,
        TETRA_NODE_ID,
    ),
    ("overlap", _overlap_frames, Code.FRAME_TIMING_OVERLAP, TETRA_NODE_ID),
    ("not_monotonic", _swap_frames, Code.FRAME_TIMING_NOT_MONOTONIC, TETRA_NODE_ID),
    ("empty_frame", lambda b: b.frames(TETRA_NODE_ID)[7].update(sample_count=0), Code.FRAME_TIMING_EMPTY_FRAME, TETRA_NODE_ID),
    ("outside_audio", _overrun_last_frame, Code.FRAME_TIMING_OUTSIDE_AUDIO, TETRA_NODE_ID),
    ("unreported_quality", _set_time_quality(None), Code.TIME_QUALITY_UNREPORTED, TETRA_NODE_ID),
    ("unknown_quality", _set_time_quality("atomic"), Code.TIME_QUALITY_UNKNOWN_VALUE, TETRA_NODE_ID),
    ("free_running", _set_time_quality("free_running", first_frame_index=50), Code.TIME_QUALITY_FREE_RUNNING, TETRA_NODE_ID),
    ("gt_invalid", lambda b: b.event.update(end_ns=b.event["start_ns"] - 1), Code.GROUND_TRUTH_INVALID, None),
    ("gt_empty", lambda b: b.ground_truth.update(events=[]), Code.GROUND_TRUTH_MISSING, None),
    ("gt_absent", lambda b: b.removed_members.add("ground_truth.json"), Code.GROUND_TRUTH_MISSING, None),
    (
        "gt_outside",
        lambda b: b.event.update(start_ns=b.event["start_ns"] + 3_600_000_000_000, end_ns=b.event["end_ns"] + 3_600_000_000_000),
        Code.GROUND_TRUTH_OUTSIDE_CAPTURE,
        None,
    ),
    ("gt_partial", lambda b: b.event.update(end_ns=b.event["end_ns"] + 5_000_000_000), Code.GROUND_TRUTH_PARTIALLY_OUTSIDE_CAPTURE, None),
    (
        "gt_far",
        lambda b: b.event.update(geometry={"type": "static", "position_m": [100_000.0, 0.0, 0.0]}),
        Code.GROUND_TRUTH_POSITION_OUTSIDE_SITE,
        None,
    ),
    ("gt_source_node", lambda b: b.event.update(source_node_id="ghost"), Code.GROUND_TRUTH_SOURCE_NODE_UNKNOWN, None),
    ("gt_coarse", lambda b: b.event["geometry"].update(position_std_m=5.0), Code.GROUND_TRUTH_ACCURACY_COARSE_FOR_RANGE, None),
]


@pytest.mark.parametrize(
    ("edit", "expected_code", "expected_node_id"),
    [pytest.param(edit, code, node_id, id=case_id) for case_id, edit, code, node_id in TARGETED_DEFECTS],
)
def test_targeted_defect_is_reported(good_bundle_path: Path, tmp_path: Path, edit: BundleEdit, expected_code: Code, expected_node_id) -> None:
    report = _validate_edited(good_bundle_path, tmp_path, edit)
    assert expected_code in report.codes(), "\n".join(f.one_line() for f in report.findings)
    if expected_node_id is not None:
        assert _finding(report, expected_code).node_id == expected_node_id


def test_unreadable_bundles_are_reported_not_raised(good_bundle_path: Path, tmp_path: Path) -> None:
    not_a_zip = tmp_path / "not_a_bundle.zip"
    not_a_zip.write_text("hello")
    assert validate_calibration_bundle(not_a_zip).codes() == {Code.BUNDLE_UNREADABLE}
    wrong_kind = _validate_edited(good_bundle_path, tmp_path, lambda b: b.manifest.update(kind="something_else"))
    assert wrong_kind.codes() == {Code.BUNDLE_UNREADABLE}


# --- clock evidence --------------------------------------------------------------


def _step_frames(step_ns: int) -> BundleEdit:
    def edit(bundle: EditableBundle) -> None:
        for frame in bundle.frames(TETRA_NODE_ID)[STEPPED_FRAME_INDEX:]:
            frame["start_time_ns"] += step_ns

    return edit


def test_e4_step_inside_one_upload_is_flagged_and_over_the_residual_limit(good_bundle_path: Path, tmp_path: Path) -> None:
    report = _validate_edited(good_bundle_path, tmp_path, _step_frames(TIMESTAMP_STEP_NS))
    step = _finding(report, Code.TIMESTAMP_STEP_DETECTED)
    assert step.node_id == TETRA_NODE_ID
    assert step.details["step_frame_indices"] == [STEPPED_FRAME_INDEX]
    assert step.details["largest_step_seconds"] == pytest.approx(TIMESTAMP_STEP_NS / 1e9)
    assert _finding(report, Code.TIMELINE_RESIDUAL_OVER_LIMIT).node_id == TETRA_NODE_ID
    assert not report.has_errors


def test_small_step_is_flagged_without_crossing_the_residual_limit(good_bundle_path: Path, tmp_path: Path) -> None:
    report = _validate_edited(good_bundle_path, tmp_path, _step_frames(SMALL_TIMESTAMP_STEP_NS))
    assert Code.TIMESTAMP_STEP_DETECTED in report.codes()
    assert Code.TIMELINE_RESIDUAL_OVER_LIMIT not in report.codes()


def test_clock_drift_is_measured_not_flagged(good_bundle_path: Path, tmp_path: Path) -> None:
    def drift(bundle: EditableBundle) -> None:
        sample_rate_hz = bundle.frame_timing_by_node_id[TETRA_NODE_ID]["sample_rate_hz"]
        for frame in bundle.frames(TETRA_NODE_ID):
            frame["start_time_ns"] += round(frame["sample_offset"] / sample_rate_hz * CLOCK_DRIFT_PPM * 1e3)

    report = _validate_edited(good_bundle_path, tmp_path, drift)
    assert not report.has_warnings
    summary = next(f for f in report.findings if f.code is Code.NODE_CLOCK_SUMMARY and f.node_id == TETRA_NODE_ID)
    assert summary.details["drift_ppm"] == pytest.approx(CLOCK_DRIFT_PPM, abs=DRIFT_TOLERANCE_PPM)


def test_clock_class_change_is_noted(good_bundle_path: Path, tmp_path: Path) -> None:
    report = _validate_edited(good_bundle_path, tmp_path, _set_time_quality("ntp_disciplined", first_frame_index=50))
    assert _finding(report, Code.TIME_QUALITY_CHANGED).details["transition_count"] == 1
    assert not report.has_warnings


def test_lost_packets_give_gap_statistics(good_bundle_path: Path, tmp_path: Path) -> None:
    def drop_frames(bundle: EditableBundle) -> None:
        frames = bundle.frames(TETRA_NODE_ID)
        frames[:] = [frame for index, frame in enumerate(frames) if index not in DROPPED_FRAME_INDICES]

    report = _validate_edited(good_bundle_path, tmp_path, drop_frames)
    statistics = next(f for f in report.findings if f.code is Code.NODE_GAP_STATISTICS and f.node_id == TETRA_NODE_ID).details
    assert statistics["gap_count"] == 1
    assert statistics["sequence_jump_count"] == 1
    assert statistics["longest_gap_seconds"] == pytest.approx(0.2)
    assert statistics["gap_fraction"] == pytest.approx(0.1)
    assert _finding(report, Code.NODE_GAP_FRACTION_HIGH).node_id == TETRA_NODE_ID
    assert Code.TIMESTAMP_STEP_DETECTED not in report.codes()  # a gap is not a step


# --- mixed rates -----------------------------------------------------------------


def test_mixed_rate_cluster_is_flagged_as_excluded_from_cross_node_pairs(mixed_rate_bundle_path: Path) -> None:
    report = validate_calibration_bundle(mixed_rate_bundle_path, SHORT_BUNDLE_CONFIG)
    assert report.codes(FindingSeverity.WARNING) == {Code.CLUSTER_SAMPLE_RATE_MISMATCH}
    groups = _finding(report, Code.CLUSTER_SAMPLE_RATE_MISMATCH).details["node_ids_by_sample_rate_hz"]
    assert groups == {str(SAMPLE_RATE_44_1_KHZ): [TETRA_NODE_ID], str(SAMPLE_RATE_48_KHZ): [PLANAR_NODE_ID, POINT_NODE_ID]}
    assert not report.has_errors


def test_separate_clusters_may_run_at_different_rates(mixed_rate_bundle_path: Path, tmp_path: Path) -> None:
    report = _validate_edited(mixed_rate_bundle_path, tmp_path, _set_node(TETRA_NODE_ID, cluster_id="mast"))
    assert Code.CLUSTER_SAMPLE_RATE_MISMATCH not in report.codes()


def test_reference_node_at_the_source_is_left_out_of_rates_and_ranges(mixed_rate_bundle_path: Path, tmp_path: Path) -> None:
    report = _validate_edited(mixed_rate_bundle_path, tmp_path, lambda b: b.event.update(source_node_id=TETRA_NODE_ID))
    assert Code.CLUSTER_SAMPLE_RATE_MISMATCH not in report.codes()
    assert Code.RANGE_LADDER_COVERAGE in report.codes()


# --- ground truth shapes and the range ladder -------------------------------------


def _as_trajectory(waypoint_offsets_seconds: list[float]) -> BundleEdit:
    def edit(bundle: EditableBundle) -> None:
        position_geo = bundle.event["geometry"]["position_geo"]
        bundle.event["geometry"] = {
            "type": "trajectory",
            "waypoints": [
                {"t_ns": bundle.event["start_ns"] + int(offset * 1e9), "position_geo": position_geo}
                for offset in waypoint_offsets_seconds
            ],
            "position_std_m": TRUTH_POSITION_STD_M,
        }

    return edit


def test_drone_style_trajectory_checks(good_bundle_path: Path, tmp_path: Path) -> None:
    dense = _validate_edited(good_bundle_path, tmp_path, _as_trajectory([-0.1, 0.2, 0.6]))
    assert not dense.has_warnings
    incomplete = _validate_edited(good_bundle_path, tmp_path, _as_trajectory([0.25]))
    assert Code.GROUND_TRUTH_TRAJECTORY_INCOMPLETE in incomplete.codes()
    sparse = _validate_edited(good_bundle_path, tmp_path, _as_trajectory([-2.0, 2.0]))
    assert Code.GROUND_TRUTH_TRAJECTORY_SPARSE in sparse.codes()


def test_truth_without_stated_accuracy_is_noted(good_bundle_path: Path, tmp_path: Path) -> None:
    report = _validate_edited(good_bundle_path, tmp_path, lambda b: b.event["geometry"].pop("position_std_m"))
    assert Code.GROUND_TRUTH_POSITION_STD_MISSING in report.codes(FindingSeverity.INFO)
    assert not report.has_warnings


def test_event_beyond_the_ladder_is_reported(good_bundle_path: Path, tmp_path: Path) -> None:
    def move_far(bundle: EditableBundle) -> None:
        planar_position_m = bundle.node(PLANAR_NODE_ID)["position_m"]
        bundle.event["geometry"] = {"type": "static", "position_m": [planar_position_m[0] + 5_000.0, *planar_position_m[1:]]}

    report = _validate_edited(good_bundle_path, tmp_path, move_far)
    assert Code.GROUND_TRUTH_OUTSIDE_RANGE_LADDER in report.codes()


@pytest.mark.parametrize(
    ("range_m", "expected_rung_m"),
    [(0.5, 1.0), (3.0, 1.0), (3.3, 10.0), (30.0, 10.0), (150.0, 100.0), (2_000.0, 1_000.0), (4_000.0, None), (0.2, None)],
)
def test_range_ladder_rungs_match_the_e5_matrix(range_m: float, expected_rung_m: float | None) -> None:
    assert range_ladder_rung_for(range_m, BundleValidationConfig()) == expected_rung_m


def _finding_tuples(report) -> list[tuple]:
    return [(finding.code, finding.node_id, finding.message) for finding in report.findings]


@pytest.mark.parametrize("edit", [None, _remove_audio, _set_time_quality("ntp_disciplined", first_frame_index=5)])
def test_artifact_directory_validation_matches_the_exported_zip(good_bundle_path: Path, tmp_path: Path, edit) -> None:
    # The API validates the session's artifact directory instead of rebuilding the zip.
    bundle_path = good_bundle_path if edit is None else rewrite_bundle(good_bundle_path, tmp_path / "edited.zip", edit)
    artifact_dir, ground_truth_payload = _extracted_artifact_directory(bundle_path, tmp_path)
    from_zip = validate_calibration_bundle(bundle_path, SHORT_BUNDLE_CONFIG)
    from_directory = validate_calibration_artifact_directory(artifact_dir, ground_truth_payload, SHORT_BUNDLE_CONFIG)
    assert _finding_tuples(from_directory) == _finding_tuples(from_zip)
