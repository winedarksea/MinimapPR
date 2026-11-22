"""Entity-aware replay metrics + ground-truth v2 bundle format on hand-built bundles."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from minimappr.calibration.bundle import load_bundle
from minimappr.calibration.ground_truth_events import (
    parse_ground_truth_events,
    site_frame_from_manifest,
)
from minimappr.sim.replay import evaluate_bundle

SECOND_NS = 1_000_000_000
T0_NS = 1_800_000_000 * SECOND_NS
ROBIN_POSITION_M = [0.0, 10.0, 0.0]
TANAGER_POSITION_M = [25.0, 10.0, 0.0]
REFLECTOR_IMAGE_POSITION_M = [-30.0, -20.0, 0.0]

MANIFEST = {
    "schema_version": 1,
    "kind": "minimappr_calibration_bundle",
    "session_id": "hand-built",
    "site": {"origin": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0}, "coordinate_mode": "flat"},
    "environment": {"temperature_c": 20.0, "humidity_fraction": 0.5},
    "nodes": [
        {"node_id": "node-a", "audio_file": "audio/node-a.wav", "position_m": [0.0, 0.0, 0.0],
         "channel_sensor_ids": ["node-a:ch0"]},
        {"node_id": "node-b", "audio_file": "audio/node-b.wav", "position_m": [20.0, 0.0, 0.0],
         "channel_sensor_ids": ["node-b:ch0"]},
    ],
}

# Entity gates enforced; legacy gates relaxed so tests isolate the new metrics.
STRICT_EXPECTATIONS = {
    "schema_version": 2,
    "classification": {"min_label_accuracy": 0.0, "match": "exact"},
    "localization": {"min_event_recall": 0.0, "max_median_position_error_m": 1e9,
                     "max_p90_position_error_m": 1e9},
    "entities": {
        "association_radius_m": 5.0,
        "min_localized_and_classified_recall": 1.0,
        "max_label_swap_rate": 0.0,
        "max_duplicate_entities": 0,
        "max_echo_promotions": 0,
    },
}


def _static_event(event_id, label, position_m, start_s, end_s, **extra):
    return {
        "event_id": event_id,
        "label": label,
        "label_category": "bird",
        "geometry": {"type": "static", "position_m": list(position_m)},
        "start_ns": T0_NS + int(start_s * SECOND_NS),
        "end_ns": T0_NS + int(end_s * SECOND_NS),
        **extra,
    }


def _detection(t_s, label, position_m=None, *, track_id=None, confidence=0.9, modality="localized",
               node_id="node-a", detection_id=None):
    return {
        "id": detection_id or f"det-{t_s}-{label}-{track_id}",
        "timestamp_ns": T0_NS + int(t_s * SECOND_NS),
        "label": label,
        "label_category": "bird",
        "label_confidence": confidence,
        "position_m": list(position_m) if position_m is not None else None,
        "reporting_modality": modality,
        "track_id": track_id,
        "source_node_id": node_id,
    }


def _write_bundle(tmp_path: Path, events: list[dict], *, schema_version: int = 2,
                  expectations: dict | None = None, name: str = "bundle.zip") -> Path:
    out = tmp_path / name
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("manifest.json", json.dumps(MANIFEST))
        zf.writestr("ground_truth.json",
                    json.dumps({"schema_version": schema_version, "events": events}))
        zf.writestr("detections.json", json.dumps({"schema_version": 1, "detections": []}))
        if expectations is not None:
            zf.writestr("expectations.json", json.dumps(expectations))
    return out


def test_v1_bundle_loads_unchanged_and_scores(tmp_path):
    frame = site_frame_from_manifest(MANIFEST)
    geo = frame.local_to_geo(ROBIN_POSITION_M)
    v1_event = {
        "event_id": "cgt-1", "label": "robin", "label_category": "bird", "source": "manual",
        "geometry": {"type": "static",
                     "position_geo": {"lat": geo.lat, "lon": geo.lon, "alt_m": geo.alt_m}},
        "start_ns": T0_NS, "end_ns": T0_NS + 2 * SECOND_NS, "notes": None,
    }
    bundle = load_bundle(_write_bundle(tmp_path, [v1_event], schema_version=1))
    assert bundle.events == [v1_event]  # raw dicts untouched
    (parsed,) = parse_ground_truth_events(bundle.events, frame)
    assert parsed.individual_id == "cgt-1"
    assert parsed.accepted_labels == {"robin"}
    assert not parsed.is_echo_source
    np.testing.assert_allclose(parsed.position_at(T0_NS), ROBIN_POSITION_M, atol=1e-6)

    report = evaluate_bundle([_detection(1.0, "robin", [0.0, 11.0, 0.0], track_id="T1")],
                             bundle, STRICT_EXPECTATIONS)
    assert report.event_recall == 1.0
    assert report.median_position_error_m == pytest.approx(1.0, abs=1e-3)
    assert report.entity_metrics.localized_and_classified_recall == 1.0
    assert report.passed, report.errors


def test_v2_validation_rejects_malformed_events(tmp_path):
    bad_cases = [
        _static_event("e1", "robin", ROBIN_POSITION_M, 0, 1, species=[]),
        _static_event("e1", "robin", ROBIN_POSITION_M, 0, 1, is_echo_source=True),
        _static_event("e1", "robin", ROBIN_POSITION_M, 0, 1,
                      species=[{"label": "robin", "status": "maybe"}]),
        {**_static_event("e1", "robin", ROBIN_POSITION_M, 0, 1),
         "geometry": {"type": "trajectory", "waypoints": [
             {"t_ns": 5, "position_m": [0, 0, 0]}, {"t_ns": 1, "position_m": [1, 0, 0]}]}},
        {**_static_event("e1", "robin", ROBIN_POSITION_M, 0, 1), "geometry": {"type": "static"}},
    ]
    for index, event in enumerate(bad_cases):
        with pytest.raises(ValueError):
            load_bundle(_write_bundle(tmp_path, [event], name=f"bad{index}.zip"))
    echo_to_nowhere = _static_event("echo", "robin", REFLECTOR_IMAGE_POSITION_M, 0, 1,
                                    is_echo_source=True, echo_of_event_id="missing")
    with pytest.raises(ValueError, match="unknown event"):
        load_bundle(_write_bundle(tmp_path, [echo_to_nowhere], name="echo.zip"))
    with pytest.raises(ValueError, match="schema_version"):
        load_bundle(_write_bundle(tmp_path, [], schema_version=99, name="v99.zip"))


def test_trajectory_position_is_interpolated_at_detection_time(tmp_path):
    event = {
        **_static_event("walk", "robin", [0, 0, 0], 0, 10),
        "geometry": {"type": "trajectory", "waypoints": [
            {"t_ns": T0_NS, "position_m": [0.0, 0.0, 0.0]},
            {"t_ns": T0_NS + 10 * SECOND_NS, "position_m": [20.0, 0.0, 0.0]},
        ]},
    }
    bundle = load_bundle(_write_bundle(tmp_path, [event]))
    (parsed,) = parse_ground_truth_events(bundle.events, site_frame_from_manifest(MANIFEST))
    np.testing.assert_allclose(parsed.position_at(T0_NS + 5 * SECOND_NS), [10.0, 0.0, 0.0])
    np.testing.assert_allclose(parsed.position_at(T0_NS - SECOND_NS), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(parsed.position_at(T0_NS + 99 * SECOND_NS), [20.0, 0.0, 0.0])

    report = evaluate_bundle([_detection(5.0, "robin", [10.0, 1.0, 0.0], track_id="T1")],
                             bundle, STRICT_EXPECTATIONS)
    assert report.median_position_error_m == pytest.approx(1.0)
    assert report.passed, report.errors


def test_localized_and_classified_recall_uses_entity_label(tmp_path):
    # Two bouts of one robin: bout 1's fix is weakly labelled, bout 2 is confident; both
    # land in track T1, so the entity is localized AND classified for both bouts.
    events = [
        _static_event("bout-1", "robin", ROBIN_POSITION_M, 0, 1, individual_id="robin-1"),
        _static_event("bout-2", "robin", ROBIN_POSITION_M, 8, 9, individual_id="robin-1"),
        _static_event("lonely", "tanager", TANAGER_POSITION_M, 20, 21),
    ]
    bundle = load_bundle(_write_bundle(tmp_path, events))
    detections = [
        _detection(0.5, "sparrow", ROBIN_POSITION_M, track_id="T1", confidence=0.25),
        _detection(8.5, "robin", ROBIN_POSITION_M, track_id="T1", confidence=0.85),
        _detection(20.5, "tanager", modality="omni", node_id="node-b"),  # heard, not localized
    ]
    metrics = evaluate_bundle(detections, bundle, STRICT_EXPECTATIONS).entity_metrics
    assert metrics.call_event_count == 3
    assert metrics.individual_count == 2
    assert metrics.localized_and_classified_event_count == 2
    assert metrics.localized_and_classified_recall == pytest.approx(2 / 3)

    # Flip confidences: the entity now displays the wrong species for both bouts.
    detections[0]["label_confidence"], detections[1]["label_confidence"] = 0.95, 0.5
    metrics = evaluate_bundle(detections, bundle, STRICT_EXPECTATIONS).entity_metrics
    assert metrics.localized_and_classified_event_count == 0


def test_label_swap_rate_counts_only_concurrent_rival_species(tmp_path):
    events = [
        _static_event("robin-call", "robin", ROBIN_POSITION_M, 0, 2),
        _static_event("tanager-call", "tanager", TANAGER_POSITION_M, 0, 2),
        _static_event("robin-solo", "robin", ROBIN_POSITION_M, 30, 31),
    ]
    bundle = load_bundle(_write_bundle(tmp_path, events))
    detections = [
        _detection(1.0, "tanager", ROBIN_POSITION_M, track_id="T-robin"),  # swap
        _detection(1.0, "tanager", TANAGER_POSITION_M, track_id="T-tanager"),  # correct
        _detection(30.5, "sparrow", ROBIN_POSITION_M, track_id="T-robin"),  # wrong, no rival
    ]
    report = evaluate_bundle(detections, bundle, STRICT_EXPECTATIONS)
    metrics = report.entity_metrics
    assert metrics.swap_eligible_detection_count == 2
    assert metrics.label_swap_count == 1
    assert metrics.label_swap_rate == pytest.approx(0.5)
    assert any("label swap rate" in error for error in report.errors)


def test_candidate_species_count_as_correct_and_never_as_swap(tmp_path):
    species = [{"label": "robin", "label_category": "bird", "status": "true"},
               {"label": "thrush", "label_category": "bird", "status": "candidate"}]
    events = [
        _static_event("a", "robin", ROBIN_POSITION_M, 0, 2, species=species),
        _static_event("b", "thrush", TANAGER_POSITION_M, 0, 2),
    ]
    bundle = load_bundle(_write_bundle(tmp_path, events))
    detections = [_detection(1.0, "thrush", ROBIN_POSITION_M, track_id="T1"),
                  _detection(1.0, "thrush", TANAGER_POSITION_M, track_id="T2")]
    metrics = evaluate_bundle(detections, bundle, STRICT_EXPECTATIONS).entity_metrics
    # Event b's species set is a subset of a's accepted set, so a's detection has no rival.
    assert metrics.label_swap_count == 0
    assert metrics.localized_and_classified_recall == 1.0


def test_duplicate_entities_count_omni_and_split_tracks(tmp_path):
    events = [_static_event("bout-1", "robin", ROBIN_POSITION_M, 0, 1, individual_id="robin-1"),
              _static_event("bout-2", "robin", ROBIN_POSITION_M, 10, 11, individual_id="robin-1")]
    bundle = load_bundle(_write_bundle(tmp_path, events))
    detections = [
        _detection(0.5, "robin", ROBIN_POSITION_M, track_id="T1"),
        _detection(0.6, "robin", modality="omni", node_id="node-a"),  # same bird, omni row
        _detection(10.5, "robin", ROBIN_POSITION_M, track_id="T2"),  # bout 2 split off
    ]
    report = evaluate_bundle(detections, bundle, STRICT_EXPECTATIONS)
    assert report.entity_metrics.entity_count_by_individual == {"robin-1": 3}
    assert report.entity_metrics.duplicate_entities == 2
    assert any("duplicate entities 2" in error for error in report.errors)

    single = [detections[0], {**detections[2], "track_id": "T1"}]
    report = evaluate_bundle(single, bundle, STRICT_EXPECTATIONS)
    assert report.entity_metrics.duplicate_entities == 0
    assert report.passed, report.errors


def test_echo_promotions_and_real_source_priority(tmp_path):
    events = [
        _static_event("call", "robin", ROBIN_POSITION_M, 0, 2),
        _static_event("call-echo", "robin", REFLECTOR_IMAGE_POSITION_M, 0, 2,
                      is_echo_source=True, echo_of_event_id="call",
                      reflector={"kind": "trunk", "position_m": [-15.0, -5.0, 0.0],
                                 "extra_delay_ms": 12.0, "attenuation_db": 8.0}),
    ]
    bundle = load_bundle(_write_bundle(tmp_path, events))
    detections = [
        _detection(1.0, "robin", ROBIN_POSITION_M, track_id="T-real"),
        _detection(1.0, "robin", REFLECTOR_IMAGE_POSITION_M, track_id="T-ghost"),
        _detection(1.2, "robin", [200.0, -200.0, 0.0], track_id="T-spurious"),
    ]
    report = evaluate_bundle(detections, bundle, STRICT_EXPECTATIONS)
    metrics = report.entity_metrics
    assert metrics.echo_event_count == 1
    assert metrics.call_event_count == 1
    assert metrics.echo_promotions == 1
    assert metrics.unassociated_localized_entities == 1
    assert metrics.duplicate_entities == 0
    assert len(report.event_results) == 1  # echo excluded from per-event metrics
    assert any("echo promotions 1" in error for error in report.errors)

    # A ghost track that also carries a real-source fix is the bird, not an echo.
    detections[1]["track_id"] = "T-real"
    assert evaluate_bundle(detections, bundle, STRICT_EXPECTATIONS).entity_metrics.echo_promotions == 0


def test_far_field_events_associate_by_bearing(tmp_path):
    far_position_m = [0.0, 400.0, 0.0]  # beyond far_field_distance_m from the centroid
    bundle = load_bundle(_write_bundle(tmp_path, [_static_event("far", "robin", far_position_m, 0, 2)]))
    near_bearing_wrong_range = _detection(1.0, "robin", [10.0, 150.0, 0.0], track_id="T1")
    metrics = evaluate_bundle([near_bearing_wrong_range], bundle, STRICT_EXPECTATIONS).entity_metrics
    assert metrics.localized_and_classified_recall == 1.0
    off_bearing = _detection(1.0, "robin", [150.0, 10.0, 0.0], track_id="T1")
    metrics = evaluate_bundle([off_bearing], bundle, STRICT_EXPECTATIONS).entity_metrics
    assert metrics.localized_and_classified_recall == 0.0
    assert metrics.unassociated_localized_entities == 1


def test_entity_gates_are_report_only_by_default(tmp_path):
    events = [_static_event("a", "robin", ROBIN_POSITION_M, 0, 1)]
    bundle = load_bundle(_write_bundle(tmp_path, events))
    detections = [_detection(0.5, "robin", ROBIN_POSITION_M, track_id="T1"),
                  _detection(0.5, "robin", modality="omni")]
    relaxed = {key: value for key, value in STRICT_EXPECTATIONS.items() if key != "entities"}
    report = evaluate_bundle(detections, bundle, relaxed)
    assert report.entity_metrics.duplicate_entities == 1
    assert report.passed, report.errors


def test_omni_detection_ambiguity_is_left_unassociated(tmp_path):
    events = [_static_event("a", "robin", ROBIN_POSITION_M, 0, 2),
              _static_event("b", "tanager", TANAGER_POSITION_M, 0, 2)]
    bundle = load_bundle(_write_bundle(tmp_path, events))
    ambiguous_wrong_label = _detection(1.0, "sparrow", modality="omni")
    metrics = evaluate_bundle([ambiguous_wrong_label], bundle, STRICT_EXPECTATIONS).entity_metrics
    assert metrics.entity_count_by_individual == {}
    label_matched = _detection(1.0, "tanager", modality="omni")
    metrics = evaluate_bundle([label_matched], bundle, STRICT_EXPECTATIONS).entity_metrics
    assert metrics.entity_count_by_individual == {"b": 1}
    assert metrics.localized_and_classified_recall == 0.0


def test_synthetic_entity_scenarios_render_loadable_v2_bundles(tmp_path):
    from minimappr.sim.synthetic_entity_scenarios import (
        SYNTHETIC_ENTITY_SCENARIOS,
        DominantFrequencySpeciesClassifier,
        render_scenario_bundle,
        species_bands_for,
    )

    for scenario in SYNTHETIC_ENTITY_SCENARIOS:
        bundle = load_bundle(render_scenario_bundle(scenario, tmp_path / f"{scenario.name}.zip"))
        assert len(bundle.events) == len(scenario.calls)
        channels, sample_rate_hz = bundle.node_audio("node-a")
        assert channels.shape == (4, int(scenario.duration_s * sample_rate_hz))
        parsed = parse_ground_truth_events(bundle.events, site_frame_from_manifest(bundle.manifest))
        assert {e.is_echo_source for e in parsed} >= {False}
        classifier = DominantFrequencySpeciesClassifier(species_bands_for(scenario))
        first_call = scenario.calls[0]
        start = int(first_call.start_s * sample_rate_hz)
        window = channels[0, start : start + sample_rate_hz]
        assert classifier.classify(window, sample_rate_hz).label == first_call.species_label
