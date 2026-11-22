"""Golden cases for the Rust ``wildlife_tracking`` / ``bird_entity`` parity gate.

Python is the reference. Component cases call the S3/S6 Python functions
directly (log-odds belief, perched filter, nearest-neighbour association,
species association gate, concurrent-perch guard, entity geometry and
absorption). Scenario cases drive the real ``TrackManager`` (S3 flags on,
dormant reacquire and fingerprints off: not ported), ``BirdEntityRegistry``
and ``EntityReportingPolicy`` in FusionNode order (see
``tests/wildlife_pipeline_reference.py``) on scripted bird scenes.

``python -m tests.wildlife_tracking_parity_cases`` writes
``tests/fixtures/wildlife_tracking_parity_goldens.npz`` holding
``case_inputs_json`` and ``expected_outputs_json`` (one JSON string per case).
The parity test feeds the inputs to the sidecar's ``wildlife-tracking-oracle``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from minimappr.core.bird_entity_absorption_geometry import covariance_gated_fix_score, fixes_are_two_birds
from minimappr.core.bird_entity_model import (
    EntityFixSample,
    bearing_estimate_toward,
    horizontal_major_sigma_m,
    uncertainty_ellipse,
)
from minimappr.core.track_associators import NearestNeighborAssociator
from minimappr.core.track_filters_perched import PerchedRandomWalkTrackFilter
from minimappr.core.track_perched_concurrency import ConcurrentPerchGuardConfig, concurrently_active_perch_conflicts
from minimappr.core.track_species_association_gate import species_claims_contradict
from minimappr.core.track_species_belief import SpeciesBeliefConfig, SpeciesLogOddsBelief
from minimappr.models import TrackState
from tests.wildlife_pipeline_reference import entity_config, run_pipeline_case
from tests.wildlife_scenario_scripts import SCENARIO_BUILDERS

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "wildlife_tracking_parity_goldens.npz"
SPECIES = ("american robin", "scarlet tanager", "rose-breasted grosbeak", "black-capped chickadee", "unknown")
SECOND_NS = 1_000_000_000
T0_NS = 1_700_000_000_000_000_000


def _random_covariance(rng: np.random.Generator, scale_m: float) -> list[list[float]]:
    basis = rng.normal(size=(3, 3)) * scale_m
    return (basis @ basis.T + np.eye(3) * 0.05).tolist()


def _random_evidence(rng: np.random.Generator) -> list[list]:
    labels = rng.choice(len(SPECIES), size=int(rng.integers(0, 4)), replace=False)
    return [[SPECIES[index], float(np.round(rng.uniform(0.0, 1.0), 3))] for index in labels]


def _track_state(track_id: str, position, covariance, last_seen_ns: int) -> TrackState:
    return TrackState(
        id=track_id, first_seen_ns=T0_NS, last_seen_ns=last_seen_ns, position_m=tuple(position),
        position_covariance_m2=covariance, velocity_mps=(0.0, 0.0, 0.0), label="x", label_category="wildlife",
        iff_category="unknown", confidence=0.5, update_count=1, status="tentative", tqi=0.5,
        capability_tier="full_3d", track_kind="acoustic",
    )


def belief_case(seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    updates = [_random_evidence(rng) for _ in range(12)]
    belief = SpeciesLogOddsBelief(SpeciesBeliefConfig())
    outputs = []
    for evidence in updates:
        belief.update({label: probability for label, probability in evidence})
        top_label, top_probability = belief.top_label()
        outputs.append({
            "exposed": [[label, value] for label, value in belief.exposed_probabilities().items()],
            "top_label": top_label, "top_probability": top_probability,
            "passes_display": belief.passes_display_threshold(),
        })
    return {"kind": "belief", "updates": updates}, {"steps": outputs}


def perched_filter_case(seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    perched = PerchedRandomWalkTrackFilter(
        position_diffusion_m2_per_second=0.05, measurement_noise_m2=1.5,
        initial_position_variance_m2=4.0, relocation_snap_distance_m=6.0,
    )
    perch = rng.uniform(-20, 20, size=3)
    initial_position_m = perch.tolist()
    perched.initialize_track("t", tuple(perch))
    state = _track_state("t", perch, None, T0_NS)
    steps, outputs = [], []
    for _ in range(10):
        if rng.uniform() < 0.2:
            perch = perch + rng.uniform(-15, 15, size=3)
        measurement = (perch + rng.normal(scale=1.0, size=3)).tolist()
        covariance = _random_covariance(rng, float(rng.uniform(0.2, 3.0))) if rng.uniform() < 0.7 else None
        dt_s = float(rng.uniform(0.5, 12.0))
        steps.append({"measurement_m": measurement, "covariance_m2": covariance, "dt_s": dt_s})
        relocations_before = perched.relocation_count
        state = perched.update(state, tuple(measurement), dt_s, covariance)
        predicted = perched.predict(state, 3.0)
        outputs.append({
            "position_m": list(state.position_m), "covariance_m2": state.position_covariance_m2,
            "relocated": perched.relocation_count > relocations_before,
            "predicted_covariance_m2": predicted.position_covariance_m2,
        })
    return {"kind": "perched_filter", "initial_position_m": initial_position_m, "steps": steps}, {"steps": outputs}


def association_case(seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    associator = NearestNeighborAssociator(8.0, max_gate_m=32.0, chi2_gate=9.0, fingerprint_weight=0.0)
    tracks = []
    for index in range(int(rng.integers(1, 6))):
        covariance = _random_covariance(rng, float(rng.uniform(0.3, 8.0)))
        tracks.append({"track_id": index + 1, "position_m": rng.uniform(-30, 30, size=3).tolist(), "covariance_m2": covariance})
    position = rng.uniform(-30, 30, size=3).tolist()
    measurement_covariance = _random_covariance(rng, float(rng.uniform(0.3, 6.0))) if rng.uniform() < 0.7 else None
    states = [_track_state(f"trk-{t['track_id']:05d}", t["position_m"], t["covariance_m2"], T0_NS) for t in tracks]
    matched = associator.associate(T0_NS, tuple(position), states, measurement_covariance)
    case = {"kind": "association", "position_m": position, "covariance_m2": measurement_covariance, "tracks": tracks}
    return case, {"matched_track_id": int(matched.split("-")[1]) if matched else None}


def species_gate_case(seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    pairs = [(_random_evidence(rng), _random_evidence(rng)) for _ in range(20)]
    decisions = [
        species_claims_contradict(dict(map(tuple, detection)), dict(map(tuple, track)), 0.5, 0.15)
        for detection, track in pairs
    ]
    return {"kind": "species_gate", "pairs": pairs}, {"contradicts": decisions}


def concurrency_case(seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    config = ConcurrentPerchGuardConfig(enabled=True, snap_distance_m=6.0, min_silence_seconds=2.0, concurrent_activity_seconds=30.0)
    now_ns = T0_NS + 60 * SECOND_NS
    tracks = []
    for index in range(int(rng.integers(1, 6))):
        tracks.append({
            "track_id": index + 1, "position_m": rng.uniform(-20, 20, size=3).tolist(),
            "covariance_m2": _random_covariance(rng, float(rng.uniform(0.3, 4.0))),
            "last_seen_ns": int(now_ns - rng.uniform(0.0, 40.0) * SECOND_NS),
            "top_species": [str(rng.choice(SPECIES[:2]))],
        })
    position = rng.uniform(-20, 20, size=3).tolist()
    covariance = _random_covariance(rng, 1.0)
    species = str(rng.choice(SPECIES[:2]))
    states = [_track_state(f"trk-{t['track_id']:05d}", t["position_m"], t["covariance_m2"], t["last_seen_ns"]) for t in tracks]
    top_by_id = {f"trk-{t['track_id']:05d}": t["top_species"] for t in tracks}
    conflicts = concurrently_active_perch_conflicts(
        timestamp_ns=now_ns, position_m=tuple(position), measurement_covariance_m2=covariance,
        detection_species=species, tracks=states, top_species_for=lambda track: top_by_id[track.id],
        is_perched=lambda _track_id: True, config=config,
    )
    case = {"kind": "concurrency", "timestamp_ns": now_ns, "position_m": position, "covariance_m2": covariance,
            "species": species, "tracks": tracks}
    return case, {"conflicts": sorted(int(track_id.split("-")[1]) for track_id in conflicts)}


def entity_geometry_case(seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    config = entity_config()
    samples = []
    for _ in range(12):
        node = rng.uniform(-30, 30, size=3).tolist()
        target = (np.asarray(node) + rng.uniform(-80, 80, size=3)).tolist()
        covariance = _random_covariance(rng, float(rng.uniform(0.2, 25.0))) if rng.uniform() < 0.8 else None
        other = (np.asarray(target) + rng.normal(scale=float(rng.uniform(0.5, 15.0)), size=3)).tolist()
        other_covariance = _random_covariance(rng, float(rng.uniform(0.2, 6.0))) if rng.uniform() < 0.8 else None
        dt_ns = int(rng.uniform(0.0, 3.0) * SECOND_NS)
        samples.append({"node_m": node, "target_m": target, "covariance_m2": covariance, "other_m": other,
                        "other_covariance_m2": other_covariance, "dt_ns": dt_ns, "range_observable": bool(rng.uniform() < 0.5)})
    outputs = []
    for sample in samples:
        bearing = bearing_estimate_toward(
            origin_node_id="n", origin_position_m=tuple(sample["node_m"]), target_position_m=tuple(sample["target_m"]),
            covariance_m2=sample["covariance_m2"], range_observable=sample["range_observable"],
            default_half_width_deg=config.bearing_default_half_width_deg, min_half_width_deg=config.bearing_min_half_width_deg,
            max_half_width_deg=config.bearing_max_half_width_deg, sigma_multiplier=config.uncertainty_sigma_multiplier,
        )
        ellipse = uncertainty_ellipse(sample["covariance_m2"], config.uncertainty_sigma_multiplier)
        score = covariance_gated_fix_score(
            tuple(sample["target_m"]), sample["covariance_m2"], tuple(sample["other_m"]), sample["other_covariance_m2"], config
        )
        two_birds = fixes_are_two_birds(
            EntityFixSample(T0_NS, tuple(sample["target_m"]), sample["covariance_m2"]),
            EntityFixSample(T0_NS + sample["dt_ns"], tuple(sample["other_m"]), sample["other_covariance_m2"]),
            config,
        )
        outputs.append({
            "horizontal_sigma_m": horizontal_major_sigma_m(sample["covariance_m2"]),
            "bearing_deg": bearing.bearing_deg, "half_width_deg": bearing.half_width_deg,
            "range_min_m": bearing.range_min_m, "range_max_m": bearing.range_max_m,
            "ellipse": list(ellipse) if ellipse is not None else None,
            "fix_score": score, "two_birds": two_birds,
        })
    return {"kind": "entity_geometry", "samples": samples}, {"samples": outputs}


def all_cases() -> list[tuple[dict, dict]]:
    cases = []
    for seed in range(4):
        cases.append(belief_case(seed))
        cases.append(perched_filter_case(100 + seed))
        cases.append(species_gate_case(200 + seed))
        cases.append(entity_geometry_case(300 + seed))
    cases.extend(association_case(400 + seed) for seed in range(24))
    cases.extend(concurrency_case(500 + seed) for seed in range(16))
    for name, builder in SCENARIO_BUILDERS.items():
        for seed in range(3):
            case = {"kind": "pipeline", "name": name, "seed": seed, "steps": builder(seed)}
            cases.append((case, run_pipeline_case(case["steps"])))
    return cases


def main() -> None:
    cases = all_cases()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        GOLDEN_PATH,
        case_inputs_json=np.asarray([json.dumps(case) for case, _ in cases]),
        expected_outputs_json=np.asarray([json.dumps(expected) for _, expected in cases]),
    )
    print(f"wrote {len(cases)} cases to {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
