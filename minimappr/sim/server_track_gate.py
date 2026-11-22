"""Server-track "S gate": every Field Mode server stream together, on sim scenes.

Each stream (S2 onset clustering, S3 perched tracker + species log-odds, S4
partial binding + two-threshold BirdNET, S5 attribution + relative trigger, S6
entity reporting) was measured in isolation. This module replays fixed-seed
Rust-simulator scenes through a full FusionNode under named flag sets
(baseline, one stream, all-on, all-on minus one) and scores them with the
entity metrics, so the combinations are measured too.

Scoring choices (``docs/field_mode_progress/server_track_gate.md``):

* Classifier: ``SnrGradedSpeciesOracle`` (confidence follows in-band SNR,
  multi-label, BirdNET floors, two-threshold wrapper when flagged), so the
  belief/threshold/LOCO streams have graded evidence to act on.
* Production reporting window (30 s) and a realistic absolute trigger
  (``GATE_TRIGGER_RMS``, the env default), not the hand-tuned per-scene levels.
* Rows are scored as the operator sees them: sub-display (two-threshold) rows
  whose track is not visible are dropped (they are persisted for belief only),
  and with entity reporting each row carries its entity id (entities are memory
  only; the DB row does not store it).
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from minimappr.classifiers.birdnet_two_threshold import detection_is_below_display_threshold
from minimappr.classifiers.factory import create_context_classifier
from minimappr.classifiers.routing import CONTEXT_DETECTION_TRIGGER, ContextSpec, apply_settings, default_routing
from minimappr.config import Settings
from minimappr.core import onset_srp_localization_stage
from minimappr.sim import sim_scenario_scoring as scoring
from minimappr.sim.fixture_audio_scenarios import (
    build_species_gate_scenario,
    load_fixture_bird_clips,
    render_scenario_bundle,
)
from minimappr.sim.graded_species_oracle import ABSTENTION_LABEL, build_graded_oracle_for_settings
from minimappr.sim.replay import build_fusion_for_bundle, evaluate_bundle, replay_bundle

# Stream -> flags it turns on. S5 attribution requires onset clustering (config
# validation), so its "alone" row carries S2 as well.
STREAM_FLAGS: dict[str, dict[str, bool]] = {
    "S2_onset_clustering": {"localization_onset_clustering_enabled": True},
    "S3_perched_filter": {"tracking_perched_filter_enabled": True},
    "S3_species_log_odds": {"tracking_species_log_odds_enabled": True},
    "S4_partial_binding": {"tracking_partial_constraint_binding_enabled": True},
    "S4_two_threshold": {"birdnet_two_threshold_enabled": True},
    "S5_attribution": {"onset_attribution_enabled": True},
    "S5_relative_trigger": {"trigger_noise_floor_relative_enabled": True},
    "S6_entity_reporting": {"entity_reporting_enabled": True},
}
STREAM_PREREQUISITES: dict[str, tuple[str, ...]] = {"S5_attribution": ("S2_onset_clustering",)}
BASELINE = "baseline"
ALL_ON = "all_on"
ALONE_PREFIX = "only:"
LEAVE_ONE_OUT_PREFIX = "all_on-minus:"

# The env default (`MINIMAPPR_TRIGGER_RMS`); the scenes' own 0.003 / 0.00015 were
# hand-set per scene, which hides what the relative trigger is for.
GATE_TRIGGER_RMS = 0.001
GATE_COMMON_OVERRIDES: dict[str, Any] = {"trigger_rms": GATE_TRIGGER_RMS, "classification_window_seconds": 1.0}
GATE_ASSOCIATION_RADIUS_M = 5.0
GATE_SCENE_SEED = 0

# Scenes: the six W2/W3 onset scenes plus the S-gate scenes (same 20 m layout
# family). Single-array scenes keep their own geometry.
GATE_SCENES: tuple[str, ...] = (
    "onset_overlap_two_species",
    "onset_trunk_echo",
    "onset_faint_far_bird",
    "onset_same_band_pair",
    "onset_same_band_two_species",
    "onset_single_tetra_overlap",
    "gate_countersinging_pair",
    "gate_flock_echo",
    "gate_omni_then_localized",
    "gate_collinear_trail",
)

# Real-audio scene: four xeno-canto fixture birds on the species-gate perches,
# two bouts each, classified by the production BirdNET v2.4 member (3 s
# windows, as in the cutover gate). Scarlet Tanager exercises the "car"
# category bug fixed alongside this gate.
FIXTURE_SCENE_PREFIX = "fixture_"
FIXTURE_SCENE = "fixture_birdnet_four_birds"
FIXTURE_SCENE_SPECIES = ("American Robin", "Scarlet Tanager", "Rose-breasted Grosbeak", "White-throated Sparrow")
FIXTURE_SCENE_BOUTS_PER_BIRD = 2
# BirdNET (plus LOCO reruns with all flags on) on a loaded machine can keep the
# classification queue busy far past the oracle scenes' 120 s drain budget.
FIXTURE_QUEUE_DRAIN_TIMEOUT_SECONDS = 900.0
FIXTURE_SCENE_OVERRIDES: dict[str, Any] = {"classification_window_seconds": 3.0}
FIXTURE_SCENE_EXPECTATIONS: dict[str, Any] = {
    "schema_version": 2,
    "classification": {"min_label_accuracy": 0.0, "match": "exact"},
    "localization": {"min_event_recall": 0.0, "far_field_distance_m": 120.0},
    "entities": {"association_time_slop_s": 2.0},
}

ENTITY_KEYS = (
    "call_event_count",
    "individual_count",
    "localized_and_classified_recall",
    "localized_and_classified_individual_recall",
    "label_swap_count",
    "swap_eligible_detection_count",
    "duplicate_entities",
    "echo_promotions",
    "unassociated_localized_entities",
    "entities_spanning_multiple_individuals",
)


def configuration_flags(name: str) -> dict[str, bool]:
    """Flag overrides for a named configuration (every stream flag set explicitly)."""
    enabled: set[str]
    if name == BASELINE:
        enabled = set()
    elif name == ALL_ON:
        enabled = set(STREAM_FLAGS)
    elif name.startswith(ALONE_PREFIX):
        stream = name[len(ALONE_PREFIX):]
        enabled = {stream, *STREAM_PREREQUISITES.get(stream, ())}
    elif name.startswith(LEAVE_ONE_OUT_PREFIX):
        dropped = name[len(LEAVE_ONE_OUT_PREFIX):]
        enabled = {s for s in STREAM_FLAGS if s != dropped}
        # Dropping a prerequisite drops what depends on it.
        enabled -= {s for s, prereqs in STREAM_PREREQUISITES.items() if dropped in prereqs}
    else:
        raise ValueError(f"unknown configuration {name!r}")
    if any(s not in STREAM_FLAGS for s in enabled):
        raise ValueError(f"unknown stream in configuration {name!r}")
    flags = {flag: False for stream_flags in STREAM_FLAGS.values() for flag in stream_flags}
    for stream in enabled:
        flags.update(STREAM_FLAGS[stream])
    return flags


def all_configuration_names(*, alone: bool = True) -> list[str]:
    """Baseline, all-on and all-on-minus-each first (the gate's core); one-stream rows last."""
    names = [BASELINE, ALL_ON, *(f"{LEAVE_ONE_OUT_PREFIX}{s}" for s in STREAM_FLAGS)]
    if alone:
        names += [f"{ALONE_PREFIX}{s}" for s in STREAM_FLAGS]
    return names


@dataclass
class OnsetClusteringTimer:
    """Wall time of each onset-clustering call (worker thread), for the cost table."""

    durations_s: list[float] = field(default_factory=list)

    @contextlib.contextmanager
    def installed(self):
        original = onset_srp_localization_stage.localize_onset_clusters

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.durations_s.append(time.perf_counter() - started)

        onset_srp_localization_stage.localize_onset_clusters = timed
        try:
            yield self
        finally:
            onset_srp_localization_stage.localize_onset_clusters = original


def _entity_ids(fusion) -> tuple[dict[str, str], dict[str, str], dict[str, list[float]]]:
    """(canonical detection -> entity, bound track -> entity, entity -> displayed position)."""
    coordinator = getattr(fusion, "entity_reporting", None)
    if coordinator is None:
        return {}, {}, {}
    entities = coordinator.registry.entities()
    by_detection = {str(e.canonical_detection_id): e.entity_id for e in entities if e.canonical_detection_id}
    by_track = {str(e.track_id): e.entity_id for e in entities if e.track_id}
    positions = {e.entity_id: [float(v) for v in e.position_m] for e in entities if e.position_m is not None}
    return by_detection, by_track, positions


def surfaced_detections(
    rows: list[dict],
    *,
    visible_track_ids: set[str],
    entity_ids: dict[str, str],
    entity_ids_by_track: dict[str, str] | None = None,
    entity_positions_m: dict[str, list[float]] | None = None,
) -> list[dict]:
    """Rows as the operator sees them.

    * unknown / zero-confidence rows are never broadcast (rules stage drop);
    * a sub-display (two-threshold) row shows only through a visible track;
    * with entity reporting, a row belongs to its entity: as the entity's
      canonical row, or through the track bound to the entity (the map draws a
      bound track only as its entity), and is placed where the entity is drawn
      (``entity_positions_m``: the bound track's filtered position). The
      canonical DB row itself holds the latest bout's raw fix.
    """
    entity_ids_by_track = entity_ids_by_track or {}
    entity_positions_m = entity_positions_m or {}
    surfaced = []
    for row in rows:
        if str(row.get("label") or ABSTENTION_LABEL).strip().lower() == ABSTENTION_LABEL:
            continue
        if not float(row.get("label_confidence") or 0.0) > 0.0:
            continue
        track_id = row.get("track_id")
        hidden_track = track_id is not None and str(track_id) not in visible_track_ids
        if detection_is_below_display_threshold(row.get("feature_summary")) and (track_id is None or hidden_track):
            continue
        row = dict(row)
        entity_id = entity_ids.get(str(row.get("id"))) or entity_ids_by_track.get(str(track_id))
        if entity_id:
            row["entity_id"] = entity_id
            if entity_id in entity_positions_m and row.get("position_m") is not None:
                row["position_m"] = entity_positions_m[entity_id]
        surfaced.append(row)
    return surfaced


def gate_expectations(bundle) -> dict[str, Any]:
    expectations = copy.deepcopy(bundle.expectations or {})
    expectations.setdefault("classification", {})["match"] = "exact"
    expectations.setdefault("entities", {})["association_radius_m"] = GATE_ASSOCIATION_RADIUS_M
    expectations["entities"]["horizontal_only"] = True
    expectations.setdefault("localization", {})["horizontal_position_error"] = True
    return expectations


async def replay_scene_configuration(
    scene: str,
    configuration: str,
    work_dir: Path,
    *,
    extra_overrides: dict[str, Any] | None = None,
    seed: int = GATE_SCENE_SEED,
) -> dict[str, Any]:
    """Render (cached per work_dir), replay and score one (scene, configuration)."""
    overrides = {**GATE_COMMON_OVERRIDES, **configuration_flags(configuration)}
    if scene.startswith(FIXTURE_SCENE_PREFIX):
        bundle, classifier, overrides = _prepare_fixture_scene(scene, work_dir, seed, overrides, extra_overrides)
    else:
        overrides.update(extra_overrides or {})
        scenario_path = scoring.SCENARIO_DIRECTORY / f"{scene}.toml"
        bundle_path = bundle_path_for(scene, work_dir, seed)
        if not bundle_path.exists():
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            scoring.render_scenario(scenario_path, bundle_path, seed=seed)
        bundle = load_scene_bundle(scene, bundle_path)
        bands = scoring.species_bands_from_scenario(scenario_path)
        classifier = build_graded_oracle_for_settings(Settings(**overrides), bands)
    replay_dir = work_dir / "replay" / f"{scene}-{configuration.replace(':', '_')}"
    # A DB left by an interrupted run would be scored together with this one.
    shutil.rmtree(replay_dir, ignore_errors=True)
    timer = OnsetClusteringTimer()
    started = time.perf_counter()
    with timer.installed():
        fusion, storage, _ = await build_fusion_for_bundle(bundle, replay_dir, overrides=overrides, classifier=classifier)
        try:
            await replay_bundle(fusion, bundle)
            drain_timeout_s = (
                FIXTURE_QUEUE_DRAIN_TIMEOUT_SECONDS
                if scene.startswith(FIXTURE_SCENE_PREFIX)
                else scoring.QUEUE_DRAIN_TIMEOUT_SECONDS
            )
            await scoring.drain_fusion_queues(fusion, timeout_seconds=drain_timeout_s)
            rows = await scoring.collect_settled_detections(storage, fusion)
            visible = {str(track.id) for track in await fusion.tracker.snapshot()}
            entity_ids, entity_ids_by_track, entity_positions = _entity_ids(fusion)
            metrics = fusion._metrics
            loco_reruns = int(metrics.onset_attribution_loco_rerun_count)
            loco_ms = float(metrics.onset_attribution_loco_rerun_total_ms)
            silent_drops = {
                stage: dict(getattr(metrics, counter_field))
                for stage, counter_field in fusion._DROP_REASON_COUNTER_FIELDS.items()
                if getattr(metrics, counter_field)
            }
        finally:
            await fusion.stop()
            await storage.close()
            classifier.close()
    wall_s = time.perf_counter() - started
    raw = {
        "rows": rows,
        "visible_track_ids": sorted(visible),
        "entity_ids": entity_ids,
        "entity_ids_by_track": entity_ids_by_track,
        "entity_positions_m": entity_positions,
    }
    return {
        "scene": scene,
        "configuration": configuration,
        "seed": seed,
        "bundle_path": str(bundle_path_for(scene, work_dir, seed)),
        **score_raw_replay(scene, bundle, raw),
        "audio_seconds": _bundle_duration_s(bundle),
        "replay_wall_s": round(wall_s, 3),
        "onset_windows": len(timer.durations_s),
        "onset_clustering_s": [round(d, 4) for d in timer.durations_s],
        "loco_reruns": loco_reruns,
        "loco_rerun_ms": round(loco_ms, 2),
        "silent_drops": silent_drops,
        "raw": raw,
    }


def bundle_path_for(scene: str, work_dir: Path, seed: int) -> Path:
    return work_dir / "bundles" / f"{scene}-s{seed}.zip"


def load_scene_bundle(scene: str, bundle_path: Path):
    bundle = scoring.load_bundle_with_echo_truth(bundle_path)
    if scene.startswith(FIXTURE_SCENE_PREFIX):
        bundle = dataclasses.replace(bundle, expectations=copy.deepcopy(FIXTURE_SCENE_EXPECTATIONS))
    return bundle


def score_raw_replay(scene: str, bundle, raw: dict[str, Any], *, use_entity_positions: bool = True) -> dict[str, Any]:
    """Metrics from a replay's stored rows (re-runnable offline when scoring rules change)."""
    detections = surfaced_detections(
        raw["rows"],
        visible_track_ids=set(raw["visible_track_ids"]),
        entity_ids=raw["entity_ids"],
        entity_ids_by_track=raw["entity_ids_by_track"],
        # Runs recorded before entity positions were captured score the DB row's fix.
        entity_positions_m=raw.get("entity_positions_m") if use_entity_positions else None,
    )
    report = evaluate_bundle(detections, bundle, gate_expectations(bundle))
    entity = asdict(report.entity_metrics)
    return {
        "stored_rows": len(raw["rows"]),
        "surfaced_detections": len(detections),
        "labels": sorted(str(d.get("label")) for d in detections),
        "event_recall": report.event_recall,
        "label_accuracy": report.label_accuracy,
        "median_position_error_m": report.median_position_error_m,
        "p90_position_error_m": report.p90_position_error_m,
        "median_bearing_error_deg": report.median_bearing_error_deg,
        **{key: entity[key] for key in ENTITY_KEYS},
        "detections_brief": [_brief(d, bundle) for d in detections],
    }


def rescore_row(row: dict[str, Any]) -> dict[str, Any]:
    bundle = load_scene_bundle(row["scene"], Path(row["bundle_path"]))
    return {**row, **score_raw_replay(row["scene"], bundle, row["raw"])}


def _prepare_fixture_scene(scene, work_dir: Path, seed: int, overrides: dict, extra_overrides: dict | None):
    """Real xeno-canto fixture birds (``fixture_audio_scenarios``) classified by production BirdNET."""
    clips_by_species = {clip.species: clip for clip in load_fixture_bird_clips()}
    clips = [clips_by_species[name] for name in FIXTURE_SCENE_SPECIES]
    bundle_path = bundle_path_for(scene, work_dir, seed)
    if not bundle_path.exists():
        scenario = build_species_gate_scenario(clips, seed=seed, bouts_per_bird=FIXTURE_SCENE_BOUTS_PER_BIRD)
        render_scenario_bundle(scenario, bundle_path, sim_binary_path=scoring.sim_binary_path())
    bundle = load_scene_bundle(scene, bundle_path)
    origin = bundle.manifest["site"]["origin"]
    overrides = {**overrides, **FIXTURE_SCENE_OVERRIDES, "site_origin_lat": origin["lat"], "site_origin_lon": origin["lon"]}
    overrides.update(extra_overrides or {})
    settings = Settings(**overrides)
    routing = default_routing()
    routing.contexts[CONTEXT_DETECTION_TRIGGER] = ContextSpec(name=CONTEXT_DETECTION_TRIGGER, run=("birdnet",))
    classifier = create_context_classifier(settings, CONTEXT_DETECTION_TRIGGER, routing=apply_settings(routing, settings))
    return bundle, classifier, overrides


def _brief(detection: dict, bundle) -> dict[str, Any]:
    """Compact per-detection record for reading a gate result by eye."""
    start_ns = min(int(n.get("audio_start_time_ns") or 0) for n in bundle.manifest.get("nodes", []))
    position = detection.get("position_m")
    features = detection.get("feature_summary") or {}
    return {
        "t_s": round((int(detection.get("timestamp_ns") or 0) - start_ns) / 1e9, 3),
        "label": detection.get("label"),
        "confidence": round(float(detection.get("label_confidence") or 0.0), 3),
        "position_m": None if position is None else [round(float(v), 2) for v in position],
        "node": detection.get("source_node_id"),
        "track_id": detection.get("track_id"),
        "entity_id": detection.get("entity_id"),
        "algorithm": features.get("resolved_algorithm") or features.get("localization_method"),
    }


def _bundle_duration_s(bundle) -> float:
    durations = []
    for node in bundle.manifest.get("nodes", []):
        channels, sample_rate_hz = bundle.node_audio(node["node_id"])
        durations.append(channels.shape[1] / sample_rate_hz)
    return max(durations) if durations else 0.0


# ── verdict logic ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SceneComparison:
    recall_delta: float
    swap_delta: int
    duplicate_delta: int
    echo_delta: int
    merged_delta: int
    median_error_delta_m: float | None


# Position error "no worse": tolerance for replay jitter between configurations.
POSITION_ERROR_TOLERANCE_M = 0.5


def compare_rows(candidate: dict, reference: dict) -> SceneComparison:
    def _num(row, key):
        return float(row.get(key) or 0.0)

    errors = (candidate.get("median_position_error_m"), reference.get("median_position_error_m"))
    return SceneComparison(
        recall_delta=_num(candidate, "localized_and_classified_individual_recall")
        - _num(reference, "localized_and_classified_individual_recall"),
        swap_delta=int(candidate["label_swap_count"]) - int(reference["label_swap_count"]),
        duplicate_delta=int(candidate["duplicate_entities"]) - int(reference["duplicate_entities"]),
        echo_delta=int(candidate["echo_promotions"]) - int(reference["echo_promotions"]),
        merged_delta=int(candidate["entities_spanning_multiple_individuals"])
        - int(reference["entities_spanning_multiple_individuals"]),
        median_error_delta_m=None if None in errors else float(errors[0]) - float(errors[1]),
    )


def all_on_gate_failures(all_on_rows: list[dict], baseline_rows: list[dict]) -> list[str]:
    """The plan's gate: recall up overall, zero swaps/duplicates/echo promotions, error no worse."""
    failures: list[str] = []
    baseline_by_scene = {row["scene"]: row for row in baseline_rows}
    recall_on = sum(float(r["localized_and_classified_individual_recall"] or 0.0) for r in all_on_rows)
    recall_off = sum(float(baseline_by_scene[r["scene"]]["localized_and_classified_individual_recall"] or 0.0) for r in all_on_rows)
    if recall_on <= recall_off:
        failures.append(f"summed individual recall all-on {recall_on:.2f} <= baseline {recall_off:.2f}")
    for row in all_on_rows:
        scene = row["scene"]
        for key in ("label_swap_count", "duplicate_entities", "echo_promotions"):
            if int(row[key]) > 0:
                failures.append(f"{scene}: {key} = {row[key]}")
        delta = compare_rows(row, baseline_by_scene[scene]).median_error_delta_m
        if delta is not None and delta > POSITION_ERROR_TOLERANCE_M:
            failures.append(f"{scene}: median position error worse by {delta:.2f} m")
    return failures


def summed(rows: list[dict], key: str) -> float:
    return sum(float(row.get(key) or 0.0) for row in rows)


def configuration_summary(rows: list[dict]) -> dict[str, Any]:
    errors = [r["median_position_error_m"] for r in rows if r.get("median_position_error_m") is not None]
    errors.sort()
    return {
        "scenes": len(rows),
        "individual_recall_sum": round(summed(rows, "localized_and_classified_individual_recall"), 3),
        "event_recall_sum": round(summed(rows, "localized_and_classified_recall"), 3),
        "swaps": int(summed(rows, "label_swap_count")),
        "duplicates": int(summed(rows, "duplicate_entities")),
        "echo_promotions": int(summed(rows, "echo_promotions")),
        "merged_entities": int(summed(rows, "entities_spanning_multiple_individuals")),
        "unassociated_localized": int(summed(rows, "unassociated_localized_entities")),
        "median_of_scene_median_error_m": errors[len(errors) // 2] if errors else None,
    }
