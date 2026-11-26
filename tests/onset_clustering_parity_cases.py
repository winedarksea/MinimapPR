"""Golden cases for the Rust ``minimappr_core::onset_clustering`` parity gate.

Windows of synthetic multi-node audio (``minimappr.sim.synthetic_entity_scenarios``,
quantised to PCM16 so the stored input is exact and small) run through the Python
reference ``localize_onset_clusters``. Executed as
``python -m tests.onset_clustering_parity_cases`` it writes
``tests/fixtures/onset_clustering_parity_goldens.npz``; the parity test feeds the
same windows to the sidecar's ``onset-clustering-oracle`` subcommand.

Besides the final clusters, each golden holds a stage trace (onsets per node,
frames with per-curve summaries, frame peaks, clusters after the echo checks),
captured by wrapping the reference module's stage functions, so a divergence is
pinned to the first stage that differs.

Golden layout: ``case_descriptors_json`` (one JSON object per case),
``audio_<render>`` (int16, sensors x samples) and ``audio_<render>_sensors``
(JSON: id, node, position) per render, and ``case{i}_result_json``.
"""

from __future__ import annotations

import base64
import dataclasses
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import minimappr.core.onset_srp_clustering as onset_module
from minimappr.config import Settings
from minimappr.sim.synthetic_entity_scenarios import (
    SYNTHETIC_ENTITY_SCENARIOS,
    _sensor_positions,
    render_node_audio,
)

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "onset_clustering_parity_goldens.npz"
SAMPLE_RATE_HZ = 12_000
SOUND_SPEED_MPS = 343.2
INTERPOLATION_FACTOR = 4
FAR_FIELD_DEFAULT_RANGE_M = 50.0
RENDER_SEED = 0
# Window around the first bout (calls start at 1.0 s and last 0.5 s).
SPAN_S = (0.9, 1.8)
PCM16_FULL_SCALE = 32768.0
# A raised fourth node makes the layout non-planar (bounded LM path).
NON_PLANAR_NODE_ORIGINS_M = {
    "node-a": (0.0, 0.0, 0.0),
    "node-b": (20.0, 0.0, 0.0),
    "node-c": (10.0, 18.0, 0.0),
    "node-d": (10.0, 6.0, 7.0),
}

RENDERS = {
    "one_bird": ("one_bird_two_bouts", None),
    "overlap": ("two_species_overlap", None),
    "trunk_echo": ("one_bird_trunk_echo", None),
    "one_bird_non_planar": ("one_bird_two_bouts", NON_PLANAR_NODE_ORIGINS_M),
}

# Attribution level evidence (S5) at the server default audibility floor.
ATTRIBUTION_AUDIBLE_SNR_DB = 3.0

# (render, node subset or None, consume the onsets of an earlier case, attach attribution evidence)
CASE_DESCRIPTORS = (
    {"render": "one_bird", "nodes": None, "consume_from_case": None, "attribution": False},
    {"render": "one_bird", "nodes": ["node-a"], "consume_from_case": None, "attribution": False},
    {"render": "overlap", "nodes": None, "consume_from_case": None, "attribution": True},
    {"render": "trunk_echo", "nodes": None, "consume_from_case": None, "attribution": False},
    {"render": "one_bird_non_planar", "nodes": None, "consume_from_case": None, "attribution": True},
    {"render": "one_bird", "nodes": None, "consume_from_case": 0, "attribution": False},
)


def render_window(render: str) -> tuple[np.ndarray, list[dict]]:
    """(int16 sensors x samples, sensor descriptors) for one render key."""
    scenario_name, node_origins = RENDERS[render]
    scenario = next(s for s in SYNTHETIC_ENTITY_SCENARIOS if s.name == scenario_name)
    if node_origins is not None:
        scenario = dataclasses.replace(scenario, node_origins_m=dict(node_origins))
    audio = render_node_audio(scenario, SAMPLE_RATE_HZ, RENDER_SEED)
    positions = _sensor_positions(scenario)
    start, stop = (int(t * SAMPLE_RATE_HZ) for t in SPAN_S)
    rows, sensors = [], []
    for node in sorted(audio):
        for index in range(audio[node].shape[0]):
            pcm = np.clip(np.round(audio[node][index, start:stop] * PCM16_FULL_SCALE), -32768, 32767)
            rows.append(pcm.astype(np.int16))
            sensors.append({"id": f"{node}:ch{index}", "node_id": node, "position_m": [float(v) for v in positions[node][index]]})
    return np.vstack(rows), sensors


def window_inputs(pcm: np.ndarray, sensors: list[dict], nodes: list[str] | None):
    signals, positions, node_ids = {}, {}, {}
    for row, sensor in zip(pcm, sensors):
        if nodes is not None and sensor["node_id"] not in nodes:
            continue
        signals[sensor["id"]] = row.astype(np.float64) / PCM16_FULL_SCALE
        positions[sensor["id"]] = np.asarray(sensor["position_m"], dtype=np.float64)
        node_ids[sensor["id"]] = sensor["node_id"]
    return signals, positions, node_ids


def _b64(values) -> str:
    return base64.b64encode(np.asarray(values, dtype="<f8").tobytes()).decode("ascii")


def oracle_case(signals, positions, node_ids, consumed, attribution_audible_snr_db=None) -> dict:
    return {
        "attribution_audible_snr_db": attribution_audible_snr_db,
        "kind": "window",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "sound_speed_mps": SOUND_SPEED_MPS,
        "interpolation_factor": INTERPOLATION_FACTOR,
        "far_field_default_range_m": FAR_FIELD_DEFAULT_RANGE_M,
        "consumed_onset_samples_by_node": consumed,
        "sensors": [
            {"id": sid, "node_id": node_ids[sid], "position_b64": _b64(positions[sid]), "signal_b64": _b64(signals[sid])}
            for sid in sorted(signals)
        ],
    }


def _curve_summary(curve) -> dict:
    peak = int(np.argmax(curve.values))
    return {
        "sensor_b": curve.sensor_b,
        "weight_sum": float(curve.weight_sum),
        "used_envelope": bool(curve.used_envelope),
        "effective_bandwidth_hz": float(curve.effective_bandwidth_hz),
        "centre_frequency_hz": float(curve.centre_frequency_hz),
        "peak_lag_s": float(curve.lags_s[peak]),
        "peak_value": float(curve.values[peak]),
        "value_sum": float(np.sum(curve.values)),
    }


@contextmanager
def _traced_stages(record: dict):
    """Wrap the reference module's stage functions to record their outputs."""
    originals = {
        name: getattr(onset_module, name)
        for name in ("detect_onsets", "_build_frame", "cluster_frame_peaks", "apply_onset_echo_and_persistence_checks",
                     "build_direction_grid", "build_window_spatial_grid")
    }

    def detect_onsets(*args, **kwargs):
        onsets = originals["detect_onsets"](*args, **kwargs)
        record["onsets"].append(onsets)
        return onsets

    def build_frame(*args, **kwargs):
        frame = originals["_build_frame"](*args, **kwargs)
        if frame is not None:
            record["frames"].append(frame)
        return frame

    def cluster_frame_peaks(peaks, grid, config):
        record["peaks"] = list(peaks)
        clusters = originals["cluster_frame_peaks"](peaks, grid, config)
        record["clusters"] = clusters
        return clusters

    def echo_checks(clusters, *args, **kwargs):
        originals["apply_onset_echo_and_persistence_checks"](clusters, *args, **kwargs)
        record["cluster_summaries"] = [
            {
                "frame_indices": sorted(cluster.frame_indices),
                "status": cluster.status,
                "status_reasons": list(cluster.status_reasons),
                "independent_onset_count": int(cluster.independent_onset_count),
                "weight": float(cluster.weight),
            }
            for cluster in clusters
        ]

    def grid_recorder(name):
        def build(*args, **kwargs):
            grid = originals[name](*args, **kwargs)
            record["grid"] = grid
            return grid
        return build

    replacements = {
        "detect_onsets": detect_onsets,
        "_build_frame": build_frame,
        "cluster_frame_peaks": cluster_frame_peaks,
        "apply_onset_echo_and_persistence_checks": echo_checks,
        "build_direction_grid": grid_recorder("build_direction_grid"),
        "build_window_spatial_grid": grid_recorder("build_window_spatial_grid"),
    }
    for name, replacement in replacements.items():
        setattr(onset_module, name, replacement)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(onset_module, name, original)


def _finite_or_none(value):
    return None if value is None or not np.isfinite(value) else float(value)


def _evidence_json(evidence) -> dict | None:
    if evidence is None:
        return None
    return {
        "node_band_snr_db": {node: float(v) for node, v in evidence.node_band_snr_db.items()},
        "node_distances_m": {node: float(v) for node, v in evidence.node_distances_m.items()},
        "node_onset_arrivals": {
            node: [[int(a.span_sample), float(a.band_min_hz), float(a.band_max_hz)] for a in arrivals]
            for node, arrivals in evidence.node_onset_arrivals.items()
        },
        "fitted_source_level_db": _finite_or_none(evidence.fitted_source_level_db),
        "spreading_residual_db": _finite_or_none(evidence.spreading_residual_db),
    }


def run_python_window(signals, positions, node_ids, consumed, attribution_audible_snr_db=None) -> dict:
    record = {"onsets": [], "frames": [], "peaks": [], "cluster_summaries": [], "grid": None}
    with _traced_stages(record):
        result = onset_module.localize_onset_clusters(
            sensor_signals=signals,
            sensor_positions=positions,
            sensor_node_ids=node_ids,
            sample_rate_hz=SAMPLE_RATE_HZ,
            sound_speed_mps=SOUND_SPEED_MPS,
            config=Settings().onset_srp_clustering_config(),
            interpolation_factor=INTERPOLATION_FACTOR,
            far_field_default_range_m=FAR_FIELD_DEFAULT_RANGE_M,
            consumed_onset_samples_by_node={node: tuple(samples) for node, samples in consumed.items()},
            attribution_audible_snr_db=attribution_audible_snr_db,
        )
    nodes = sorted(set(node_ids.values()))
    grid = record["grid"]
    trace = {
        "onsets_by_node": [
            {"node_id": node, "onsets": [[o.sample_index, o.strength, o.band_min_hz, o.band_max_hz] for o in onsets]}
            for node, onsets in zip(nodes, record["onsets"])
        ],
        "frames": [
            {
                "frame_index": frame.frame_index,
                "template_node_id": frame.template_node_id,
                "reference_sensor": frame.reference_sensor,
                "onset_sample": int(frame.onset_sample),
                "weight": float(frame.weight),
                "curves": [_curve_summary(curve) for curve in frame.curves],
            }
            for frame in record["frames"]
        ],
        "peaks": [
            {
                "frame_index": peak.frame_index,
                "position_m": [float(v) for v in peak.position_m],
                "range_m": _finite_or_none(peak.range_m),
                "value": float(peak.value),
                "weight": float(peak.weight),
                "is_frame_primary": bool(peak.is_frame_primary),
            }
            for peak in record["peaks"]
        ],
        "is_direction_grid": bool(grid.is_direction_grid) if grid is not None else False,
        "grid_point_count": int(grid.points_m.shape[0]) if grid is not None else 0,
        "clusters": record["cluster_summaries"],
    }
    clusters = [
        {
            "position_m": [float(v) for v in cluster.position_m],
            "covariance_m2": cluster.covariance_m2,
            "confidence": _finite_or_none(cluster.confidence),
            "gdop": _finite_or_none(cluster.gdop),
            "range_projection_mode": cluster.range_projection_mode,
            "range_observability": _finite_or_none(cluster.range_observability),
            "residual_rms_seconds": _finite_or_none(cluster.residual_rms_seconds),
            "reference_sensor": cluster.reference_sensor,
            "tdoa_s": {key: float(value) for key, value in cluster.tdoa_s.items()},
            "status": cluster.status,
            "status_reasons": list(cluster.status_reasons),
            "onset_samples": [int(v) for v in cluster.onset_samples],
            "template_node_ids": list(cluster.template_node_ids),
            "band_min_hz": float(cluster.band_min_hz),
            "band_max_hz": float(cluster.band_max_hz),
            "weight": float(cluster.weight),
            "frame_count": int(cluster.frame_count),
            "independent_onset_count": int(cluster.independent_onset_count),
            "refined_by_least_squares": bool(cluster.refined_by_least_squares),
            "used_envelope_fraction": float(cluster.used_envelope_fraction),
            "contributing_node_ids": list(cluster.contributing_node_ids),
            "loudest_node_id": cluster.loudest_node_id,
            "height_resolution": cluster.height_resolution,
            "range_interval_m": list(cluster.range_interval_m) if cluster.range_interval_m is not None else None,
            "range_lower_bound_m": cluster.range_lower_bound_m,
            "attribution_evidence": _evidence_json(cluster.attribution_evidence),
        }
        for cluster in result.clusters
    ]
    return {
        "clusters": clusters,
        "onset_count": int(result.onset_count),
        "frame_count": int(result.frame_count),
        "consumed_onset_samples_by_node": {node: [int(v) for v in samples] for node, samples in result.consumed_onset_samples_by_node.items()},
        "trace": trace,
    }


def case_inputs(descriptors, audio_by_render, sensors_by_render, results):
    """Yield (descriptor, signals, positions, node_ids, consumed, attribution floor) per case."""
    for descriptor in descriptors:
        signals, positions, node_ids = window_inputs(
            audio_by_render[descriptor["render"]], sensors_by_render[descriptor["render"]], descriptor["nodes"]
        )
        source = descriptor["consume_from_case"]
        consumed = {} if source is None else results[source]["consumed_onset_samples_by_node"]
        audible = ATTRIBUTION_AUDIBLE_SNR_DB if descriptor["attribution"] else None
        yield descriptor, signals, positions, node_ids, consumed, audible


def build_goldens() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {"case_descriptors_json": np.asarray([json.dumps(d) for d in CASE_DESCRIPTORS])}
    audio, sensors = {}, {}
    for render in RENDERS:
        audio[render], sensors[render] = render_window(render)
        arrays[f"audio_{render}"] = audio[render]
        arrays[f"audio_{render}_sensors"] = np.asarray(json.dumps(sensors[render]))
    results: list[dict] = []
    for index, (_, signals, positions, node_ids, consumed, audible) in enumerate(case_inputs(CASE_DESCRIPTORS, audio, sensors, results)):
        results.append(run_python_window(signals, positions, node_ids, consumed, audible))
        arrays[f"case{index}_result_json"] = np.asarray(json.dumps(results[-1]))
    return arrays


if __name__ == "__main__":
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_PATH, **build_goldens())
    print(f"wrote {GOLDEN_PATH}")
