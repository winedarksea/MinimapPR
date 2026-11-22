//! Hidden `onset-clustering-oracle` subcommand: runs
//! `minimappr_core::onset_clustering` on analysis windows supplied as JSON on
//! stdin and prints one JSON result per window (final clusters plus the
//! stage trace). Used by `tests/test_onset_clustering_cross_language_parity.py`
//! against goldens from the Python `localize_onset_clusters`.
//!
//! Float arrays (signals, positions) travel as base64 little-endian f64 so the
//! inputs are bit-exact (decimal JSON parsing may be off by an ulp).
//! Non-finite floats are emitted as JSON `null`.

use std::collections::BTreeMap;

use base64::Engine;
use minimappr_core::onset_clustering::attribution_evidence::ClusterAttributionEvidence;
use minimappr_core::onset_clustering::trace::OnsetClusteringTrace;
use minimappr_core::onset_clustering::{
    localize_onset_clusters_traced, OnsetClusteringConfig, OnsetClusteringInput,
};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct SensorInput {
    id: String,
    node_id: String,
    position_b64: String,
    signal_b64: String,
}

#[derive(Deserialize)]
struct WindowCase {
    sample_rate_hz: u32,
    sound_speed_mps: f64,
    interpolation_factor: usize,
    far_field_default_range_m: f64,
    sensors: Vec<SensorInput>,
    #[serde(default)]
    consumed_onset_samples_by_node: BTreeMap<String, Vec<usize>>,
    #[serde(default)]
    config: Value,
    #[serde(default)]
    attribution_audible_snr_db: Option<f64>,
}

fn decode_f64s(encoded: &str) -> Result<Vec<f64>, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|error| error.to_string())?;
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8-byte chunk")))
        .collect())
}

fn finite(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        Value::Null
    }
}

fn evidence_json(evidence: &ClusterAttributionEvidence) -> Value {
    json!({
        "node_band_snr_db": evidence.node_band_snr_db,
        "node_distances_m": evidence.node_distances_m,
        "node_onset_arrivals": evidence.node_onset_arrivals.iter().map(|(node, arrivals)| {
            (node.clone(), arrivals.iter().map(|a| json!([a.span_sample, a.band_min_hz, a.band_max_hz])).collect::<Vec<_>>())
        }).collect::<BTreeMap<_, _>>(),
        "fitted_source_level_db": evidence.fitted_source_level_db,
        "spreading_residual_db": evidence.spreading_residual_db,
    })
}

fn trace_json(trace: &OnsetClusteringTrace) -> Value {
    json!({
        "onsets_by_node": trace.onsets_by_node.iter().map(|(node, onsets)| json!({
            "node_id": node,
            "onsets": onsets.iter().map(|o| json!([o.sample_index, o.strength, o.band_min_hz, o.band_max_hz])).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "frames": trace.frames.iter().map(|frame| json!({
            "frame_index": frame.frame_index,
            "template_node_id": frame.template_node_id,
            "reference_sensor": frame.reference_sensor,
            "onset_sample": frame.onset_sample,
            "weight": frame.weight,
            "curves": frame.curves.iter().map(|c| json!({
                "sensor_b": c.sensor_b,
                "weight_sum": c.weight_sum,
                "used_envelope": c.used_envelope,
                "effective_bandwidth_hz": c.effective_bandwidth_hz,
                "centre_frequency_hz": c.centre_frequency_hz,
                "peak_lag_s": c.peak_lag_s,
                "peak_value": c.peak_value,
                "value_sum": c.value_sum,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "peaks": trace.peaks.iter().map(|p| json!({
            "frame_index": p.frame_index,
            "position_m": p.position_m,
            "range_m": finite(p.range_m),
            "value": p.value,
            "weight": p.weight,
            "is_frame_primary": p.is_frame_primary,
        })).collect::<Vec<_>>(),
        "is_direction_grid": trace.is_direction_grid,
        "grid_point_count": trace.grid_point_count,
        "clusters": trace.clusters.iter().map(|c| json!({
            "frame_indices": c.frame_indices,
            "status": c.status,
            "status_reasons": c.status_reasons,
            "independent_onset_count": c.independent_onset_count,
            "weight": c.weight,
        })).collect::<Vec<_>>(),
    })
}

fn run_window(case: WindowCase) -> Result<Value, String> {
    let config: OnsetClusteringConfig = if case.config.is_null() {
        OnsetClusteringConfig::default()
    } else {
        serde_json::from_value(case.config).map_err(|error| error.to_string())?
    };
    let mut input = OnsetClusteringInput {
        sensor_signals: BTreeMap::new(),
        sensor_positions_m: BTreeMap::new(),
        sensor_node_ids: BTreeMap::new(),
        sample_rate_hz: case.sample_rate_hz,
        sound_speed_mps: case.sound_speed_mps,
        interpolation_factor: case.interpolation_factor,
        far_field_default_range_m: case.far_field_default_range_m,
        consumed_onset_samples_by_node: case.consumed_onset_samples_by_node,
        attribution_audible_snr_db: case.attribution_audible_snr_db,
    };
    for sensor in case.sensors {
        let position = decode_f64s(&sensor.position_b64)?;
        if position.len() != 3 {
            return Err(format!("sensor {} position must have 3 values", sensor.id));
        }
        input
            .sensor_positions_m
            .insert(sensor.id.clone(), [position[0], position[1], position[2]]);
        input
            .sensor_signals
            .insert(sensor.id.clone(), decode_f64s(&sensor.signal_b64)?);
        input.sensor_node_ids.insert(sensor.id, sensor.node_id);
    }
    let mut trace = OnsetClusteringTrace::default();
    let result = localize_onset_clusters_traced(&input, &config, Some(&mut trace));
    let clusters: Vec<Value> = result
        .clusters
        .iter()
        .map(|c| {
            json!({
                "position_m": c.position_m,
                "covariance_m2": c.covariance_m2,
                "confidence": finite(c.confidence),
                "gdop": finite(c.gdop),
                "range_projection_mode": c.range_projection_mode,
                "range_observability": c.range_observability.map(finite),
                "residual_rms_seconds": c.residual_rms_seconds.map(finite),
                "reference_sensor": c.reference_sensor,
                "tdoa_s": c.tdoa_s,
                "status": c.status.as_str(),
                "status_reasons": c.status_reasons,
                "onset_samples": c.onset_samples,
                "template_node_ids": c.template_node_ids,
                "band_min_hz": c.band_min_hz,
                "band_max_hz": c.band_max_hz,
                "weight": c.weight,
                "frame_count": c.frame_count,
                "independent_onset_count": c.independent_onset_count,
                "refined_by_least_squares": c.refined_by_least_squares,
                "used_envelope_fraction": c.used_envelope_fraction,
                "contributing_node_ids": c.contributing_node_ids,
                "loudest_node_id": c.loudest_node_id,
                "height_resolution": c.height_resolution,
                "range_interval_m": c.range_interval_m.map(|(near, far)| [near, far]),
                "attribution_evidence": c.attribution_evidence.as_ref().map(evidence_json),
            })
        })
        .collect();
    Ok(json!({
        "clusters": clusters,
        "onset_count": result.onset_count,
        "frame_count": result.frame_count,
        "consumed_onset_samples_by_node": result.consumed_onset_samples_by_node,
        "trace": trace_json(&trace),
    }))
}

pub fn run_onset_clustering_oracle() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::io::Read;

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let value: Value = serde_json::from_str(&input)?;
    let mut results = Vec::new();
    for case in value["cases"].as_array().ok_or("cases")? {
        let result = match case["kind"].as_str().ok_or("kind")? {
            "window" => match serde_json::from_value::<WindowCase>(case.clone()) {
                Ok(window) => run_window(window).unwrap_or_else(|error| json!({ "error": error })),
                Err(error) => json!({ "error": error.to_string() }),
            },
            other => return Err(format!("unknown case kind {other:?}").into()),
        };
        results.push(result);
    }
    println!("{}", serde_json::to_string(&json!({ "results": results }))?);
    Ok(())
}
