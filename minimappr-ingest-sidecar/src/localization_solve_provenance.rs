//! `localization.solve_provenance` and the localization manifests that carry
//! it (`docs/localization_solve_contract.md`).
//!
//! Provenance is built from the layout snapshot the solve ran on (taken at
//! ingest, so a parked solve reports the geometry it triggered with), never
//! from the registry's current state. A classifier render and the
//! `localization_result` of the same payload carry one `solve_id`; a render
//! that covers an earlier event-triggered solve repeats that solve's id (see
//! `dsp_covered_event_solve`).

use minimappr_core::localization_sensor_entry::{
    ExcludedSensor, SolveExclusionReason, SolveProvenance, SOLVE_PROVENANCE_SCHEMA_VERSION,
};
use uuid::Uuid;

use crate::{
    dsp_worker::{ComputePayload, PairTdoa, SolveSensorLayout},
    manifests::{DspManifest, LocalizationManifestPayload, PairTdoaDiagnostic},
    srp_phat::{HalfSpace, SrpPhatLocalization},
};

/// A steering vector shorter than this carries no direction.
const MINIMUM_STEERING_DIRECTION_MAGNITUDE: f32 = 1.0e-6;

/// What one solve ran on, beyond its layout snapshot.
pub(crate) struct SolveProvenanceInputs<'a> {
    pub(crate) solve_id: &'a str,
    pub(crate) attempt_kind: minimappr_core::localization_sensor_entry::SolveAttemptKind,
    pub(crate) active_sensor_indices: &'a [usize],
    pub(crate) window_center_time_ns: i128,
    pub(crate) window_seconds: f64,
}

/// Every sensor of the layout is either active or excluded with a reason:
/// the layout's own exclusions (geometry, rate, cap) first, and any other
/// inactive sensor because its window coverage fell under the minimum.
pub(crate) fn build_solve_provenance(
    layout: &SolveSensorLayout,
    inputs: SolveProvenanceInputs<'_>,
) -> SolveProvenance {
    let mut excluded_sensors = layout.excluded_sensors.clone();
    for (sensor_index, sensor) in layout.sensors.iter().enumerate() {
        let accounted_for = inputs.active_sensor_indices.contains(&sensor_index)
            || excluded_sensors
                .iter()
                .any(|excluded| excluded.sensor_index == sensor_index);
        if !accounted_for {
            excluded_sensors.push(ExcludedSensor {
                sensor_index,
                sensor_id: sensor.sensor_id.clone(),
                reason: SolveExclusionReason::CoverageBelowThreshold,
            });
        }
    }
    excluded_sensors.sort_by_key(|excluded| excluded.sensor_index);
    SolveProvenance {
        schema_version: SOLVE_PROVENANCE_SCHEMA_VERSION,
        solve_id: inputs.solve_id.to_string(),
        attempt_kind: inputs.attempt_kind,
        buffer_scope: layout.buffer_scope,
        cluster_id: layout.cluster_id.clone(),
        reporting_node_id: layout.reporting_node_id.clone(),
        node_local_routing_reason: layout.node_local_routing_reason,
        coordinate_frame: layout.coordinate_frame.clone(),
        layout_revision: layout.layout_revision,
        geometry_push_revision: layout.geometry_push_revision,
        sensors: layout.sensors.clone(),
        active_sensor_indices: inputs.active_sensor_indices.to_vec(),
        excluded_sensors,
        // Clock evidence and pair admission land with milestone 2 (R5 / R7).
        excluded_pairs: Vec::new(),
        node_timing: Vec::new(),
        window_center_time_ns: i64::try_from(inputs.window_center_time_ns).unwrap_or(i64::MAX),
        window_seconds: inputs.window_seconds,
        solve_sample_rate_hz: layout.solve_sample_rate_hz,
    }
}

/// The `localization` payload one compute payload publishes, with its provenance.
pub(crate) fn localization_manifest_payload(
    payload: &ComputePayload,
    localization: &SrpPhatLocalization,
    pair_tdoas: &[PairTdoa],
    received_level_dbfs: Option<f32>,
) -> LocalizationManifestPayload {
    let layout = &payload.sensor_layout;
    let pair_tdoas = pair_tdoas
        .iter()
        .map(|pair| PairTdoaDiagnostic {
            ch_a: pair.ch_a,
            ch_b: pair.ch_b,
            delay_samples: pair.tdoa.delay_samples,
            lag_seconds: pair.tdoa.lag_seconds,
            confidence: pair.tdoa.confidence,
            is_cross_node: Some(layout.is_cross_node_pair(pair.ch_a, pair.ch_b)),
            timing_uncertainty_seconds: None,
        })
        .collect();
    let steering_magnitude = localization
        .steering_direction
        .iter()
        .map(|axis| axis * axis)
        .sum::<f32>()
        .sqrt();
    LocalizationManifestPayload {
        attempted_algorithm: localization.attempted_algorithm.clone(),
        resolved_algorithm: localization.resolved_algorithm.clone(),
        steering_direction: (steering_magnitude > MINIMUM_STEERING_DIRECTION_MAGNITUDE)
            .then_some(localization.steering_direction),
        position_m: localization.position_m,
        position_covariance_m2: localization.position_covariance_m2,
        range_observability: localization.range_observability,
        range_projection_mode: localization.range_projection_mode.clone(),
        confidence: localization.confidence,
        residual_rms_seconds: localization
            .residual_rms_seconds
            .is_finite()
            .then_some(localization.residual_rms_seconds),
        sound_speed_mps: localization.sound_speed_mps,
        effective_band_hz: Some(payload.config.localization_band_hz),
        dominant_frequency_hz: localization.dominant_frequency_hz,
        received_level_dbfs,
        half_space_applied: layout.solve_half_space(&payload.active_channels) != HalfSpace::None,
        pair_tdoas,
        solve_provenance: Some(build_solve_provenance(
            layout,
            SolveProvenanceInputs {
                solve_id: &payload.solve_id,
                attempt_kind: payload.attempt_kind,
                active_sensor_indices: &payload.active_channels,
                window_center_time_ns: payload.localization_window_center_time_ns,
                window_seconds: payload.config.window_seconds,
            },
        )),
    }
}

/// The `localization_result` manifest of one payload. `render_derived_handle`
/// lets Python pair it with the classifier render built from the same payload.
pub(crate) fn localization_result_manifest(
    payload: &ComputePayload,
    localization: LocalizationManifestPayload,
    render_derived_handle: Option<crate::journal_reader::JournalPayloadHandle>,
    coverage_stats: Option<serde_json::Value>,
) -> DspManifest {
    DspManifest {
        manifest_id: format!("manifest-{}", Uuid::new_v4()),
        manifest_type: "localization_result".to_string(),
        created_ns: payload.now_ns,
        source_handles: payload.manifest.source_handles.clone(),
        derived_handle: render_derived_handle,
        localization: Some(localization),
        classifier_render: None,
        birdnet: None,
        coverage_stats,
        promotion_ready: false,
        env_samples: None,
        node_context: payload.manifest.node_context.clone(),
        cluster_id: payload.manifest.cluster_id.clone(),
        cluster_sensor_positions: payload.manifest.cluster_sensor_positions.clone(),
        raw_payload: None,
        raw_render_bytes: None,
        raw_audio_frame: None,
        raw_audio_bytes: None,
    }
}

#[cfg(test)]
#[path = "localization_solve_provenance_tests.rs"]
mod localization_solve_provenance_tests;
