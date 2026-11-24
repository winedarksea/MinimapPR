//! Wire types of the Rust <-> Python localization solve contract
//! (`docs/localization_solve_contract.md`).
//!
//! Two payloads live here:
//! - [`SolveProvenance`]: emitted by the sidecar as `localization.solve_provenance`
//!   on `localization_result` and `classifier_render` manifests. It names the
//!   exact sensors, positions, frame and timing evidence one solve used, so the
//!   Python handoff never has to re-derive geometry from its own registry.
//! - [`SiteGeometryPush`]: the full-snapshot body Python sends to
//!   `POST /api/v1/dsp/geometry` (and at launch in `MINIMAPPR_SITE_GEOMETRY_JSON`).
//!
//! Field names and enum spellings are the contract; Python parses against the
//! doc, so renaming anything here is a breaking change on both sides.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Bumped on any incompatible change to [`SolveProvenance`].
pub const SOLVE_PROVENANCE_SCHEMA_VERSION: u32 = 1;

/// Why a solve ran. Event triggers are distinguished by gate so the trigger
/// funnel can be measured per gate; a scheduled render solves only because a
/// classifier render was due on that frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveAttemptKind {
    EventTriggerAbsoluteRms,
    EventTriggerNoiseFloorRelative,
    ScheduledClassifierRender,
    NotAttempted,
}

/// Which shared buffer the solve's audio came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveBufferScope {
    /// The reporting node's own channels only.
    NodeLocal,
    /// An operator-declared cluster (manifest `cluster_id` + positions).
    ExplicitCluster,
    /// The sidecar's automatic cluster of every positioned node on a stream.
    AutomaticSiteCluster,
}

/// Why a node solved alone (`buffer_scope` `node_local`) instead of joining a
/// cluster buffer; `solve_provenance.node_local_routing_reason`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeLocalRoutingReason {
    /// 2-3 channels without offsets, so no world placement is possible.
    NodeArrayGeometryUnusable,
    /// No trusted world position: a `fallback_static` (or missing) fix and no
    /// pushed placement. Two nodes at one built-in point cannot form a baseline.
    NodePositionUntrustedNoOverride,
    /// The manifest carries only half of the explicit cluster fields.
    ExplicitClusterIncomplete,
    /// Fewer declared sensors than an SRP solve needs.
    ExplicitClusterBelowMinimumSensors,
    /// Some of the node's channels are not in the declared sensor list.
    NodeChannelsMissingFromExplicitCluster,
}

impl NodeLocalRoutingReason {
    /// The wire spelling, for logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NodeArrayGeometryUnusable => "node_array_geometry_unusable",
            Self::NodePositionUntrustedNoOverride => "node_position_untrusted_no_override",
            Self::ExplicitClusterIncomplete => "explicit_cluster_incomplete",
            Self::ExplicitClusterBelowMinimumSensors => "explicit_cluster_below_minimum_sensors",
            Self::NodeChannelsMissingFromExplicitCluster => {
                "node_channels_missing_from_explicit_cluster"
            }
        }
    }
}

/// Geodetic -> local ENU conversion. Rust implements exactly the projection
/// Python's `LocalCoordinateFrame` is configured with (`flat` / `geodetic`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateProjection {
    /// Python `mode="flat"`: WGS84 series metres-per-degree about the origin.
    FlatWgs84Series,
    /// Python `mode="geodetic"`: LLA -> ECEF -> ENU about the origin.
    GeodeticEcefEnu,
}

/// What the solve's `position_m` values are relative to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateFrameKind {
    /// East/north/up metres about the pushed `site_origin`.
    SiteEnu,
    /// East/north/up metres about an origin the sidecar chose (no push yet).
    SidecarAutomaticEnu,
    /// Positions copied verbatim from the manifest's explicit cluster geometry.
    ExplicitClusterPositions,
    /// The reporting node's own body-frame mic offsets (node-local solve).
    NodeBody,
}

/// Geodetic point; same field names as Python `GeoPoint`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeodeticOrigin {
    pub lat: f64,
    pub lon: f64,
    pub alt_m: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolveCoordinateFrame {
    /// Echo of the pushed `site_frame_id`; null until Python has pushed one.
    /// Python accepts a solve's positions only when this matches its frame.
    pub frame_id: Option<String>,
    pub kind: CoordinateFrameKind,
    /// Null for frames without a geodetic anchor (`node_body`,
    /// `explicit_cluster_positions`).
    pub origin: Option<GeodeticOrigin>,
    /// Null exactly when `origin` is null.
    pub projection: Option<CoordinateProjection>,
}

/// Where a sensor's world position came from, highest precedence first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorPositionSource {
    /// Pushed placement entered by an operator (hand-measured or surveyed).
    OperatorOverride,
    /// Pushed placement from Python's filtered stationary-node estimate.
    StationaryEstimate,
    /// Raw per-packet GPS (only for nodes with no pushed placement).
    ReportedGps,
    /// Firmware's built-in static position; untrusted for cross-node baselines.
    FallbackStatic,
    /// Copied from an explicit cluster's `cluster_sensor_positions`.
    ExplicitClusterManifest,
    /// Node body-frame offset (node-local solve; no world position).
    NodeBodyOffset,
}

/// Node timing class; spellings match firmware / Python `TimeQuality`.
/// Legacy `ntp_sync` is NTP-disciplined (Python converts it the same way on
/// load); any other value deserializes as `unknown`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeTimeQuality {
    GpsLocked,
    GpsHoldover,
    #[serde(alias = "ntp_sync")]
    NtpDisciplined,
    FreeRunning,
    #[serde(other)]
    Unknown,
}

/// One row of `sensors[]`. Its index in that array equals the solve buffer
/// channel and is what `pair_tdoas[].ch_a/ch_b` and every `*_index` field refer to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolveSensorEntry {
    /// `<node_id>:ch<node_channel_index>`.
    pub sensor_id: String,
    pub node_id: String,
    pub node_channel_index: usize,
    /// The snapshot the solve used, in `coordinate_frame`, metres.
    pub position_m: [f64; 3],
    pub sample_rate_hz: u32,
    pub position_source: SensorPositionSource,
    /// 1-sigma position accuracy; null when nothing states one.
    pub position_std_m: Option<f64>,
    pub clock_source: NodeTimeQuality,
    /// Timebase the packet timestamps are expressed in (today always `utc`).
    pub clock_domain: String,
    /// 1-sigma clock offset uncertainty the pair weights used; null if unknown.
    pub clock_offset_std_s: Option<f64>,
}

/// Every exclusion (sensor or pair) carries exactly one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveExclusionReason {
    /// Sensor or pair: sensor rate differs from `solve_sample_rate_hz`.
    SampleRateMismatch,
    /// Pair: sensor separation under `localization_min_pair_baseline_m`.
    PairBaselineBelowMinimum,
    /// Sensor or pair: window coverage under the minimum ratio.
    CoverageBelowThreshold,
    /// Pair: a node's measured timeline residual exceeds
    /// `localization_cross_node_max_timeline_residual_seconds`.
    TimelineResidualOverLimit,
    /// Sensor: `fallback_static` node with no pushed placement.
    NodePositionUntrustedNoOverride,
    /// Sensor: the node's own offsets fail validation for its channel count.
    NodeArrayGeometryUnusable,
    /// Sensor: beyond the `localization_max_solve_sensors` cap.
    MaxSolveSensorsExceeded,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExcludedSensor {
    pub sensor_index: usize,
    pub sensor_id: String,
    pub reason: SolveExclusionReason,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExcludedSensorPair {
    pub sensor_index_a: usize,
    pub sensor_index_b: usize,
    pub reason: SolveExclusionReason,
}

/// Clock evidence for one node contributing to the solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeTimingEvidence {
    pub node_id: String,
    pub time_quality: NodeTimeQuality,
    pub clock_offset_std_s: Option<f64>,
    /// Measured packet-timeline residual; null until measured.
    pub timeline_residual_ns: Option<i64>,
    /// Signed size of the most recent detected timestamp step; null if none.
    pub last_timestamp_step_ns: Option<i64>,
    pub timestamp_step_count: u64,
}

/// `localization.solve_provenance`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolveProvenance {
    pub schema_version: u32,
    /// Identifies one solve; a classifier render covering it repeats the id.
    pub solve_id: String,
    pub attempt_kind: SolveAttemptKind,
    pub buffer_scope: SolveBufferScope,
    /// Null for node-local solves.
    pub cluster_id: Option<String>,
    /// Node whose packet produced the manifest (not necessarily every contributor).
    pub reporting_node_id: String,
    /// Why the reporting node solved node-local; null for cluster solves.
    pub node_local_routing_reason: Option<NodeLocalRoutingReason>,
    pub coordinate_frame: SolveCoordinateFrame,
    /// Bumps whenever any sensor position in the layout changes (GPS included).
    pub layout_revision: u64,
    /// Revision of the geometry push in effect; null before the first push.
    pub geometry_push_revision: Option<u64>,
    pub sensors: Vec<SolveSensorEntry>,
    pub active_sensor_indices: Vec<usize>,
    pub excluded_sensors: Vec<ExcludedSensor>,
    pub excluded_pairs: Vec<ExcludedSensorPair>,
    pub node_timing: Vec<NodeTimingEvidence>,
    /// Packet-derived window centre: the stored observation time.
    pub window_center_time_ns: i64,
    pub window_seconds: f64,
    pub solve_sample_rate_hz: u32,
}

/// Who entered a pushed node placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodePlacementSource {
    OperatorOverride,
    StationaryEstimate,
}

impl NodePlacementSource {
    pub fn sensor_position_source(self) -> SensorPositionSource {
        match self {
            Self::OperatorOverride => SensorPositionSource::OperatorOverride,
            Self::StationaryEstimate => SensorPositionSource::StationaryEstimate,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodePlacement {
    /// Node origin in site ENU metres about `site_origin`. Absent = an
    /// orientation-only placement: the node's position stays its reported GPS
    /// and only `body_to_world_quaternion_wxyz` is pinned (an operator
    /// orientation override on an automatically positioned node).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_m: Option<[f64; 3]>,
    /// 1-sigma accuracy of `position_m`; only meaningful with it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_std_m: Option<f64>,
    pub placement_source: NodePlacementSource,
    /// Body -> world rotation (w, x, y, z); absent means identity. Sensor world
    /// position = position_m + R * sensor_offset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_to_world_quaternion_wxyz: Option<[f64; 4]>,
}

/// `POST /api/v1/dsp/geometry` body and `MINIMAPPR_SITE_GEOMETRY_JSON` value.
/// A full snapshot: nodes absent from `node_placements` lose any earlier push.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteGeometryPush {
    pub site_frame_id: String,
    pub geometry_push_revision: u64,
    pub site_origin: GeodeticOrigin,
    pub projection: CoordinateProjection,
    pub node_placements: BTreeMap<String, NodePlacement>,
}

/// Response to a geometry push: the revision now in effect.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteGeometryPushAcknowledgement {
    pub site_frame_id: String,
    pub applied_geometry_push_revision: u64,
    pub node_placement_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_spellings_match_the_contract_doc() {
        let attempt_kinds = [
            (
                SolveAttemptKind::EventTriggerAbsoluteRms,
                "event_trigger_absolute_rms",
            ),
            (
                SolveAttemptKind::EventTriggerNoiseFloorRelative,
                "event_trigger_noise_floor_relative",
            ),
            (
                SolveAttemptKind::ScheduledClassifierRender,
                "scheduled_classifier_render",
            ),
            (SolveAttemptKind::NotAttempted, "not_attempted"),
        ];
        for (kind, spelling) in attempt_kinds {
            assert_eq!(serde_json::to_value(kind).unwrap(), spelling);
        }
        assert_eq!(
            serde_json::to_value(CoordinateProjection::FlatWgs84Series).unwrap(),
            "flat_wgs84_series"
        );
        assert_eq!(
            serde_json::to_value(CoordinateProjection::GeodeticEcefEnu).unwrap(),
            "geodetic_ecef_enu"
        );
        assert_eq!(
            serde_json::to_value(SolveExclusionReason::PairBaselineBelowMinimum).unwrap(),
            "pair_baseline_below_minimum"
        );
        let routing_reasons = [
            NodeLocalRoutingReason::NodeArrayGeometryUnusable,
            NodeLocalRoutingReason::NodePositionUntrustedNoOverride,
            NodeLocalRoutingReason::ExplicitClusterIncomplete,
            NodeLocalRoutingReason::ExplicitClusterBelowMinimumSensors,
            NodeLocalRoutingReason::NodeChannelsMissingFromExplicitCluster,
        ];
        for reason in routing_reasons {
            assert_eq!(serde_json::to_value(reason).unwrap(), reason.as_str());
        }
    }

    /// Same JSON as the example in `docs/localization_solve_contract.md`.
    fn contract_doc_solve_provenance_example() -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "solve_id": "solve-3f2a",
            "attempt_kind": "event_trigger_absolute_rms",
            "buffer_scope": "automatic_site_cluster",
            "cluster_id": "auto-gps",
            "reporting_node_id": "planar-a",
            "node_local_routing_reason": null,
            "coordinate_frame": {
                "frame_id": "site-7",
                "kind": "site_enu",
                "origin": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0},
                "projection": "flat_wgs84_series"
            },
            "layout_revision": 12,
            "geometry_push_revision": 3,
            "sensors": [
                {
                    "sensor_id": "planar-a:ch0",
                    "node_id": "planar-a",
                    "node_channel_index": 0,
                    "position_m": [1.0177, 2.0177, 1.5],
                    "sample_rate_hz": 16000,
                    "position_source": "operator_override",
                    "position_std_m": 0.05,
                    "clock_source": "gps_locked",
                    "clock_domain": "utc",
                    "clock_offset_std_s": 1e-6
                },
                {
                    "sensor_id": "planar-a:ch4",
                    "node_id": "planar-a",
                    "node_channel_index": 4,
                    "position_m": [1.0, 2.0, 1.5],
                    "sample_rate_hz": 16000,
                    "position_source": "operator_override",
                    "position_std_m": 0.05,
                    "clock_source": "gps_locked",
                    "clock_domain": "utc",
                    "clock_offset_std_s": 1e-6
                },
                {
                    "sensor_id": "point-b:ch0",
                    "node_id": "point-b",
                    "node_channel_index": 0,
                    "position_m": [6.0, 0.0, 1.5],
                    "sample_rate_hz": 16000,
                    "position_source": "reported_gps",
                    "position_std_m": null,
                    "clock_source": "ntp_disciplined",
                    "clock_domain": "utc",
                    "clock_offset_std_s": 0.002
                }
            ],
            "active_sensor_indices": [0, 1, 2],
            "excluded_sensors": [],
            "excluded_pairs": [
                {"sensor_index_a": 0, "sensor_index_b": 2, "reason": "timeline_residual_over_limit"},
                {"sensor_index_a": 1, "sensor_index_b": 2, "reason": "timeline_residual_over_limit"}
            ],
            "node_timing": [
                {
                    "node_id": "planar-a",
                    "time_quality": "gps_locked",
                    "clock_offset_std_s": 1e-6,
                    "timeline_residual_ns": 1200,
                    "last_timestamp_step_ns": null,
                    "timestamp_step_count": 0
                },
                {
                    "node_id": "point-b",
                    "time_quality": "ntp_disciplined",
                    "clock_offset_std_s": 0.002,
                    "timeline_residual_ns": 15000000,
                    "last_timestamp_step_ns": -21000000,
                    "timestamp_step_count": 1
                }
            ],
            "window_center_time_ns": 1764000000016000000_i64,
            "window_seconds": 0.032,
            "solve_sample_rate_hz": 16000
        })
    }

    #[test]
    fn contract_doc_example_round_trips_field_for_field() {
        let example = contract_doc_solve_provenance_example();
        let parsed: SolveProvenance = serde_json::from_value(example.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), example);
    }

    #[test]
    fn unrecognised_time_quality_parses_as_unknown() {
        let parsed: NodeTimeQuality = serde_json::from_str("\"sundial\"").unwrap();
        assert_eq!(parsed, NodeTimeQuality::Unknown);
    }

    #[test]
    fn legacy_ntp_sync_parses_as_ntp_disciplined() {
        let parsed: NodeTimeQuality = serde_json::from_str("\"ntp_sync\"").unwrap();
        assert_eq!(parsed, NodeTimeQuality::NtpDisciplined);
    }

    #[test]
    fn geometry_push_round_trips_with_optional_rotation() {
        let body = serde_json::json!({
            "site_frame_id": "site-7",
            "geometry_push_revision": 3,
            "site_origin": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0},
            "projection": "flat_wgs84_series",
            "node_placements": {
                "planar-a": {
                    "position_m": [1.0, 2.0, 1.5],
                    "position_std_m": 0.05,
                    "placement_source": "operator_override",
                    "body_to_world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]
                },
                "point-b": {
                    "position_m": [6.0, 0.0, 1.5],
                    "position_std_m": 0.4,
                    "placement_source": "stationary_estimate"
                },
                "planar-gps": {
                    "placement_source": "operator_override",
                    "body_to_world_quaternion_wxyz": [0.9537, 0.0, 0.0, 0.3007]
                }
            }
        });
        let push: SiteGeometryPush = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(
            push.node_placements["point-b"].body_to_world_quaternion_wxyz,
            None
        );
        let orientation_only = &push.node_placements["planar-gps"];
        assert_eq!(orientation_only.position_m, None);
        assert_eq!(orientation_only.position_std_m, None);
        assert_eq!(serde_json::to_value(&push).unwrap(), body);
    }
}
