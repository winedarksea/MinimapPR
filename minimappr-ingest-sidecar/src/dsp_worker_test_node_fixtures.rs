//! Parameterised node fixtures for worker-level localization tests: a Sirith
//! planar 5-mic node, a Sirith tetra node or a single-mic point node, placed
//! at a site ENU position and yaw, all hearing one coherent broadband point
//! source with true per-sensor propagation delays.
//!
//! Yaw is a body -> world rotation about +Z, the matrix convention of Python
//! `orientation_rotation_matrix` (`minimappr/spatial_audio/geometry.py`).

use super::dsp_worker_tests::{encode_pcm16le_b64, raw_manifest_for_payload_with_metadata};
use super::*;
use minimappr_core::{
    localization_sensor_entry::{
        CoordinateProjection, GeodeticOrigin, NodePlacement, NodePlacementSource, SiteGeometryPush,
    },
    sirith_array_geometry::SIRITH_PLANAR_MIC_OFFSETS_M,
    site_frame_projection::local_enu_to_geodetic,
};

pub(super) const FIXTURE_SAMPLE_RATE_HZ: u32 = 16_000;
pub(super) const FIXTURE_FRAME_SAMPLES: usize = 512;
pub(super) const FIXTURE_FRAME_DURATION_NS: u64 = 32_000_000;
const FIXTURE_STREAM_START_NS: u64 = 1_000_000_000;
/// Source audio is pre-rolled this far so the farthest sensor's delayed read
/// never indexes before the start of the signal.
const SOURCE_PRE_ROLL_SAMPLES: usize = 16_000;
const SOURCE_PEAK_AMPLITUDE: f32 = 0.3;
const SOURCE_NOISE_SEED: u32 = 0x5eed_1234;
pub(super) const FIXTURE_SITE_ORIGIN: GeodeticOrigin = GeodeticOrigin {
    lat: 45.0,
    lon: -93.0,
    alt_m: 2.0,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FixtureArrayKind {
    SirithPlanar,
    SirithTetra,
    Point,
}

#[derive(Clone, Debug)]
pub(super) struct FixtureNode {
    pub(super) node_id: String,
    pub(super) array_kind: FixtureArrayKind,
    pub(super) site_position_m: [f64; 3],
    pub(super) yaw_deg: f64,
}

impl FixtureNode {
    pub(super) fn planar(node_id: &str, site_position_m: [f64; 3], yaw_deg: f64) -> Self {
        Self::new(
            node_id,
            FixtureArrayKind::SirithPlanar,
            site_position_m,
            yaw_deg,
        )
    }

    pub(super) fn tetra(node_id: &str, site_position_m: [f64; 3]) -> Self {
        Self::new(node_id, FixtureArrayKind::SirithTetra, site_position_m, 0.0)
    }

    pub(super) fn point(node_id: &str, site_position_m: [f64; 3]) -> Self {
        Self::new(node_id, FixtureArrayKind::Point, site_position_m, 0.0)
    }

    fn new(
        node_id: &str,
        array_kind: FixtureArrayKind,
        site_position_m: [f64; 3],
        yaw_deg: f64,
    ) -> Self {
        Self {
            node_id: node_id.to_string(),
            array_kind,
            site_position_m,
            yaw_deg,
        }
    }

    pub(super) fn body_offsets_m(&self) -> Vec<[f32; 3]> {
        match self.array_kind {
            FixtureArrayKind::SirithPlanar => SIRITH_PLANAR_MIC_OFFSETS_M.to_vec(),
            FixtureArrayKind::SirithTetra => SIRITH_MIC_POSITIONS_M.to_vec(),
            FixtureArrayKind::Point => vec![[0.0, 0.0, 0.0]],
        }
    }

    pub(super) fn channel_count(&self) -> usize {
        self.body_offsets_m().len()
    }

    pub(super) fn sensor_id(&self, channel_index: usize) -> String {
        format!("{}:ch{channel_index}", self.node_id)
    }

    pub(super) fn stream_key(&self) -> String {
        format!("{}__audio_main__0001", self.node_id)
    }

    /// Rotation about +Z by `yaw_deg`, as (w, x, y, z).
    pub(super) fn body_to_world_quaternion_wxyz(&self) -> [f64; 4] {
        let half_yaw = self.yaw_deg.to_radians() / 2.0;
        [half_yaw.cos(), 0.0, 0.0, half_yaw.sin()]
    }

    /// `(sensor_id, world position)` with the yaw applied to each body offset.
    pub(super) fn sensor_world_positions_m(&self) -> Vec<(String, [f64; 3])> {
        let (sin_yaw, cos_yaw) = self.yaw_deg.to_radians().sin_cos();
        self.body_offsets_m()
            .iter()
            .enumerate()
            .map(|(channel_index, offset)| {
                let [x, y, z] = offset.map(f64::from);
                let world = [
                    self.site_position_m[0] + cos_yaw * x - sin_yaw * y,
                    self.site_position_m[1] + sin_yaw * x + cos_yaw * y,
                    self.site_position_m[2] + z,
                ];
                (self.sensor_id(channel_index), world)
            })
            .collect()
    }

    /// GPS fix that projects back onto `site_position_m` about
    /// `FIXTURE_SITE_ORIGIN` with Python's default (flat) projection.
    pub(super) fn reported_gps(&self) -> GeodeticOrigin {
        local_enu_to_geodetic(
            &FIXTURE_SITE_ORIGIN,
            CoordinateProjection::FlatWgs84Series,
            self.site_position_m,
        )
    }

    /// `node_context.node` as the ingest backend embeds it: body offsets,
    /// GPS fix, orientation and (planar only) the coplanar half-space.
    fn node_context_json(&self) -> serde_json::Value {
        let gps = self.reported_gps();
        let mut node = serde_json::json!({
            "id": self.node_id,
            "sensor_offsets_m": self.body_offsets_m(),
            "position_geo": {"lat": gps.lat, "lon": gps.lon, "alt_m": gps.alt_m},
            "orientation": {"yaw_deg": self.yaw_deg, "pitch_deg": 0.0, "roll_deg": 0.0},
            "metadata": {}
        });
        if self.array_kind == FixtureArrayKind::SirithPlanar {
            node["half_space"] = serde_json::Value::from("upper");
        }
        node
    }

    pub(super) fn operator_placement(&self, position_std_m: f64) -> NodePlacement {
        NodePlacement {
            position_m: Some(self.site_position_m),
            position_std_m: Some(position_std_m),
            placement_source: NodePlacementSource::OperatorOverride,
            body_to_world_quaternion_wxyz: Some(self.body_to_world_quaternion_wxyz()),
        }
    }
}

/// Operator placements for `nodes` in the fixture site frame.
pub(super) fn fixture_site_geometry_push(
    site_frame_id: &str,
    geometry_push_revision: u64,
    nodes: &[&FixtureNode],
) -> SiteGeometryPush {
    const OPERATOR_PLACEMENT_STD_M: f64 = 0.05;
    SiteGeometryPush {
        site_frame_id: site_frame_id.to_string(),
        geometry_push_revision,
        site_origin: FIXTURE_SITE_ORIGIN,
        projection: CoordinateProjection::FlatWgs84Series,
        node_placements: nodes
            .iter()
            .map(|node| {
                (
                    node.node_id.clone(),
                    node.operator_placement(OPERATOR_PLACEMENT_STD_M),
                )
            })
            .collect(),
    }
}

/// One broadband source heard by every fixture sensor with an integer-sample
/// propagation delay, so expected pair lags are exact.
pub(super) struct CoherentPointSource {
    position_m: [f64; 3],
    sound_speed_mps: f64,
    signal: Vec<f32>,
}

impl CoherentPointSource {
    pub(super) fn new(position_m: [f64; 3], frame_count: usize) -> Self {
        let mut state = SOURCE_NOISE_SEED;
        let signal = (0..SOURCE_PRE_ROLL_SAMPLES + frame_count * FIXTURE_FRAME_SAMPLES)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                SOURCE_PEAK_AMPLITUDE * (state as i32 as f32 / i32::MAX as f32)
            })
            .collect();
        Self {
            position_m,
            sound_speed_mps: f64::from(DspWorkerConfig::default().sound_speed_mps),
            signal,
        }
    }

    pub(super) fn delay_samples(&self, sensor_position_m: [f64; 3]) -> usize {
        let distance_m = (0..3)
            .map(|axis| (sensor_position_m[axis] - self.position_m[axis]).powi(2))
            .sum::<f64>()
            .sqrt();
        (distance_m / self.sound_speed_mps * f64::from(FIXTURE_SAMPLE_RATE_HZ)).round() as usize
    }

    /// Arrival at `a` minus arrival at `b`: the sign of `pair_tdoas[].lag_seconds`.
    pub(super) fn expected_lag_seconds(
        &self,
        position_a_m: [f64; 3],
        position_b_m: [f64; 3],
    ) -> f64 {
        (self.delay_samples(position_a_m) as f64 - self.delay_samples(position_b_m) as f64)
            / f64::from(FIXTURE_SAMPLE_RATE_HZ)
    }

    pub(super) fn frame_channels(&self, node: &FixtureNode, frame_index: usize) -> Vec<Vec<f32>> {
        node.sensor_world_positions_m()
            .into_iter()
            .map(|(_, position_m)| {
                let first_sample = SOURCE_PRE_ROLL_SAMPLES + frame_index * FIXTURE_FRAME_SAMPLES
                    - self.delay_samples(position_m);
                self.signal[first_sample..first_sample + FIXTURE_FRAME_SAMPLES].to_vec()
            })
            .collect()
    }
}

/// How a fixture manifest reaches the worker's geometry.
pub(super) enum FixtureClusterGeometry {
    /// No cluster fields: the GPS fix in `node_context` drives the automatic cluster.
    AutomaticGps,
    /// Operator-declared cluster with these world sensor positions.
    Explicit {
        cluster_id: String,
        sensor_positions: Vec<(String, [f32; 3])>,
    },
}

pub(super) fn explicit_cluster_sensor_positions(nodes: &[&FixtureNode]) -> Vec<(String, [f32; 3])> {
    nodes
        .iter()
        .flat_map(|node| node.sensor_world_positions_m())
        .map(|(sensor_id, position)| (sensor_id, position.map(|axis| axis as f32)))
        .collect()
}

/// One 32 ms GPS-timed frame of `node` hearing `source`; every node shares
/// the same packet timeline, so frame `k` is simultaneous across nodes.
pub(super) async fn fixture_frame_manifest(
    storage_root: &std::path::Path,
    node: &FixtureNode,
    frame_index: usize,
    source: &CoherentPointSource,
    geometry: &FixtureClusterGeometry,
) -> DspManifest {
    fixture_manifest_with_channels(
        storage_root,
        node,
        frame_index,
        FIXTURE_SAMPLE_RATE_HZ,
        source.frame_channels(node, frame_index),
        geometry,
    )
    .await
}

/// A silent frame of `node` on the shared packet timeline at another rate
/// (mixed-rate clusters).
pub(super) async fn fixture_silent_frame_manifest_at_rate(
    storage_root: &std::path::Path,
    node: &FixtureNode,
    frame_index: usize,
    sample_rate_hz: u32,
) -> DspManifest {
    let samples_per_frame =
        (u64::from(sample_rate_hz) * FIXTURE_FRAME_DURATION_NS / 1_000_000_000) as usize;
    fixture_manifest_with_channels(
        storage_root,
        node,
        frame_index,
        sample_rate_hz,
        vec![vec![0.0; samples_per_frame]; node.channel_count()],
        &FixtureClusterGeometry::AutomaticGps,
    )
    .await
}

/// `fixture_frame_manifest` with caller-built channels (one per sensor).
pub(super) async fn fixture_frame_manifest_from_channels(
    storage_root: &std::path::Path,
    node: &FixtureNode,
    frame_index: usize,
    channels: &[Vec<f32>],
    geometry: &FixtureClusterGeometry,
) -> DspManifest {
    fixture_manifest_with_channels(
        storage_root,
        node,
        frame_index,
        FIXTURE_SAMPLE_RATE_HZ,
        channels.to_vec(),
        geometry,
    )
    .await
}

async fn fixture_manifest_with_channels(
    storage_root: &std::path::Path,
    node: &FixtureNode,
    frame_index: usize,
    sample_rate_hz: u32,
    channels: Vec<Vec<f32>>,
    geometry: &FixtureClusterGeometry,
) -> DspManifest {
    let frame_index_u64 = frame_index as u64;
    let samples_per_frame = channels.first().map_or(0, Vec::len) as u64;
    let start_time_ns = FIXTURE_STREAM_START_NS + frame_index_u64 * FIXTURE_FRAME_DURATION_NS;
    let start_sample_index = frame_index_u64 * samples_per_frame;
    let node_json = node.node_context_json();
    let payload = serde_json::json!({
        "node": node_json,
        "buffered_frames": [{
            "frame": {
                "start_time_ns": start_time_ns,
                "utc_start_ns": start_time_ns,
                "utc_end_ns": start_time_ns + FIXTURE_FRAME_DURATION_NS,
                "start_sample_index": start_sample_index,
                "end_sample_index": start_sample_index + samples_per_frame,
                "sample_rate_hz": sample_rate_hz,
                "channels": node.channel_count(),
                "encoding": "pcm16le",
                "samples_per_channel": samples_per_frame,
                "samples_b64": encode_pcm16le_b64(&channels),
                "sequence": frame_index_u64 + 1,
                "time_quality": "gps_locked"
            }
        }]
    })
    .to_string();
    let (cluster_id, cluster_sensor_positions) = match geometry {
        FixtureClusterGeometry::AutomaticGps => (None, None),
        FixtureClusterGeometry::Explicit {
            cluster_id,
            sensor_positions,
        } => (Some(cluster_id.clone()), Some(sensor_positions.clone())),
    };
    let mut manifest = raw_manifest_for_payload_with_metadata(
        storage_root,
        &format!("manifest-{}-{frame_index}", node.node_id),
        &format!("seg-{}-{frame_index}", node.node_id),
        &node.stream_key(),
        payload,
        cluster_id,
        cluster_sensor_positions,
    )
    .await;
    manifest.node_context = Some(serde_json::json!({ "node": node_json }));
    manifest
}

#[test]
fn rotated_planar_fixture_places_sensors_around_its_site_position() {
    let node = FixtureNode::planar("planar", [10.0, -4.0, 1.5], 90.0);
    let positions = node.sensor_world_positions_m();
    assert_eq!(positions.len(), 5);
    // ch0 (body NE corner) lands in the world NW quadrant after a +90 deg yaw.
    let corner_xy_m = f64::from(SIRITH_PLANAR_MIC_OFFSETS_M[0][0]);
    assert!((positions[0].1[0] - (10.0 - corner_xy_m)).abs() < 1e-9);
    assert!((positions[0].1[1] - (-4.0 + corner_xy_m)).abs() < 1e-9);
    assert_eq!(positions[4].1, [10.0, -4.0, 1.5]);
    assert_eq!(positions[4].0, "planar:ch4");
}
