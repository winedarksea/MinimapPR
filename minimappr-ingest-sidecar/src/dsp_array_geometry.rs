//! Mic-array geometry: per-manifest sensor positions, GPS-defined automatic
//! clusters and cluster buffer routing.

use super::*;

/// Incrementally-built geometry for the automatic, GPS-defined array. Sensor
/// order never changes after insertion because buffer indices are long-lived.
/// Re-sorting after a new node joins would silently associate prior audio with
/// the wrong position.
#[derive(Default)]
pub(super) struct AutomaticGpsGeometry {
    pub(super) origin: Option<ReportedGpsPosition>,
    pub(super) sensor_positions: Vec<(String, [f32; 3])>,
    pub(super) sensor_indices_by_id: HashMap<String, usize>,
}

#[derive(Clone, Copy)]
pub(super) struct ReportedGpsPosition {
    pub(super) lat_deg: f64,
    pub(super) lon_deg: f64,
    pub(super) alt_m: f64,
}

impl AutomaticGpsGeometry {
    pub(super) fn origin_as_geodetic(
        &self,
    ) -> Option<minimappr_core::localization_sensor_entry::GeodeticOrigin> {
        self.origin.map(
            |origin| minimappr_core::localization_sensor_entry::GeodeticOrigin {
                lat: origin.lat_deg,
                lon: origin.lon_deg,
                alt_m: origin.alt_m,
            },
        )
    }

    pub(super) fn update_from_manifest(
        &mut self,
        manifest: &DspManifest,
        node_id: &str,
        channel_count: usize,
    ) -> Option<Vec<(String, [f32; 3])>> {
        let node = manifest.node_context.as_ref()?.get("node")?;
        let position = reported_gps_position(node)?;
        let origin = *self.origin.get_or_insert(position);
        let node_origin_m = gps_position_to_local_m(position, origin);
        let offsets = sensor_offsets_for_channel_count(node, channel_count);
        if offsets.len() != channel_count {
            return None;
        }

        for (channel_index, offset_m) in offsets.into_iter().enumerate() {
            let sensor_id = format!("{node_id}:ch{channel_index}");
            let sensor_position = [
                node_origin_m[0] + offset_m[0],
                node_origin_m[1] + offset_m[1],
                node_origin_m[2] + offset_m[2],
            ];
            if let Some(index) = self.sensor_indices_by_id.get(&sensor_id).copied() {
                self.sensor_positions[index].1 = sensor_position;
            } else {
                let index = self.sensor_positions.len();
                self.sensor_indices_by_id.insert(sensor_id.clone(), index);
                self.sensor_positions.push((sensor_id, sensor_position));
            }
        }
        Some(self.sensor_positions.clone())
    }
}

/// Extract per-channel mic positions from a manifest, checking cluster positions
/// first, then falling back to node_context.node.sensor_offsets_m, and finally
/// to the hardcoded Sirith tetrahedral positions for legacy payloads.
pub(crate) fn mic_positions_from_manifest(
    manifest: &crate::manifests::DspManifest,
) -> Vec<[f32; 3]> {
    // Cluster-resolved positions are authoritative when present.
    if let Some(ref cluster_pos) = manifest.cluster_sensor_positions {
        if !cluster_pos.is_empty() {
            return cluster_pos.iter().map(|(_, pos)| *pos).collect();
        }
    }
    mic_positions_from_node_context(&manifest.node_context)
}

/// Extract per-channel mic positions from node_context.node.sensor_offsets_m.
/// Falls back to the hardcoded Sirith tetrahedral positions if absent.
pub(crate) fn mic_positions_from_node_context(
    node_context: &Option<serde_json::Value>,
) -> Vec<[f32; 3]> {
    let positions = node_context
        .as_ref()
        .and_then(|ctx| ctx.get("node"))
        .and_then(|node| node.get("sensor_offsets_m"))
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            arr.iter()
                .map(|v| {
                    let arr = v.as_array()?;
                    if arr.len() < 3 {
                        return None;
                    }
                    Some([
                        arr[0].as_f64()? as f32,
                        arr[1].as_f64()? as f32,
                        arr[2].as_f64()? as f32,
                    ])
                })
                .collect::<Option<Vec<[f32; 3]>>>()
        })
        .filter(|v| !v.is_empty());
    positions.unwrap_or_else(|| SIRITH_MIC_POSITIONS_M.to_vec())
}

/// Extract the coplanar half-space constraint from node_context.node.half_space
/// ("upper" | "lower" | "none"). Absent or unrecognized values default to
/// `HalfSpace::None` (no constraint), matching non-planar arrays.
pub(crate) fn half_space_from_node_context(
    node_context: &Option<serde_json::Value>,
) -> crate::srp_phat::HalfSpace {
    node_context
        .as_ref()
        .and_then(|ctx| ctx.get("node"))
        .and_then(|node| node.get("half_space"))
        .and_then(|v| v.as_str())
        .map(crate::srp_phat::HalfSpace::from_wire_str)
        .unwrap_or(crate::srp_phat::HalfSpace::None)
}

pub(super) fn resolve_effective_sound_speed_mps(
    config: &DspWorkerConfig,
    temperature_c: Option<f32>,
    humidity_fraction: Option<f32>,
) -> f32 {
    if temperature_c.is_none() && humidity_fraction.is_none() {
        return config.sound_speed_mps;
    }
    speed_of_sound_mps(
        temperature_c.unwrap_or(config.default_temperature_c),
        humidity_fraction.unwrap_or(config.default_humidity_fraction),
    )
}

pub(super) fn resolve_buffer_start_time_ns(
    decoded: &DecodedAudioPayload,
    first_handle: &JournalPayloadHandle,
    sample_rate_hz: u32,
    now_ns: u128,
) -> i128 {
    decoded
        .start_time_ns
        .filter(|start_time_ns| *start_time_ns > 0)
        .or_else(|| first_handle.toa_ns.map(i128::from))
        .or_else(|| {
            let anchor_ns = first_handle
                .toa_ns
                .unwrap_or_else(|| first_handle.tor_ns.unwrap_or(now_ns as u64));
            decoded.start_sample_index.map(|sample_index| {
                sample_index_to_absolute_time_from_now_ns(
                    sample_index,
                    sample_rate_hz,
                    anchor_ns.into(),
                )
            })
        })
        .or_else(|| {
            let anchor_ns = first_handle
                .toa_ns
                .unwrap_or_else(|| first_handle.tor_ns.unwrap_or(now_ns as u64));
            first_handle.sample_index_start.map(|sample_index| {
                sample_index_to_absolute_time_from_now_ns(
                    sample_index as i64,
                    sample_rate_hz,
                    anchor_ns.into(),
                )
            })
        })
        .or_else(|| first_handle.tor_ns.map(i128::from))
        .unwrap_or(now_ns as i128)
}

pub(super) fn resolve_buffer_end_time_ns(
    buffers: &[SensorStreamBuffer],
    end_sample_index: Option<i64>,
    fallback_end_time_ns: i128,
    _sample_rate_hz: u32,
    reference_buffer_index: usize,
) -> i128 {
    // The buffer is anchored to the sample-index timeline, so windowing must
    // ride that timeline regardless of how far the publish-time TOA has drifted.
    // GPS-disciplined nodes recompute utcStart from a freshly-sampled clock
    // correction on every publish (NodeRunner::buildFrameForPacket), which can
    // shift consecutive packets' TOAs by tens of milliseconds while the sample
    // counter advances at a fixed rate. Trusting TOA here makes
    // coverage_centered_at / window_ending_at overshoot the buffer tail and
    // return None, which silently drops the entire classifier_render.
    end_sample_index
        .and_then(|sample_index| {
            buffers
                .get(reference_buffer_index)
                .or_else(|| buffers.first())
                .and_then(|buffer| buffer.time_for_sample_index(sample_index))
        })
        .unwrap_or(fallback_end_time_ns)
}

pub(super) fn resolve_cluster_buffer_routing(
    manifest: &crate::manifests::DspManifest,
    stream_key: &str,
    decoded_channel_count: usize,
) -> Option<ManifestBufferRouting> {
    // Route any cluster-aware manifest into the shared sensor buffer when the
    // cluster exposes enough geometry for SRP. This lets a tetrahedral node
    // contribute its four channels alongside separate point nodes instead of
    // forcing Rust to localize each manifest in isolation.
    if decoded_channel_count == 0 {
        return None;
    }
    let cluster_id = manifest.cluster_id.as_deref()?;
    let cluster_sensor_positions = manifest.cluster_sensor_positions.as_ref()?;
    // An automatic GPS array must begin sharing audio as soon as its first
    // node arrives. Waiting for four sensors routes those early frames into
    // isolated buffers, so the fourth node sees only itself. Explicit legacy
    // clusters retain their historical four-sensor minimum.
    if cluster_sensor_positions.len() < 4 && cluster_id != AUTOMATIC_GPS_CLUSTER_ID {
        return None;
    }
    let node_id = stream_key_node_id(stream_key);
    let stream_id = stream_key_stream_id(stream_key).unwrap_or("audio_main");
    let buffer_channel_indices = (0..decoded_channel_count)
        .map(|channel_index| {
            let sensor_id = format!("{node_id}:ch{channel_index}");
            cluster_sensor_positions
                .iter()
                .position(|(candidate_sensor_id, _)| candidate_sensor_id == &sensor_id)
        })
        .collect::<Option<Vec<_>>>()?;
    Some(ManifestBufferRouting {
        buffer_key: format!("{CLUSTER_BUFFER_KEY_PREFIX}{cluster_id}::{stream_id}"),
        buffer_channel_count: cluster_sensor_positions.len(),
        buffer_channel_indices,
    })
}

pub(super) fn reported_gps_position(node: &serde_json::Value) -> Option<ReportedGpsPosition> {
    let position = node.get("position_geo")?;
    let lat_deg = position.get("lat")?.as_f64()?;
    let lon_deg = position.get("lon")?.as_f64()?;
    let alt_m = position
        .get("alt_m")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    if !lat_deg.is_finite()
        || !lon_deg.is_finite()
        || !alt_m.is_finite()
        || !(-90.0..=90.0).contains(&lat_deg)
        || !(-180.0..=180.0).contains(&lon_deg)
    {
        return None;
    }
    Some(ReportedGpsPosition {
        lat_deg,
        lon_deg,
        alt_m,
    })
}

pub(super) fn sensor_offsets_for_channel_count(
    node: &serde_json::Value,
    channel_count: usize,
) -> Vec<[f32; 3]> {
    let offsets = node
        .get("sensor_offsets_m")
        .and_then(serde_json::Value::as_array)
        .and_then(|values| {
            values
                .iter()
                .take(channel_count)
                .map(|value| {
                    let values = value.as_array()?;
                    if values.len() < 3 {
                        return None;
                    }
                    Some([
                        values[0].as_f64()? as f32,
                        values[1].as_f64()? as f32,
                        values[2].as_f64()? as f32,
                    ])
                })
                .collect::<Option<Vec<_>>>()
        });
    match offsets {
        Some(offsets) if offsets.len() == channel_count => offsets,
        _ if channel_count == SIRITH_MIC_POSITIONS_M.len() => SIRITH_MIC_POSITIONS_M.to_vec(),
        _ if channel_count == 1 => vec![[0.0, 0.0, 0.0]],
        _ => Vec::new(),
    }
}

pub(super) fn gps_position_to_local_m(
    position: ReportedGpsPosition,
    origin: ReportedGpsPosition,
) -> [f32; 3] {
    let radians_per_degree = std::f64::consts::PI / 180.0;
    let east_m = (position.lon_deg - origin.lon_deg)
        * radians_per_degree
        * EARTH_RADIUS_M
        * origin.lat_deg.to_radians().cos();
    let north_m = (position.lat_deg - origin.lat_deg) * radians_per_degree * EARTH_RADIUS_M;
    [
        east_m as f32,
        north_m as f32,
        (position.alt_m - origin.alt_m) as f32,
    ]
}
