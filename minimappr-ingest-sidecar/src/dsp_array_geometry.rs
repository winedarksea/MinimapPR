//! Automatic-GPS site frame (until the R2 geometry push) plus the sound-speed
//! and buffer-timeline helpers ingest uses.

use super::*;

/// R2 seam: the frame automatic-GPS sensor positions are expressed in until
/// the geometry push lands. Mirrors the pinned `SiteGeometryState` surface
/// (`coordinate_frame`, `geometry_push_revision`, origin adoption on the
/// first trusted fix) so the integration can swap the type wholesale.
#[derive(Default)]
pub(super) struct AutomaticSiteFrame {
    origin: Option<ReportedGpsPosition>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ReportedGpsPosition {
    pub(super) lat_deg: f64,
    pub(super) lon_deg: f64,
    pub(super) alt_m: f64,
}

impl AutomaticSiteFrame {
    pub(super) fn coordinate_frame(
        &self,
    ) -> minimappr_core::localization_sensor_entry::SolveCoordinateFrame {
        minimappr_core::localization_sensor_entry::SolveCoordinateFrame {
            frame_id: None,
            kind:
                minimappr_core::localization_sensor_entry::CoordinateFrameKind::SidecarAutomaticEnu,
            origin: self.origin.map(|origin| {
                minimappr_core::localization_sensor_entry::GeodeticOrigin {
                    lat: origin.lat_deg,
                    lon: origin.lon_deg,
                    alt_m: origin.alt_m,
                }
            }),
            // The legacy spherical projection is not a contract projection.
            projection: None,
        }
    }

    pub(super) fn geometry_push_revision(&self) -> Option<u64> {
        None
    }

    /// Node origin in the automatic frame; the first fix seen becomes the origin.
    pub(super) fn local_position_m(&mut self, fix: ReportedGpsPosition) -> [f64; 3] {
        let origin = *self.origin.get_or_insert(fix);
        gps_position_to_local_m(fix, origin)
    }
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

fn gps_position_to_local_m(position: ReportedGpsPosition, origin: ReportedGpsPosition) -> [f64; 3] {
    let radians_per_degree = std::f64::consts::PI / 180.0;
    let east_m = (position.lon_deg - origin.lon_deg)
        * radians_per_degree
        * EARTH_RADIUS_M
        * origin.lat_deg.to_radians().cos();
    let north_m = (position.lat_deg - origin.lat_deg) * radians_per_degree * EARTH_RADIUS_M;
    [east_m, north_m, position.alt_m - origin.alt_m]
}
