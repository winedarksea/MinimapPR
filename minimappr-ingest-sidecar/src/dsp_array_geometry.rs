//! Sound-speed and buffer-timeline helpers ingest uses.

use super::*;

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
