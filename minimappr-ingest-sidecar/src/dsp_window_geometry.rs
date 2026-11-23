//! Pure window extraction, coverage and timestamp helpers.

use super::*;

#[allow(dead_code)]
pub(super) fn localization_channel_states(
    buffers: &[SensorStreamBuffer],
    end_ns: i128,
    window_seconds: f64,
) -> Vec<LocalizationChannelState> {
    buffers
        .iter()
        .map(|buf| LocalizationChannelState {
            coverage: buf.coverage_ending_at(end_ns, window_seconds),
            window: buf
                .window_ending_at(end_ns, window_seconds)
                .unwrap_or_default(),
        })
        .collect()
}

/// Build localization channel states using centered windows (matching Python's
/// `get_window(center_time_ns, window_seconds)`). TDOA is computed relative to
/// the window center, so centering on the frame midpoint gives symmetric coverage.
pub(crate) fn localization_channel_states_centered(
    buffers: &[SensorStreamBuffer],
    center_time_ns: i128,
    window_seconds: f64,
) -> Vec<LocalizationChannelState> {
    buffers
        .iter()
        .map(|buf| LocalizationChannelState {
            coverage: buf.coverage_centered_at(center_time_ns, window_seconds),
            window: buf
                .window_centered_at(center_time_ns, window_seconds)
                .unwrap_or_default(),
        })
        .collect()
}

pub(super) fn resolve_localization_center_time_ns(
    buffers: &[SensorStreamBuffer],
    clamp_buffer_index: usize,
    buffer_start_time_ns: i128,
    end_ns: i128,
    window_seconds: f64,
) -> i128 {
    let window_duration_ns = (window_seconds * 1_000_000_000.0).round() as i128;
    let half_window_duration_ns = window_duration_ns / 2;
    let clamp_buffer = buffers.get(clamp_buffer_index).or_else(|| buffers.first());
    let buffered_range_start_ns = clamp_buffer
        .and_then(|buffer| buffer.start_time_ns())
        .unwrap_or(buffer_start_time_ns);
    let buffered_range_end_ns = clamp_buffer
        .and_then(|buffer| buffer.end_time_ns())
        .unwrap_or(end_ns);
    let unclamped_center_time_ns = end_ns.saturating_sub(half_window_duration_ns);
    let centered_window_start_limit_ns =
        buffered_range_start_ns.saturating_add(half_window_duration_ns);
    let centered_window_end_limit_ns =
        buffered_range_end_ns.saturating_sub(half_window_duration_ns);

    if centered_window_start_limit_ns <= centered_window_end_limit_ns {
        unclamped_center_time_ns.clamp(centered_window_start_limit_ns, centered_window_end_limit_ns)
    } else {
        unclamped_center_time_ns
    }
}

pub(crate) fn channel_windows_ending_at(
    buffers: &[SensorStreamBuffer],
    end_ns: i128,
    window_seconds: f64,
) -> Vec<Vec<f32>> {
    buffers
        .iter()
        .map(|buf| {
            buf.window_ending_at(end_ns, window_seconds)
                .unwrap_or_default()
        })
        .collect()
}

pub(crate) fn channel_windows_ending_at_with_gap_concealment(
    buffers: &[SensorStreamBuffer],
    end_ns: i128,
    window_seconds: f64,
) -> Vec<Vec<f32>> {
    buffers
        .iter()
        .map(|buf| {
            buf.window_ending_at_with_gap_concealment(end_ns, window_seconds)
                .unwrap_or_default()
        })
        .collect()
}

pub(crate) fn channel_coverage_ending_at(
    buffers: &[SensorStreamBuffer],
    end_ns: i128,
    window_seconds: f64,
) -> Vec<Option<AudioCoverageStats>> {
    buffers
        .iter()
        .map(|buf| buf.coverage_ending_at(end_ns, window_seconds))
        .collect()
}

pub(crate) fn classifier_render_bounds_from_windows(
    end_ns: i128,
    windows: &[Vec<f32>],
    sample_rate_hz: u32,
) -> (Option<u128>, Option<u128>) {
    let sample_count = windows.iter().map(|window| window.len()).max().unwrap_or(0);
    if sample_count == 0 || sample_rate_hz == 0 {
        return (None, None);
    }
    let duration_ns =
        ((sample_count as f64 / sample_rate_hz as f64) * 1_000_000_000.0).round() as i128;
    let render_end_ns = end_ns.max(0);
    let render_start_ns = render_end_ns.saturating_sub(duration_ns).max(0);
    (Some(render_start_ns as u128), Some(render_end_ns as u128))
}

pub(crate) fn eligible_localization_channels(
    channel_states: &[LocalizationChannelState],
    min_coverage_ratio: f64,
) -> Vec<usize> {
    channel_states
        .iter()
        .enumerate()
        .filter_map(|(channel_index, state)| {
            let coverage = state.coverage.as_ref()?;
            (coverage.coverage_ratio >= min_coverage_ratio && !state.window.is_empty())
                .then_some(channel_index)
        })
        .collect()
}

pub(super) fn eligible_coverage_channels(
    channel_coverage: &[Option<AudioCoverageStats>],
    min_coverage_ratio: f64,
) -> Vec<usize> {
    channel_coverage
        .iter()
        .enumerate()
        .filter_map(|(channel_index, coverage)| {
            let coverage = coverage.as_ref()?;
            (coverage.coverage_ratio >= min_coverage_ratio).then_some(channel_index)
        })
        .collect()
}

pub(super) fn stream_key_node_id(stream_key: &str) -> &str {
    stream_key.split("__").next().unwrap_or(stream_key)
}

pub(super) fn stream_key_stream_id(stream_key: &str) -> Option<&str> {
    stream_key.split("__").nth(1)
}

pub(super) fn should_use_receipt_time_alignment(
    node_timestamp_is_available: bool,
    skew_ns: u128,
    max_trusted_node_clock_skew_ns: u128,
) -> bool {
    // Preserve packet/node timing for TDOA whenever firmware timing exists.
    // Receipt-time alignment is a last-resort fallback only when node timing
    // is absent and skew is beyond the trusted horizon.
    //
    // Receipt-time-fallback policy — intentional asymmetry with the Python ingest
    // path. Python's `_buffer_timestamps_for_frame` (ingest.py:688) *overrides*
    // firmware timestamps when time_quality=FREE_RUNNING + gps_optional capability
    // + node clock skew > max_trusted, so debug audio stays playable on dev nodes
    // that lack GPS/NTP lock. The Rust sidecar deliberately does *not* override:
    // TDOA correctness depends on consistent packet-time alignment across nodes,
    // and silently rewriting timestamps masks clock issues that operators must
    // see to act on. Do not align the two implementations — this divergence is
    // covered by parity tests as an expected difference, not a bug.
    !node_timestamp_is_available && skew_ns > max_trusted_node_clock_skew_ns
}

pub(super) fn sample_index_to_absolute_time_from_now_ns(
    sample_index: i64,
    sample_rate_hz: u32,
    now_ns: u128,
) -> i128 {
    let relative_duration_ns = sample_index_to_relative_time_ns(sample_index, sample_rate_hz);
    if relative_duration_ns <= 0 {
        return now_ns as i128;
    }
    now_ns.saturating_sub(relative_duration_ns as u128) as i128
}

pub(super) fn manifest_is_older_than_buffer_horizon(
    manifest: &DspManifest,
    now_ns: u128,
    max_buffer_seconds: f64,
) -> bool {
    let horizon_ns = (max_buffer_seconds.max(0.0) * 1_000_000_000.0).round() as u128;
    if horizon_ns == 0 {
        return false;
    }
    let newest_source_ns = manifest
        .source_handles
        .iter()
        .filter_map(|handle| handle.tor_ns.or(handle.toa_ns))
        .map(u128::from)
        .max()
        .unwrap_or(manifest.created_ns);
    let newest_received_ns = manifest
        .source_handles
        .iter()
        .filter_map(|handle| handle.received_ns)
        .max();

    // Use explicit ingest receipt time for freshness when available.
    // Source timing fields come from node/packet clocks and can legitimately
    // be far in the past (replay/tests/GPS-anchored captures). Staleness
    // gating is meant to protect live buffering from old queue backlog, so
    // a manifest that just arrived should not be dropped solely because its
    // embedded capture epoch is old.
    let freshness_anchor_ns =
        newest_received_ns.unwrap_or_else(|| newest_source_ns.max(manifest.created_ns));
    now_ns.saturating_sub(freshness_anchor_ns) > horizon_ns
}

pub(super) fn classifier_render_min_interval_ns(
    classifier_render_min_interval_seconds: f64,
    _pending_backlog_depth: usize,
) -> u128 {
    if classifier_render_min_interval_seconds <= 0.0 {
        return 0;
    }
    (classifier_render_min_interval_seconds * 1_000_000_000.0).round() as u128
}

pub(super) fn sample_index_to_relative_time_ns(sample_index: i64, sample_rate_hz: u32) -> i128 {
    if sample_rate_hz == 0 {
        return 0;
    }
    i128::from(sample_index).saturating_mul(1_000_000_000) / i128::from(sample_rate_hz)
}

pub(super) fn speed_of_sound_mps(temperature_c: f32, humidity_fraction: f32) -> f32 {
    let humidity_percent = humidity_fraction.clamp(0.0, 1.0) * 100.0;
    331.3 + (0.606 * temperature_c) + (0.0124 * humidity_percent)
}
