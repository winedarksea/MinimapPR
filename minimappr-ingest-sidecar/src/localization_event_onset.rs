//! Where an event-triggered solve (either gate) reads its window.
//!
//! The relative trigger judges the reporter's newest frame, so it fires on the
//! frame in which an event begins. The node-local window ends at that frame,
//! so an event beginning late in it left the window mostly pre-onset noise:
//! a 100 m burst put 8 ms of signal in the 80 ms window, and the noise lags
//! became the event's TDOAs (E5 range matrix, `100m-single_tetra-*`). The 5 s
//! cooldown then blocked the frame that held the event. The absolute gate
//! fails the same way on a cross-node cluster: it judges only the reporter's
//! window, which can hold the nearest node's onset in its last few
//! milliseconds while the farther nodes' arrivals fall after it (E5
//! `100m-four_node_spread-gps_locked-hand_measured` with tetra-d reporting:
//! tetra-d's onset 56 ms into the 80 ms window, tetra-a's 86 ms, so the solve
//! returned a 9.8 ms residual and an upward bearing while tetra-b's
//! onset-centred solve was right).
//!
//! A triggered window (either gate) therefore starts at the onset whenever the
//! frame located one: the first short block of the frame that reaches the
//! level the frame had to clear over its noise floor.
//! The coverage wait holds the solve until the audio after the onset has
//! arrived (the event span grows around this centre for cross-node layouts).
//! With the wait off nothing later can be read, so the window stays on the
//! newest audio.

use super::*;

/// Onset resolution: short enough to place a call's attack within a few
/// milliseconds of an 80 ms window, long enough (80 samples at 16 kHz) for a
/// block RMS that noise alone rarely lifts to the trigger level.
const ONSET_SCAN_BLOCK_SECONDS: f64 = 0.005;
const NANOSECONDS_PER_SECOND_F64: f64 = 1_000_000_000.0;

/// Seconds from the frame's first sample to the first block whose strongest
/// channel reaches `level_rms`; `None` for an empty frame.
///
/// A frame that cleared `level_rms` as a whole has at least one block at or
/// above it (its mean square is the blocks' length-weighted mean), so the
/// loudest block answers only when float rounding hides that block.
pub(crate) fn event_onset_offset_seconds(
    channels: &[Vec<f32>],
    sample_rate_hz: u32,
    level_rms: f64,
) -> Option<f64> {
    let frame_samples = channels.iter().map(Vec::len).max().unwrap_or(0);
    if frame_samples == 0 || sample_rate_hz == 0 {
        return None;
    }
    let block_samples =
        ((ONSET_SCAN_BLOCK_SECONDS * f64::from(sample_rate_hz)).round() as usize).max(1);
    let mut loudest_block: Option<(usize, f64)> = None;
    for block_start in (0..frame_samples).step_by(block_samples) {
        let block_rms = channels
            .iter()
            .filter_map(|channel| {
                let block_end = (block_start + block_samples).min(channel.len());
                (block_start < block_end)
                    .then(|| f64::from(window_rms(&channel[block_start..block_end])))
            })
            .fold(0.0, f64::max);
        if block_rms >= level_rms {
            return Some(block_start as f64 / f64::from(sample_rate_hz));
        }
        if loudest_block.map_or(true, |(_, loudest_rms)| block_rms > loudest_rms) {
            loudest_block = Some((block_start, block_rms));
        }
    }
    loudest_block.map(|(block_start, _)| block_start as f64 / f64::from(sample_rate_hz))
}

/// Centre of the event solve's node-local window: `onset .. onset + window`
/// for a trigger (absolute or relative) whose frame located an onset while
/// the coverage wait can collect the audio after it; otherwise the buffered
/// placement (window ending at the trigger frame). The onset exists only when
/// the frame stood above its node's noise floor (relative trigger enabled).
pub(crate) fn event_window_center_time_ns(
    trigger: Option<LocalizationTriggerSource>,
    node_frame: NodeFrameTrigger,
    frame_end_time_ns: i128,
    buffered_center_time_ns: i128,
    node_local_window_seconds: f64,
    coverage_wait_enabled: bool,
) -> i128 {
    let (Some(_), Some(onset_offset_seconds), true) = (
        trigger,
        node_frame.onset_offset_seconds,
        coverage_wait_enabled,
    ) else {
        return buffered_center_time_ns;
    };
    let seconds_from_frame_end_to_center =
        node_frame.frame_seconds - onset_offset_seconds - 0.5 * node_local_window_seconds;
    frame_end_time_ns
        - (seconds_from_frame_end_to_center * NANOSECONDS_PER_SECOND_F64).round() as i128
}

impl DspWorker {
    /// Moves the buffered localization window to `center_time_ns` and returns
    /// the channels it now covers; `None` when the buffer has gone.
    pub(super) fn recenter_localization_window(
        &self,
        owned: &OwnedManifestAudio,
        buffered: &mut BufferedManifestAudio,
        center_time_ns: i128,
    ) -> Option<Vec<usize>> {
        let buffers = self.buffers.get(&owned.buffer_key)?;
        let mut channel_states = localization_channel_states_centered(
            buffers,
            center_time_ns,
            self.config.window_seconds,
        );
        owned
            .sensor_layout
            .clear_solve_excluded_channels(&mut channel_states);
        let active_channels =
            eligible_localization_channels(&channel_states, self.config.min_coverage_ratio);
        buffered.localization_center_time_ns = center_time_ns;
        buffered.channel_states = channel_states;
        Some(active_channels)
    }
}

#[cfg(test)]
#[path = "localization_event_onset_tests.rs"]
mod localization_event_onset_tests;
