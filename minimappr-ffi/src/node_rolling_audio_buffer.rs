//! The node's PCM16 rolling buffer on the 48 kHz device clock, plus the
//! Stage-1 detector fed from the same blocks. Implements the core's
//! `NodeAudioWindowSource` so the state machine can answer readiness queries
//! and clip requests without the shell copying audio across the FFI.

use minimappr_core::clip_codec::DecimationFilterId;
use minimappr_core::mesh::node_state_machine::{EncodedLocalClip, LocalAudioReadiness, NodeAudioWindowSource};
use minimappr_core::protocol::detection_messages::ClipOnset;
use minimappr_core::protocol::wire_codes::FrequencyBandHz;
use minimappr_core::stage1::{Stage1Configuration, Stage1DetectionMode, Stage1Detector, Stage1Event};

use crate::clip_codec::encode_clip_from_rolling_buffer_slice;

/// Compaction slack as a fraction of the retained span: the front is drained
/// only once this much extra audio piles up, so the memmove cost is paid
/// every few seconds instead of on every 20 ms capture block.
const COMPACTION_SLACK_FRACTION_OF_RETAINED_SPAN: usize = 4;

pub struct NodeRollingAudioBuffer {
    retained_sample_count: usize,
    samples: Vec<i16>,
    first_buffered_device_sample_index: u64,
    capture_gap_count: u64,
    stage1_detector: Stage1Detector,
}

impl NodeRollingAudioBuffer {
    pub fn new(retained_sample_count: usize, stage1_mode: Stage1DetectionMode) -> Self {
        Self {
            retained_sample_count,
            samples: Vec::with_capacity(
                retained_sample_count + retained_sample_count / COMPACTION_SLACK_FRACTION_OF_RETAINED_SPAN,
            ),
            first_buffered_device_sample_index: 0,
            capture_gap_count: 0,
            stage1_detector: Stage1Detector::new(Stage1Configuration::for_mode(stage1_mode)),
        }
    }

    /// Device index one past the newest buffered sample (0 before capture).
    pub fn next_device_sample_index(&self) -> u64 {
        self.first_buffered_device_sample_index + self.samples.len() as u64
    }

    pub fn capture_gap_count(&self) -> u64 {
        self.capture_gap_count
    }

    /// Appends a capture block. A block that does not continue the buffer
    /// restarts it: a clip spanning a gap would carry wrong sample timing,
    /// which is worse for TDOA than having no clip.
    pub fn push_capture_block(&mut self, first_device_sample_index: u64, block: &[i16]) -> Vec<Stage1Event> {
        if !self.samples.is_empty() && first_device_sample_index != self.next_device_sample_index() {
            self.capture_gap_count += 1;
            self.samples.clear();
        }
        if self.samples.is_empty() {
            self.first_buffered_device_sample_index = first_device_sample_index;
        }
        self.samples.extend_from_slice(block);
        let compaction_threshold =
            self.retained_sample_count + self.retained_sample_count / COMPACTION_SLACK_FRACTION_OF_RETAINED_SPAN;
        if self.samples.len() > compaction_threshold {
            let excess = self.samples.len() - self.retained_sample_count;
            self.samples.drain(..excess);
            self.first_buffered_device_sample_index += excess as u64;
        }
        self.stage1_detector.process_pcm16_block(first_device_sample_index, block)
    }

    /// Full-scale-normalised copy of a buffered window, for chirp detection
    /// outside the device lock. `None` unless the whole window is buffered.
    pub fn copy_window_as_f32(&self, window_first_device_sample_index: u64, window_sample_count: usize) -> Option<Vec<f32>> {
        let window_end_exclusive = window_first_device_sample_index.checked_add(window_sample_count as u64)?;
        if window_sample_count == 0 || !self.window_is_buffered(window_first_device_sample_index, window_end_exclusive) {
            return None;
        }
        let start = (window_first_device_sample_index - self.first_buffered_device_sample_index) as usize;
        Some(crate::chirp_detector::pcm16_to_unit_f32(&self.samples[start..start + window_sample_count]))
    }

    /// Oldest device index still buffered.
    pub fn first_buffered_device_sample_index(&self) -> u64 {
        self.first_buffered_device_sample_index
    }

    fn window_is_buffered(&self, window_first: u64, window_end_exclusive: u64) -> bool {
        !self.samples.is_empty()
            && window_first >= self.first_buffered_device_sample_index
            && window_end_exclusive <= self.next_device_sample_index()
    }
}

impl NodeAudioWindowSource for NodeRollingAudioBuffer {
    fn readiness(&self, centre_device_sample_index: u64, half_window_samples: u32, _band: FrequencyBandHz) -> LocalAudioReadiness {
        // Indices can come from a remote host's request: overflow must read as
        // "not buffered", never panic (a panic would poison the node mutex).
        let window_first = centre_device_sample_index.saturating_sub(u64::from(half_window_samples));
        let audio_still_buffered = centre_device_sample_index
            .checked_add(u64::from(half_window_samples))
            .is_some_and(|window_end_exclusive| self.window_is_buffered(window_first, window_end_exclusive));
        LocalAudioReadiness {
            audio_still_buffered,
            // Skeleton: no on-device VAD or motion blanking in the shell yet
            // (see docs/field_mode_progress/android_skeleton.md, open issues).
            speech_free: true,
            handling_noise_free: true,
            in_band_level_over_floor_db: 0.0,
        }
    }

    fn encode_clip(&self, window_first_device_sample_index: u64, window_sample_count: u32, decimation_filter_id: u8) -> Option<EncodedLocalClip> {
        let window_end_exclusive = window_first_device_sample_index.checked_add(u64::from(window_sample_count))?;
        if window_sample_count == 0 || !self.window_is_buffered(window_first_device_sample_index, window_end_exclusive) {
            return None;
        }
        let decimation_filter_id = DecimationFilterId::from_wire_byte(decimation_filter_id)?;
        let clip_container = encode_clip_from_rolling_buffer_slice(
            &self.samples,
            self.first_buffered_device_sample_index,
            window_first_device_sample_index,
            window_sample_count,
            decimation_filter_id,
        )
        .ok()?;
        let onsets = self
            .stage1_detector
            .onsets_in_device_range(window_first_device_sample_index, window_end_exclusive)
            .into_iter()
            .map(|onset| ClipOnset {
                device_sample_index: onset.device_sample_index,
                positive_flux_db: onset.positive_flux_db as f32,
            })
            .collect();
        Some(EncodedLocalClip {
            clip_container,
            onsets,
            // Sensor frames on the audio timeline land with the sensor stream.
            sensor_frames: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RETAINED_SAMPLES: usize = 48_000;
    const BLOCK_SAMPLES: usize = 960;
    const BIRD_BAND: FrequencyBandHz = FrequencyBandHz { low_hz: 1_500.0, high_hz: 10_000.0 };

    fn filled_buffer(block_count: usize) -> NodeRollingAudioBuffer {
        let mut buffer = NodeRollingAudioBuffer::new(RETAINED_SAMPLES, Stage1DetectionMode::Birds);
        for block_index in 0..block_count {
            buffer.push_capture_block((block_index * BLOCK_SAMPLES) as u64, &[0; BLOCK_SAMPLES]);
        }
        buffer
    }

    #[test]
    fn retained_span_is_bounded_and_tracks_device_index() {
        let total_blocks = 200;
        let buffer = filled_buffer(total_blocks);
        assert_eq!(buffer.next_device_sample_index(), (total_blocks * BLOCK_SAMPLES) as u64);
        assert!(buffer.samples.len() <= RETAINED_SAMPLES + RETAINED_SAMPLES / COMPACTION_SLACK_FRACTION_OF_RETAINED_SPAN);
        assert_eq!(buffer.capture_gap_count(), 0);
    }

    #[test]
    fn gap_restarts_buffer_and_old_windows_are_not_ready() {
        let mut buffer = filled_buffer(10);
        let after_gap = buffer.next_device_sample_index() + BLOCK_SAMPLES as u64;
        buffer.push_capture_block(after_gap, &[0; BLOCK_SAMPLES]);
        assert_eq!(buffer.capture_gap_count(), 1);
        assert!(!buffer.readiness(BLOCK_SAMPLES as u64, 100, BIRD_BAND).audio_still_buffered);
        assert!(buffer.readiness(after_gap + 480, 100, BIRD_BAND).audio_still_buffered);
    }

    #[test]
    fn overflowing_remote_window_indices_are_not_buffered_and_do_not_panic() {
        let buffer = filled_buffer(10);
        assert!(!buffer.readiness(u64::MAX, u32::MAX, BIRD_BAND).audio_still_buffered);
        let filter = DecimationFilterId::BirdBandWithRumbleCut.wire_byte();
        assert!(buffer.encode_clip(u64::MAX - 10, u32::MAX, filter).is_none());
        assert!(buffer.encode_clip(0, 0, filter).is_none());
    }

    #[test]
    fn window_copy_requires_the_whole_window_buffered() {
        let mut buffer = NodeRollingAudioBuffer::new(RETAINED_SAMPLES, Stage1DetectionMode::Birds);
        buffer.push_capture_block(0, &[i16::MAX; BLOCK_SAMPLES]);
        let window = buffer.copy_window_as_f32(10, 100).expect("buffered");
        assert_eq!(window.len(), 100);
        assert!((window[0] - 1.0).abs() < 1e-6);
        assert!(buffer.copy_window_as_f32(BLOCK_SAMPLES as u64 - 10, 100).is_none());
        assert!(buffer.copy_window_as_f32(u64::MAX - 5, 100).is_none());
    }

    #[test]
    fn buffered_window_encodes_a_clip() {
        let buffer = filled_buffer(50);
        let clip = buffer
            .encode_clip(9_600, 9_600, DecimationFilterId::BirdBandWithRumbleCut.wire_byte())
            .expect("clip");
        assert!(!clip.clip_container.is_empty());
    }
}
