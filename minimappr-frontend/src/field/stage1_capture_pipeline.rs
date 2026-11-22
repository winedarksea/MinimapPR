//! Capture -> Stage-1 -> classification windows + clips, all on the device
//! clock (sample indices, never wall time).
//!
//! Pure Rust over `minimappr-core`: the worklet hands `(first_frame,
//! samples)` chunks in; out come Stage-1 events and 3 s windows ready for
//! BirdNET. A gap in `first_frame` (dropped worklet messages) restarts the
//! rolling buffer at the new index and is counted, so a clip never silently
//! splices audio across a gap.

use std::collections::VecDeque;

use minimappr_core::clip_codec::field_clip::encode_field_clip;
use minimappr_core::clip_codec::{DecimationFilterId, FieldClipDecimator};
use minimappr_core::stage1::{Stage1Configuration, Stage1Counters, Stage1DetectionMode, Stage1Detector, Stage1Event};

use super::field_configuration::{
    CLASSIFICATION_PRE_ROLL_SECONDS, CLASSIFICATION_WINDOW_HOP_SECONDS, CLASSIFICATION_WINDOW_SECONDS,
    CLIP_CODEC_REQUIRED_SAMPLE_RATE_HZ, CLIP_WINDOW_SECONDS, MAXIMUM_QUEUED_CLASSIFICATION_WINDOWS,
    ROLLING_BUFFER_SECONDS,
};

const FLOAT_TO_PCM16_SCALE: f32 = 32_767.0;

/// A window of capture-rate audio for the classifier, with Stage-1 context.
#[derive(Debug, Clone)]
pub struct ClassificationWindow {
    pub first_device_sample_index: u64,
    pub samples: Vec<f32>,
    pub stage1_contrast_db: f32,
    pub stage1_active_band_hz: Option<(f32, f32)>,
}

/// What one chunk produced, for the UI and the classifier queue.
#[derive(Debug, Clone, Default)]
pub struct CaptureChunkOutcome {
    pub trigger_opened_count: u32,
    pub onset_count: u32,
    pub last_trigger_contrast_db: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CapturePipelineCounters {
    pub chunks_received: u64,
    pub samples_received: u64,
    pub capture_gaps: u64,
    /// Where the newest gap was: expected vs received device index.
    pub last_gap_expected_index: u64,
    pub last_gap_received_index: u64,
    pub windows_queued: u64,
    pub windows_dropped_classifier_busy: u64,
}

/// The trigger that the next classification window belongs to.
struct OpenTriggerContext {
    next_window_first_index: u64,
    contrast_db: f32,
    active_band_hz: (f32, f32),
}

pub struct Stage1CapturePipeline {
    sample_rate_hz: u32,
    detector: Stage1Detector,
    rolling_buffer: VecDeque<f32>,
    rolling_buffer_first_index: u64,
    rolling_buffer_capacity: usize,
    open_trigger: Option<OpenTriggerContext>,
    ready_windows: VecDeque<ClassificationWindow>,
    counters: CapturePipelineCounters,
    clip_decimator: Option<FieldClipDecimator>,
}

impl Stage1CapturePipeline {
    pub fn new(sample_rate_hz: u32) -> Self {
        let mut configuration = Stage1Configuration::for_mode(Stage1DetectionMode::Birds);
        configuration.input_sample_rate_hz = sample_rate_hz;
        // The clip codec's decimator is exact 48 -> 24 kHz only.
        let clip_decimator = (sample_rate_hz == CLIP_CODEC_REQUIRED_SAMPLE_RATE_HZ)
            .then(|| FieldClipDecimator::for_filter(DecimationFilterId::BirdBandWithRumbleCut));
        Self {
            sample_rate_hz,
            detector: Stage1Detector::new(configuration),
            rolling_buffer: VecDeque::new(),
            rolling_buffer_first_index: 0,
            rolling_buffer_capacity: seconds_to_samples(ROLLING_BUFFER_SECONDS, sample_rate_hz),
            open_trigger: None,
            ready_windows: VecDeque::new(),
            counters: CapturePipelineCounters::default(),
            clip_decimator,
        }
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    pub fn stage1_counters(&self) -> Stage1Counters {
        self.detector.counters()
    }

    pub fn counters(&self) -> CapturePipelineCounters {
        self.counters
    }

    pub fn trigger_is_open(&self) -> bool {
        self.detector.trigger_is_open()
    }

    pub fn last_contrast_db(&self) -> f64 {
        self.detector.last_frame_diagnostics().contrast_db
    }

    /// Device-clock index just past the newest sample. Not `samples_received`:
    /// that excludes gap frames and would drift from the device clock.
    pub fn next_expected_device_sample_index(&self) -> u64 {
        self.rolling_buffer_first_index + self.rolling_buffer.len() as u64
    }

    pub fn clips_supported(&self) -> bool {
        self.clip_decimator.is_some()
    }

    /// Next window for the classifier, oldest first.
    pub fn take_ready_window(&mut self) -> Option<ClassificationWindow> {
        self.ready_windows.pop_front()
    }

    pub fn push_chunk(&mut self, first_device_sample_index: u64, samples: &[f32]) -> CaptureChunkOutcome {
        self.counters.chunks_received += 1;
        self.counters.samples_received += samples.len() as u64;
        self.append_to_rolling_buffer(first_device_sample_index, samples);
        let mut outcome = CaptureChunkOutcome::default();
        for event in self.detector.process_block(first_device_sample_index, samples) {
            match event {
                Stage1Event::TriggerOpened {
                    device_sample_index,
                    contrast_db,
                    active_band_low_hz,
                    active_band_high_hz,
                    ..
                } => {
                    outcome.trigger_opened_count += 1;
                    outcome.last_trigger_contrast_db = Some(contrast_db);
                    let pre_roll = seconds_to_samples(CLASSIFICATION_PRE_ROLL_SECONDS, self.sample_rate_hz) as u64;
                    self.open_trigger = Some(OpenTriggerContext {
                        next_window_first_index: device_sample_index.saturating_sub(pre_roll),
                        contrast_db: contrast_db as f32,
                        active_band_hz: (active_band_low_hz as f32, active_band_high_hz as f32),
                    });
                }
                Stage1Event::TriggerClosed { .. } => {}
                Stage1Event::Onset(_) => outcome.onset_count += 1,
            }
        }
        self.cut_due_windows();
        outcome
    }

    /// Emits every window whose audio is complete. A window that started
    /// while the trigger was open is cut even if the trigger closed since.
    fn cut_due_windows(&mut self) {
        let window_length = seconds_to_samples(CLASSIFICATION_WINDOW_SECONDS, self.sample_rate_hz);
        let hop = seconds_to_samples(CLASSIFICATION_WINDOW_HOP_SECONDS, self.sample_rate_hz) as u64;
        let buffer_end = self.rolling_buffer_first_index + self.rolling_buffer.len() as u64;
        let trigger_open = self.detector.trigger_is_open();
        let Some(context) = self.open_trigger.as_mut() else { return };
        let first_index = context.next_window_first_index.max(self.rolling_buffer_first_index);
        if first_index + window_length as u64 > buffer_end {
            return;
        }
        let window = ClassificationWindow {
            first_device_sample_index: first_index,
            samples: copy_range(&self.rolling_buffer, self.rolling_buffer_first_index, first_index, window_length),
            stage1_contrast_db: context.contrast_db,
            stage1_active_band_hz: Some(context.active_band_hz),
        };
        context.next_window_first_index = first_index + hop;
        if !trigger_open {
            self.open_trigger = None;
        }
        if self.ready_windows.len() >= MAXIMUM_QUEUED_CLASSIFICATION_WINDOWS {
            self.ready_windows.pop_front();
            self.counters.windows_dropped_classifier_busy += 1;
        }
        self.ready_windows.push_back(window);
        self.counters.windows_queued += 1;
    }

    /// FLAC+MMCL clip of `CLIP_WINDOW_SECONDS` centred in the window, cut
    /// from the rolling buffer with real neighbouring audio as filter context.
    pub fn encode_clip_centred_in(&self, window: &ClassificationWindow) -> Option<Result<Vec<u8>, String>> {
        let decimator = self.clip_decimator.as_ref()?;
        let clip_length = seconds_to_samples(CLIP_WINDOW_SECONDS, self.sample_rate_hz);
        let centre = window.first_device_sample_index + window.samples.len() as u64 / 2;
        let clip_first_index = centre.saturating_sub(clip_length as u64 / 2);
        let pcm16: Vec<i16> = self.rolling_buffer.iter().map(|&sample| float_to_pcm16(sample)).collect();
        Some(
            decimator
                .decimate_window_from_rolling_buffer(&pcm16, self.rolling_buffer_first_index, clip_first_index, clip_length)
                .map_err(|error| error.to_string())
                .and_then(|decimated| encode_field_clip(&decimated).map_err(|error| error.to_string())),
        )
    }

    fn append_to_rolling_buffer(&mut self, first_device_sample_index: u64, samples: &[f32]) {
        let expected_index = self.next_expected_device_sample_index();
        if self.rolling_buffer.is_empty() || first_device_sample_index != expected_index {
            if !self.rolling_buffer.is_empty() {
                self.counters.capture_gaps += 1;
                self.counters.last_gap_expected_index = expected_index;
                self.counters.last_gap_received_index = first_device_sample_index;
                // A window spanning the gap would splice unrelated audio.
                self.open_trigger = None;
            }
            self.rolling_buffer.clear();
            self.rolling_buffer_first_index = first_device_sample_index;
        }
        self.rolling_buffer.extend(samples.iter().copied());
        let overflow = self.rolling_buffer.len().saturating_sub(self.rolling_buffer_capacity);
        if overflow > 0 {
            self.rolling_buffer.drain(..overflow);
            self.rolling_buffer_first_index += overflow as u64;
        }
    }
}

fn copy_range(buffer: &VecDeque<f32>, buffer_first_index: u64, first_index: u64, length: usize) -> Vec<f32> {
    let start = (first_index - buffer_first_index) as usize;
    buffer.range(start..start + length).copied().collect()
}

fn seconds_to_samples(seconds: f64, sample_rate_hz: u32) -> usize {
    (seconds * f64::from(sample_rate_hz)).round() as usize
}

fn float_to_pcm16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * FLOAT_TO_PCM16_SCALE).round() as i16
}
