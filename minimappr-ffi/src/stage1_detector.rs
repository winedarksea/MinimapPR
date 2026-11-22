//! Stage-1 trigger exposed as a UniFFI object. The capture thread pushes
//! every PCM16 block with its device-clock sample index; events come back
//! synchronously (the detector is cheap enough for the capture path).

use std::sync::Mutex;

use minimappr_core::stage1::{Stage1Configuration, Stage1DetectionMode, Stage1Detector, Stage1Event};

use crate::error::FieldFfiError;
use crate::pcm16_bytes::pcm16_samples_from_little_endian_bytes;

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FieldStage1Mode {
    Birds,
    Drone,
}

impl From<FieldStage1Mode> for Stage1DetectionMode {
    fn from(mode: FieldStage1Mode) -> Self {
        match mode {
            FieldStage1Mode::Birds => Stage1DetectionMode::Birds,
            FieldStage1Mode::Drone => Stage1DetectionMode::Drone,
        }
    }
}

#[derive(Debug, Clone, PartialEq, uniffi::Enum)]
pub enum FieldStage1Event {
    TriggerOpened {
        frame_index: u64,
        device_sample_index: u64,
        contrast_db: f64,
        active_band_low_hz: f64,
        active_band_high_hz: f64,
    },
    TriggerClosed {
        frame_index: u64,
        device_sample_index: u64,
    },
    Onset {
        frame_index: u64,
        device_sample_index: u64,
        positive_flux_db: f64,
    },
}

impl From<Stage1Event> for FieldStage1Event {
    fn from(event: Stage1Event) -> Self {
        match event {
            Stage1Event::TriggerOpened {
                frame_index,
                device_sample_index,
                contrast_db,
                active_band_low_hz,
                active_band_high_hz,
            } => Self::TriggerOpened {
                frame_index,
                device_sample_index,
                contrast_db,
                active_band_low_hz,
                active_band_high_hz,
            },
            Stage1Event::TriggerClosed { frame_index, device_sample_index } => {
                Self::TriggerClosed { frame_index, device_sample_index }
            }
            Stage1Event::Onset(onset) => Self::Onset {
                frame_index: onset.frame_index,
                device_sample_index: onset.device_sample_index,
                positive_flux_db: onset.positive_flux_db,
            },
        }
    }
}

/// Miss-rate audit counters, mirrored 1:1 from the core.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Record)]
pub struct FieldStage1Counters {
    pub frames_evaluated: u64,
    pub frames_while_triggered: u64,
    pub frames_rejected: u64,
    pub triggers_opened: u64,
    pub onsets_detected: u64,
    pub onsets_dropped_from_full_list: u64,
    pub stream_discontinuities: u64,
}

#[derive(uniffi::Object)]
pub struct FieldStage1Detector {
    detector: Mutex<Stage1Detector>,
}

#[uniffi::export]
impl FieldStage1Detector {
    /// Detector with the core's default tuning for `mode` (48 kHz input).
    #[uniffi::constructor]
    pub fn new(mode: FieldStage1Mode) -> Self {
        Self {
            detector: Mutex::new(Stage1Detector::new(Stage1Configuration::for_mode(mode.into()))),
        }
    }

    /// Feeds one contiguous PCM16 block; a gap or overlap in
    /// `first_device_sample_index` is counted as a stream discontinuity.
    pub fn process_pcm16_block(
        &self,
        first_device_sample_index: u64,
        pcm16_little_endian_bytes: Vec<u8>,
    ) -> Result<Vec<FieldStage1Event>, FieldFfiError> {
        let samples = pcm16_samples_from_little_endian_bytes(&pcm16_little_endian_bytes)?;
        let mut detector = self.detector.lock().expect("stage-1 detector mutex poisoned");
        Ok(detector
            .process_pcm16_block(first_device_sample_index, &samples)
            .into_iter()
            .map(FieldStage1Event::from)
            .collect())
    }

    pub fn trigger_is_open(&self) -> bool {
        self.detector.lock().expect("stage-1 detector mutex poisoned").trigger_is_open()
    }

    pub fn counters(&self) -> FieldStage1Counters {
        let counters = self.detector.lock().expect("stage-1 detector mutex poisoned").counters();
        FieldStage1Counters {
            frames_evaluated: counters.frames_evaluated,
            frames_while_triggered: counters.frames_while_triggered,
            frames_rejected: counters.frames_rejected,
            triggers_opened: counters.triggers_opened,
            onsets_detected: counters.onsets_detected,
            onsets_dropped_from_full_list: counters.onsets_dropped_from_full_list,
            stream_discontinuities: counters.stream_discontinuities,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE_SECOND_AT_48_KHZ: usize = 48_000;
    const PCM16_BYTES_PER_SAMPLE: usize = 2;

    fn one_second_of_silence_bytes() -> Vec<u8> {
        vec![0; ONE_SECOND_AT_48_KHZ * PCM16_BYTES_PER_SAMPLE]
    }

    #[test]
    fn silence_evaluates_frames_without_opening_the_trigger() {
        let detector = FieldStage1Detector::new(FieldStage1Mode::Birds);
        detector.process_pcm16_block(0, one_second_of_silence_bytes()).unwrap();
        let counters = detector.counters();
        assert!(counters.frames_evaluated > 0);
        assert_eq!(counters.triggers_opened, 0);
        assert!(!detector.trigger_is_open());
    }

    #[test]
    fn sample_index_gap_counts_a_discontinuity() {
        let detector = FieldStage1Detector::new(FieldStage1Mode::Birds);
        detector.process_pcm16_block(0, one_second_of_silence_bytes()).unwrap();
        detector
            .process_pcm16_block(2 * ONE_SECOND_AT_48_KHZ as u64, one_second_of_silence_bytes())
            .unwrap();
        assert_eq!(detector.counters().stream_discontinuities, 1);
    }
}
