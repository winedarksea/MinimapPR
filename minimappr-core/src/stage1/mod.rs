//! Stage-1 activity trigger: cheap, always-on, relative-to-noise-floor.
//!
//! Wakes the on-device classifier, keeps a 32-48-band 20 ms spectrogram
//! history, marks note onsets (positive spectral flux peaks) that travel with
//! clips, and provides noise-fill blanking at the running floor.

pub mod band_analyzer;
pub mod band_history;
pub mod configuration;
pub mod detector;
pub mod noise_fill;
pub mod noise_floor;
// W4-port-server-track (S5): per-node noise-floor-relative trigger.
pub mod noise_floor_relative_trigger;

pub use configuration::{Stage1Configuration, Stage1DetectionMode};
pub use detector::{
    NoteOnset, Stage1Counters, Stage1Detector, Stage1Event, Stage1FrameDiagnostics,
};
pub use noise_fill::{
    blank_span_with_noise_fill, DeterministicNoiseSource, NoiseFillConfiguration,
    DEFAULT_NOISE_FILL_CONFIGURATION,
};
