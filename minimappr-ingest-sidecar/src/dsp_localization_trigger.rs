//! The live localization event trigger: today's absolute RMS gate OR the
//! node's noise-floor-relative trigger (`NoiseFloorRelativeTrigger`, the
//! Python ingest trigger's port), then the per-buffer cooldown.
//!
//! Why relative: a faint far call 10 dB over a quiet site's floor sits far
//! below any fixed gate that also keeps a noisy site calm, so long-range
//! events never opened a solve. The node floors are tracked even when the
//! relative trigger is off, because the solver's reference floor is stated
//! relative to them (`dsp_solver_reference_floor.rs`).
//!
//! Deciding and committing are separate: the cooldown starts only when a
//! solve is actually dispatched, so a trigger a later gate suppresses does
//! not silence the next 0.8 s of audio.

use std::collections::{BTreeMap, HashMap};

use super::*;
use minimappr_core::stage1::noise_floor_relative_trigger::{
    NoiseFloorRelativeTrigger, RelativeTriggerConfig,
};

const NANOSECONDS_PER_SECOND_F64: f64 = 1_000_000_000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LocalizationTriggerSource {
    AbsoluteRms,
    NoiseFloorRelative,
}

impl LocalizationTriggerSource {
    pub(crate) fn attempt_kind(self) -> SolveAttemptKind {
        match self {
            Self::AbsoluteRms => SolveAttemptKind::EventTriggerAbsoluteRms,
            Self::NoiseFloorRelative => SolveAttemptKind::EventTriggerNoiseFloorRelative,
        }
    }
}

pub(crate) struct LocalizationEventTrigger {
    /// `<= 0` disables the energy check, so every frame passes the absolute gate.
    absolute_rms_gate: f32,
    noise_floor_relative_enabled: bool,
    cooldown_ns: u128,
    node_noise_floors: NoiseFloorRelativeTrigger,
    last_dispatched_trigger_ns_by_stream: HashMap<String, u128>,
}

impl LocalizationEventTrigger {
    pub(crate) fn new(config: &DspWorkerConfig) -> Self {
        let relative_trigger_config = RelativeTriggerConfig {
            snr_db: f64::from(
                config
                    .localization_solve
                    .localization_trigger_noise_floor_relative_db,
            ),
            ..RelativeTriggerConfig::default()
        };
        Self {
            absolute_rms_gate: config.localization_rms_gate,
            noise_floor_relative_enabled: config
                .localization_solve
                .localization_trigger_noise_floor_relative_enabled,
            cooldown_ns: (config.trigger_cooldown_seconds * NANOSECONDS_PER_SECOND_F64).round()
                as u128,
            node_noise_floors: NoiseFloorRelativeTrigger::new(relative_trigger_config),
            last_dispatched_trigger_ns_by_stream: HashMap::new(),
        }
    }

    /// Feeds the node's newly decoded (preprocessed) audio to its floor, once
    /// per frame. The level is the strongest channel's RMS, as in Python's
    /// `trigger_rms` (averaging channels can cancel narrowband birdsong).
    /// True when the frame stands above the node's floor and the relative
    /// trigger is enabled.
    pub(crate) fn observe_node_frame(
        &mut self,
        node_id: &str,
        channels: &[Vec<f32>],
        sample_rate_hz: u32,
    ) -> bool {
        let frame_samples = channels.iter().map(Vec::len).max().unwrap_or(0);
        if frame_samples == 0 || sample_rate_hz == 0 {
            return false;
        }
        let frame_seconds = frame_samples as f64 / f64::from(sample_rate_hz);
        let frame_rms = channels
            .iter()
            .map(|channel| f64::from(window_rms(channel)))
            .fold(0.0, f64::max);
        let relatively_loud = self
            .node_noise_floors
            .observe(node_id, frame_rms, frame_seconds);
        relatively_loud && self.noise_floor_relative_enabled
    }

    /// Which gate opens a solve on this buffer now, without starting the
    /// cooldown. Absolute wins a tie so a frame that triggered before this
    /// change keeps its `attempt_kind`.
    pub(crate) fn evaluate(
        &self,
        stream_key: &str,
        audio_ns: u128,
        windows: &[Vec<f32>],
        node_frame_relatively_loud: bool,
    ) -> Option<LocalizationTriggerSource> {
        let source = if self.absolute_gate_passes(windows) {
            LocalizationTriggerSource::AbsoluteRms
        } else if node_frame_relatively_loud {
            LocalizationTriggerSource::NoiseFloorRelative
        } else {
            return None;
        };
        let cooling_down = self
            .last_dispatched_trigger_ns_by_stream
            .get(stream_key)
            .is_some_and(|last_ns| audio_ns.saturating_sub(*last_ns) < self.cooldown_ns);
        (!cooling_down).then_some(source)
    }

    /// Starts the cooldown; call only once the triggered solve is dispatched
    /// (or parked for the coverage wait, which dispatches it on release).
    pub(crate) fn commit_dispatched_trigger(&mut self, stream_key: &str, audio_ns: u128) {
        self.last_dispatched_trigger_ns_by_stream
            .insert(stream_key.to_string(), audio_ns);
    }

    /// Tracked floors of the layout's nodes, snapshotted at trigger so a
    /// parked solve is judged against the floor it triggered over.
    pub(crate) fn node_noise_floor_snapshot(
        &self,
        layout: &SolveSensorLayout,
    ) -> Arc<BTreeMap<String, f32>> {
        let floors = layout
            .sensors
            .iter()
            .filter_map(|sensor| {
                let floor_rms = self.node_noise_floors.floor_rms(&sensor.node_id)?;
                Some((sensor.node_id.clone(), floor_rms as f32))
            })
            .collect();
        Arc::new(floors)
    }

    pub(crate) fn forget_stream(&mut self, stream_key: &str) {
        self.last_dispatched_trigger_ns_by_stream.remove(stream_key);
    }

    #[cfg(test)]
    pub(crate) fn last_dispatched_trigger_ns(&self, stream_key: &str) -> Option<u128> {
        self.last_dispatched_trigger_ns_by_stream
            .get(stream_key)
            .copied()
    }

    /// Today's gate: the strongest channel of the node-local localization windows.
    fn absolute_gate_passes(&self, windows: &[Vec<f32>]) -> bool {
        if self.absolute_rms_gate <= 0.0 {
            return true;
        }
        windows
            .iter()
            .map(|window| window_rms(window))
            .fold(0.0_f32, f32::max)
            >= self.absolute_rms_gate
    }
}

/// What the trigger decides on for one buffered frame.
#[derive(Clone, Copy)]
pub(crate) struct LocalizationTriggerEvidence<'frame> {
    pub(crate) windows: &'frame [Vec<f32>],
    /// The reporting node's new frame stood above its noise floor.
    pub(crate) node_frame_relatively_loud: bool,
}

impl DspWorker {
    /// The trigger gate that opens a solve now, if the localization cadence
    /// is also open. Commits nothing: see `commit_localization_trigger`.
    pub(crate) fn localization_trigger_opening(
        &self,
        stream_key: &str,
        audio_ns: u128,
        evidence: LocalizationTriggerEvidence<'_>,
    ) -> Option<LocalizationTriggerSource> {
        let source = self.localization_trigger.evaluate(
            stream_key,
            audio_ns,
            evidence.windows,
            evidence.node_frame_relatively_loud,
        )?;
        let cadence_ns = (self.config.localization_cadence_ms as u128) * 1_000_000;
        let cadence_closed = cadence_ns > 0
            && self
                .last_localization_ns_by_stream
                .get(stream_key)
                .is_some_and(|last_ns| audio_ns.saturating_sub(*last_ns) < cadence_ns);
        (!cadence_closed).then_some(source)
    }

    /// Starts the trigger cooldown and the localization cadence once the
    /// triggered solve is dispatched (or parked, which dispatches on release).
    pub(crate) fn commit_localization_trigger(&mut self, stream_key: &str, audio_ns: u128) {
        self.localization_trigger
            .commit_dispatched_trigger(stream_key, audio_ns);
        self.last_localization_ns_by_stream
            .insert(stream_key.to_string(), audio_ns);
    }
}

/// RMS of one window; 0 for an empty one.
pub(crate) fn window_rms(window: &[f32]) -> f32 {
    if window.is_empty() {
        return 0.0;
    }
    (window.iter().map(|sample| sample * sample).sum::<f32>() / window.len() as f32).sqrt()
}

#[cfg(test)]
#[path = "dsp_localization_trigger_tests.rs"]
mod dsp_localization_trigger_tests;
