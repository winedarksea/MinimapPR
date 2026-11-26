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
//!
//! The cooldown is geometry-aware (`localization_event_span.rs`): a trigger on
//! another node is the same event while it falls within the cooldown plus the
//! sound's travel time between the two nodes, either order. So a far node's
//! onset of a km-spread event never supersedes the parked solve that is
//! waiting for its audio, while a second event on the same node still only
//! waits out the plain cooldown.

use std::collections::{BTreeMap, HashMap};

use super::*;
use minimappr_core::stage1::noise_floor_relative_trigger::{
    peak_short_block_rms, NoiseFloorRelativeTrigger, RelativeTriggerConfig,
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
    /// `Some(block seconds)` when the short-block excess candidate is on
    /// (only consulted with the relative trigger on).
    short_block_seconds: Option<f64>,
    node_noise_floors: NoiseFloorRelativeTrigger,
    last_dispatched_trigger_by_stream: HashMap<String, DispatchedTrigger>,
    /// Per node: the firmware boot session and receipt time of its last frame,
    /// so a reboot or a long silence resets the floor instead of leaving one
    /// learnt under another gain state.
    node_floor_sessions: HashMap<String, NodeFloorSession>,
}

/// The last trigger that dispatched (or parked) a solve on a buffer.
#[derive(Clone, Debug, PartialEq)]
struct DispatchedTrigger {
    audio_ns: u128,
    node_id: String,
}

#[derive(Clone, Copy)]
struct NodeFloorSession {
    boot_session: Option<u32>,
    last_heard_ns: u128,
}

/// What the reporter's newest frame says about an event.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct NodeFrameTrigger {
    /// The frame stood above the node's floor and the relative trigger is on.
    pub(crate) relatively_loud: bool,
    /// Seconds from the frame's first sample to the event onset (the first
    /// short block that reaches the trigger level); set when `relatively_loud`.
    pub(crate) onset_offset_seconds: Option<f64>,
    pub(crate) frame_seconds: f64,
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
            short_block_seconds: config
                .localization_solve
                .localization_trigger_short_block_excess_enabled
                .then_some(config.localization_solve.localization_trigger_short_block_seconds),
            node_noise_floors: NoiseFloorRelativeTrigger::new(relative_trigger_config),
            last_dispatched_trigger_by_stream: HashMap::new(),
            node_floor_sessions: HashMap::new(),
        }
    }

    /// Resets the node's floor when its firmware rebooted (new boot session)
    /// or it was silent past `stale_after_ns`: the old floor describes another
    /// gain state, and a stale floor would mute or flood the relative trigger
    /// until it re-converged. Call before `observe_node_frame`.
    pub(crate) fn observe_node_session(
        &mut self,
        node_id: &str,
        boot_session: Option<u32>,
        now_ns: u128,
        stale_after_ns: u128,
    ) {
        let session = NodeFloorSession {
            boot_session,
            last_heard_ns: now_ns,
        };
        if let Some(previous) = self
            .node_floor_sessions
            .insert(node_id.to_string(), session)
        {
            let rebooted = previous.boot_session.is_some()
                && boot_session.is_some()
                && previous.boot_session != boot_session;
            let stale = now_ns.saturating_sub(previous.last_heard_ns) >= stale_after_ns;
            if rebooted || stale {
                self.node_noise_floors.forget_node(node_id);
            }
        }
    }

    /// Forgets the floors (and sessions) of nodes silent past `stale_after_ns`,
    /// alongside the stream purge, so a node that left takes no floor with it
    /// when it returns.
    pub(crate) fn evict_stale_node_floors(&mut self, now_ns: u128, stale_after_ns: u128) {
        let node_floors = &mut self.node_noise_floors;
        self.node_floor_sessions.retain(|node_id, session| {
            let keep = now_ns.saturating_sub(session.last_heard_ns) < stale_after_ns;
            if !keep {
                node_floors.forget_node(node_id);
            }
            keep
        });
    }

    /// Feeds the node's newly decoded (preprocessed) audio to its floor, once
    /// per frame. The level is the strongest channel's RMS, as in Python's
    /// `trigger_rms` (averaging channels can cancel narrowband birdsong).
    /// A frame that stands above the node's floor (relative trigger enabled)
    /// also reports where in it the event began.
    pub(crate) fn observe_node_frame(
        &mut self,
        node_id: &str,
        channels: &[Vec<f32>],
        sample_rate_hz: u32,
    ) -> NodeFrameTrigger {
        let frame_samples = channels.iter().map(Vec::len).max().unwrap_or(0);
        if frame_samples == 0 || sample_rate_hz == 0 {
            return NodeFrameTrigger::default();
        }
        let frame_seconds = frame_samples as f64 / f64::from(sample_rate_hz);
        let frame_rms = channels
            .iter()
            .map(|channel| f64::from(window_rms(channel)))
            .fold(0.0, f64::max);
        // The bar this frame is judged against, before the frame moves the floor.
        let trigger_level_rms = self.node_noise_floors.trigger_level_rms(node_id);
        // Off (or relative off): `None`, so the floors evolve exactly as before.
        let frame_peak_short_block_rms = self
            .short_block_seconds
            .filter(|_| self.noise_floor_relative_enabled)
            .map(|block_seconds| {
                peak_short_block_rms(channels, f64::from(sample_rate_hz), block_seconds)
            });
        let relatively_loud = self.node_noise_floors.observe_with_short_blocks(
            node_id,
            frame_rms,
            frame_seconds,
            frame_peak_short_block_rms,
        ) && self.noise_floor_relative_enabled;
        let onset_offset_seconds = trigger_level_rms
            .filter(|_| relatively_loud)
            .and_then(|level_rms| event_onset_offset_seconds(channels, sample_rate_hz, level_rms));
        NodeFrameTrigger {
            relatively_loud,
            onset_offset_seconds,
            frame_seconds,
        }
    }

    /// Which gate opens a solve on this buffer now, without starting the
    /// cooldown. Absolute wins a tie so a frame that triggered before this
    /// change keeps its `attempt_kind`. `same_event_window_seconds(last_node)`
    /// is how far from the last trigger (either order) a trigger on
    /// `trigger_node_id` is still that event.
    pub(crate) fn evaluate(
        &self,
        stream_key: &str,
        audio_ns: u128,
        windows: &[Vec<f32>],
        node_frame_relatively_loud: bool,
        same_event_window_seconds: impl Fn(&str) -> f64,
    ) -> Option<LocalizationTriggerSource> {
        let source = if self.absolute_gate_passes(windows) {
            LocalizationTriggerSource::AbsoluteRms
        } else if node_frame_relatively_loud {
            LocalizationTriggerSource::NoiseFloorRelative
        } else {
            return None;
        };
        let same_event_as_last = self
            .last_dispatched_trigger_by_stream
            .get(stream_key)
            .is_some_and(|last| {
                let window_ns = (same_event_window_seconds(&last.node_id)
                    * NANOSECONDS_PER_SECOND_F64)
                    .round() as u128;
                audio_ns.abs_diff(last.audio_ns) < window_ns
            });
        (!same_event_as_last).then_some(source)
    }

    /// Starts the cooldown; call only once the triggered solve is dispatched
    /// (or parked for the coverage wait, which dispatches it on release).
    pub(crate) fn commit_dispatched_trigger(
        &mut self,
        stream_key: &str,
        audio_ns: u128,
        trigger_node_id: &str,
    ) {
        self.last_dispatched_trigger_by_stream.insert(
            stream_key.to_string(),
            DispatchedTrigger {
                audio_ns,
                node_id: trigger_node_id.to_string(),
            },
        );
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
        self.last_dispatched_trigger_by_stream.remove(stream_key);
    }

    #[cfg(test)]
    pub(crate) fn node_floor_rms(&self, node_id: &str) -> Option<f64> {
        self.node_noise_floors.floor_rms(node_id)
    }

    #[cfg(test)]
    pub(crate) fn last_dispatched_trigger_ns(&self, stream_key: &str) -> Option<u128> {
        self.last_dispatched_trigger_by_stream
            .get(stream_key)
            .map(|last| last.audio_ns)
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
    /// The buffer's layout (its reporting node is the trigger node) and the
    /// sound speed, for the geometry-aware same-event window.
    pub(crate) layout: &'frame SolveSensorLayout,
    pub(crate) sound_speed_mps: f32,
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
            |last_trigger_node_id| {
                same_event_window_seconds(
                    evidence.layout,
                    &self.config,
                    evidence.sound_speed_mps,
                    last_trigger_node_id,
                    &evidence.layout.reporting_node_id,
                    |node_id| {
                        self.node_clock_quality
                            .assessment(node_id)
                            .timing_uncertainty_seconds
                    },
                )
            },
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
    pub(crate) fn commit_localization_trigger(
        &mut self,
        stream_key: &str,
        audio_ns: u128,
        trigger_node_id: &str,
    ) {
        self.localization_trigger
            .commit_dispatched_trigger(stream_key, audio_ns, trigger_node_id);
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
