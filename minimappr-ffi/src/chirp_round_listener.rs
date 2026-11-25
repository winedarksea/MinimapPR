//! Turns a `ListenForChirpRound` into a `LocalChirpRound`: each slot's
//! session-clock start → local monotonic time (session alignment) → capture
//! sample (capture anchor) gives a search window in the rolling buffer.
//!
//! Locking: the device copies the windows under its lock (`plan_searches`),
//! runs the detector outside it (`run_searches`, ~ms per slot), then
//! re-locks to report. Capture pushes never wait on detection.
//!
//! Contract with the host (`finish_round` mutes "inaudible" nodes): a slot
//! this phone did not play is reported as not emitted (no own emission
//! index), never as `own_chirp_inaudible`. Only a chirp that was played and
//! then not heard by its own mic marks the phone inaudible (volume 0).

use std::collections::BTreeMap;

use minimappr_core::mesh::node_state_machine::{LocalChirpRound, ScheduledChirpSlot};
use minimappr_core::protocol::session_messages::ChirpArrival;
use minimappr_core::protocol::NodeId;

use crate::capture_clock_anchor::CaptureClockAnchor;
use crate::chirp_detector::{FieldChirpDetection, FieldChirpDetector};
use crate::node_rolling_audio_buffer::NodeRollingAudioBuffer;
use minimappr_core::mesh::session_clock_alignment::SessionClockAlignment;

const NANOSECONDS_PER_MILLISECOND: i64 = 1_000_000;
const SAMPLES_PER_MILLISECOND: f64 = 48.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmissionOutcome {
    pub played: bool,
    /// Estimated local time the chirp's first frame left the speaker.
    pub first_frame_local_ns: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PendingOwnEmission {
    pub emission_id: u64,
    pub slot_index: usize,
    pub outcome: Option<EmissionOutcome>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PendingChirpRound {
    pub round_id: u32,
    pub slots: Vec<ScheduledChirpSlot>,
    pub listen_until_session_ms: u64,
    pub own_emissions: Vec<PendingOwnEmission>,
}

/// Search geometry shared by every slot of a session.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ListenerGeometry {
    pub slot_duration_ms: u64,
    pub coarse_schedule_uncertainty_ms: u64,
    pub chirp_sample_count: usize,
    pub self_path_length_m: f64,
    /// Rounds still waiting this long after their listen window (capture
    /// lagging, outcome never reported) are reported with what is known.
    pub abandon_after_listen_ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlotSearchJob {
    pub slot_index: usize,
    pub emitter_node_id: NodeId,
    pub is_self: bool,
    pub window_first_device_sample_index: u64,
    pub samples: Vec<f32>,
    pub search_lag_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlotSearchResult {
    pub slot_index: usize,
    pub emitter_node_id: NodeId,
    pub is_self: bool,
    /// Detection with `sample_index` already in device samples.
    pub detection: Option<FieldChirpDetection>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlannedRound {
    pub round_id: u32,
    pub jobs: Vec<SlotSearchJob>,
    pub own_emission_device_sample_index: Option<u64>,
    pub any_own_emission_played: bool,
    /// Slots whose window was not (fully) buffered.
    pub unsearchable_slot_count: usize,
}

pub enum RoundReadiness {
    Waiting,
    Ready(PlannedRound),
}

pub struct ListenerClocks<'a> {
    pub alignment: &'a SessionClockAlignment,
    pub capture_anchor: &'a CaptureClockAnchor,
    pub buffer: &'a NodeRollingAudioBuffer,
    pub local_now_ns: i64,
}

impl PendingChirpRound {
    /// Search windows for every slot, or `Waiting` while the window's audio
    /// or an emission outcome is still due.
    pub fn plan_searches(&self, own_node: NodeId, geometry: &ListenerGeometry, clocks: &ListenerClocks<'_>) -> RoundReadiness {
        let Some(listen_until_local_ns) = clocks.alignment.local_ns_for_session(self.listen_until_session_ms as i64 * NANOSECONDS_PER_MILLISECOND) else {
            return RoundReadiness::Waiting;
        };
        if clocks.local_now_ns < listen_until_local_ns {
            return RoundReadiness::Waiting;
        }
        let abandon = clocks.local_now_ns >= listen_until_local_ns + geometry.abandon_after_listen_ms as i64 * NANOSECONDS_PER_MILLISECOND;
        let outcomes_known = self.own_emissions.iter().all(|emission| emission.outcome.is_some());
        let uncertainty_samples = geometry.coarse_schedule_uncertainty_ms as f64 * SAMPLES_PER_MILLISECOND;
        let chirp_ms = geometry.chirp_sample_count as f64 / SAMPLES_PER_MILLISECOND;
        let maximum_flight_samples =
            ((geometry.slot_duration_ms as f64 - chirp_ms - 2.0 * geometry.coarse_schedule_uncertainty_ms as f64) * SAMPLES_PER_MILLISECOND).max(0.0);
        let mut jobs = Vec::new();
        let mut unsearchable_slot_count = 0;
        let mut window_audio_pending = false;
        for (slot_index, slot) in self.slots.iter().enumerate() {
            let is_self = slot.emitter_node_id == own_node;
            let expected_local_ns = if is_self {
                let emission = self.own_emissions.iter().find(|emission| emission.slot_index == slot_index);
                match emission.and_then(|emission| emission.outcome) {
                    Some(outcome) if outcome.played => Some(outcome.first_frame_local_ns),
                    // Not played (or never scheduled): nothing to search for.
                    _ => None,
                }
            } else {
                clocks.alignment.local_ns_for_session(slot.slot_start_ms as i64 * NANOSECONDS_PER_MILLISECOND)
            };
            let Some(expected_local_ns) = expected_local_ns else { continue };
            let Some(expected_sample) = clocks.capture_anchor.sample_index_at(expected_local_ns) else {
                unsearchable_slot_count += 1;
                continue;
            };
            let window_first = (expected_sample - uncertainty_samples).max(0.0) as u64;
            // A self chirp travels d_ii only; keep its window tight so the
            // previous or next slot's chirp cannot fall inside it.
            let flight_samples = if is_self { 2.0 * SAMPLES_PER_MILLISECOND } else { maximum_flight_samples };
            let search_lag_count = (2.0 * uncertainty_samples + flight_samples) as usize;
            let window_sample_count = search_lag_count + geometry.chirp_sample_count;
            match clocks.buffer.copy_window_as_f32(window_first, window_sample_count) {
                Some(samples) => jobs.push(SlotSearchJob {
                    slot_index,
                    emitter_node_id: slot.emitter_node_id,
                    is_self,
                    window_first_device_sample_index: window_first,
                    samples,
                    search_lag_count,
                }),
                None => {
                    if window_first + window_sample_count as u64 > clocks.buffer.next_device_sample_index() {
                        window_audio_pending = true;
                    }
                    unsearchable_slot_count += 1;
                }
            }
        }
        if !abandon && (window_audio_pending || !outcomes_known) {
            return RoundReadiness::Waiting;
        }
        let played: Vec<EmissionOutcome> = self
            .own_emissions
            .iter()
            .filter_map(|emission| emission.outcome)
            .filter(|outcome| outcome.played)
            .collect();
        let own_emission_device_sample_index = played
            .first()
            .and_then(|outcome| clocks.capture_anchor.sample_index_at(outcome.first_frame_local_ns))
            .map(|sample| sample.max(0.0).round() as u64);
        RoundReadiness::Ready(PlannedRound {
            round_id: self.round_id,
            jobs,
            own_emission_device_sample_index,
            any_own_emission_played: !played.is_empty(),
            unsearchable_slot_count,
        })
    }
}

/// Runs outside the device lock.
pub fn run_searches(jobs: &[SlotSearchJob], detector: &FieldChirpDetector, self_path_length_m: f64) -> Vec<SlotSearchResult> {
    jobs.iter()
        .map(|job| {
            let detection = detector
                .detect_in_f32_slice(&job.samples, 0, job.search_lag_count, Some(self_path_length_m), job.is_self)
                .map(|detection| FieldChirpDetection {
                    sample_index: job.window_first_device_sample_index as f64 + detection.sample_index,
                    ..detection
                });
            SlotSearchResult { slot_index: job.slot_index, emitter_node_id: job.emitter_node_id, is_self: job.is_self, detection }
        })
        .collect()
}

pub fn chirp_arrival_from_detection(emitter_node_id: NodeId, detection: &FieldChirpDetection) -> ChirpArrival {
    let integer = detection.sample_index.round();
    let mut fraction = (detection.sample_index - integer) as f32;
    // round() maps x.5 away from zero; keep the wire's [-0.5, 0.5) contract.
    if fraction >= 0.5 {
        fraction -= 1.0;
    }
    ChirpArrival {
        emitter_node_id,
        arrival_device_sample_index: integer.max(0.0) as u64,
        arrival_fractional_sample: fraction,
        peak_to_noise_db: detection.snr_db as f32,
    }
}

pub fn local_round_from_results(planned: &PlannedRound, results: &[SlotSearchResult]) -> LocalChirpRound {
    let heard_self = results.iter().any(|result| result.is_self && result.detection.is_some());
    LocalChirpRound {
        round_id: planned.round_id,
        own_emission_device_sample_index: planned.own_emission_device_sample_index,
        arrivals: results
            .iter()
            .filter_map(|result| result.detection.as_ref().map(|d| chirp_arrival_from_detection(result.emitter_node_id, d)))
            .collect(),
        own_chirp_inaudible: planned.any_own_emission_played && !heard_self,
    }
}

/// Rounds the node asked to listen to, keyed by round id.
#[derive(Debug, Default)]
pub struct ChirpRoundListener {
    pub pending: BTreeMap<u32, PendingChirpRound>,
}

impl ChirpRoundListener {
    pub fn record_emission_outcome(&mut self, emission_id: u64, outcome: EmissionOutcome) -> bool {
        for round in self.pending.values_mut() {
            if let Some(emission) = round.own_emissions.iter_mut().find(|emission| emission.emission_id == emission_id) {
                emission.outcome = Some(outcome);
                return true;
            }
        }
        false
    }

    pub fn take_ready_rounds(&mut self, own_node: NodeId, geometry: &ListenerGeometry, clocks: &ListenerClocks<'_>) -> Vec<PlannedRound> {
        let mut ready = Vec::new();
        let round_ids: Vec<u32> = self.pending.keys().copied().collect();
        for round_id in round_ids {
            if let RoundReadiness::Ready(planned) = self.pending[&round_id].plan_searches(own_node, geometry, clocks) {
                self.pending.remove(&round_id);
                ready.push(planned);
            }
        }
        ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrival_fraction_stays_in_wire_range() {
        for sample_index in [10.5, 10.49, 9.51, 0.0, 3.25] {
            let detection = FieldChirpDetection { sample_index, snr_db: 30.0, doppler_skew_samples: 0.0, peak_fraction: 0.1 };
            let arrival = chirp_arrival_from_detection(NodeId(1), &detection);
            assert!((-0.5..0.5).contains(&arrival.arrival_fractional_sample), "{sample_index}");
            let rebuilt = arrival.arrival_device_sample_index as f64 + arrival.arrival_fractional_sample as f64;
            assert!((rebuilt - sample_index).abs() < 1e-5);
        }
    }

    #[test]
    fn unplayed_emission_is_reported_as_not_emitted_not_inaudible() {
        let planned = PlannedRound {
            round_id: 1,
            jobs: Vec::new(),
            own_emission_device_sample_index: None,
            any_own_emission_played: false,
            unsearchable_slot_count: 0,
        };
        let round = local_round_from_results(&planned, &[]);
        assert!(!round.own_chirp_inaudible);
        assert_eq!(round.own_emission_device_sample_index, None);
    }

    #[test]
    fn played_but_unheard_emission_is_inaudible() {
        let planned = PlannedRound {
            round_id: 1,
            jobs: Vec::new(),
            own_emission_device_sample_index: Some(100),
            any_own_emission_played: true,
            unsearchable_slot_count: 0,
        };
        let results = vec![SlotSearchResult { slot_index: 0, emitter_node_id: NodeId(2), is_self: true, detection: None }];
        assert!(local_round_from_results(&planned, &results).own_chirp_inaudible);
    }
}
