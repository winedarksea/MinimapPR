//! A classifier render reports the event-triggered solve it covers.
//!
//! Renders run on a fixed schedule over a long trailing window, while the
//! event that made the window worth classifying triggered a solve on its
//! onset frame. Left alone, a render solves only the short localization
//! window ending at its own frame, which is often after the event, so the
//! stored detection got a position (and time) from trailing noise. Instead a
//! render whose span contains the latest event-triggered solve on its buffer
//! solves that event's window:
//!
//! - when the buffers now cover exactly the sensors the event solve used, it
//!   re-runs that solve (same windows, layout snapshot, centre, sample rate,
//!   sound speed and `solve_id`), so the stored detection names the solve it
//!   came from. The re-run is deterministic and on the render's own payload,
//!   which avoids a cross-task wait on the trigger's result;
//! - when a member whose audio was late now covers the window too, it solves
//!   the fuller sensor set as a new solve of the same window (new `solve_id`)
//!   and that solve becomes the one later renders repeat. Repeating the
//!   poorer trigger-time solve would throw the late member away.
//!
//! A render reported at another sample rate than the event (mixed-rate
//! cluster) never takes the event over: its windows and the event's are
//! different sample grids, and the render beamforms at its own rate.
//!
//! An event solve parked by the coverage wait is only dispatched at a batch
//! boundary (or later), so a covering render dispatched in the meantime would
//! find no event and solve its own trailing window. Such a render is held
//! until the parked event is dispatched, then takes it over as above.
//!
//! A repeated solve has exactly the inputs of the one it repeats, so its SRP
//! evaluation is computed once and shared (`SharedEventSolveEvaluation`):
//! whichever payload reaches compute first solves, the others reuse its
//! result (waiting for it while it is still running). Re-solving instead
//! ran a full multi-node SRP per covering render, the dominant compute cost
//! of a multi-node cluster.

use std::sync::OnceLock;

use super::*;
use crate::srp_phat::SrpPhatEvaluation;

/// One solve's SRP evaluation, computed by the first payload of that solve to
/// reach compute and reused by the rest.
pub(crate) type SharedEventSolveEvaluation = Arc<OnceLock<SrpPhatEvaluation>>;

/// The inputs of the latest event-triggered solve on one buffer.
struct RecentEventTriggerSolve {
    solve_id: String,
    attempt_kind: SolveAttemptKind,
    window_center_time_ns: i128,
    window_seconds: f64,
    node_noise_floor_rms: Arc<std::collections::BTreeMap<String, f32>>,
    channel_states: Vec<LocalizationChannelState>,
    active_channels: Vec<usize>,
    sensor_layout: Arc<SolveSensorLayout>,
    pair_admission: SolvePairAdmission,
    node_clock_assessments: Arc<NodeClockAssessments>,
    /// The rate the event's windows are sampled at; lags are counted in it.
    sample_rate_hz: u32,
    sound_speed_mps: f32,
    evaluation: SharedEventSolveEvaluation,
}

/// One entry per buffer key, dropped with the buffer.
#[derive(Default)]
pub(crate) struct RecentEventTriggerSolves {
    /// Keyed by dispatch gate (`dsp_dispatch_gate_key`).
    by_buffer_key: HashMap<String, RecentEventTriggerSolve>,
    /// Renders covering a parked event solve, in dispatch order. Bounded by
    /// the coverage wait's total cap: the parked event is always released.
    held_renders_by_buffer_key: HashMap<String, Vec<ComputePayload>>,
}

fn is_event_trigger(attempt_kind: SolveAttemptKind) -> bool {
    matches!(
        attempt_kind,
        SolveAttemptKind::EventTriggerAbsoluteRms
            | SolveAttemptKind::EventTriggerNoiseFloorRelative
    )
}

fn same_sensor_order(a: &SolveSensorLayout, b: &SolveSensorLayout) -> bool {
    a.sensors.len() == b.sensors.len()
        && a.sensors
            .iter()
            .zip(&b.sensors)
            .all(|(left, right)| left.sensor_id == right.sensor_id)
}

/// The span of a render that could take over an event solve; omni renders
/// (override channels) and sub-SRP buffers never solve.
fn covering_render_span_ns(payload: &ComputePayload) -> Option<std::ops::RangeInclusive<i128>> {
    if !payload.run_classifier_render
        || payload.omni_channels_override.is_some()
        || payload.sensor_layout.sensor_count() < MIN_SRP_CHANNEL_COUNT
    {
        return None;
    }
    Some(payload.classifier_render_start_ns? as i128..=payload.classifier_render_end_ns? as i128)
}

fn render_covers_window(payload: &ComputePayload, window_center_time_ns: i128) -> bool {
    covering_render_span_ns(payload).is_some_and(|span| span.contains(&window_center_time_ns))
}

/// Per dispatch gate, so each rate group of a mixed-rate cluster keeps its own event.
fn remembered_solve_key(payload: &ComputePayload) -> String {
    payload.sensor_layout.dispatch_gate_key(&payload.stream_key)
}

/// Remembers `payload`'s solve for later renders to repeat, sharing its SRP
/// evaluation with them.
fn remembered_solve_of(payload: &mut ComputePayload) -> RecentEventTriggerSolve {
    let evaluation = SharedEventSolveEvaluation::default();
    payload.shared_event_solve = Some(evaluation.clone());
    RecentEventTriggerSolve {
        solve_id: payload.solve_id.clone(),
        attempt_kind: payload.attempt_kind,
        window_center_time_ns: payload.localization_window_center_time_ns,
        window_seconds: payload.localization_window_seconds,
        node_noise_floor_rms: payload.node_noise_floor_rms.clone(),
        channel_states: payload.channel_states.clone(),
        active_channels: payload.active_channels.clone(),
        sensor_layout: payload.sensor_layout.clone(),
        pair_admission: payload.pair_admission.clone(),
        node_clock_assessments: payload.node_clock_assessments.clone(),
        sample_rate_hz: payload.sr,
        sound_speed_mps: payload.effective_sound_speed_mps,
        evaluation,
    }
}

impl RecentEventTriggerSolves {
    /// Called for every payload in dispatch order with its buffer's current
    /// audio: an event-triggered solve is remembered; a scheduled render that
    /// covers the remembered solve takes over that event's window.
    pub(crate) fn remember_or_attach(
        &mut self,
        payload: &mut ComputePayload,
        buffers: Option<&[SensorStreamBuffer]>,
    ) {
        if payload.run_srp && is_event_trigger(payload.attempt_kind) {
            self.by_buffer_key
                .insert(remembered_solve_key(payload), remembered_solve_of(payload));
            return;
        }
        let Some(render_span_ns) = covering_render_span_ns(payload) else {
            return;
        };
        let Some(event) = self.by_buffer_key.get(&remembered_solve_key(payload)) else {
            return;
        };
        let Some(buffers) = buffers else {
            return;
        };
        let covered = render_span_ns.contains(&event.window_center_time_ns);
        // The render beamforms its windows with the solve's positions, so the
        // snapshot must index the same buffer channels as the render does, on
        // the same sample grid.
        if !covered
            || event.sample_rate_hz != payload.sr
            || !same_sensor_order(&event.sensor_layout, &payload.sensor_layout)
        {
            return;
        }
        let layout_buffers = &buffers[..payload.sensor_layout.sensor_count().min(buffers.len())];
        let mut channel_states = localization_channel_states_centered(
            layout_buffers,
            event.window_center_time_ns,
            event.window_seconds,
        );
        payload
            .sensor_layout
            .clear_solve_excluded_channels(&mut channel_states);
        let active_channels =
            eligible_localization_channels(&channel_states, payload.config.min_coverage_ratio);
        let late_member_now_covers = active_channels.len() > event.active_channels.len()
            && event
                .active_channels
                .iter()
                .all(|channel| active_channels.contains(channel));
        payload.attempt_kind = event.attempt_kind;
        payload.localization_window_center_time_ns = event.window_center_time_ns;
        payload.localization_window_seconds = event.window_seconds;
        payload.node_noise_floor_rms = event.node_noise_floor_rms.clone();
        payload.run_srp = true;
        if late_member_now_covers {
            // The render's own fresh solve_id and current layout stand.
            payload.channel_states = channel_states;
            payload.active_channels = active_channels;
            payload.pair_admission = SolvePairAdmission::compute(
                &payload.sensor_layout,
                &payload.active_channels,
                // A new solve of the window: the render's own (current) clock snapshot.
                &payload.node_clock_assessments,
                &payload.config.localization_solve,
            );
            self.by_buffer_key
                .insert(remembered_solve_key(payload), remembered_solve_of(payload));
            return;
        }
        payload.solve_id = event.solve_id.clone();
        payload.channel_states = event.channel_states.clone();
        payload.active_channels = event.active_channels.clone();
        payload.sensor_layout = event.sensor_layout.clone();
        payload.pair_admission = event.pair_admission.clone();
        payload.node_clock_assessments = event.node_clock_assessments.clone();
        payload.effective_sound_speed_mps = event.sound_speed_mps;
        payload.shared_event_solve = Some(event.evaluation.clone());
    }

    /// Held renders are kept: the batch boundary releases them once their
    /// parked event is gone, so a source manifest is never dropped unconsumed.
    pub(crate) fn forget(&mut self, buffer_key: &str) {
        self.by_buffer_key
            .retain(|key, _| buffer_key_of_dispatch_gate_key(key) != buffer_key);
    }

    fn hold_render(&mut self, payload: ComputePayload) {
        self.held_renders_by_buffer_key
            .entry(payload.stream_key.clone())
            .or_default()
            .push(payload);
    }

    fn take_held_renders(&mut self, buffer_key: &str) -> Vec<ComputePayload> {
        self.held_renders_by_buffer_key
            .remove(buffer_key)
            .unwrap_or_default()
    }

    fn held_render_buffer_keys(&self) -> Vec<String> {
        self.held_renders_by_buffer_key.keys().cloned().collect()
    }
}

impl DspWorker {
    /// The covered-event step of dispatch: returns the payloads to dispatch
    /// now, in order (none when `payload` is a render held for a parked
    /// event; the event plus its released renders when the event goes out).
    pub(super) fn route_through_covered_event_solves(
        &mut self,
        mut payload: ComputePayload,
    ) -> Vec<ComputePayload> {
        let buffer_key = payload.stream_key.clone();
        if self.parked_event_covered_by(&payload) {
            self.recent_event_trigger_solves.hold_render(payload);
            return Vec::new();
        }
        let buffers = self.buffers.get(&buffer_key).map(Vec::as_slice);
        self.recent_event_trigger_solves
            .remember_or_attach(&mut payload, buffers);
        let mut ready = vec![payload];
        ready.extend(self.release_held_renders_no_longer_waiting(&buffer_key));
        ready
    }

    /// Batch boundary: a parked event can also end without a dispatch (its
    /// buffer was evicted); renders held for it then go out on their own.
    pub(super) fn release_every_held_render_no_longer_waiting(&mut self) -> Vec<ComputePayload> {
        self.recent_event_trigger_solves
            .held_render_buffer_keys()
            .into_iter()
            .flat_map(|buffer_key| self.release_held_renders_no_longer_waiting(&buffer_key))
            .collect()
    }

    fn parked_event_covered_by(&self, payload: &ComputePayload) -> bool {
        self.deferred_localizations
            .pending_window_center_times_ns(&payload.stream_key)
            .into_iter()
            .any(|center_time_ns| render_covers_window(payload, center_time_ns))
    }

    fn release_held_renders_no_longer_waiting(&mut self, buffer_key: &str) -> Vec<ComputePayload> {
        let held_renders = self
            .recent_event_trigger_solves
            .take_held_renders(buffer_key);
        let mut released = Vec::with_capacity(held_renders.len());
        for mut render in held_renders {
            if self.parked_event_covered_by(&render) {
                self.recent_event_trigger_solves.hold_render(render);
                continue;
            }
            let buffers = self.buffers.get(buffer_key).map(Vec::as_slice);
            self.recent_event_trigger_solves
                .remember_or_attach(&mut render, buffers);
            released.push(render);
        }
        released
    }
}
