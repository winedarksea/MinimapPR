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
//!   re-runs that solve (same windows, layout snapshot, centre and
//!   `solve_id`), so the stored detection names the solve it came from. The
//!   re-run is deterministic and on the render's own payload, which avoids a
//!   cross-task wait on the trigger's result;
//! - when a member whose audio was late now covers the window too, it solves
//!   the fuller sensor set as a new solve of the same window (new `solve_id`)
//!   and that solve becomes the one later renders repeat. Repeating the
//!   poorer trigger-time solve would throw the late member away.

use super::*;

/// The inputs of the latest event-triggered solve on one buffer.
struct RecentEventTriggerSolve {
    solve_id: String,
    attempt_kind: SolveAttemptKind,
    window_center_time_ns: i128,
    channel_states: Vec<LocalizationChannelState>,
    active_channels: Vec<usize>,
    sensor_layout: Arc<SolveSensorLayout>,
    pair_admission: SolvePairAdmission,
}

/// One entry per buffer key, dropped with the buffer.
#[derive(Default)]
pub(crate) struct RecentEventTriggerSolves {
    by_buffer_key: HashMap<String, RecentEventTriggerSolve>,
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

fn remembered_solve_of(payload: &ComputePayload) -> RecentEventTriggerSolve {
    RecentEventTriggerSolve {
        solve_id: payload.solve_id.clone(),
        attempt_kind: payload.attempt_kind,
        window_center_time_ns: payload.localization_window_center_time_ns,
        channel_states: payload.channel_states.clone(),
        active_channels: payload.active_channels.clone(),
        sensor_layout: payload.sensor_layout.clone(),
        pair_admission: payload.pair_admission.clone(),
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
                .insert(payload.stream_key.clone(), remembered_solve_of(payload));
            return;
        }
        // Omni renders (override channels) and sub-SRP buffers never solve.
        if !payload.run_classifier_render
            || payload.omni_channels_override.is_some()
            || payload.sensor_layout.sensor_count() < MIN_SRP_CHANNEL_COUNT
        {
            return;
        }
        let (Some(render_start_ns), Some(render_end_ns)) = (
            payload.classifier_render_start_ns,
            payload.classifier_render_end_ns,
        ) else {
            return;
        };
        let Some(event) = self.by_buffer_key.get(&payload.stream_key) else {
            return;
        };
        let Some(buffers) = buffers else {
            return;
        };
        let covered = (render_start_ns as i128..=render_end_ns as i128)
            .contains(&event.window_center_time_ns);
        // The render beamforms its windows with the solve's positions, so the
        // snapshot must index the same buffer channels as the render does.
        if !covered || !same_sensor_order(&event.sensor_layout, &payload.sensor_layout) {
            return;
        }
        let layout_buffers = &buffers[..payload.sensor_layout.sensor_count().min(buffers.len())];
        let mut channel_states = localization_channel_states_centered(
            layout_buffers,
            event.window_center_time_ns,
            payload.config.window_seconds,
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
        payload.run_srp = true;
        if late_member_now_covers {
            // The render's own fresh solve_id and current layout stand.
            payload.channel_states = channel_states;
            payload.active_channels = active_channels;
            payload.pair_admission = SolvePairAdmission::compute(
                &payload.sensor_layout,
                &payload.active_channels,
                &NodeClockAssessments::new(), // R5 seam: the render payload's clock snapshot
                &payload.config.localization_solve,
            );
            self.by_buffer_key
                .insert(payload.stream_key.clone(), remembered_solve_of(payload));
            return;
        }
        payload.solve_id = event.solve_id.clone();
        payload.channel_states = event.channel_states.clone();
        payload.active_channels = event.active_channels.clone();
        payload.sensor_layout = event.sensor_layout.clone();
        payload.pair_admission = event.pair_admission.clone();
    }

    pub(crate) fn forget(&mut self, buffer_key: &str) {
        self.by_buffer_key.remove(buffer_key);
    }
}
