//! Mic-path visibility (`active_microphones` events): which microphones and
//! pre-processing effects AudioRecord ran over the run, and whether the
//! configuration changed right after the phone played a given channel. A
//! change that follows bottom-speaker emissions would support the
//! mic-switch explanation for the multi-copy remote picks.

use crate::bundle_loader::PhoneBundle;
use crate::bundle_schema::{BundleEvent, MicrophonePathState};

/// A configuration change this soon after an emission is attributed to it.
const CHANGE_ATTRIBUTION_WINDOW_NS: i64 = 1_000_000_000;
const NANOSECONDS_PER_SECOND: f64 = 1e9;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct MicrophonePathObservation {
    pub state: MicrophonePathState,
    pub event_count: usize,
    pub first_s: f64,
    pub last_s: f64,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct MicrophonePathSummary {
    pub event_count: usize,
    /// Distinct states, in order of first appearance.
    pub states: Vec<MicrophonePathObservation>,
    /// State changes between consecutive events.
    pub change_count: usize,
    /// (channel of the emission ≤ 1 s before a change, count).
    pub changes_after_channel: Vec<(String, usize)>,
}

pub fn microphone_path_summary(bundle: &PhoneBundle) -> MicrophonePathSummary {
    let origin_ns = bundle.session.started_ns.unwrap_or_else(|| bundle.events.first().map_or(0, BundleEvent::t_ns));
    let seconds = |t_ns: i64| (t_ns - origin_ns) as f64 / NANOSECONDS_PER_SECOND;
    let samples: Vec<(i64, &MicrophonePathState)> = bundle
        .events
        .iter()
        .filter_map(|event| match event {
            BundleEvent::ActiveMicrophones { t_ns, state } => Some((*t_ns, state)),
            _ => None,
        })
        .collect();
    let mut states: Vec<MicrophonePathObservation> = Vec::new();
    for (t_ns, state) in &samples {
        match states.iter_mut().find(|o| &o.state == *state) {
            Some(observation) => {
                observation.event_count += 1;
                observation.last_s = seconds(*t_ns);
            }
            None => states.push(MicrophonePathObservation { state: (*state).clone(), event_count: 1, first_s: seconds(*t_ns), last_s: seconds(*t_ns) }),
        }
    }
    let emissions = bundle.emissions();
    let mut changes_after_channel: Vec<(String, usize)> = Vec::new();
    let mut change_count = 0;
    for pair in samples.windows(2).filter(|pair| pair[0].1 != pair[1].1) {
        change_count += 1;
        let change_ns = pair[1].0;
        let preceding = emissions.iter().filter(|e| e.played).filter_map(|e| Some((e.target_ns?, e.channel.clone()))).filter(|(t, _)| *t <= change_ns).max_by_key(|(t, _)| *t);
        let channel = match preceding {
            Some((t, channel)) if change_ns - t <= CHANGE_ATTRIBUTION_WINDOW_NS => channel.unwrap_or_else(|| "?".into()),
            _ => "none".into(),
        };
        match changes_after_channel.iter_mut().find(|(c, _)| *c == channel) {
            Some((_, count)) => *count += 1,
            None => changes_after_channel.push((channel, 1)),
        }
    }
    MicrophonePathSummary { event_count: samples.len(), states, change_count, changes_after_channel }
}
