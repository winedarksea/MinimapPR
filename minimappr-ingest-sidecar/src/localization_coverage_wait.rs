//! Cross-node coverage wait for cluster-routed localization (Rust counterpart
//! of `minimappr/core/sensor_span_coverage_wait.py`).
//!
//! Why: cluster members share one buffer and every manifest localizes a window
//! centred on *its own* end time. Whichever node delivers frame k first saw
//! the other nodes' channels below `min_coverage_ratio`, so it solved alone,
//! and the trigger cooldown (keyed on the cluster buffer) then blocked the
//! later node from re-running with both: "0 cross-node pairs" in Mode 2.
//!
//! Contract (same meaning as the Python keys of the same names):
//! - keep a localization pending while any uncovered member channel that can
//!   still cover the window is advancing;
//! - release a channel whose buffer end has not advanced for
//!   `stall_budget_ns` (a dead node costs one stall budget once, because the
//!   per-channel tracker persists across windows);
//! - never hold a window longer than its cap: the window's own geometry-derived
//!   `coverage_total_cap_ns` (cross-node lag + delivery margin), else `total_cap_ns`;
//! - channels that never buffered audio are not streaming and are ignored.
//!
//! The wait is a *deferral*, never a sleep: the DSP worker parks one pending
//! localization per cluster buffer and re-evaluates it on later manifests and
//! poll ticks, so the ingest/audio hot path never blocks.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use crate::dsp::SensorStreamBuffer;

pub(crate) const STOP_REASON_ALL_COVERED: &str = "all_covered";
pub(crate) const STOP_REASON_STALLED: &str = "stalled";
pub(crate) const STOP_REASON_TOTAL_CAP: &str = "total_cap";
/// A newer localization candidate arrived while an older one was still
/// pending; the older one is dispatched with whatever coverage it has so at
/// most one pending localization exists per buffer.
pub(crate) const STOP_REASON_SUPERSEDED: &str = "superseded";
const NANOSECONDS_PER_SECOND_F64: f64 = 1_000_000_000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LocalizationCoverageWaitConfig {
    pub(crate) enabled: bool,
    pub(crate) stall_budget_ns: u128,
    pub(crate) total_cap_ns: u128,
}

impl LocalizationCoverageWaitConfig {
    pub(crate) fn from_seconds(
        enabled: bool,
        stall_budget_seconds: f64,
        total_cap_seconds: f64,
    ) -> Self {
        Self {
            enabled,
            stall_budget_ns: seconds_to_ns(stall_budget_seconds),
            total_cap_ns: seconds_to_ns(total_cap_seconds),
        }
    }
}

pub(crate) fn seconds_to_ns(seconds: f64) -> u128 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * NANOSECONDS_PER_SECOND_F64).round() as u128
}

/// Wall-clock source for stall/cap accounting. `Manual` lets tests step time
/// deterministically instead of sleeping.
#[derive(Clone, Debug)]
pub(crate) enum CoverageWaitClock {
    System(Instant),
    #[cfg_attr(not(test), allow(dead_code))]
    Manual(Arc<AtomicU64>),
}

impl Default for CoverageWaitClock {
    fn default() -> Self {
        Self::System(Instant::now())
    }
}

impl CoverageWaitClock {
    pub(crate) fn now_ns(&self) -> u128 {
        match self {
            Self::System(origin) => origin.elapsed().as_nanos(),
            Self::Manual(now_ns) => u128::from(now_ns.load(Ordering::Acquire)),
        }
    }
}

/// Remembers, across windows, when each buffer channel's end time last
/// advanced. Observed on every manifest for the buffer (O(channels)).
#[derive(Debug, Default)]
pub(crate) struct ChannelStreamActivityTracker {
    last_end_time_ns_by_channel: Vec<Option<i128>>,
    last_advance_ns_by_channel: Vec<Option<u128>>,
}

impl ChannelStreamActivityTracker {
    pub(crate) fn observe(&mut self, buffers: &[SensorStreamBuffer], now_ns: u128) {
        if self.last_end_time_ns_by_channel.len() < buffers.len() {
            self.last_end_time_ns_by_channel.resize(buffers.len(), None);
            self.last_advance_ns_by_channel.resize(buffers.len(), None);
        }
        for (channel_index, buffer) in buffers.iter().enumerate() {
            let Some(end_time_ns) = buffer.end_time_ns() else {
                continue;
            };
            let advanced = self.last_end_time_ns_by_channel[channel_index]
                .is_none_or(|previous_end_ns| end_time_ns > previous_end_ns);
            if advanced {
                self.last_advance_ns_by_channel[channel_index] = Some(now_ns);
            }
            self.last_end_time_ns_by_channel[channel_index] = Some(end_time_ns);
        }
    }

    fn ns_since_advance(&self, channel_index: usize, now_ns: u128) -> Option<u128> {
        self.last_advance_ns_by_channel
            .get(channel_index)
            .copied()
            .flatten()
            .map(|last_advance_ns| now_ns.saturating_sub(last_advance_ns))
    }
}

/// The localization window a pending candidate is waiting on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LocalizationWindowSpan {
    pub(crate) center_time_ns: i128,
    pub(crate) window_seconds: f64,
    pub(crate) min_coverage_ratio: f64,
    /// This window's wait cap, derived from its layout's cross-node lag
    /// (`localization_event_span.rs`); `None` uses the config's cap.
    pub(crate) coverage_total_cap_ns: Option<u128>,
}

impl LocalizationWindowSpan {
    fn start_and_end_ns(&self) -> (i128, i128) {
        let half_window_ns =
            ((self.window_seconds * NANOSECONDS_PER_SECOND_F64) / 2.0).round() as i128;
        (
            self.center_time_ns.saturating_sub(half_window_ns),
            self.center_time_ns.saturating_add(half_window_ns),
        )
    }

    fn channel_is_covered(&self, buffer: &SensorStreamBuffer) -> bool {
        buffer
            .coverage_centered_at(self.center_time_ns, self.window_seconds)
            .is_some_and(|coverage| coverage.coverage_ratio >= self.min_coverage_ratio)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CoverageWaitDecision {
    /// Keep the localization pending; at least one streaming member channel
    /// is still expected to cover the window.
    Wait,
    /// Dispatch now.
    Ready { stop_reason: &'static str },
}

/// Decide whether a localization window should be dispatched now.
///
/// `waited_ns` is the wall time since the candidate was first deferred.
/// Eligible-to-wait-for = has buffered audio, buffer start at or before the
/// window start (ring buffer starts only move forward), trails the window end
/// by no more audio than the remaining cap (audio arrives at most ~real time),
/// and advanced within the stall budget.
pub(crate) fn evaluate_window_coverage(
    buffers: &[SensorStreamBuffer],
    span: LocalizationWindowSpan,
    tracker: &ChannelStreamActivityTracker,
    config: LocalizationCoverageWaitConfig,
    waited_ns: u128,
    now_ns: u128,
) -> CoverageWaitDecision {
    if !config.enabled {
        return CoverageWaitDecision::Ready {
            stop_reason: STOP_REASON_ALL_COVERED,
        };
    }
    let (span_start_ns, span_end_ns) = span.start_and_end_ns();
    let total_cap_ns = span.coverage_total_cap_ns.unwrap_or(config.total_cap_ns);
    let remaining_cap_ns = total_cap_ns.saturating_sub(waited_ns) as i128;
    let mut any_uncovered_streaming_channel = false;
    let mut any_channel_worth_waiting_for = false;
    for (channel_index, buffer) in buffers.iter().enumerate() {
        let (Some(start_ns), Some(end_ns)) = (buffer.start_time_ns(), buffer.end_time_ns()) else {
            continue;
        };
        // Audio up to the window end has arrived: nothing more to wait for.
        // A gap inside the window is never backfilled, so waiting on it
        // would only burn the stall budget; coverage ratio decides at solve.
        if end_ns >= span_end_ns || span.channel_is_covered(buffer) {
            continue;
        }
        any_uncovered_streaming_channel = true;
        let audio_deficit_ns = span_end_ns.saturating_sub(end_ns);
        let can_catch_up = start_ns <= span_start_ns && audio_deficit_ns <= remaining_cap_ns;
        let recently_advanced = tracker
            .ns_since_advance(channel_index, now_ns)
            .is_some_and(|since_advance_ns| since_advance_ns < config.stall_budget_ns);
        if can_catch_up && recently_advanced {
            any_channel_worth_waiting_for = true;
        }
    }
    if !any_uncovered_streaming_channel {
        return CoverageWaitDecision::Ready {
            stop_reason: STOP_REASON_ALL_COVERED,
        };
    }
    if waited_ns >= total_cap_ns {
        return CoverageWaitDecision::Ready {
            stop_reason: STOP_REASON_TOTAL_CAP,
        };
    }
    if !any_channel_worth_waiting_for {
        return CoverageWaitDecision::Ready {
            stop_reason: STOP_REASON_STALLED,
        };
    }
    CoverageWaitDecision::Wait
}

#[cfg(test)]
#[path = "localization_coverage_wait_tests.rs"]
mod localization_coverage_wait_tests;
