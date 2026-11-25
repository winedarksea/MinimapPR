//! Coarse session clock on a joiner: the host's monotonic clock, estimated
//! from probe/echo exchanges over the transport (NTP-style four timestamps).
//!
//! Why: the host stamps a chirp schedule with its own `now_ms`, and every
//! node reads it back as its own `now_ms`. Two phones' monotonic clocks
//! share no epoch (each counts from its own boot), so without alignment the
//! slots land hours away. Only the coarse clock is aligned here; chirp
//! rounds still measure the fine audio-clock offsets.
//!
//! Estimator: the lowest-RTT exchange in a sliding window. Queueing delay
//! only ever adds to the RTT, and an asymmetric delay biases the offset by at
//! most half the RTT, so the fastest exchange is the least biased. Wi-Fi
//! power save adds 100–300 ms spikes that this rejects outright.
//!
//! Session time must never step back (state-machine timers subtract
//! timestamps), so after the first lock the applied offset moves toward the
//! estimate at a bounded slew rate below 1, which keeps session time
//! strictly increasing. Large forward corrections step at once.

use std::collections::VecDeque;

const NANOSECONDS_PER_MILLISECOND: i64 = 1_000_000;

#[derive(Debug, Clone, PartialEq)]
pub struct SessionClockAlignmentConfig {
    /// Exchanges kept for the lowest-RTT pick.
    pub window_exchange_count: usize,
    /// Exchanges older than this (local clock) leave the window, so a slowly
    /// drifting pair of crystals keeps being tracked.
    pub exchange_max_age_ns: i64,
    /// Exchanges with a longer RTT are discarded (a stalled link).
    pub maximum_accepted_round_trip_ns: i64,
    /// Applied offset changes by at most this fraction of elapsed local time.
    pub maximum_slew_fraction: f64,
    /// A correction this far forward is applied as a step (time may jump
    /// forward, never back).
    pub forward_step_threshold_ns: i64,
}

impl Default for SessionClockAlignmentConfig {
    fn default() -> Self {
        Self {
            window_exchange_count: 32,
            exchange_max_age_ns: 120_000 * NANOSECONDS_PER_MILLISECOND,
            maximum_accepted_round_trip_ns: 1_000 * NANOSECONDS_PER_MILLISECOND,
            maximum_slew_fraction: 0.05,
            forward_step_threshold_ns: 20 * NANOSECONDS_PER_MILLISECOND,
        }
    }
}

/// One probe/echo: t1 and t4 on the joiner's clock, t2 and t3 on the host's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockProbeExchange {
    pub probe_sent_local_ns: i64,
    pub probe_received_host_ns: i64,
    pub echo_sent_host_ns: i64,
    pub echo_received_local_ns: i64,
}

impl ClockProbeExchange {
    /// host − local, assuming symmetric one-way delays.
    pub fn offset_ns(&self) -> i64 {
        let outbound = self.probe_received_host_ns - self.probe_sent_local_ns;
        let inbound = self.echo_sent_host_ns - self.echo_received_local_ns;
        outbound / 2 + inbound / 2 + (outbound % 2 + inbound % 2) / 2
    }

    /// Network round trip, host processing time excluded.
    pub fn round_trip_ns(&self) -> i64 {
        (self.echo_received_local_ns - self.probe_sent_local_ns) - (self.echo_sent_host_ns - self.probe_received_host_ns)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeOutcome {
    Accepted,
    /// Negative RTT, reversed timestamps or a stalled link.
    Rejected,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionClockAlignment {
    config: SessionClockAlignmentConfig,
    window: VecDeque<ClockProbeExchange>,
    applied_offset_ns: Option<i64>,
    last_advance_local_ns: Option<i64>,
    pub accepted_exchange_count: u64,
    pub rejected_exchange_count: u64,
}

impl SessionClockAlignment {
    pub fn new(config: SessionClockAlignmentConfig) -> Self {
        Self {
            config,
            window: VecDeque::new(),
            applied_offset_ns: None,
            last_advance_local_ns: None,
            accepted_exchange_count: 0,
            rejected_exchange_count: 0,
        }
    }

    /// The host's own clock is the session clock.
    pub fn identity() -> Self {
        let mut alignment = Self::new(SessionClockAlignmentConfig::default());
        alignment.applied_offset_ns = Some(0);
        alignment
    }

    pub fn is_aligned(&self) -> bool {
        self.applied_offset_ns.is_some()
    }

    pub fn add_exchange(&mut self, exchange: ClockProbeExchange) -> ExchangeOutcome {
        let round_trip_ns = exchange.round_trip_ns();
        let ordered = exchange.echo_received_local_ns >= exchange.probe_sent_local_ns
            && exchange.echo_sent_host_ns >= exchange.probe_received_host_ns;
        if !ordered || round_trip_ns < 0 || round_trip_ns > self.config.maximum_accepted_round_trip_ns {
            self.rejected_exchange_count += 1;
            return ExchangeOutcome::Rejected;
        }
        self.accepted_exchange_count += 1;
        self.window.push_back(exchange);
        while self.window.len() > self.config.window_exchange_count.max(1) {
            self.window.pop_front();
        }
        let newest_local_ns = exchange.echo_received_local_ns;
        let max_age = self.config.exchange_max_age_ns;
        // Always keep the newest exchange, however old the rest are.
        while self.window.len() > 1 && self.window.front().is_some_and(|e| newest_local_ns - e.echo_received_local_ns > max_age) {
            self.window.pop_front();
        }
        if self.applied_offset_ns.is_none() {
            self.applied_offset_ns = self.best_exchange().map(|e| e.offset_ns());
            self.last_advance_local_ns = Some(newest_local_ns);
        }
        ExchangeOutcome::Accepted
    }

    fn best_exchange(&self) -> Option<&ClockProbeExchange> {
        self.window.iter().min_by_key(|exchange| exchange.round_trip_ns())
    }

    /// Offset the window currently supports (host − local).
    pub fn target_offset_ns(&self) -> Option<i64> {
        self.best_exchange().map(|exchange| exchange.offset_ns())
    }

    pub fn best_round_trip_ns(&self) -> Option<i64> {
        self.best_exchange().map(|exchange| exchange.round_trip_ns())
    }

    pub fn applied_offset_ns(&self) -> Option<i64> {
        self.applied_offset_ns
    }

    /// Move the applied offset toward the target; call with a
    /// non-decreasing local clock before reading session time.
    pub fn advance(&mut self, local_now_ns: i64) {
        let (Some(applied), Some(target)) = (self.applied_offset_ns, self.target_offset_ns()) else {
            return;
        };
        let elapsed_ns = self.last_advance_local_ns.map_or(0, |last| (local_now_ns - last).max(0));
        self.last_advance_local_ns = Some(self.last_advance_local_ns.map_or(local_now_ns, |last| last.max(local_now_ns)));
        let error_ns = target - applied;
        if error_ns >= self.config.forward_step_threshold_ns {
            self.applied_offset_ns = Some(target);
            return;
        }
        let maximum_change_ns = (elapsed_ns as f64 * self.config.maximum_slew_fraction) as i64;
        self.applied_offset_ns = Some(applied + error_ns.clamp(-maximum_change_ns, maximum_change_ns));
    }

    pub fn session_ns_at(&self, local_ns: i64) -> Option<i64> {
        self.applied_offset_ns.map(|offset| local_ns + offset)
    }

    pub fn local_ns_for_session(&self, session_ns: i64) -> Option<i64> {
        self.applied_offset_ns.map(|offset| session_ns - offset)
    }

    /// Session milliseconds for the state machines (clamped at 0).
    pub fn session_ms_at(&self, local_ns: i64) -> Option<u64> {
        self.session_ns_at(local_ns).map(|session_ns| (session_ns.max(0) / NANOSECONDS_PER_MILLISECOND) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = NANOSECONDS_PER_MILLISECOND;

    /// Host clock = local + true offset; delays in ns.
    fn exchange(sent_local_ns: i64, true_offset_ns: i64, outbound_ns: i64, inbound_ns: i64, processing_ns: i64) -> ClockProbeExchange {
        let received_host = sent_local_ns + outbound_ns + true_offset_ns;
        let echo_host = received_host + processing_ns;
        ClockProbeExchange {
            probe_sent_local_ns: sent_local_ns,
            probe_received_host_ns: received_host,
            echo_sent_host_ns: echo_host,
            echo_received_local_ns: echo_host - true_offset_ns + inbound_ns,
        }
    }

    #[test]
    fn symmetric_exchange_recovers_offset_and_rtt() {
        let probe = exchange(1_000 * MS, 3_600_000 * MS, 4 * MS, 4 * MS, MS);
        assert_eq!(probe.offset_ns(), 3_600_000 * MS);
        assert_eq!(probe.round_trip_ns(), 8 * MS);
    }

    #[test]
    fn lowest_rtt_exchange_wins_over_jittery_asymmetric_ones() {
        let true_offset = -7_200_000 * MS;
        let mut alignment = SessionClockAlignment::new(SessionClockAlignmentConfig::default());
        // Power-save spikes: large, asymmetric delays.
        let delays = [(250, 5), (3, 180), (120, 2), (2, 3), (90, 60), (300, 4)];
        for (index, (outbound, inbound)) in delays.iter().enumerate() {
            let probe = exchange(index as i64 * 500 * MS, true_offset, outbound * MS, inbound * MS, MS);
            assert_eq!(alignment.add_exchange(probe), ExchangeOutcome::Accepted);
        }
        let error = alignment.target_offset_ns().unwrap() - true_offset;
        assert!(error.abs() <= MS, "error {error} ns");
        assert_eq!(alignment.best_round_trip_ns(), Some(5 * MS));
    }

    #[test]
    fn negative_or_stalled_round_trips_are_rejected() {
        let mut alignment = SessionClockAlignment::new(SessionClockAlignmentConfig::default());
        let mut reversed = exchange(0, 0, MS, MS, 0);
        reversed.echo_received_local_ns = -MS;
        assert_eq!(alignment.add_exchange(reversed), ExchangeOutcome::Rejected);
        assert_eq!(alignment.add_exchange(exchange(0, 0, 900 * MS, 900 * MS, 0)), ExchangeOutcome::Rejected);
        assert!(!alignment.is_aligned());
    }

    #[test]
    fn session_time_never_steps_back_when_the_estimate_drops() {
        let mut alignment = SessionClockAlignment::new(SessionClockAlignmentConfig::default());
        // First exchange is slow and asymmetric: offset over-estimated by 40 ms.
        alignment.add_exchange(exchange(0, 10_000 * MS, 82 * MS, 2 * MS, 0));
        let first_offset = alignment.applied_offset_ns().unwrap();
        alignment.add_exchange(exchange(100 * MS, 10_000 * MS, 2 * MS, 2 * MS, 0));
        assert!(alignment.target_offset_ns().unwrap() < first_offset);
        let mut previous_session = i64::MIN;
        for step in 0..400 {
            let local = 200 * MS + step * 10 * MS;
            alignment.advance(local);
            let session = alignment.session_ns_at(local).unwrap();
            assert!(session > previous_session, "session time stepped back at step {step}");
            previous_session = session;
        }
        assert_eq!(alignment.applied_offset_ns(), alignment.target_offset_ns());
    }

    #[test]
    fn large_forward_correction_steps_at_once() {
        let mut alignment = SessionClockAlignment::new(SessionClockAlignmentConfig::default());
        alignment.add_exchange(exchange(0, 1_000 * MS, 2 * MS, 80 * MS, 0));
        alignment.add_exchange(exchange(10 * MS, 1_000 * MS, MS, MS, 0));
        alignment.advance(20 * MS);
        assert_eq!(alignment.applied_offset_ns(), Some(1_000 * MS));
    }

    #[test]
    fn identity_alignment_maps_local_to_itself() {
        let alignment = SessionClockAlignment::identity();
        assert_eq!(alignment.session_ms_at(5 * MS), Some(5));
        assert_eq!(alignment.local_ns_for_session(7 * MS), Some(7 * MS));
    }
}
