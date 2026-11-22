//! When an island runs a chirp round (plan §Schedule).
//!
//! Silent until needed: a slow heartbeat round keeps drift tracked and
//! finds islands back in earshot; a round runs at once when the group stops
//! walking, once right after each detection, and soon after any clock in
//! the island is invalidated. Periodic adds a fixed cadence on top.

use super::config::{ChirpScheduleMode, ChirpSchedulerConfig};
use super::MeshMilliseconds;

/// Why a round was started (diagnostics and tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundTrigger {
    FirstRound,
    Heartbeat,
    Periodic,
    PostDetection,
    GroupStopped,
    ClockInvalid,
    /// A joiner, returning or lone node has not chirped in a round yet.
    NewMember,
}

#[derive(Debug, Clone, Default)]
pub struct IslandChirpScheduler {
    last_round_started_ms: Option<MeshMilliseconds>,
    last_triggered_round_ms: Option<MeshMilliseconds>,
    post_detection_due_ms: Option<MeshMilliseconds>,
    group_stopped_pending: bool,
    clock_invalid_pending: bool,
    group_was_still: bool,
}

impl IslandChirpScheduler {
    pub fn note_detection(&mut self, now_ms: MeshMilliseconds, config: &ChirpSchedulerConfig) {
        if self.post_detection_due_ms.is_none() {
            self.post_detection_due_ms = Some(now_ms + config.post_detection_round_delay_ms);
        }
    }

    pub fn note_group_motion(&mut self, group_is_still: bool) {
        if group_is_still && !self.group_was_still {
            self.group_stopped_pending = true;
        }
        self.group_was_still = group_is_still;
    }

    pub fn note_clock_invalid(&mut self) {
        self.clock_invalid_pending = true;
    }

    fn triggered_gap_elapsed(&self, now_ms: MeshMilliseconds, config: &ChirpSchedulerConfig) -> bool {
        self.last_triggered_round_ms
            .is_none_or(|last| now_ms.saturating_sub(last) >= config.triggered_round_minimum_gap_ms)
    }

    /// The trigger for a round due now, if any.
    pub fn due(&self, now_ms: MeshMilliseconds, config: &ChirpSchedulerConfig) -> Option<RoundTrigger> {
        let Some(last) = self.last_round_started_ms else {
            return Some(RoundTrigger::FirstRound);
        };
        let since_last = now_ms.saturating_sub(last);
        if self.post_detection_due_ms.is_some_and(|due| now_ms >= due) {
            return Some(RoundTrigger::PostDetection);
        }
        if self.triggered_gap_elapsed(now_ms, config) {
            if self.group_stopped_pending {
                return Some(RoundTrigger::GroupStopped);
            }
            if self.clock_invalid_pending {
                return Some(RoundTrigger::ClockInvalid);
            }
        }
        if config.mode == ChirpScheduleMode::Periodic
            && since_last >= config.periodic_round_interval_ms
        {
            return Some(RoundTrigger::Periodic);
        }
        (since_last >= config.heartbeat_round_interval_ms).then_some(RoundTrigger::Heartbeat)
    }

    pub fn mark_round_started(&mut self, now_ms: MeshMilliseconds, trigger: RoundTrigger) {
        self.last_round_started_ms = Some(now_ms);
        match trigger {
            RoundTrigger::PostDetection => self.post_detection_due_ms = None,
            RoundTrigger::GroupStopped => {
                self.group_stopped_pending = false;
                self.last_triggered_round_ms = Some(now_ms);
            }
            RoundTrigger::ClockInvalid => {
                self.clock_invalid_pending = false;
                self.last_triggered_round_ms = Some(now_ms);
            }
            _ => {}
        }
        // Any round also serves pending triggers it overlaps.
        if self.post_detection_due_ms.is_some_and(|due| due <= now_ms) {
            self.post_detection_due_ms = None;
        }
        self.clock_invalid_pending = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silent_schedule_runs_heartbeat_detection_and_stillness_rounds() {
        let config = ChirpSchedulerConfig::default();
        let mut scheduler = IslandChirpScheduler::default();
        assert_eq!(scheduler.due(0, &config), Some(RoundTrigger::FirstRound));
        scheduler.mark_round_started(0, RoundTrigger::FirstRound);
        assert_eq!(scheduler.due(10_000, &config), None);
        scheduler.note_detection(10_000, &config);
        assert_eq!(scheduler.due(10_600, &config), Some(RoundTrigger::PostDetection));
        scheduler.mark_round_started(10_600, RoundTrigger::PostDetection);
        scheduler.note_group_motion(false);
        scheduler.note_group_motion(true);
        assert_eq!(scheduler.due(11_000, &config), Some(RoundTrigger::GroupStopped));
        scheduler.mark_round_started(11_000, RoundTrigger::GroupStopped);
        assert_eq!(scheduler.due(11_000 + 90_000, &config), Some(RoundTrigger::Heartbeat));
    }
}
