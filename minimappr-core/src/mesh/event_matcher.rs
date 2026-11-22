//! Event matcher: turns detections into calls to localize.
//!
//! One detection above the localize-to-track score starts a call unless an
//! already pending or pulled window of the same island covers its time
//! (plan: the host does not re-trigger while a pulled window covers it);
//! then it merges into that call. The shared window serves every species
//! active in it, so matching is by time and island, not by species.

use std::collections::VecDeque;

use crate::protocol::wire_codes::SpeciesScore;

use super::config::MatcherConfig;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CallWindow {
    pub call_id: u64,
    pub island_id: u32,
    pub centre_reference_s: f64,
    pub half_window_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchDecision {
    NewCall { call_id: u64 },
    MergedInto { call_id: u64 },
    BelowThreshold,
}

#[derive(Debug, Clone, Default)]
pub struct EventMatcher {
    remembered: VecDeque<CallWindow>,
    next_call_id: u64,
}

pub fn top_score(scores: &[SpeciesScore]) -> f32 {
    scores.iter().map(|score| score.score).fold(0.0, f32::max)
}

/// Max-merge species beliefs from several detections of one call.
pub fn merge_species(belief: &mut Vec<SpeciesScore>, scores: &[SpeciesScore]) {
    for score in scores {
        match belief.iter_mut().find(|existing| existing.label_id == score.label_id) {
            Some(existing) => existing.score = existing.score.max(score.score),
            None => belief.push(score.clone()),
        }
    }
}

impl EventMatcher {
    pub fn decide(
        &mut self,
        island_id: u32,
        centre_reference_s: f64,
        half_window_s: f64,
        scores: &[SpeciesScore],
        config: &MatcherConfig,
    ) -> MatchDecision {
        if top_score(scores) < config.localize_to_track_score {
            return MatchDecision::BelowThreshold;
        }
        if let Some(window) = self.remembered.iter().find(|window| {
            window.island_id == island_id
                && (window.centre_reference_s - centre_reference_s).abs() <= window.half_window_s
        }) {
            return MatchDecision::MergedInto { call_id: window.call_id };
        }
        self.next_call_id += 1;
        let call_id = self.next_call_id;
        self.remembered.push_back(CallWindow {
            call_id,
            island_id,
            centre_reference_s,
            half_window_s,
        });
        while self.remembered.len() > config.remembered_window_count {
            self.remembered.pop_front();
        }
        MatchDecision::NewCall { call_id }
    }

    /// A successor host keeps numbering above every call it restored.
    pub fn reserve_call_ids_through(&mut self, highest_call_id: u64) {
        self.next_call_id = self.next_call_id.max(highest_call_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(value: f32) -> Vec<SpeciesScore> {
        vec![SpeciesScore { label_id: 1, score: value }]
    }

    #[test]
    fn covered_detections_merge_and_weak_ones_do_not_start_calls() {
        let config = MatcherConfig::default();
        let mut matcher = EventMatcher::default();
        assert_eq!(matcher.decide(1, 10.0, 0.9, &scores(0.2), &config), MatchDecision::BelowThreshold);
        let MatchDecision::NewCall { call_id } = matcher.decide(1, 10.0, 0.9, &scores(0.8), &config) else {
            panic!("expected a call")
        };
        assert_eq!(matcher.decide(1, 10.4, 0.9, &scores(0.7), &config), MatchDecision::MergedInto { call_id });
        assert!(matches!(matcher.decide(2, 10.4, 0.9, &scores(0.7), &config), MatchDecision::NewCall { .. }));
        assert!(matches!(matcher.decide(1, 13.0, 0.9, &scores(0.7), &config), MatchDecision::NewCall { .. }));
    }
}
