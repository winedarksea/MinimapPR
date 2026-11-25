//! Step re-anchor and drift slew of one node's buffer timeline (owner
//! decision 2026-11-25, "step + drift re-anchor").
//!
//! A node's buffered audio sits on a nominal-rate timeline: one origin (a
//! packet time at a sample index) extrapolated by the node's own sample count.
//! A frame's *timeline residual* is its packet timestamp minus where that
//! timeline put it. Both terms come from the same node, so the residual is
//! that node's packet clock against its own sample clock: it carries no
//! acoustic delay and no other node, and absorbing it cannot fight a genuine
//! TDOA.
//!
//! Both corrections move only the timeline origin. The mapping stays one
//! straight line over the sample index, so buffered audio stays contiguous
//! (no zero gap, no overwritten samples) and every buffered sample is retimed
//! onto the corrected clock. That is the best estimate for recent pre-step
//! audio too: the sample clock never jumped, and a step is usually a
//! correction (SNTP/GPS re-sync) of an offset the earlier packets already had.
//!
//! - **Step re-anchor.** A frame the step detector flags opens a pending step.
//!   It is confirmed once `step_confirm_frames` frames agree with the stepped
//!   clock (within the step threshold), and the origin then shifts by their
//!   median residual. A frame back on the pre-step timeline cancels it: one
//!   late-stamped frame is jitter, not a step.
//! - **Drift slew.** When the windowed residual magnitude passes
//!   `drift_slew_threshold_ns` (a fraction of the cross-node hard limit) the
//!   origin shifts by the windowed residual, so steady crystal drift never
//!   reaches the limit. A windowed residual above the hard limit is not drift
//!   (an unconfirmed step, a flapping clock) and is left to the pair gate.
//!
//! The caller applies each correction to every buffer of the node and to the
//! residuals it has recorded. The Python mirror is
//! `minimappr/core/timeline_discipline.py`; both are pinned by
//! `tests/fixtures/timeline_discipline_parity_cases.json`.

const NANOSECONDS_PER_SECOND_F64: f64 = 1.0e9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimelineDisciplinePolicy {
    pub step_reanchor_enabled: bool,
    pub step_threshold_ns: i128,
    /// Frames on the stepped clock (the flagged frame included) before a step
    /// re-anchors; at least 1.
    pub step_confirm_frames: usize,
    /// 0 disables the drift slew.
    pub drift_slew_threshold_ns: i128,
    pub hard_limit_ns: i128,
}

fn seconds_to_ns(seconds: f64) -> i128 {
    (seconds.max(0.0) * NANOSECONDS_PER_SECOND_F64).round() as i128
}

impl TimelineDisciplinePolicy {
    pub fn from_seconds(
        step_reanchor_enabled: bool,
        step_threshold_seconds: f64,
        step_confirm_frames: usize,
        drift_slew_fraction_of_hard_limit: f64,
        hard_limit_seconds: f64,
    ) -> Self {
        Self {
            step_reanchor_enabled,
            step_threshold_ns: seconds_to_ns(step_threshold_seconds),
            step_confirm_frames: step_confirm_frames.max(1),
            drift_slew_threshold_ns: seconds_to_ns(
                drift_slew_fraction_of_hard_limit.max(0.0) * hard_limit_seconds,
            ),
            hard_limit_ns: seconds_to_ns(hard_limit_seconds),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimelineCorrectionKind {
    StepReanchor,
    DriftSlew,
}

/// Add `origin_shift_ns` to the timeline origin time; every residual measured
/// against the old timeline drops by the same amount.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimelineCorrection {
    pub kind: TimelineCorrectionKind,
    pub origin_shift_ns: i128,
}

#[derive(Clone, Debug)]
struct PendingStep {
    baseline_residual_ns: i128,
    stepped_residuals_ns: Vec<i128>,
}

/// Lower median (nearest rank), so both languages pick the same element.
fn lower_median(values: &[i128]) -> i128 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[(sorted.len() - 1) / 2]
}

#[derive(Clone, Debug)]
pub struct TimelineDiscipline {
    policy: TimelineDisciplinePolicy,
    pending_step: Option<PendingStep>,
}

impl TimelineDiscipline {
    pub fn new(policy: TimelineDisciplinePolicy) -> Self {
        Self {
            policy,
            pending_step: None,
        }
    }

    pub fn step_pending(&self) -> bool {
        self.pending_step.is_some()
    }

    /// A reboot or a timeline the caller replaced makes the pending step's
    /// baseline meaningless.
    pub fn cancel_pending_step(&mut self) {
        self.pending_step = None;
    }

    /// One frame, in arrival order. `frame_residual_ns` is measured against
    /// the current timeline; `detected_step_ns` is the step detector's verdict
    /// on this frame (`None`: no step).
    pub fn observe_frame(
        &mut self,
        frame_residual_ns: i128,
        detected_step_ns: Option<i128>,
    ) -> Option<TimelineCorrection> {
        if !self.policy.step_reanchor_enabled {
            return None;
        }
        let threshold_ns = self.policy.step_threshold_ns;
        match self.pending_step.as_mut() {
            Some(pending) => {
                if (frame_residual_ns - pending.baseline_residual_ns).abs() <= threshold_ns {
                    self.pending_step = None;
                    return None;
                }
                if (frame_residual_ns - lower_median(&pending.stepped_residuals_ns)).abs()
                    <= threshold_ns
                {
                    pending.stepped_residuals_ns.push(frame_residual_ns);
                } else {
                    // A further step before confirmation: restart the count on
                    // the newest clock, still measured from the old baseline.
                    pending.stepped_residuals_ns = vec![frame_residual_ns];
                }
            }
            None => {
                let step_ns = detected_step_ns?;
                self.pending_step = Some(PendingStep {
                    baseline_residual_ns: frame_residual_ns - step_ns,
                    stepped_residuals_ns: vec![frame_residual_ns],
                });
            }
        }
        let pending = self.pending_step.as_ref()?;
        if pending.stepped_residuals_ns.len() < self.policy.step_confirm_frames {
            return None;
        }
        let origin_shift_ns = lower_median(&pending.stepped_residuals_ns);
        self.pending_step = None;
        (origin_shift_ns != 0).then_some(TimelineCorrection {
            kind: TimelineCorrectionKind::StepReanchor,
            origin_shift_ns,
        })
    }

    /// Drift slew on the caller's windowed residual (after this frame was
    /// recorded). Never while a step is pending: its frames are not drift.
    pub fn drift_slew(&self, windowed_residual_ns: Option<i128>) -> Option<TimelineCorrection> {
        if self.pending_step.is_some() || self.policy.drift_slew_threshold_ns <= 0 {
            return None;
        }
        let windowed_residual_ns = windowed_residual_ns?;
        let magnitude_ns = windowed_residual_ns.abs();
        (magnitude_ns > self.policy.drift_slew_threshold_ns
            && magnitude_ns <= self.policy.hard_limit_ns)
            .then_some(TimelineCorrection {
                kind: TimelineCorrectionKind::DriftSlew,
                origin_shift_ns: windowed_residual_ns,
            })
    }
}

#[cfg(test)]
#[path = "timeline_discipline_tests.rs"]
mod timeline_discipline_tests;
