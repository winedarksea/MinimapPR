//! One island's published map anchor: each round's `fit_layout_to_gps`
//! fused through [`MapAnchorFilter`], plus a per-phone GPS fix history that
//! catches a still phone's multipath jump the fit alone cannot.
//!
//! Why a history: a 20 m jump at 6 m GPS σ is only 3.3σ against the fit,
//! and the fit has already absorbed part of it, so the 3σ hard drop often
//! misses it and Huber still pulls ~2 m per round. But a still phone does
//! not move, and its GPS error is mostly a slowly varying multipath bias:
//! against the running mean of its *own* fixes the jump stands out, at a
//! lower threshold than the fit's. The phone has not moved (its step
//! detector says so), so the jump is subtracted from its fixes for as long
//! as it lasts, which keeps the phone's earlier pull on the fit. Simply
//! dropping it would not: on a 10 m layout under 6 m GPS, a 4-of-5 fit
//! sits ~bias/4 and up to ~0.25 rad away from the 5-phone one. Any motion
//! clears the histories.
//!
//! Frames: every layout solve has its own MDS frame. Call [`reframe`] with
//! the previous→new frame map before fitting a new solve (fix histories
//! are world coordinates and need no change).
//!
//! [`reframe`]: FilteredMapAnchor::reframe

use std::collections::BTreeMap;

use super::anchor_filter::{AnchorEstimate, AnchorFilterConfig, MapAnchorFilter};
use super::gps_anchor_fit::{fit_layout_to_gps, AnchorFit, AnchorFitConfig, AnchorFitError, ChiralityCue, GpsFix, RigidTransform2d};

#[derive(Clone, Debug, PartialEq)]
pub struct FilteredMapAnchorConfig {
    pub filter: AnchorFilterConfig,
    /// While the group is still, a fix that departs from the phone's own
    /// running mean by more than this many fix σ has that jump subtracted.
    /// Below the fit's 3σ drop: the bias cancels against the phone's past,
    /// so only GPS jitter (< σ) remains in the deviation.
    pub still_jump_threshold_sigma: f64,
    /// A held phone is released only once its deviation falls below this
    /// (hysteresis): a 20 m jump plus jitter dips under the hold threshold
    /// now and then, and each such fix would drag the history toward it.
    pub still_jump_release_threshold_sigma: f64,
    /// Still fits a phone's history needs before it can judge a jump.
    pub minimum_history_fits: u32,
    /// Weight of a new fix in a phone's running mean (EWMA, ~1/weight fits).
    pub history_smoothing_weight: f64,
}

impl Default for FilteredMapAnchorConfig {
    fn default() -> Self {
        Self {
            filter: AnchorFilterConfig::default(),
            still_jump_threshold_sigma: 2.0,
            still_jump_release_threshold_sigma: 1.0,
            minimum_history_fits: 6,
            history_smoothing_weight: 0.05,
        }
    }
}

/// One round's inputs: `layout_xy` and `fixes` have one entry per `nodes`.
#[derive(Clone, Copy, Debug)]
pub struct AnchorRound<'a, Key> {
    pub time_s: f64,
    pub nodes: &'a [Key],
    pub layout_xy: &'a [[f64; 2]],
    pub fixes: &'a [Option<GpsFix>],
    pub cue: ChiralityCue,
    /// Every member's step detector and GNSS speed say still since the last round.
    pub group_is_still: bool,
}

/// Running mean of one still phone's own GPS fixes (world east, north).
#[derive(Clone, Copy, Debug, PartialEq)]
struct FixHistory {
    mean_m: [f64; 2],
    fits: u32,
}

#[derive(Clone, Debug)]
pub struct FilteredMapAnchor<Key: Ord + Copy> {
    pub config: FilteredMapAnchorConfig,
    filter: MapAnchorFilter,
    histories: BTreeMap<Key, FixHistory>,
    /// A held phone's jump (its fixes minus its pre-jump mean), averaged
    /// while held and subtracted from its fixes: the phone keeps its real
    /// jitter in the fit, so only the offset estimate's error remains.
    jump_offsets: BTreeMap<Key, FixHistory>,
    last_fit: Option<AnchorFit>,
    held_last_update: Vec<Key>,
}

impl<Key: Ord + Copy> FilteredMapAnchor<Key> {
    pub fn new(config: FilteredMapAnchorConfig) -> Self {
        let filter = MapAnchorFilter::new(config.filter.clone());
        Self { config, filter, histories: BTreeMap::new(), jump_offsets: BTreeMap::new(), last_fit: None, held_last_update: Vec::new() }
    }

    pub fn reset(&mut self) {
        self.filter.reset();
        self.histories.clear();
        self.jump_offsets.clear();
        self.last_fit = None;
        self.held_last_update.clear();
    }

    /// Move the filter into a new layout frame; `None` (no common nodes to
    /// relate the frames) restarts it, since the old anchor cannot transfer.
    pub fn reframe(&mut self, previous_to_new: Option<&RigidTransform2d>) {
        match previous_to_new {
            Some(frame_change) => self.filter.reframe(frame_change),
            None => self.reset(),
        }
    }

    /// Fit the round's layout to its fixes and fuse it.
    /// Returns the raw per-round fit; the filtered anchor is [`Self::filtered_fit`].
    pub fn update(&mut self, round: &AnchorRound<'_, Key>, fit_config: &AnchorFitConfig) -> Result<AnchorFit, AnchorFitError> {
        let AnchorRound { time_s, nodes, layout_xy, fixes, cue, group_is_still } = *round;
        if !group_is_still {
            self.histories.clear();
            self.jump_offsets.clear();
        }
        let held = if group_is_still { self.jumped_fixes(nodes, fixes) } else { Vec::new() };
        // A majority "jumping" together is the group moving (or chance
        // outliers on top of a real jump), not multipath on single phones:
        // hold nothing this round and let every history learn it.
        let fix_count = fixes.iter().flatten().count();
        let held = if 2 * held.len() >= fix_count { Vec::new() } else { held };
        self.jump_offsets.retain(|node, _| held.iter().any(|&index| nodes[index] == *node));
        let mut used_fixes = fixes.to_vec();
        for &index in &held {
            let history = self.histories[&nodes[index]];
            let Some(fix) = used_fixes[index].as_mut() else { continue };
            let deviation = [fix.east_m - history.mean_m[0], fix.north_m - history.mean_m[1]];
            let offset = self.jump_offsets.entry(nodes[index]).or_insert(FixHistory { mean_m: deviation, fits: 0 });
            let weight = (1.0 / (offset.fits + 1) as f64).max(self.config.history_smoothing_weight);
            offset.mean_m[0] += weight * (deviation[0] - offset.mean_m[0]);
            offset.mean_m[1] += weight * (deviation[1] - offset.mean_m[1]);
            offset.fits += 1;
            fix.east_m -= offset.mean_m[0];
            fix.north_m -= offset.mean_m[1];
        }
        let fit = fit_layout_to_gps(layout_xy, &used_fixes, cue, fit_config)?;
        self.filter.update(time_s, &fit, group_is_still);
        if group_is_still {
            self.record_fixes(nodes, fixes, &held);
        }
        self.held_last_update = held.iter().map(|&index| nodes[index]).collect();
        self.last_fit = Some(fit.clone());
        Ok(fit)
    }

    fn jumped_fixes(&self, nodes: &[Key], fixes: &[Option<GpsFix>]) -> Vec<usize> {
        (0..nodes.len())
            .filter(|&index| {
                let (Some(fix), Some(history)) = (fixes[index], self.histories.get(&nodes[index])) else { return false };
                if history.fits < self.config.minimum_history_fits || fix.sigma_m <= 0.0 {
                    return false;
                }
                let deviation = (fix.east_m - history.mean_m[0]).hypot(fix.north_m - history.mean_m[1]);
                let threshold = if self.held_last_update.contains(&nodes[index]) {
                    self.config.still_jump_release_threshold_sigma
                } else {
                    self.config.still_jump_threshold_sigma
                };
                deviation / fix.sigma_m > threshold
            })
            .collect()
    }

    /// Held fixes stay out of their own history, so a jump that persists
    /// while the group is still never becomes the phone's "normal".
    fn record_fixes(&mut self, nodes: &[Key], fixes: &[Option<GpsFix>], held: &[usize]) {
        for index in 0..nodes.len() {
            let Some(fix) = fixes[index] else { continue };
            if held.contains(&index) || fix.sigma_m <= 0.0 {
                continue;
            }
            self.histories
                .entry(nodes[index])
                .and_modify(|history| {
                    // Plain average until the EWMA window fills, so early
                    // means are not dominated by the first fix.
                    let weight = (1.0 / (history.fits + 1) as f64).max(self.config.history_smoothing_weight);
                    history.mean_m[0] += weight * (fix.east_m - history.mean_m[0]);
                    history.mean_m[1] += weight * (fix.north_m - history.mean_m[1]);
                    history.fits += 1;
                })
                .or_insert(FixHistory { mean_m: [fix.east_m, fix.north_m], fits: 1 });
        }
    }

    pub fn estimate(&self) -> Option<AnchorEstimate> {
        self.filter.estimate()
    }

    /// Phones held at their history in the last update.
    pub fn held_last_update(&self) -> &[Key] {
        &self.held_last_update
    }

    /// The last round's fit with its transform and σ replaced by the
    /// filter's: what the map should draw with (chirality confidence and
    /// residuals stay the round's own).
    pub fn filtered_fit(&self) -> Option<AnchorFit> {
        let estimate = self.filter.estimate()?;
        let mut fit = self.last_fit.clone()?;
        fit.transform = estimate.transform;
        fit.translation_sigma_m = estimate.translation_sigma_m[0].max(estimate.translation_sigma_m[1]);
        fit.rotation_sigma_rad = estimate.rotation_sigma_rad;
        Some(fit)
    }
}

#[cfg(test)]
#[path = "filtered_map_anchor_tests.rs"]
mod tests;
