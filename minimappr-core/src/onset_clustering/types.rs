//! Shared value types for onset-frame SRP clustering (port of
//! `minimappr/core/onset_srp_types.py`).

use std::collections::BTreeMap;

use super::attribution_evidence::ClusterAttributionEvidence;
use super::frame_correlation::PairCorrelationCurve;
use super::srp_grid::{norm, Point3};

/// Python divides by `max(weight_sum, 1e-12)` when averaging peak positions.
const WEIGHT_FLOOR: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterStatus {
    Accepted,
    Tentative,
    Echo,
}

impl ClusterStatus {
    /// The Python status string (`accepted` / `tentative` / `echo`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Tentative => "tentative",
            Self::Echo => "echo",
        }
    }
}

/// One onset's template-vs-search correlations (the unit of localization).
#[derive(Debug, Clone, PartialEq)]
pub struct OnsetFrame {
    pub frame_index: usize,
    pub template_node_id: String,
    pub reference_sensor: String,
    /// Index into the analysis span (shared timeline).
    pub onset_sample: usize,
    pub strength: f64,
    pub band_min_hz: f64,
    pub band_max_hz: f64,
    pub curves: Vec<PairCorrelationCurve>,
    pub weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FramePeak {
    pub frame_index: usize,
    pub position_m: Point3,
    pub direction: Point3,
    pub range_m: f64,
    pub value: f64,
    pub weight: f64,
    pub is_frame_primary: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnsetCluster {
    pub peaks: Vec<FramePeak>,
    pub status: ClusterStatus,
    pub status_reasons: Vec<String>,
    pub independent_onset_count: usize,
}

impl OnsetCluster {
    pub fn with_peak(peak: FramePeak) -> Self {
        Self {
            peaks: vec![peak],
            status: ClusterStatus::Accepted,
            status_reasons: Vec::new(),
            independent_onset_count: 0,
        }
    }

    pub fn weight(&self) -> f64 {
        self.peaks.iter().map(|peak| peak.weight).sum()
    }

    /// Distinct frame indices, ascending (Python iterates sets in this order for small ints).
    pub fn frame_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self.peaks.iter().map(|peak| peak.frame_index).collect();
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    pub fn led_frame_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self
            .peaks
            .iter()
            .filter(|peak| peak.is_frame_primary)
            .map(|peak| peak.frame_index)
            .collect();
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    pub fn weighted_direction(&self) -> Point3 {
        let mut stacked = [0.0; 3];
        for peak in &self.peaks {
            for axis in 0..3 {
                stacked[axis] += peak.direction[axis] * peak.weight;
            }
        }
        let length = norm(&stacked);
        if length > 0.0 {
            [
                stacked[0] / length,
                stacked[1] / length,
                stacked[2] / length,
            ]
        } else {
            self.peaks[0].direction
        }
    }

    pub fn weighted_position_m(&self) -> Point3 {
        let mut weighted = [0.0; 3];
        let mut total = 0.0;
        for peak in &self.peaks {
            total += peak.weight;
            for axis in 0..3 {
                weighted[axis] += peak.weight * peak.position_m[axis];
            }
        }
        let divisor = total.max(WEIGHT_FLOOR);
        [
            weighted[0] / divisor,
            weighted[1] / divisor,
            weighted[2] / divisor,
        ]
    }
}

/// A clustered, refined candidate bird.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalizedOnsetCluster {
    pub position_m: Point3,
    pub covariance_m2: Option<[[f64; 3]; 3]>,
    pub confidence: f64,
    pub gdop: f64,
    pub range_projection_mode: Option<String>,
    pub range_observability: Option<f64>,
    pub residual_rms_seconds: Option<f64>,
    pub reference_sensor: String,
    pub tdoa_s: BTreeMap<String, f64>,
    pub status: ClusterStatus,
    pub status_reasons: Vec<String>,
    /// Led onsets at their template nodes (span timeline).
    pub onset_samples: Vec<usize>,
    pub template_node_ids: Vec<String>,
    pub band_min_hz: f64,
    pub band_max_hz: f64,
    pub weight: f64,
    pub frame_count: usize,
    pub independent_onset_count: usize,
    pub refined_by_least_squares: bool,
    pub used_envelope_fraction: f64,
    pub contributing_node_ids: Vec<String>,
    /// Node to classify from: nearest the fix when range is known, else the
    /// most frequent template node.
    pub loudest_node_id: String,
    pub height_resolution: Option<String>,
    /// SRP-supported range interval (m from the layout centroid); None = not judged.
    pub range_interval_m: Option<(f64, f64)>,
    /// Per-node level evidence for attribution (S5); None unless requested.
    pub attribution_evidence: Option<ClusterAttributionEvidence>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnsetClusteringResult {
    pub clusters: Vec<LocalizedOnsetCluster>,
    pub onset_count: usize,
    pub frame_count: usize,
    pub consumed_onset_samples_by_node: BTreeMap<String, Vec<usize>>,
}
