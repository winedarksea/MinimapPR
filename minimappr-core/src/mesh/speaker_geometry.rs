//! Speaker–mic geometry in the live chirp-ranging solve.
//!
//! Each phone ranges with its mic but chirps from a speaker offset by a
//! per-model body vector v (an earpiece sits ~0.13 m up the body). The
//! exchange then measures speaker_A→mic_B and speaker_B→mic_A, so the range
//! is biased by ≈ (v_B − v_A)·û/2 and the clock offset by ≈ (v_A + v_B)·û/(2c)
//! (û = unit vector mic A → mic B). Removing it needs v in the world frame
//! (attitude) *and* û in that same frame. Attitude gives the first; û is
//! not observable from the ranges of a small group, and GPS at a few metres
//! is noise. So:
//!
//! - **Corrected** only when the island's previous anchored layout gives the
//!   pair's ENU bearing to within `maximum_pair_bearing_sigma_rad` (anchor
//!   rotation σ plus relative-position σ, chirality confidently resolved)
//!   and every phone with a speaker vector reported a usable heading. The
//!   previous round's layout is used because phones ranging are still and
//!   the anchor needs a solved layout first; re-solving inside the round
//!   would add a second layout solve for a bearing that moves ~0 between
//!   rounds. The residual from bearing and heading error stays in the σ.
//! - **Widened** otherwise: the range keeps its bias but its σ grows by the
//!   bound 0.5(|v_A|+|v_B|) and the clock offset's by (|v_A|+|v_B|)/(2c), so
//!   the layout, clock graph and downstream TDOA weighting know.

use std::collections::BTreeMap;

use crate::protocol::session_messages::{ChirpArrivalReport, DeviceAttitude};
use crate::protocol::NodeId;
use crate::sync::two_way_ranging::{correct_for_speaker_mic_offsets, TwoWayRangingResult};

use super::config::IslandSyncConfig;
use super::host_membership::SessionMembership;
use super::island_sync_engine::{IslandLayout, IslandSyncEngine};
use super::MeshMilliseconds;

/// Mic positions for the exact correction are placed at the measured
/// distance, which is itself biased; one more pass at the corrected
/// distance leaves an error of order |v|²/d² of the correction (< 0.1 mm).
const MIC_DISTANCE_REFINEMENT_PASSES: usize = 2;
/// A body vector shorter than this is "speaker on the mic" (no correction).
const NEGLIGIBLE_SPEAKER_OFFSET_M: f64 = 1e-6;

#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerGeometryConfig {
    /// A node attaches its attitude to a round report only if it is younger
    /// than this (phones ranging are still; the sensor is sampled ~1 Hz).
    pub attitude_maximum_age_ms: MeshMilliseconds,
    /// Pair bearing σ above which the correction is not applied. The
    /// residual is linear in the bearing error only for small errors, and
    /// at 20° it still removes ~2/3 of the bias.
    pub maximum_pair_bearing_sigma_rad: f64,
    /// Heading σ (reported accuracy ⊕ declination) above which a phone's
    /// world speaker vector counts as unknown (uncalibrated magnetometer).
    pub maximum_heading_sigma_rad: f64,
    /// The anchored bearing of a pair flips with the layout's chirality
    /// when three or more nodes are placed; below this P(chirality) the
    /// bearing is not trusted.
    pub minimum_chirality_confidence: f64,
    /// Android rotation vectors point at magnetic north while the anchor
    /// is GPS (true) north and no declination model is on the phone:
    /// added in quadrature to a magnetic heading. 15° covers most of the
    /// inhabited mid-latitudes (US/Europe declinations are within ±15°).
    pub magnetic_declination_sigma_rad: f64,
}

impl Default for SpeakerGeometryConfig {
    fn default() -> Self {
        Self {
            attitude_maximum_age_ms: 3_000,
            maximum_pair_bearing_sigma_rad: 20f64.to_radians(),
            maximum_heading_sigma_rad: 30f64.to_radians(),
            minimum_chirality_confidence: 0.95,
            magnetic_declination_sigma_rad: 15f64.to_radians(),
        }
    }
}

/// One phone's speaker offset from its ranging mic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeSpeakerGeometry {
    pub body_offset_m: [f64; 3],
    /// ENU offset and heading σ, when attitude made it usable.
    pub world_offset: Option<WorldSpeakerOffset>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldSpeakerOffset {
    pub offset_east_north_up_m: [f64; 3],
    pub heading_sigma_rad: f64,
}

/// ENU bearing mic first → mic second (keyed with first < second).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KnownPairBearing {
    pub unit_east_north: [f64; 2],
    pub sigma_rad: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerGeometryTreatment {
    /// Neither phone has a speaker vector: unchanged.
    NoSpeakerOffset,
    Corrected,
    /// σ widened: no confident ENU bearing for the pair.
    WidenedBearingUnknown,
    /// σ widened: a phone with a speaker vector has no usable attitude.
    WidenedAttitudeUnusable,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairSpeakerGeometryOutcome {
    pub treatment: SpeakerGeometryTreatment,
    /// σ for the layout range (base range σ included).
    pub range_sigma_m: f64,
    /// Added in quadrature to the pair's clock-offset constraint σ.
    pub additional_offset_sigma_s: f64,
}

/// Everything one island round needs; the default changes nothing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RoundSpeakerGeometry {
    pub nodes: BTreeMap<NodeId, NodeSpeakerGeometry>,
    pub pair_bearings: BTreeMap<(NodeId, NodeId), KnownPairBearing>,
}

/// v_enu = q · v_body · q*, q normalised; `None` for a degenerate quaternion.
pub fn rotate_body_to_east_north_up(quaternion_wxyz: [f32; 4], body: [f64; 3]) -> Option<[f64; 3]> {
    let q = quaternion_wxyz.map(f64::from);
    let norm = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    if !norm.is_finite() || norm < f64::EPSILON {
        return None;
    }
    let (w, u) = (q[0] / norm, [q[1] / norm, q[2] / norm, q[3] / norm]);
    let t = cross(u, body).map(|c| 2.0 * c);
    let u_cross_t = cross(u, t);
    Some([0, 1, 2].map(|axis| body[axis] + w * t[axis] + u_cross_t[axis]))
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|c| c * c).sum::<f64>().sqrt()
}

/// Per-model body speaker vectors of the session's members.
pub fn speaker_body_offsets_by_node(membership: &SessionMembership, config: &IslandSyncConfig) -> BTreeMap<NodeId, [f64; 3]> {
    membership
        .members
        .values()
        .filter_map(|member| {
            let offset = config.speaker_offset_body_frame_by_platform_m.get(&member.descriptor.platform_name)?;
            Some((member.node_id, *offset))
        })
        .collect()
}

impl NodeSpeakerGeometry {
    pub fn from_attitude(body_offset_m: [f64; 3], attitude: Option<&DeviceAttitude>, config: &SpeakerGeometryConfig) -> Self {
        let world_offset = attitude.and_then(|attitude| {
            let reported = f64::from(attitude.heading_accuracy_rad?);
            if !reported.is_finite() || reported < 0.0 {
                return None;
            }
            let declination = if attitude.heading_is_true_north { 0.0 } else { config.magnetic_declination_sigma_rad };
            let heading_sigma_rad = reported.hypot(declination);
            if heading_sigma_rad > config.maximum_heading_sigma_rad {
                return None;
            }
            Some(WorldSpeakerOffset {
                offset_east_north_up_m: rotate_body_to_east_north_up(attitude.body_to_east_north_up_wxyz, body_offset_m)?,
                heading_sigma_rad,
            })
        });
        Self { body_offset_m, world_offset }
    }

    fn has_offset(&self) -> bool {
        norm(self.body_offset_m) > NEGLIGIBLE_SPEAKER_OFFSET_M
    }

    /// World offset and heading σ; a speaker on its mic needs no attitude.
    fn usable_world_offset(&self) -> Option<WorldSpeakerOffset> {
        if self.has_offset() {
            self.world_offset
        } else {
            Some(WorldSpeakerOffset { offset_east_north_up_m: [0.0; 3], heading_sigma_rad: 0.0 })
        }
    }
}

/// ENU bearings between every pair of the layout's nodes, from its anchor.
/// Empty without an anchor or with an unresolved chirality.
pub fn pair_bearings_from_layout(layout: &IslandLayout, config: &SpeakerGeometryConfig) -> BTreeMap<(NodeId, NodeId), KnownPairBearing> {
    let Some(anchor) = &layout.anchor else { return BTreeMap::new() };
    if anchor.chirality_confidence < config.minimum_chirality_confidence {
        return BTreeMap::new();
    }
    let nodes: Vec<(NodeId, [f64; 3], f64)> = layout
        .local_positions_m
        .iter()
        .map(|(node, local)| {
            let sigma = layout.sigmas_m.get(node).copied().unwrap_or(f64::INFINITY);
            (*node, IslandSyncEngine::anchored(layout, *local), sigma)
        })
        .collect();
    let mut bearings = BTreeMap::new();
    for (index, (first, first_position, first_sigma)) in nodes.iter().enumerate() {
        for (second, second_position, second_sigma) in &nodes[index + 1..] {
            let (east, north) = (second_position[0] - first_position[0], second_position[1] - first_position[1]);
            let distance = east.hypot(north);
            if distance <= 0.0 {
                continue;
            }
            let relative_sigma_rad = first_sigma.hypot(*second_sigma).atan2(distance);
            bearings.insert(
                (*first, *second),
                KnownPairBearing {
                    unit_east_north: [east / distance, north / distance],
                    sigma_rad: anchor.rotation_sigma_rad.hypot(relative_sigma_rad),
                },
            );
        }
    }
    bearings
}

impl RoundSpeakerGeometry {
    /// `layout`: the island's layout before this round (bearing source).
    pub fn for_island_round(
        reports: &BTreeMap<NodeId, ChirpArrivalReport>,
        membership: &SessionMembership,
        layout: Option<&IslandLayout>,
        config: &IslandSyncConfig,
    ) -> Self {
        Self::from_body_offsets(reports, &speaker_body_offsets_by_node(membership, config), layout, config)
    }

    pub fn from_body_offsets(
        reports: &BTreeMap<NodeId, ChirpArrivalReport>,
        body_offsets_by_node: &BTreeMap<NodeId, [f64; 3]>,
        layout: Option<&IslandLayout>,
        config: &IslandSyncConfig,
    ) -> Self {
        let geometry = &config.speaker_geometry;
        let nodes = body_offsets_by_node
            .iter()
            .map(|(node, body)| {
                let attitude = reports.get(node).and_then(|report| report.reporter_attitude.as_ref());
                (*node, NodeSpeakerGeometry::from_attitude(*body, attitude, geometry))
            })
            .collect();
        let pair_bearings = layout.map(|layout| pair_bearings_from_layout(layout, geometry)).unwrap_or_default();
        Self { nodes, pair_bearings }
    }

    fn node(&self, node: NodeId) -> NodeSpeakerGeometry {
        self.nodes
            .get(&node)
            .copied()
            .unwrap_or(NodeSpeakerGeometry { body_offset_m: [0.0; 3], world_offset: None })
    }

    /// ENU bearing first → second, if known.
    fn bearing(&self, first: NodeId, second: NodeId) -> Option<KnownPairBearing> {
        if first < second {
            self.pair_bearings.get(&(first, second)).copied()
        } else {
            self.pair_bearings.get(&(second, first)).map(|bearing| KnownPairBearing {
                unit_east_north: bearing.unit_east_north.map(|c| -c),
                ..*bearing
            })
        }
    }

    /// Correct `result` (first = A of the exchange) in place where the
    /// geometry is observable; otherwise leave it and widen the σ.
    pub fn adjust_pair(&self, first: NodeId, second: NodeId, result: &mut TwoWayRangingResult, config: &IslandSyncConfig) -> PairSpeakerGeometryOutcome {
        let (a, b) = (self.node(first), self.node(second));
        let speed = config.sound_speed_mps;
        if !a.has_offset() && !b.has_offset() {
            return PairSpeakerGeometryOutcome {
                treatment: SpeakerGeometryTreatment::NoSpeakerOffset,
                range_sigma_m: config.range_sigma_m,
                additional_offset_sigma_s: 0.0,
            };
        }
        let offset_sum_m = norm(a.body_offset_m) + norm(b.body_offset_m);
        let widened = |treatment| PairSpeakerGeometryOutcome {
            treatment,
            range_sigma_m: config.range_sigma_m.hypot(0.5 * offset_sum_m),
            additional_offset_sigma_s: offset_sum_m / (2.0 * speed),
        };
        let (Some(world_a), Some(world_b)) = (a.usable_world_offset(), b.usable_world_offset()) else {
            return widened(SpeakerGeometryTreatment::WidenedAttitudeUnusable);
        };
        let Some(bearing) = self
            .bearing(first, second)
            .filter(|bearing| bearing.sigma_rad <= config.speaker_geometry.maximum_pair_bearing_sigma_rad)
        else {
            return widened(SpeakerGeometryTreatment::WidenedBearingUnknown);
        };
        let residual = correction_residual(&world_a, &world_b, &bearing, speed);
        // Only worth applying when what is left is smaller than the bound.
        if residual.range_m >= 0.5 * offset_sum_m {
            return widened(SpeakerGeometryTreatment::WidenedBearingUnknown);
        }
        let unit = [bearing.unit_east_north[0], bearing.unit_east_north[1], 0.0];
        let raw = *result;
        let mut mic_distance_m = raw.distance_m;
        for _pass in 0..MIC_DISTANCE_REFINEMENT_PASSES {
            let mic_b = unit.map(|c| c * mic_distance_m);
            *result = correct_for_speaker_mic_offsets(&raw, [0.0; 3], mic_b, world_a.offset_east_north_up_m, world_b.offset_east_north_up_m, speed);
            mic_distance_m = result.distance_m.max(0.0);
        }
        PairSpeakerGeometryOutcome {
            treatment: SpeakerGeometryTreatment::Corrected,
            range_sigma_m: config.range_sigma_m.hypot(residual.range_m),
            additional_offset_sigma_s: residual.offset_s,
        }
    }
}

struct CorrectionResidual {
    range_m: f64,
    offset_s: f64,
}

/// Linearised 1σ left after the correction. Range bias (v_B − v_A)·û/2 and
/// offset bias (v_A + v_B)·û/(2c): a bearing error δθ moves û by û⊥·δθ, a
/// heading error δψ rotates v by ẑ×v·δψ. Tilt (roll/pitch) error from the
/// accelerometer is a degree or so and ignored.
fn correction_residual(a: &WorldSpeakerOffset, b: &WorldSpeakerOffset, bearing: &KnownPairBearing, speed_of_sound_mps: f64) -> CorrectionResidual {
    let (va, vb) = (a.offset_east_north_up_m, b.offset_east_north_up_m);
    let unit = bearing.unit_east_north;
    let perpendicular = [-unit[1], unit[0]];
    let dot = |v: [f64; 3], u: [f64; 2]| v[0] * u[0] + v[1] * u[1];
    let heading_term = |v: [f64; 3], sigma: f64| dot([-v[1], v[0], 0.0], unit) * sigma;
    let heading_a = heading_term(va, a.heading_sigma_rad);
    let heading_b = heading_term(vb, b.heading_sigma_rad);
    let difference = [vb[0] - va[0], vb[1] - va[1], 0.0];
    let sum = [va[0] + vb[0], va[1] + vb[1], 0.0];
    let range_bearing = dot(difference, perpendicular) * bearing.sigma_rad;
    let offset_bearing = dot(sum, perpendicular) * bearing.sigma_rad;
    CorrectionResidual {
        range_m: 0.5 * (heading_a.powi(2) + heading_b.powi(2) + range_bearing.powi(2)).sqrt(),
        offset_s: (heading_a.powi(2) + heading_b.powi(2) + offset_bearing.powi(2)).sqrt() / (2.0 * speed_of_sound_mps),
    }
}

#[cfg(test)]
#[path = "speaker_geometry_tests.rs"]
mod tests;
