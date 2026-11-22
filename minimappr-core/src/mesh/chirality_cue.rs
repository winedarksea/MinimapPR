//! Layout chirality from chirp bearings (plan §Mirror images: "any
//! direction measurement locks the chirality").
//!
//! One bearing cannot do it when the map rotation itself comes from noisy
//! GPS: a rotation error of a few tens of degrees makes either chirality
//! fit. Two bearings taken by the same stereo phone or array node in the
//! same round can: the signed angle between them is rotation-invariant (the
//! phone's heading error cancels), and a mirrored layout flips its sign.
//! The comparison is made in the layout's own frame, so GPS is not needed.

use std::collections::BTreeMap;

use crate::layout::gps_anchor_fit::ChiralityCue;
use crate::protocol::mesh_messages::ChirpDirectionReport;
use crate::protocol::NodeId;

/// Wrap an angle into (−π, π].
fn wrapped(angle_rad: f64) -> f64 {
    let wrapped = angle_rad.rem_euclid(std::f64::consts::TAU);
    if wrapped > std::f64::consts::PI { wrapped - std::f64::consts::TAU } else { wrapped }
}

/// Cue from two reports of one reporter and round, or `None` when the pair
/// is not decisive (nearly collinear emitters, or noise larger than the
/// separation between the two chirality hypotheses).
pub fn chirality_from_bearing_pair(
    layout_local_m: &BTreeMap<NodeId, [f64; 3]>,
    first: &ChirpDirectionReport,
    second: &ChirpDirectionReport,
    relative_bearing_noise_std_rad: f64,
    decisive_sigma_multiplier: f64,
) -> Option<ChiralityCue> {
    if first.reporting_node_id != second.reporting_node_id
        || first.round_id != second.round_id
        || first.emitter_node_id == second.emitter_node_id
    {
        return None;
    }
    let reporter = layout_local_m.get(&first.reporting_node_id)?;
    let a = layout_local_m.get(&first.emitter_node_id)?;
    let b = layout_local_m.get(&second.emitter_node_id)?;
    let local_angle = |p: &[f64; 3]| (p[1] - reporter[1]).atan2(p[0] - reporter[0]);
    let layout_turn = wrapped(local_angle(b) - local_angle(a));
    let measured_turn = wrapped((second.world_azimuth_deg as f64 - first.world_azimuth_deg as f64).to_radians());
    let proper_error = wrapped(measured_turn - layout_turn).abs();
    let mirrored_error = wrapped(measured_turn + layout_turn).abs();
    // The difference of two bearings carries √2 × the per-bearing noise.
    let pair_noise = relative_bearing_noise_std_rad * std::f64::consts::SQRT_2;
    if (proper_error - mirrored_error).abs() <= decisive_sigma_multiplier * pair_noise {
        return None;
    }
    Some(ChiralityCue::Locked { mirrored: mirrored_error < proper_error })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(emitter: u32, azimuth_deg: f32) -> ChirpDirectionReport {
        ChirpDirectionReport {
            round_id: 4,
            reporting_node_id: NodeId(0),
            emitter_node_id: NodeId(emitter),
            world_azimuth_deg: azimuth_deg,
            azimuth_standard_deviation_deg: 20.0,
        }
    }

    #[test]
    fn heading_error_cancels_and_mirror_is_detected() {
        let layout: BTreeMap<NodeId, [f64; 3]> =
            [(0, [0.0, 0.0]), (1, [5.0, 0.0]), (2, [0.0, 5.0])].into_iter().map(|(id, [x, y])| (NodeId(id), [x, y, 1.4])).collect();
        let noise = 5f64.to_radians();
        // World = layout rotated by 37° with a 25° heading error on both bearings.
        let (rotation, heading_error) = (37.0f32, 25.0f32);
        let proper = chirality_from_bearing_pair(&layout, &report(1, rotation + heading_error), &report(2, rotation + 90.0 + heading_error), noise, 3.0);
        assert_eq!(proper, Some(ChiralityCue::Locked { mirrored: false }));
        let mirrored = chirality_from_bearing_pair(&layout, &report(1, rotation + heading_error), &report(2, rotation - 90.0 + heading_error), noise, 3.0);
        assert_eq!(mirrored, Some(ChiralityCue::Locked { mirrored: true }));
        let collinear: BTreeMap<NodeId, [f64; 3]> =
            [(0, [0.0, 0.0]), (1, [5.0, 0.0]), (2, [9.0, 0.2])].into_iter().map(|(id, [x, y])| (NodeId(id), [x, y, 1.4])).collect();
        assert_eq!(chirality_from_bearing_pair(&collinear, &report(1, 10.0), &report(2, 12.0), noise, 3.0), None);
    }
}
