//! Round-robin classification inside chirp-ranged clumps (plan: a tight
//! clump of three classifies each call once, not three times).
//!
//! The host groups an island's phones by single linkage within the clump
//! radius; each clump shares duty by time slot on the island reference
//! clock, so every member decides locally, with no per-call message. Two
//! members hear one call within milliseconds of each other on that clock,
//! so they land in the same slot and exactly one classifies.

use std::collections::{BTreeMap, BTreeSet};

use crate::protocol::mesh_messages::{ClassificationClump, ClassificationRota};
use crate::protocol::NodeId;

use super::clock_mapping::NodeClockMapping;

const MILLISECONDS_PER_SECOND: f64 = 1_000.0;
const MINIMUM_CLUMP_SIZE: usize = 2;

/// Clumps of classifier-capable nodes within `clump_radius_m` (single linkage).
pub fn clumps_from_layout(positions_m: &BTreeMap<NodeId, [f64; 3]>, classifiers: &BTreeSet<NodeId>, clump_radius_m: f64) -> Vec<ClassificationClump> {
    let nodes: Vec<NodeId> = positions_m.keys().copied().filter(|node| classifiers.contains(node)).collect();
    let mut parent: Vec<usize> = (0..nodes.len()).collect();
    fn root(parent: &mut [usize], index: usize) -> usize {
        let mut current = index;
        while parent[current] != current {
            current = parent[current];
        }
        current
    }
    for i in 0..nodes.len() {
        for j in i + 1..nodes.len() {
            let (a, b) = (positions_m[&nodes[i]], positions_m[&nodes[j]]);
            if ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt() <= clump_radius_m {
                let (ri, rj) = (root(&mut parent, i), root(&mut parent, j));
                parent[ri.max(rj)] = ri.min(rj);
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<NodeId>> = BTreeMap::new();
    for (index, node) in nodes.iter().enumerate() {
        groups.entry(root(&mut parent, index)).or_default().push(*node);
    }
    groups
        .into_values()
        .filter(|members| members.len() >= MINIMUM_CLUMP_SIZE)
        .map(|member_node_ids| ClassificationClump { member_node_ids })
        .collect()
}

/// Node side: should this node classify a trigger at `trigger_device_seconds`?
pub fn has_classification_duty(
    rota: Option<&ClassificationRota>,
    own_node: NodeId,
    own_clock: Option<&NodeClockMapping>,
    trigger_device_seconds: f64,
) -> bool {
    let (Some(rota), Some(clock)) = (rota, own_clock) else { return true };
    let Some(clump) = rota.clumps.iter().find(|clump| clump.member_node_ids.contains(&own_node)) else {
        return true;
    };
    let rank = clump.member_node_ids.iter().position(|node| *node == own_node).unwrap_or(0);
    let slot_s = rota.slot_duration_milliseconds as f64 / MILLISECONDS_PER_SECOND;
    let reference_s = clock.reference_seconds_from_node_seconds(trigger_device_seconds);
    let slot = (reference_s / slot_s).floor().max(0.0) as usize;
    slot % clump.member_node_ids.len() == rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exactly_one_clump_member_has_duty_per_slot() {
        let positions: BTreeMap<NodeId, [f64; 3]> = [(1, [0.0, 0.0]), (2, [1.5, 0.5]), (3, [2.5, 2.0]), (4, [20.0, 0.0])]
            .into_iter()
            .map(|(id, [x, y])| (NodeId(id), [x, y, 1.4]))
            .collect();
        let classifiers: BTreeSet<NodeId> = positions.keys().copied().collect();
        let clumps = clumps_from_layout(&positions, &classifiers, 3.0);
        assert_eq!(clumps.len(), 1);
        assert_eq!(clumps[0].member_node_ids, vec![NodeId(1), NodeId(2), NodeId(3)]);
        let rota = ClassificationRota { island_id: 1, rota_epoch: 1, slot_duration_milliseconds: 1_500, clumps };
        let identity = NodeClockMapping::identity(0, 0.0);
        for slot in 0..9 {
            let t = slot as f64 * 1.5 + 0.7;
            let on_duty = (1..=3).filter(|id| has_classification_duty(Some(&rota), NodeId(*id), Some(&identity), t)).count();
            assert_eq!(on_duty, 1);
            assert!(has_classification_duty(Some(&rota), NodeId(4), Some(&identity), t));
        }
    }
}
