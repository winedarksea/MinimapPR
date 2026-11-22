//! Acoustic islands: connected components of the "who hears whom" graph.
//!
//! Clumps of phones out of chirp range of each other but connected to one
//! host each get their own clock domain, layout, anchor and scheduler. An
//! edge needs valid SNR in *both* directions (two-way closure), because
//! ranging and clock offsets need both arrivals. TDOA pairs may only be
//! formed inside one clock domain.
//!
//! `AcousticIslandTracker` keeps island identities stable across rounds
//! and reports merges and splits. A merged island always starts a fresh
//! clock-domain epoch, so no clock model from before the merge is reused.

use std::collections::{BTreeMap, BTreeSet};

pub type AcousticNodeId = u32;

/// Measured hearing between two nodes in one round (dB SNR per direction).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HearingObservation {
    pub node_a: AcousticNodeId,
    pub node_b: AcousticNodeId,
    pub snr_a_hears_b_db: Option<f64>,
    pub snr_b_hears_a_db: Option<f64>,
}

impl HearingObservation {
    pub fn has_two_way_closure(&self, minimum_snr_db: f64) -> bool {
        self.snr_a_hears_b_db.is_some_and(|s| s >= minimum_snr_db)
            && self.snr_b_hears_a_db.is_some_and(|s| s >= minimum_snr_db)
    }
}

/// Union–find over node ids.
#[derive(Clone, Debug, Default)]
struct DisjointSets {
    parent: BTreeMap<AcousticNodeId, AcousticNodeId>,
}

impl DisjointSets {
    fn find(&mut self, node: AcousticNodeId) -> AcousticNodeId {
        let parent = *self.parent.entry(node).or_insert(node);
        if parent == node {
            return node;
        }
        let root = self.find(parent);
        self.parent.insert(node, root);
        root
    }

    fn union(&mut self, a: AcousticNodeId, b: AcousticNodeId) {
        let (root_a, root_b) = (self.find(a), self.find(b));
        if root_a != root_b {
            let (low, high) = (root_a.min(root_b), root_a.max(root_b));
            self.parent.insert(high, low);
        }
    }
}

/// Connected components (each sorted; components ordered by smallest id).
pub fn acoustic_components(
    nodes: &[AcousticNodeId],
    observations: &[HearingObservation],
    minimum_snr_db: f64,
) -> Vec<BTreeSet<AcousticNodeId>> {
    let mut sets = DisjointSets::default();
    for node in nodes {
        sets.find(*node);
    }
    let known: BTreeSet<AcousticNodeId> = nodes.iter().copied().collect();
    for observation in observations {
        if known.contains(&observation.node_a)
            && known.contains(&observation.node_b)
            && observation.has_two_way_closure(minimum_snr_db)
        {
            sets.union(observation.node_a, observation.node_b);
        }
    }
    let mut groups: BTreeMap<AcousticNodeId, BTreeSet<AcousticNodeId>> = BTreeMap::new();
    for node in nodes {
        let root = sets.find(*node);
        groups.entry(root).or_default().insert(*node);
    }
    groups.into_values().collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcousticIsland {
    pub island_id: u64,
    pub members: BTreeSet<AcousticNodeId>,
    /// Bumped whenever the island's clock domain must be rebuilt from
    /// scratch (merge, or a split-off fragment).
    pub clock_domain_epoch: u64,
}

/// Map summary: green full fix, amber bearing/couple, grey detection only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IslandCapability {
    FullFix,
    BearingOrCouple,
    DetectionOnly,
}

/// Minimum island size for a full TDOA fix.
pub const MINIMUM_NODES_FOR_FULL_FIX: usize = 3;

impl AcousticIsland {
    pub fn capability(&self) -> IslandCapability {
        match self.members.len() {
            n if n >= MINIMUM_NODES_FOR_FULL_FIX => IslandCapability::FullFix,
            2 => IslandCapability::BearingOrCouple,
            _ => IslandCapability::DetectionOnly,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IslandEvent {
    Formed {
        island_id: u64,
    },
    Merged {
        surviving_island_id: u64,
        absorbed_island_ids: Vec<u64>,
    },
    SplitOff {
        from_island_id: u64,
        new_island_id: u64,
    },
    Dissolved {
        island_id: u64,
    },
}

#[derive(Clone, Debug, Default)]
pub struct AcousticIslandTracker {
    islands: Vec<AcousticIsland>,
    next_island_id: u64,
    next_epoch: u64,
}

impl AcousticIslandTracker {
    pub fn islands(&self) -> &[AcousticIsland] {
        &self.islands
    }

    pub fn island_of(&self, node: AcousticNodeId) -> Option<&AcousticIsland> {
        self.islands
            .iter()
            .find(|island| island.members.contains(&node))
    }

    /// TDOA pairs need one clock domain: the same island and epoch.
    pub fn share_clock_domain(&self, a: AcousticNodeId, b: AcousticNodeId) -> bool {
        match (self.island_of(a), self.island_of(b)) {
            (Some(x), Some(y)) => {
                x.island_id == y.island_id && x.clock_domain_epoch == y.clock_domain_epoch
            }
            _ => false,
        }
    }

    fn allocate_island_id(&mut self) -> u64 {
        self.next_island_id += 1;
        self.next_island_id
    }

    fn allocate_epoch(&mut self) -> u64 {
        self.next_epoch += 1;
        self.next_epoch
    }

    /// Replace the island set with this round's components.
    pub fn update(&mut self, components: Vec<BTreeSet<AcousticNodeId>>) -> Vec<IslandEvent> {
        let previous = std::mem::take(&mut self.islands);
        let mut events = Vec::new();
        let mut claimed_previous: BTreeSet<u64> = BTreeSet::new();
        // Larger components claim identities first so a split's main body keeps the id.
        let mut ordered = components;
        ordered.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
        for members in ordered {
            let mut overlaps: Vec<(usize, u64, u64)> = previous
                .iter()
                .map(|old| {
                    (
                        old.members.intersection(&members).count(),
                        old.island_id,
                        old.clock_domain_epoch,
                    )
                })
                .filter(|(count, _, _)| *count > 0)
                .collect();
            overlaps.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            let unclaimed: Vec<&(usize, u64, u64)> = overlaps
                .iter()
                .filter(|(_, id, _)| !claimed_previous.contains(id))
                .collect();
            let island = match (overlaps.len(), unclaimed.first()) {
                (0, _) => {
                    let island_id = self.allocate_island_id();
                    events.push(IslandEvent::Formed { island_id });
                    AcousticIsland {
                        island_id,
                        members,
                        clock_domain_epoch: self.allocate_epoch(),
                    }
                }
                (1, Some(&&(_, island_id, epoch))) => {
                    claimed_previous.insert(island_id);
                    AcousticIsland {
                        island_id,
                        members,
                        clock_domain_epoch: epoch,
                    }
                }
                (_, Some(&&(_, surviving_island_id, _))) if overlaps.len() > 1 => {
                    // Only islands no larger fragment kept are absorbed; an island
                    // whose main body survives elsewhere merely lost members here.
                    let absorbed: Vec<u64> = unclaimed
                        .iter()
                        .map(|o| o.1)
                        .filter(|id| *id != surviving_island_id)
                        .collect();
                    for (_, id, _) in &overlaps {
                        claimed_previous.insert(*id);
                    }
                    events.push(IslandEvent::Merged {
                        surviving_island_id,
                        absorbed_island_ids: absorbed,
                    });
                    AcousticIsland {
                        island_id: surviving_island_id,
                        members,
                        clock_domain_epoch: self.allocate_epoch(),
                    }
                }
                _ => {
                    // Every overlapping island already went to a larger fragment: split-off.
                    let island_id = self.allocate_island_id();
                    events.push(IslandEvent::SplitOff {
                        from_island_id: overlaps[0].1,
                        new_island_id: island_id,
                    });
                    AcousticIsland {
                        island_id,
                        members,
                        clock_domain_epoch: self.allocate_epoch(),
                    }
                }
            };
            self.islands.push(island);
        }
        for old in &previous {
            if !self
                .islands
                .iter()
                .any(|island| island.island_id == old.island_id)
            {
                let absorbed = events.iter().any(|event| match event {
                    IslandEvent::Merged {
                        absorbed_island_ids,
                        ..
                    } => absorbed_island_ids.contains(&old.island_id),
                    _ => false,
                });
                if !absorbed {
                    events.push(IslandEvent::Dissolved {
                        island_id: old.island_id,
                    });
                }
            }
        }
        self.islands.sort_by_key(|island| island.island_id);
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hear(a: AcousticNodeId, b: AcousticNodeId, snr_ab: f64, snr_ba: f64) -> HearingObservation {
        HearingObservation {
            node_a: a,
            node_b: b,
            snr_a_hears_b_db: Some(snr_ab),
            snr_b_hears_a_db: Some(snr_ba),
        }
    }

    const MINIMUM_SNR_DB: f64 = 12.0;

    fn two_clumps() -> Vec<BTreeSet<AcousticNodeId>> {
        let nodes = [1, 2, 3, 4, 5, 6];
        let observations = [
            hear(1, 2, 30.0, 30.0),
            hear(2, 3, 25.0, 28.0),
            hear(4, 5, 30.0, 30.0),
            hear(5, 6, 20.0, 22.0),
        ];
        acoustic_components(&nodes, &observations, MINIMUM_SNR_DB)
    }

    #[test]
    fn one_way_hearing_does_not_link() {
        let observations = [hear(1, 2, 30.0, 5.0)];
        let components = acoustic_components(&[1, 2], &observations, MINIMUM_SNR_DB);
        assert_eq!(components.len(), 2);
        let mut missing = hear(1, 2, 30.0, 30.0);
        missing.snr_b_hears_a_db = None;
        assert_eq!(
            acoustic_components(&[1, 2], &[missing], MINIMUM_SNR_DB).len(),
            2
        );
    }

    #[test]
    fn split_keeps_main_id_and_merge_starts_fresh_epoch() {
        let mut tracker = AcousticIslandTracker::default();
        let all: BTreeSet<AcousticNodeId> = (1..=6).collect();
        let events = tracker.update(vec![all]);
        assert_eq!(events, vec![IslandEvent::Formed { island_id: 1 }]);
        let original_epoch = tracker.islands()[0].clock_domain_epoch;
        assert!(tracker.share_clock_domain(1, 6));

        let events = tracker.update(two_clumps());
        assert!(events.iter().any(|e| matches!(
            e,
            IslandEvent::SplitOff {
                from_island_id: 1,
                ..
            }
        )));
        assert_eq!(tracker.islands().len(), 2);
        assert!(
            !tracker.share_clock_domain(1, 6),
            "no TDOA pair across islands"
        );
        assert!(tracker.share_clock_domain(1, 3));
        let main = tracker.island_of(1).unwrap();
        assert_eq!(main.island_id, 1);
        assert_eq!(
            main.clock_domain_epoch, original_epoch,
            "main body keeps its clock domain"
        );
        assert_eq!(main.capability(), IslandCapability::FullFix);

        let nodes = [1, 2, 3, 4, 5, 6];
        let mut observations = vec![
            hear(1, 2, 30.0, 30.0),
            hear(2, 3, 25.0, 28.0),
            hear(4, 5, 30.0, 30.0),
        ];
        observations.push(hear(3, 4, 15.0, 14.0));
        observations.push(hear(5, 6, 20.0, 22.0));
        let events = tracker.update(acoustic_components(&nodes, &observations, MINIMUM_SNR_DB));
        assert!(events.iter().any(|e| matches!(
            e,
            IslandEvent::Merged {
                surviving_island_id: 1,
                ..
            }
        )));
        assert_eq!(tracker.islands().len(), 1);
        assert!(tracker.share_clock_domain(1, 6));
        let merged_epoch = tracker.islands()[0].clock_domain_epoch;
        assert!(
            merged_epoch != original_epoch,
            "merge must not reuse a pre-merge clock model"
        );
    }

    #[test]
    fn fragment_joining_another_island_does_not_report_its_old_island_absorbed() {
        let mut tracker = AcousticIslandTracker::default();
        tracker.update(vec![
            [1, 2, 3, 4].into_iter().collect(),
            [7, 8].into_iter().collect(),
        ]);
        let big = tracker.island_of(1).unwrap().island_id;
        let small = tracker.island_of(7).unwrap().island_id;
        // Node 4 walks off from the big island and joins 7–8.
        let events = tracker.update(vec![
            [1, 2, 3].into_iter().collect(),
            [4, 7, 8].into_iter().collect(),
        ]);
        assert_eq!(tracker.island_of(1).unwrap().island_id, big);
        assert_eq!(tracker.island_of(4).unwrap().island_id, small);
        for event in &events {
            if let IslandEvent::Merged {
                absorbed_island_ids,
                ..
            } = event
            {
                assert!(
                    !absorbed_island_ids.contains(&big),
                    "big island still exists: {events:?}"
                );
            }
        }
        assert!(!tracker.share_clock_domain(1, 4));
    }

    #[test]
    fn lone_node_is_detection_only_and_disappearing_island_dissolves() {
        let mut tracker = AcousticIslandTracker::default();
        tracker.update(vec![
            [1, 2].into_iter().collect(),
            [9].into_iter().collect(),
        ]);
        assert_eq!(
            tracker.island_of(9).unwrap().capability(),
            IslandCapability::DetectionOnly
        );
        assert_eq!(
            tracker.island_of(1).unwrap().capability(),
            IslandCapability::BearingOrCouple
        );
        let events = tracker.update(vec![[1, 2].into_iter().collect()]);
        assert!(events
            .iter()
            .any(|e| matches!(e, IslandEvent::Dissolved { .. })));
        assert!(tracker.island_of(9).is_none());
    }
}
