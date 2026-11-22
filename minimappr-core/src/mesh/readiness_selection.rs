//! Which ready nodes contribute a clip when more answer yes than the cap
//! allows (plan: "best in-band SNR and spread kept; the rest counted").
//!
//! The classifying node is always in. The rest are picked greedily by
//! in-band level plus a capped bonus for distance from nodes already
//! chosen, so a tight clump of loud phones does not crowd out the far
//! phone that fixes the geometry. In-band level only ranks; it never
//! excludes a node by itself.

use crate::protocol::NodeId;

use super::config::ClipFetchConfig;

#[derive(Debug, Clone, PartialEq)]
pub struct ReadinessCandidate {
    pub node_id: NodeId,
    pub in_band_level_over_floor_db: f32,
    pub position_m: Option<[f64; 3]>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReadinessSelection {
    pub selected: Vec<NodeId>,
    pub excluded_by_cap: Vec<NodeId>,
}

fn horizontal_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// `cap` counts the classifying node.
pub fn select_clip_nodes(
    classifier: &ReadinessCandidate,
    ready_others: &[ReadinessCandidate],
    cap: usize,
    config: &ClipFetchConfig,
) -> ReadinessSelection {
    let mut chosen = vec![classifier.clone()];
    let mut remaining: Vec<ReadinessCandidate> = ready_others
        .iter()
        .filter(|candidate| candidate.node_id != classifier.node_id)
        .cloned()
        .collect();
    while chosen.len() < cap && !remaining.is_empty() {
        let score = |candidate: &ReadinessCandidate| {
            let spread_m = candidate.position_m.and_then(|position| {
                chosen
                    .iter()
                    .filter_map(|picked| picked.position_m.map(|p| horizontal_distance(p, position)))
                    .min_by(f64::total_cmp)
            });
            let bonus = spread_m.map_or(0.0, |metres| (metres * config.spread_bonus_db_per_m).min(config.spread_bonus_cap_db));
            candidate.in_band_level_over_floor_db as f64 + bonus
        };
        let best = (0..remaining.len())
            .max_by(|&a, &b| {
                score(&remaining[a])
                    .total_cmp(&score(&remaining[b]))
                    .then(remaining[b].node_id.cmp(&remaining[a].node_id))
            })
            .expect("non-empty");
        chosen.push(remaining.remove(best));
    }
    ReadinessSelection {
        selected: chosen.iter().map(|candidate| candidate.node_id).collect(),
        excluded_by_cap: remaining.iter().map(|candidate| candidate.node_id).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: u32, level: f32, x: f64) -> ReadinessCandidate {
        ReadinessCandidate {
            node_id: NodeId(id),
            in_band_level_over_floor_db: level,
            position_m: Some([x, 0.0, 1.4]),
        }
    }

    #[test]
    fn cap_keeps_classifier_loudest_and_spread() {
        let classifier = candidate(0, 20.0, 0.0);
        let others: Vec<_> = (1..8).map(|id| candidate(id, 10.0 + id as f32, id as f64)).collect();
        let selection = select_clip_nodes(&classifier, &others, 6, &ClipFetchConfig::default());
        assert_eq!(selection.selected.len(), 6);
        assert_eq!(selection.selected[0], NodeId(0));
        assert_eq!(selection.excluded_by_cap.len(), 2);
        assert!(selection.selected.contains(&NodeId(7)));
    }
}
