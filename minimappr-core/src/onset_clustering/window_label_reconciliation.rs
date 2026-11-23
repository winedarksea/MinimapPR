//! Post-window label reconciliation for sibling clusters of one analysis window
//! (port of `minimappr/core/onset_window_label_reconciliation.py`).
//!
//! The SNR-gradient tier proves a cluster's classifier mostly heard *its* bird,
//! which justifies the cluster's top label but not the rest of the window's
//! score list: a loud robin's window still carries the quieter tanager next to
//! it, and that secondary score lets the robin's fix absorb into the tanager's
//! entity (the same-band two-species merge). So a gradient-settled cluster
//! drops every label that another *settled* (gradient or LOCO) cluster of the
//! same window claims as its top label. Its own top label is always kept, so
//! two same-species birds in one window both stay labelled.

use super::attribution_cascade::AttributionTier;

/// Tiers whose top label the window trusts as that cluster's own bird.
fn tier_claims_top_label(tier: AttributionTier) -> bool {
    matches!(tier, AttributionTier::SnrGradient | AttributionTier::Loco)
}

/// For each cluster `(tier, top label)`, the labels to remove from its scores.
/// Only gradient-settled clusters lose labels; a cluster without a top label
/// (unsettled, abstained) claims nothing.
pub fn labels_to_drop_by_cluster<L: PartialEq + Clone>(clusters: &[(AttributionTier, Option<L>)]) -> Vec<Vec<L>> {
    clusters
        .iter()
        .enumerate()
        .map(|(index, (tier, own_top))| {
            if *tier != AttributionTier::SnrGradient {
                return Vec::new();
            }
            let mut dropped: Vec<L> = Vec::new();
            for (other_index, (other_tier, other_top)) in clusters.iter().enumerate() {
                let Some(other_top) = other_top else { continue };
                if other_index == index || !tier_claims_top_label(*other_tier) || Some(other_top) == own_top.as_ref() {
                    continue;
                }
                if !dropped.contains(other_top) {
                    dropped.push(other_top.clone());
                }
            }
            dropped
        })
        .collect()
}

/// The highest-scoring label (the first one on a tie, as the classifier's argmax).
pub fn top_scoring_label<L: Clone>(scores: &[(L, f64)]) -> Option<L> {
    scores
        .iter()
        .fold(None::<&(L, f64)>, |best, entry| match best {
            Some(current) if current.1 >= entry.1 => Some(current),
            _ => Some(entry),
        })
        .map(|(label, _)| label.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use AttributionTier::{Loco, SingleCluster, SnrGradient, Unsettled};

    #[test]
    fn gradient_cluster_drops_a_loco_siblings_top_label() {
        let dropped = labels_to_drop_by_cluster(&[(SnrGradient, Some("robin")), (Loco, Some("tanager"))]);
        assert_eq!(dropped, vec![vec!["tanager"], vec![]]);
    }

    #[test]
    fn own_top_label_is_kept_when_siblings_share_it() {
        let dropped = labels_to_drop_by_cluster(&[(SnrGradient, Some("robin")), (SnrGradient, Some("robin"))]);
        assert_eq!(dropped, vec![Vec::<&str>::new(), vec![]]);
    }

    #[test]
    fn unsettled_and_single_cluster_siblings_claim_nothing() {
        let dropped = labels_to_drop_by_cluster(&[
            (SnrGradient, Some("robin")),
            (Unsettled, None),
            (SingleCluster, Some("wren")),
            (SnrGradient, Some("vireo")),
        ]);
        assert_eq!(dropped, vec![vec!["vireo"], vec![], vec![], vec!["robin"]]);
    }

    #[test]
    fn top_label_takes_the_first_maximum() {
        assert_eq!(top_scoring_label(&[("a", 0.5), ("b", 0.7), ("c", 0.7)]), Some("b"));
        assert_eq!(top_scoring_label::<&str>(&[]), None);
    }
}
