//! Entity-level scoring of a run (test strategy §What to assert: label swaps,
//! echo promotions, duplicate entities). Each host call is traced to the
//! scripted call it localized (the solve trace) or, unlocalized, to its first
//! detection's call; each call to its bird (or to an echo); each bird to the
//! host's bird-entity ids from `HostEntityTracking`.

use std::collections::{BTreeMap, BTreeSet};

use minimappr_core::protocol::wire_codes::{BirdEntityState, SpeciesScore};
use minimappr_core::protocol::FieldMessage;

use crate::invariants::{ensure, InvariantResult};
use crate::scenario_outcome::ScenarioOutcome;

/// Scenario ground truth for entity scoring.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EntityTruth {
    /// Per scripted call: the bird that sang it, or None for an echo/reflection.
    pub bird_of_call: Vec<Option<usize>>,
    pub bird_species: Vec<u32>,
}

impl EntityTruth {
    pub fn add_bird(&mut self, species_label: u32) -> usize {
        self.bird_species.push(species_label);
        self.bird_species.len() - 1
    }
    pub fn note_call(&mut self, bird: Option<usize>) {
        self.bird_of_call.push(bird);
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct EntityMetrics {
    pub entities_by_bird: Vec<BTreeSet<u64>>,
    /// Extra entities per bird, summed (one bird, one entity is zero).
    pub duplicate_entities: usize,
    /// Entities holding calls from more than one bird.
    pub merged_entities: usize,
    /// Single-bird entities whose shown species is not that bird's.
    pub label_swaps: usize,
    /// Entities built only from echoes.
    pub echo_promotions: usize,
    /// Birds whose entity was pinpointed at some point.
    pub pinpointed_birds: usize,
    pub calls_scored: usize,
}

fn top_label(scores: &[SpeciesScore]) -> Option<u32> {
    let mut best: Option<&SpeciesScore> = None;
    for score in scores {
        if best.is_none_or(|current| score.score > current.score) {
            best = Some(score);
        }
    }
    best.map(|score| score.label_id)
}

pub fn entity_metrics(outcome: &ScenarioOutcome, truth: &EntityTruth) -> EntityMetrics {
    // Successful solve traces of a call are its clusters, in cluster order.
    let mut localized_call_of: BTreeMap<(u64, usize), usize> = BTreeMap::new();
    for trace in outcome.traces.iter().filter(|trace| trace.fix.is_some()) {
        let cluster_index = localized_call_of.keys().filter(|(call, _)| *call == trace.host_call_id).count();
        localized_call_of.insert((trace.host_call_id, cluster_index), trace.true_call_index as usize);
    }
    let records: BTreeMap<u64, _> = outcome.call_records().into_iter().map(|record| (record.call_id, record)).collect();
    let mut sources_by_entity: BTreeMap<u64, BTreeSet<Option<usize>>> = BTreeMap::new();
    let mut metrics = EntityMetrics { entities_by_bird: vec![BTreeSet::new(); truth.bird_species.len()], ..EntityMetrics::default() };
    for (_, host) in outcome.all_hosts() {
        for call_outcome in &host.entities.call_outcomes {
            let Some(entity) = call_outcome.bird_entity_id else { continue };
            let Some(record) = records.get(&call_outcome.call_id) else { continue };
            let scripted_call = localized_call_of.get(&(record.call_id, call_outcome.cluster_index)).copied().or_else(|| {
                record.detections.iter().find_map(|key| outcome.metrics.call_index_by_detection.get(key).copied())
            });
            let Some(scripted_call) = scripted_call else { continue };
            let bird = truth.bird_of_call.get(scripted_call).copied().flatten();
            metrics.calls_scored += 1;
            sources_by_entity.entry(entity).or_default().insert(bird);
            if let Some(bird) = bird {
                metrics.entities_by_bird[bird].insert(entity);
            }
        }
    }
    let mut last_update_label: BTreeMap<u64, Option<u32>> = BTreeMap::new();
    let mut pinpointed_entities: BTreeSet<u64> = BTreeSet::new();
    for record in outcome.host_log_records() {
        if let FieldMessage::BirdEntityUpdate(update) = record {
            if update.state == BirdEntityState::Pinpointed {
                pinpointed_entities.insert(update.bird_entity_id);
            }
            if update.state != BirdEntityState::Expired {
                last_update_label.insert(update.bird_entity_id, top_label(&update.species_belief));
            }
        }
    }
    for (entity, sources) in &sources_by_entity {
        let birds: Vec<usize> = sources.iter().flatten().copied().collect();
        match birds.as_slice() {
            [] => metrics.echo_promotions += 1,
            [bird] => {
                let shown = last_update_label.get(entity).copied().flatten();
                if shown.is_some_and(|label| label != truth.bird_species[*bird]) {
                    metrics.label_swaps += 1;
                }
            }
            _ => metrics.merged_entities += 1,
        }
    }
    for entities in &metrics.entities_by_bird {
        metrics.duplicate_entities += entities.len().saturating_sub(1);
        if entities.iter().any(|entity| pinpointed_entities.contains(entity)) {
            metrics.pinpointed_birds += 1;
        }
    }
    metrics
}

/// No duplicates, no merges (swaps between birds), no wrong labels, no echo promotions.
pub fn entity_invariants(metrics: &EntityMetrics) -> InvariantResult {
    ensure(metrics.duplicate_entities == 0, || format!("{} duplicate entities: {:?}", metrics.duplicate_entities, metrics.entities_by_bird))?;
    ensure(metrics.merged_entities == 0, || format!("{} entities merged several birds: {:?}", metrics.merged_entities, metrics.entities_by_bird))?;
    ensure(metrics.label_swaps == 0, || format!("{} entities show another bird's species", metrics.label_swaps))?;
    ensure(metrics.echo_promotions == 0, || format!("{} echoes promoted to entities", metrics.echo_promotions))
}

impl EntityMetrics {
    pub fn summary(&self) -> String {
        format!(
            "entities/bird {:?}, dup {}, merged {}, swaps {}, echo {}, pinpointed birds {}",
            self.entities_by_bird.iter().map(BTreeSet::len).collect::<Vec<_>>(),
            self.duplicate_entities,
            self.merged_entities,
            self.label_swaps,
            self.echo_promotions,
            self.pinpointed_birds
        )
    }
}
