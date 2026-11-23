//! Host call pipeline: detection → event matcher → readiness query to the
//! island → capped, budgeted clip fetch → localization → entity update.
//!
//! Every clip remembers the node's clock epoch at request time; a clip
//! whose node was invalidated (route change, gap, merge) before it is used
//! is discarded and counted, never solved against the new clock model.
//! Late clips are abandoned, not retried. Nothing is dropped silently.

use std::collections::{BTreeMap, VecDeque};

use crate::protocol::detection_messages::{ClipPayload, ClipRequest, DetectionEvent, ReadinessAnswer, ReadinessQuery};
use crate::protocol::wire_codes::{FrequencyBandHz, SpeciesScore};
use crate::protocol::{FieldMessage, NodeId};

use super::call_attribution::attribute_window_clusters;
use super::clip_budget::IslandClipBudget;
use crate::onset_clustering::attribution_cascade::LocoRerunBudget;
use super::config::DEVICE_SAMPLE_RATE_HZ;
use super::event_matcher::{merge_species, EventMatcher, MatchDecision};
use super::readiness_selection::{select_clip_nodes, ReadinessCandidate};
use super::{MeshMilliseconds, MessageRecipient, OutboundMessage};

pub use super::call_pipeline_types::*;

/// Unanswered readiness queries are re-sent once at timeout / this.
const READINESS_RETRY_FRACTION_DIVISOR: u64 = 2;

#[derive(Debug, Clone, Default)]
pub struct CallPipeline {
    pending: VecDeque<PendingCall>,
    pub matcher: EventMatcher,
    budgets: BTreeMap<u32, IslandClipBudget>,
    next_message_id: u32,
    pub completed: Vec<LocalizedCallRecord>,
    pub counters: CallPipelineCounters,
    /// S5 LOCO reruns per window (a call's clusters share one budget).
    loco_budget: LocoRerunBudget,
}

fn island_aperture_m(ctx: &CallContext<'_>, island_id: u32) -> f64 {
    let Some(layout) = ctx.engine.islands.get(&island_id).and_then(|island| island.layout.as_ref()) else {
        return 0.0;
    };
    let positions: Vec<[f64; 3]> = layout.local_positions_m.values().copied().collect();
    let mut widest: f64 = 0.0;
    for (i, a) in positions.iter().enumerate() {
        for b in &positions[i + 1..] {
            widest = widest.max(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt());
        }
    }
    widest
}

fn walked_at_call(payload: &ClipPayload, threshold: f32) -> bool {
    if payload.sensor_frames.is_empty() {
        return false;
    }
    let mean_square: f32 = payload
        .sensor_frames
        .iter()
        .map(|frame| frame.linear_acceleration_m_per_s2.iter().map(|a| a * a).sum::<f32>())
        .sum::<f32>()
        / payload.sensor_frames.len() as f32;
    mean_square.sqrt() > threshold
}

impl CallPipeline {
    /// A successor host numbers readiness queries and clip requests from
    /// here, so a late answer or clip addressed to the dead host's id space
    /// never matches one of its own requests.
    pub fn reserve_message_ids_from(&mut self, first_message_id: u32) {
        self.next_message_id = self.next_message_id.max(first_message_id);
    }

    fn allocate_message_id(&mut self) -> u32 {
        self.next_message_id = self.next_message_id.wrapping_add(1);
        self.next_message_id
    }

    /// A detection the host just logged. Returns readiness queries.
    pub fn on_detection(&mut self, detection: &DetectionEvent, ctx: &CallContext<'_>, now_ms: MeshMilliseconds) -> Vec<OutboundMessage> {
        let config = &ctx.config.clip_fetch;
        let Some(island) = ctx.engine.island_of(detection.node_id) else { return Vec::new() };
        let Some(mapping) = ctx.engine.clock_mapping(detection.node_id) else { return Vec::new() };
        let island_id = island.island_id;
        let centre_sample = detection.window_first_device_sample_index as f64 + detection.window_sample_count as f64 / 2.0;
        let centre_reference_s = mapping.reference_seconds_from_node_sample(centre_sample);
        let half_window_s = config.clip_half_window_s
            + island_aperture_m(ctx, island_id) / ctx.config.island_sync.sound_speed_mps
            + config.inter_node_delay_margin_s;
        let decision = ctx.config.matcher.clone();
        match self.matcher.decide(island_id, centre_reference_s, half_window_s, &detection.species_scores, &decision) {
            MatchDecision::BelowThreshold => Vec::new(),
            MatchDecision::MergedInto { call_id } => {
                if let Some(call) = self.pending.iter_mut().find(|call| call.call_id == call_id) {
                    merge_species(&mut call.species_belief, &detection.species_scores);
                    call.detections.push((detection.node_id, detection.local_sequence));
                }
                self.counters.detections_merged_into_calls += 1;
                Vec::new()
            }
            MatchDecision::NewCall { call_id } => {
                if !ctx.clip_fetch_enabled {
                    self.counters.calls_without_clip_fetch += 1;
                    self.complete_unlocalized(call_id, island_id, detection, centre_reference_s, now_ms, "clip fetch disabled on this transport");
                    return Vec::new();
                }
                if self.pending.len() >= ctx.config.matcher.pending_call_capacity {
                    if let Some(dropped) = self.pending.pop_front() {
                        self.counters.calls_dropped_backpressure += 1;
                        self.completed.push(LocalizedCallRecord::abandoned(&dropped, now_ms, "dropped oldest pending call"));
                    }
                }
                let band = detection.stage1_active_band.unwrap_or(FrequencyBandHz {
                    low_hz: config.default_query_band_hz[0],
                    high_hz: config.default_query_band_hz[1],
                });
                let mut queries = BTreeMap::new();
                let mut messages = Vec::new();
                for node in ctx.present.iter().copied() {
                    if node == detection.node_id || !ctx.engine.share_clock_domain(node, detection.node_id) {
                        continue;
                    }
                    let Some(node_mapping) = ctx.engine.clock_mapping(node) else { continue };
                    let query_id = self.allocate_message_id();
                    let query = ReadinessQuery {
                        query_id,
                        target_node_id: node,
                        centre_device_sample_index: node_mapping.node_sample_index_from_reference_seconds(centre_reference_s),
                        half_window_samples: (half_window_s * DEVICE_SAMPLE_RATE_HZ) as u32,
                        band,
                    };
                    queries.insert(node, query.clone());
                    self.counters.readiness_queries_sent += 1;
                    messages.push(OutboundMessage::control(MessageRecipient::Node(node), FieldMessage::ReadinessQuery(query)));
                }
                let mut species_belief: Vec<SpeciesScore> = Vec::new();
                merge_species(&mut species_belief, &detection.species_scores);
                self.pending.push_back(PendingCall {
                    call_id,
                    island_id,
                    island_epoch: island.tracker_epoch,
                    classifying_node: detection.node_id,
                    centre_reference_s,
                    half_window_s,
                    received_ms: now_ms,
                    species_belief,
                    detections: vec![(detection.node_id, detection.local_sequence)],
                    stage: PendingStage::AwaitingReadiness {
                        deadline_ms: now_ms + config.readiness_answer_timeout_ms,
                        retry_at_ms: Some(now_ms + config.readiness_answer_timeout_ms / READINESS_RETRY_FRACTION_DIVISOR),
                        queries,
                        answers: BTreeMap::new(),
                    },
                });
                messages
            }
        }
    }

    fn complete_unlocalized(
        &mut self,
        call_id: u64,
        island_id: u32,
        detection: &DetectionEvent,
        centre_reference_s: f64,
        now_ms: MeshMilliseconds,
        reason: &str,
    ) {
        let mut species_belief = Vec::new();
        merge_species(&mut species_belief, &detection.species_scores);
        self.completed.push(LocalizedCallRecord {
            call_id,
            island_id,
            completed_ms: now_ms,
            received_ms: now_ms,
            centre_reference_s,
            classifying_node: detection.node_id,
            species_belief,
            failure: Some(reason.to_string()),
            ..LocalizedCallRecord::default()
        });
    }

    pub fn on_readiness_answer(&mut self, answer: &ReadinessAnswer) {
        self.counters.readiness_answers_received += 1;
        for call in self.pending.iter_mut() {
            if let PendingStage::AwaitingReadiness { queries, answers, .. } = &mut call.stage {
                if queries.get(&answer.node_id).map(|query| query.query_id) == Some(answer.query_id) {
                    answers.insert(answer.node_id, answer.clone());
                    return;
                }
            }
        }
    }

    pub fn on_clip_payload(&mut self, payload: ClipPayload, ctx: &CallContext<'_>) {
        for call in self.pending.iter_mut() {
            if let PendingStage::FetchingClips { requests, received, .. } = &mut call.stage {
                let Some(request) = requests.get(&payload.node_id) else { continue };
                if request.request_id != payload.request_id {
                    continue;
                }
                if ctx.engine.node_clock_epoch(payload.node_id) != request.clock_epoch_at_request {
                    self.counters.stale_clips_discarded += 1;
                } else {
                    self.counters.clips_received += 1;
                    received.insert(payload.node_id, payload);
                }
                return;
            }
        }
        self.counters.late_clips_discarded += 1;
    }

    fn request_clips(&mut self, index: usize, ctx: &CallContext<'_>, now_ms: MeshMilliseconds) -> Vec<OutboundMessage> {
        let config = &ctx.config.clip_fetch;
        let call = &self.pending[index];
        let PendingStage::AwaitingReadiness { answers, .. } = &call.stage else { return Vec::new() };
        let position_of = |node: NodeId| {
            ctx.engine
                .node_position_for_solve(node, call.received_ms, false, &ctx.config.island_sync)
                .map(|estimate| estimate.position_m)
        };
        let classifier = ReadinessCandidate {
            node_id: call.classifying_node,
            in_band_level_over_floor_db: f32::INFINITY,
            position_m: position_of(call.classifying_node),
        };
        let ready: Vec<ReadinessCandidate> = answers
            .values()
            .filter(|a| a.clock_in_sync && a.audio_still_buffered && a.speech_free && a.handling_noise_free)
            .filter(|a| ctx.engine.share_clock_domain(a.node_id, call.classifying_node))
            .map(|a| ReadinessCandidate {
                node_id: a.node_id,
                in_band_level_over_floor_db: a.in_band_level_over_floor_db,
                position_m: position_of(a.node_id),
            })
            .collect();
        self.counters.readiness_not_ready += (answers.len() - ready.len()) as u64;
        let selection = select_clip_nodes(&classifier, &ready, config.readiness_clip_cap, config);
        self.counters.readiness_excluded_by_cap += selection.excluded_by_cap.len() as u64;
        let island_id = call.island_id;
        let budget = self
            .budgets
            .entry(island_id)
            .or_insert_with(|| IslandClipBudget::new(config.island_clip_budget_burst, config.island_clip_budget_refill_per_s, now_ms));
        let wanted = selection.selected.len();
        if wanted < config.minimum_clips_for_fix || budget.available(now_ms) < config.minimum_clips_for_fix {
            let reason = if wanted < config.minimum_clips_for_fix { "too few ready nodes" } else { "island clip budget exhausted" };
            if wanted >= config.minimum_clips_for_fix {
                self.counters.calls_skipped_for_clip_budget += 1;
            }
            let call = self.pending.remove(index).expect("index valid");
            self.completed.push(LocalizedCallRecord::abandoned(&call, now_ms, reason));
            return Vec::new();
        }
        let granted = budget.take(wanted, now_ms);
        self.counters.clips_denied_by_budget += (wanted - granted) as u64;
        let excluded_by_cap = selection.excluded_by_cap.len();
        let (centre, half) = (call.centre_reference_s, call.half_window_s);
        let mut requests = BTreeMap::new();
        let mut messages = Vec::new();
        for node in selection.selected.into_iter().take(granted) {
            let Some(mapping) = ctx.engine.clock_mapping(node) else { continue };
            let request_id = self.allocate_message_id();
            let first = mapping.node_sample_index_from_reference_seconds(centre - half);
            let last = mapping.node_sample_index_from_reference_seconds(centre + half);
            requests.insert(node, RequestedClip { request_id, clock_epoch_at_request: mapping.clock_epoch });
            self.counters.clip_requests_sent += 1;
            messages.push(OutboundMessage::control(
                MessageRecipient::Node(node),
                FieldMessage::ClipRequest(ClipRequest {
                    request_id,
                    target_node_id: node,
                    window_first_device_sample_index: first,
                    window_sample_count: last.saturating_sub(first) as u32,
                    decimation_filter_id: config.decimation_filter_id,
                    abandon_after_milliseconds: config.clip_abandon_after_ms as u32,
                }),
            ));
        }
        let call = &mut self.pending[index];
        call.stage = PendingStage::FetchingClips {
            deadline_ms: now_ms + config.clip_abandon_after_ms,
            requests,
            received: BTreeMap::new(),
            readiness_excluded_by_cap: excluded_by_cap,
        };
        messages
    }

    fn localize(&mut self, index: usize, ctx: &CallContext<'_>, localizer: &dyn ClipSetLocalizer, now_ms: MeshMilliseconds) {
        let call = self.pending.remove(index).expect("index valid");
        let PendingStage::FetchingClips { requests, received, readiness_excluded_by_cap, .. } = &call.stage else { return };
        let mut clips = Vec::new();
        for (node, payload) in received {
            let Some(mapping) = ctx.engine.clock_mapping(*node) else {
                self.counters.stale_clips_discarded += 1;
                continue;
            };
            if mapping.clock_epoch != requests[node].clock_epoch_at_request || !ctx.engine.share_clock_domain(*node, call.classifying_node) {
                self.counters.stale_clips_discarded += 1;
                continue;
            }
            let walked = walked_at_call(payload, ctx.config.clip_fetch.walking_linear_acceleration_threshold_m_per_s2);
            let Some(position) = ctx.engine.node_position_for_solve(*node, call.received_ms, walked, &ctx.config.island_sync) else { continue };
            clips.push(ClipLocalizationInput {
                node_id: *node,
                payload: payload.clone(),
                clock: mapping,
                position,
                walked_at_call: walked,
                island_id: ctx.engine.island_of(*node).map_or(0, |island| island.island_id),
                island_epoch: ctx.engine.island_of(*node).map_or(0, |island| island.tracker_epoch),
                clock_epoch_at_request: requests[node].clock_epoch_at_request,
            });
        }
        let mut record = LocalizedCallRecord::from_call(&call, now_ms, *readiness_excluded_by_cap);
        record.requested_nodes = requests.keys().copied().collect();
        record.clip_nodes = clips.iter().map(|clip| clip.node_id).collect();
        record.walked_nodes = clips.iter().filter(|clip| clip.walked_at_call).map(|clip| clip.node_id).collect();
        record.chirality_confidence = ctx.engine.chirality_confidence(call.classifying_node);
        record.chirality_locked = ctx.engine.chirality_locked(call.classifying_node);
        record.anchor_uncertainty = ctx.engine.anchor_uncertainty(call.classifying_node);
        record.clip_domains = clips
            .iter()
            .map(|clip| (clip.node_id, clip.island_id, clip.island_epoch, clip.clock_epoch_at_request, clip.clock.clock_epoch))
            .collect();
        if clips.len() < ctx.config.clip_fetch.minimum_clips_for_fix {
            record.failure = Some(format!("only {} usable clips", clips.len()));
            self.counters.calls_abandoned += 1;
        } else {
            let request = CallLocalizationRequest {
                call_id: call.call_id,
                island_id: call.island_id,
                classifying_node: call.classifying_node,
                centre_reference_s: call.centre_reference_s,
                half_window_s: call.half_window_s,
                clips,
                chirality_confidence: record.chirality_confidence,
                chirality_locked: record.chirality_locked,
                anchor_uncertainty: record.anchor_uncertainty,
            };
            match localizer.localize_clusters(&request) {
                Ok(mut clusters) if !clusters.is_empty() => {
                    self.counters.calls_localized += 1;
                    attribute_window_clusters(&request, &mut clusters, localizer, &ctx.config.attribution, &mut self.loco_budget);
                    record.fix = Some(clusters[0].fix.clone());
                    record.clusters = clusters;
                }
                Ok(_) => {
                    self.counters.calls_abandoned += 1;
                    record.failure = Some("no source cluster".to_string());
                }
                Err(reason) => {
                    self.counters.calls_abandoned += 1;
                    record.failure = Some(reason);
                }
            }
        }
        self.completed.push(record);
    }

    /// Advance timeouts. Returns messages and how many calls completed.
    pub fn tick(&mut self, ctx: &CallContext<'_>, localizer: &dyn ClipSetLocalizer, now_ms: MeshMilliseconds) -> Vec<OutboundMessage> {
        let mut messages = Vec::new();
        let mut index = 0;
        while index < self.pending.len() {
            let before = self.pending.len();
            match &self.pending[index].stage {
                PendingStage::AwaitingReadiness { deadline_ms, queries, answers, retry_at_ms } => {
                    if now_ms >= *deadline_ms || answers.len() == queries.len() {
                        messages.extend(self.request_clips(index, ctx, now_ms));
                    } else if retry_at_ms.is_some_and(|retry| now_ms >= retry) {
                        let unanswered: Vec<ReadinessQuery> = queries
                            .iter()
                            .filter(|(node, _)| !answers.contains_key(node))
                            .map(|(_, query)| query.clone())
                            .collect();
                        if let PendingStage::AwaitingReadiness { retry_at_ms, .. } = &mut self.pending[index].stage {
                            *retry_at_ms = None;
                        }
                        self.counters.readiness_queries_sent += unanswered.len() as u64;
                        messages.extend(unanswered.into_iter().map(|query| {
                            OutboundMessage::control(MessageRecipient::Node(query.target_node_id), FieldMessage::ReadinessQuery(query))
                        }));
                    }
                }
                PendingStage::FetchingClips { deadline_ms, requests, received, .. } => {
                    if now_ms >= *deadline_ms || received.len() == requests.len() {
                        self.localize(index, ctx, localizer, now_ms);
                    }
                }
            }
            if self.pending.len() == before {
                index += 1;
            }
        }
        messages
    }

    pub fn pending_call_count(&self) -> usize {
        self.pending.len()
    }
}
