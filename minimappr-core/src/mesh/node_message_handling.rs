//! Node reactions to inbound messages: join acceptance, host announcements
//! and handover, event-log appends (which also acknowledge detections and
//! deliver clock models, rotas and membership), chirp schedules, readiness
//! queries and clip requests.

use crate::protocol::detection_messages::{ClipPayload, ReadinessAnswer};
use crate::protocol::mesh_messages::EventLogEntry;
use crate::protocol::{FieldMessage, NodeId};

use super::clock_mapping::NodeClockMapping;
use super::config::DEVICE_SAMPLE_RATE_HZ;
use super::event_log::decode_entry;
use super::host_membership::descriptor_is_host_eligible;
use super::node_state_machine::{FieldNodeStateMachine, NodeAction, NodeAudioWindowSource, NodeUserNotice};
use super::{MeshMilliseconds, MessageRecipient, OutboundMessage};

const SAMPLES_PER_MILLISECOND: u64 = (DEVICE_SAMPLE_RATE_HZ as u64) / 1_000;

impl FieldNodeStateMachine {
    /// Own detections in entries a newer term discarded go back in the outbox.
    fn requeue_discarded(&mut self, discarded: &[EventLogEntry]) {
        for entry in discarded {
            if let Some(FieldMessage::DetectionEvent(detection)) = decode_entry(entry) {
                if detection.node_id == self.node_id {
                    self.outbox.push(detection);
                }
            }
        }
    }

    fn apply_logged_record(&mut self, record: FieldMessage) {
        match record {
            FieldMessage::DetectionEvent(detection) if detection.node_id == self.node_id => {
                self.outbox.acknowledge((detection.node_id, detection.local_sequence));
            }
            FieldMessage::ClockModel(model) => {
                if let Some(estimate) = model.node_clocks.iter().find(|clock| clock.node_id == self.node_id) {
                    let postdates_invalidation = self
                        .invalidated_after_sample
                        .is_none_or(|marker| estimate.anchor_node_sample_index > marker);
                    if postdates_invalidation {
                        self.own_clock = Some(NodeClockMapping::from_wire(model.clock_epoch, estimate));
                    }
                }
            }
            FieldMessage::ClassificationRota(rota) => {
                let mine = rota.clumps.iter().any(|clump| clump.member_node_ids.contains(&self.node_id));
                let replaces_mine = self.rota.as_ref().is_some_and(|current| current.island_id == rota.island_id);
                if mine || replaces_mine {
                    self.rota = Some(rota);
                }
            }
            FieldMessage::SessionMemberRecord(record) => {
                if descriptor_is_host_eligible(&record.node_descriptor) && !self.succession.contains(&record.node_id) {
                    self.succession.push(record.node_id);
                }
                if record.node_id == self.node_id {
                    self.joined = true;
                }
            }
            _ => {}
        }
    }

    pub(super) fn on_message(
        &mut self,
        from: NodeId,
        message: FieldMessage,
        now_ms: MeshMilliseconds,
        audio: &dyn NodeAudioWindowSource,
    ) -> Vec<NodeAction> {
        self.link.note_message_from(from, now_ms);
        match message {
            FieldMessage::JoinAccepted(accepted) => {
                self.joined = true;
                self.link.host_head_sequence = self.link.host_head_sequence.max(accepted.event_log_head_sequence);
                Vec::new()
            }
            FieldMessage::HostAnnouncement(announcement) => {
                let order = |node: NodeId| self.succession.iter().position(|n| *n == node).unwrap_or(usize::MAX);
                let announcer_wins_tie = order(announcement.host_node_id) < order(self.link.host_node_id);
                let outcome = self.link.on_announcement(&announcement, announcer_wins_tie, now_ms);
                let mut actions = Vec::new();
                if !outcome.accepted {
                    return actions;
                }
                if announcement.event_log_head_sequence < self.mirror.head_sequence() && outcome.host_changed {
                    let discarded = self.mirror.truncate_to(announcement.event_log_head_sequence);
                    self.requeue_discarded(&discarded);
                }
                if outcome.rejoined_from_solo {
                    self.counters.rejoins += 1;
                    actions.push(NodeAction::UserNotice(NodeUserNotice::Rejoined));
                }
                if outcome.host_changed {
                    self.joined = false;
                }
                if announcement.event_log_head_sequence > self.mirror.head_sequence() {
                    actions.extend(self.request_catch_up(now_ms));
                }
                actions
            }
            FieldMessage::HostHandover(handover) if handover.to_node_id == self.node_id => {
                self.counters.takeovers += 1;
                self.link.adopt_self_as_host(self.node_id, handover.host_term, now_ms);
                vec![NodeAction::BecomeHost { host_term: handover.host_term, reason: handover.reason }]
            }
            FieldMessage::EventLogAppend(append) => {
                if append.host_term < self.link.host_term {
                    return Vec::new();
                }
                self.link.host_head_sequence = self.link.host_head_sequence.max(append.host_head_sequence);
                let outcome = self.mirror.apply(&append);
                self.requeue_discarded(&outcome.discarded_entries);
                for record in outcome.applied_records {
                    self.apply_logged_record(record);
                }
                match outcome.catch_up_since {
                    Some(_) => {
                        self.last_catch_up_request_ms = None;
                        self.request_catch_up(now_ms)
                    }
                    None => Vec::new(),
                }
            }
            FieldMessage::ChirpSchedule(schedule) => {
                let round_start_ms = schedule.round_start_host_sample_index / SAMPLES_PER_MILLISECOND;
                let mut actions = Vec::new();
                let mut listen_until_ms = round_start_ms;
                for slot in &schedule.slots {
                    let slot_ms = round_start_ms + slot.slot_offset_samples as u64 / SAMPLES_PER_MILLISECOND;
                    listen_until_ms = listen_until_ms.max(slot_ms + self.config.scheduler.slot_duration_ms);
                    if slot.emitter_node_id == self.node_id && !self.listen_only {
                        actions.push(NodeAction::EmitChirp {
                            round_id: schedule.round_id,
                            emit_at_ms: slot_ms,
                            chirp_low_hz: slot.chirp_low_hz,
                            chirp_high_hz: slot.chirp_high_hz,
                            chirp_code_id: slot.chirp_code_id,
                        });
                    }
                }
                actions.push(NodeAction::ListenForChirpRound {
                    round_id: schedule.round_id,
                    listen_until_ms: listen_until_ms + self.config.scheduler.round_listen_tail_ms,
                });
                actions
            }
            FieldMessage::ReadinessQuery(query) => {
                let local = audio.readiness(query.centre_device_sample_index, query.half_window_samples, query.band);
                self.counters.readiness_answers_sent += 1;
                vec![self.to_host(FieldMessage::ReadinessAnswer(ReadinessAnswer {
                    query_id: query.query_id,
                    node_id: self.node_id,
                    clock_in_sync: self.clock_in_sync(),
                    audio_still_buffered: local.audio_still_buffered,
                    speech_free: local.speech_free,
                    handling_noise_free: local.handling_noise_free,
                    in_band_level_over_floor_db: local.in_band_level_over_floor_db,
                }))]
            }
            FieldMessage::ClipRequest(request) => {
                let Some(clip) = audio.encode_clip(
                    request.window_first_device_sample_index,
                    request.window_sample_count,
                    request.decimation_filter_id,
                ) else {
                    return Vec::new();
                };
                self.counters.clips_sent += 1;
                vec![NodeAction::Send(OutboundMessage::bulk(
                    MessageRecipient::Node(self.link.host_node_id),
                    FieldMessage::ClipPayload(ClipPayload {
                        request_id: request.request_id,
                        node_id: self.node_id,
                        clip_container: clip.clip_container,
                        onsets: clip.onsets,
                        sensor_frames: clip.sensor_frames,
                    }),
                ))]
            }
            _ => Vec::new(),
        }
    }
}
