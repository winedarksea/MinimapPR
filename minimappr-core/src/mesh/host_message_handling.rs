//! Host reactions to inbound messages: joins, heartbeats, detections and
//! idempotent queued uploads, chirp reports, readiness answers, clips,
//! status changes, catch-up requests, chirp direction cues and rival hosts.

use crate::protocol::detection_messages::DetectionEvent;
use crate::protocol::mesh_messages::{EventLogAppend, HostAnnouncement};
use crate::protocol::session_messages::JoinAccepted;
use crate::protocol::wire_codes::NodeStatusChangeCode;
use crate::protocol::{FieldMessage, NodeId, SessionId};

use super::host_state_machine::{FieldHostStateMachine, HostAction};
use super::{MeshMilliseconds, MessageRecipient, OutboundMessage};

impl FieldHostStateMachine {
    fn accept_detection(&mut self, detection: &DetectionEvent, now_ms: MeshMilliseconds) -> Vec<HostAction> {
        if !self.event_log.append_detection_if_new(self.host_term, detection) {
            return Vec::new();
        }
        self.sync.note_detection(detection.node_id, now_ms, &self.config);
        let present = self.present_nodes(now_ms);
        let mut calls = std::mem::take(&mut self.calls);
        let messages = calls.on_detection(detection, &self.call_context(&present), now_ms);
        self.calls = calls;
        messages.into_iter().map(HostAction::Send).collect()
    }

    /// A rival host's announcement: the higher term wins; on a tie the
    /// earlier join order wins.
    fn yields_to(&self, announcement: &HostAnnouncement) -> bool {
        if announcement.host_node_id == self.host_node_id {
            return false;
        }
        if announcement.host_term != self.host_term {
            return announcement.host_term > self.host_term;
        }
        let order = |node: NodeId| self.membership.members.get(&node).map_or(u32::MAX, |m| m.join_order);
        order(announcement.host_node_id) < order(self.host_node_id)
    }

    pub(super) fn on_message(&mut self, from: NodeId, message: FieldMessage, now_ms: MeshMilliseconds) -> Vec<HostAction> {
        self.membership.note_heard(from, now_ms);
        match message {
            FieldMessage::JoinRequest(request) => {
                let self_path = self.self_path_length_for(&request.node_descriptor);
                if let Some(record) = self.membership.admit(from, request.node_descriptor, self_path) {
                    self.append(&FieldMessage::SessionMemberRecord(record));
                }
                self.membership.note_heard(from, now_ms);
                let slot = self.membership.members.get(&from).map_or(0, |m| m.join_order as u16);
                vec![HostAction::Send(OutboundMessage::control(
                    MessageRecipient::Node(from),
                    FieldMessage::JoinAccepted(JoinAccepted {
                        session_id: SessionId(request.session_id.0),
                        assigned_node_id: from,
                        host_node_id: self.host_node_id,
                        chirp_slot_index: slot,
                        event_log_head_sequence: self.event_log.head_sequence(),
                    }),
                ))]
            }
            FieldMessage::Heartbeat(heartbeat) => {
                if self.membership.apply_heartbeat(&heartbeat).capture_overrun_increased {
                    self.sync.note_clock_invalid(heartbeat.node_id);
                }
                Vec::new()
            }
            FieldMessage::DetectionEvent(detection) => self.accept_detection(&detection, now_ms),
            FieldMessage::QueuedDetectionUpload(upload) => upload
                .detections
                .iter()
                .flat_map(|detection| self.accept_detection(detection, now_ms))
                .collect(),
            FieldMessage::ChirpArrivalReport(report) => {
                self.sync.on_report(report);
                Vec::new()
            }
            FieldMessage::ReadinessAnswer(answer) => {
                self.calls.on_readiness_answer(&answer);
                Vec::new()
            }
            FieldMessage::ClipPayload(payload) => {
                let present = self.present_nodes(now_ms);
                let mut calls = std::mem::take(&mut self.calls);
                calls.on_clip_payload(payload, &self.call_context(&present));
                self.calls = calls;
                Vec::new()
            }
            FieldMessage::NodeStatusChange(change) => {
                match change.change {
                    NodeStatusChangeCode::InputRouteChanged
                    | NodeStatusChangeCode::CaptureGapReported
                    | NodeStatusChangeCode::AudioSessionInterrupted => {
                        if let Some(member) = self.membership.members.get_mut(&change.node_id) {
                            member.last_capture_gap_device_sample = Some(change.device_sample_index);
                        }
                        self.sync.note_clock_invalid(change.node_id)
                    }
                    NodeStatusChangeCode::OutputRouteChangedListenOnly => {
                        if let Some(member) = self.membership.members.get_mut(&change.node_id) {
                            member.listen_only = true;
                        }
                    }
                    NodeStatusChangeCode::ThermalMicOnly => {
                        if let Some(member) = self.membership.members.get_mut(&change.node_id) {
                            member.mic_only = true;
                        }
                    }
                    _ => {}
                }
                self.append(&FieldMessage::NodeStatusChange(change));
                Vec::new()
            }
            FieldMessage::EventLogCatchUpRequest(request) => {
                self.catch_up_requests_served += 1;
                let entries = self
                    .event_log
                    .entries_since(request.since_sequence, self.config.event_log.catch_up_batch_entries)
                    .to_vec();
                vec![HostAction::Send(OutboundMessage::control(
                    MessageRecipient::Node(request.node_id),
                    FieldMessage::EventLogAppend(EventLogAppend {
                        host_node_id: self.host_node_id,
                        host_term: self.host_term,
                        host_head_sequence: self.event_log.head_sequence(),
                        entries,
                    }),
                ))]
            }
            FieldMessage::ChirpDirectionReport(report) => {
                let fixes = self.sync.gps_fixes(&self.membership);
                self.sync.engine.apply_direction_report(&report, &fixes, &self.config.island_sync);
                Vec::new()
            }
            FieldMessage::HostAnnouncement(announcement) if self.yields_to(&announcement) => {
                self.stepped_down = true;
                vec![HostAction::StepDown { successor: Some(announcement.host_node_id) }]
            }
            _ => Vec::new(),
        }
    }
}
