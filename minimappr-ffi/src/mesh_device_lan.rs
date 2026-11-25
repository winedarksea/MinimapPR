//! LAN control on a device: Hello admission, session-clock probes and
//! echoes, inbound field frames, and the status snapshot.
//!
//! Admission (unsealed dev transport): the host answers every Hello with its
//! own, so a discovering joiner (zero session id) can adopt it, but accepts
//! field frames only from senders whose Hello carried the right session id.
//! A joiner accepts frames only from the host that answered it.

use minimappr_core::mesh::host_state_machine::HostInput;
use minimappr_core::mesh::message_routing::delivery_for;
use minimappr_core::mesh::node_state_machine::NodeInput;
use minimappr_core::mesh::session_clock_alignment::ClockProbeExchange;
use minimappr_core::protocol::{decode_field_message_frame, DecodedFrame, NodeId};

use crate::chirp_round_listener::{PlannedRound, SlotSearchResult};
use crate::error::FieldFfiError;
use crate::lan_envelope::{LanEnvelope, LanPayload, LAN_PROTOCOL_VERSION, SESSION_ID_LENGTH_BYTES};
use crate::mesh_device::{DeviceState, FieldDeviceAction, FieldDeviceNotice};
use crate::sync_status::{FieldDeviceRole, FieldLocalRoundStatus, FieldSlotArrivalStatus, FieldSyncStatus};

const NANOSECONDS_PER_MILLISECOND: i64 = 1_000_000;
const MICROSECONDS_PER_SECOND: f64 = 1e6;
const HELLO_RETRY_INTERVAL_NS: i64 = 1_000 * NANOSECONDS_PER_MILLISECOND;
/// Fast probing until the window has enough exchanges, then a slow cadence
/// that keeps the lowest-RTT pick fresh as the two monotonic clocks drift.
const FAST_PROBE_INTERVAL_NS: i64 = 200 * NANOSECONDS_PER_MILLISECOND;
const SLOW_PROBE_INTERVAL_NS: i64 = 2_000 * NANOSECONDS_PER_MILLISECOND;
const FAST_PROBE_EXCHANGE_COUNT: u64 = 10;

pub(crate) fn local_round_status(planned: &PlannedRound, results: &[SlotSearchResult], own_chirp_inaudible: bool) -> FieldLocalRoundStatus {
    FieldLocalRoundStatus {
        round_id: planned.round_id,
        own_emission_played: planned.any_own_emission_played,
        own_chirp_inaudible,
        unsearchable_slot_count: planned.unsearchable_slot_count as u32,
        slots: results
            .iter()
            .map(|result| FieldSlotArrivalStatus {
                emitter_node_id: result.emitter_node_id.0,
                is_self: result.is_self,
                detected: result.detection.is_some(),
                snr_db: result.detection.map(|d| d.snr_db),
                doppler_skew_samples: result.detection.map(|d| d.doppler_skew_samples),
                peak_fraction: result.detection.map(|d| d.peak_fraction),
            })
            .collect(),
    }
}

impl DeviceState {
    fn hello_payload(&self) -> LanPayload {
        LanPayload::Hello {
            session_id: self.session_id.unwrap_or([0; SESSION_ID_LENGTH_BYTES]),
            node_id: self.node_id.0,
            protocol_version: LAN_PROTOCOL_VERSION,
            can_host: self.role == FieldDeviceRole::Host,
        }
    }

    /// Joiner: Hello until answered, then clock probes to the host.
    pub(crate) fn lan_control_tick(&mut self, local_ns: i64) -> Vec<FieldDeviceAction> {
        let mut actions = Vec::new();
        if self.role != FieldDeviceRole::Joiner || !self.link_up {
            return actions;
        }
        let Some(host) = self.host_node_id else {
            if self.last_hello_local_ns.is_none_or(|last| local_ns - last >= HELLO_RETRY_INTERVAL_NS) {
                self.last_hello_local_ns = Some(local_ns);
                let envelope = self.envelope(None, self.hello_payload());
                actions.push(FieldDeviceAction::SendEnvelope { recipient_node_id: None, envelope, is_bulk_traffic: false });
            }
            return actions;
        };
        let interval = if self.alignment.accepted_exchange_count < FAST_PROBE_EXCHANGE_COUNT { FAST_PROBE_INTERVAL_NS } else { SLOW_PROBE_INTERVAL_NS };
        if self.last_probe_local_ns.is_none_or(|last| local_ns - last >= interval) {
            self.last_probe_local_ns = Some(local_ns);
            self.next_probe_id = self.next_probe_id.wrapping_add(1);
            let payload = LanPayload::ClockProbe { probe_id: self.next_probe_id, probe_sent_local_ns: local_ns };
            let envelope = self.envelope(Some(host.0), payload);
            actions.push(FieldDeviceAction::SendEnvelope { recipient_node_id: Some(host.0), envelope, is_bulk_traffic: false });
        }
        actions
    }

    pub(crate) fn receive_envelope_locked(&mut self, local_ns: i64, envelope: LanEnvelope) -> Result<Vec<FieldDeviceAction>, FieldFfiError> {
        let sender = envelope.sender_node_id;
        if sender == self.node_id.0 {
            return Ok(Vec::new());
        }
        let addressed_here = envelope.recipient_node_id == self.node_id.0 || envelope.recipient_node_id == NodeId::BROADCAST.0;
        if !addressed_here {
            return Ok(Vec::new());
        }
        let mut actions = Vec::new();
        match envelope.payload {
            LanPayload::Hello { session_id, node_id, .. } if node_id == sender => match self.role {
                FieldDeviceRole::Host => {
                    let own = self.session_id.expect("hosts always have a session id");
                    if session_id == own {
                        self.admitted_peers.insert(sender);
                    } else if session_id != [0; SESSION_ID_LENGTH_BYTES] {
                        self.counters.rejected_hellos += 1;
                        return Ok(actions);
                    }
                    let envelope = self.envelope(Some(sender), self.hello_payload());
                    actions.push(FieldDeviceAction::SendEnvelope { recipient_node_id: Some(sender), envelope, is_bulk_traffic: false });
                }
                FieldDeviceRole::Joiner => {
                    if self.session_id.is_some_and(|expected| expected != session_id) {
                        actions.push(FieldDeviceAction::UserNotice { notice: FieldDeviceNotice::SessionIdMismatch });
                        return Ok(actions);
                    }
                    let first_answer = self.session_id.is_none();
                    self.session_id = Some(session_id);
                    if self.host_node_id.is_none() {
                        self.host_node_id = Some(NodeId(sender));
                    }
                    if first_answer {
                        // Re-introduce with the adopted id so the host admits us.
                        let envelope = self.envelope(Some(sender), self.hello_payload());
                        actions.push(FieldDeviceAction::SendEnvelope { recipient_node_id: Some(sender), envelope, is_bulk_traffic: false });
                    }
                    actions.extend(self.lan_control_tick(local_ns));
                }
            },
            LanPayload::Hello { .. } => {}
            LanPayload::ClockProbe { probe_id, probe_sent_local_ns } => {
                if self.role == FieldDeviceRole::Host {
                    let payload = LanPayload::ClockEcho { probe_id, probe_sent_local_ns, probe_received_host_ns: local_ns, echo_sent_host_ns: local_ns };
                    let envelope = self.envelope(Some(sender), payload);
                    actions.push(FieldDeviceAction::SendEnvelope { recipient_node_id: Some(sender), envelope, is_bulk_traffic: false });
                }
            }
            LanPayload::ClockEcho { probe_sent_local_ns, probe_received_host_ns, echo_sent_host_ns, .. } => {
                if self.host_node_id == Some(NodeId(sender)) {
                    self.alignment.add_exchange(ClockProbeExchange {
                        probe_sent_local_ns,
                        probe_received_host_ns,
                        echo_sent_host_ns,
                        echo_received_local_ns: local_ns,
                    });
                    self.ensure_joiner_node(local_ns, &mut actions);
                }
            }
            LanPayload::FieldFrame(frame) => {
                let admitted = match self.role {
                    FieldDeviceRole::Host => self.admitted_peers.contains(&sender),
                    FieldDeviceRole::Joiner => self.host_node_id == Some(NodeId(sender)),
                };
                if !admitted {
                    self.counters.dropped_unadmitted_frames += 1;
                    return Ok(actions);
                }
                actions.extend(self.deliver_field_frame(local_ns, NodeId(sender), &frame)?);
            }
        }
        Ok(actions)
    }

    fn deliver_field_frame(&mut self, local_ns: i64, from: NodeId, frame: &[u8]) -> Result<Vec<FieldDeviceAction>, FieldFfiError> {
        let decoded = decode_field_message_frame(frame).map_err(|error| FieldFfiError::ProtocolFrame { detail: error.to_string() })?;
        let DecodedFrame::Known { message, .. } = decoded else { return Ok(Vec::new()) };
        let Some(session_ms) = self.session_ms(local_ns) else { return Ok(Vec::new()) };
        let delivery = delivery_for(&message);
        let host_actions = if delivery.to_host && self.host.is_some() {
            self.step_host(session_ms, HostInput::MessageReceived { from, message: message.clone() })
        } else {
            Vec::new()
        };
        let node_actions = if delivery.to_node { self.step_node(session_ms, NodeInput::MessageReceived { from, message }) } else { Vec::new() };
        Ok(self.apply_actions(session_ms, node_actions, host_actions))
    }

    pub(crate) fn sync_status(&self) -> FieldSyncStatus {
        let own_clock = self.node.as_ref().and_then(|node| node.own_clock_mapping());
        let host = self.host.as_ref();
        FieldSyncStatus {
            role: self.role,
            node_id: self.node_id.0,
            host_node_id: self.host_node_id.map(|node| node.0),
            session_id_hex: self.session_id.map(|id| id.iter().map(|byte| format!("{byte:02x}")).collect()).unwrap_or_default(),
            joined: self.joined(),
            session_clock_aligned: self.alignment.is_aligned(),
            session_clock_offset_ms: self.alignment.applied_offset_ns().map(|ns| ns as f64 / NANOSECONDS_PER_MILLISECOND as f64),
            session_clock_best_round_trip_ms: self.alignment.best_round_trip_ns().map(|ns| ns as f64 / NANOSECONDS_PER_MILLISECOND as f64),
            clock_probe_exchanges_accepted: self.alignment.accepted_exchange_count,
            own_clock_in_sync: self.node.as_ref().is_some_and(|node| node.clock_in_sync()),
            own_clock_offset_us: own_clock.map(|clock| clock.offset_s * MICROSECONDS_PER_SECOND),
            own_clock_rate_ppm: own_clock.map(|clock| clock.rate_offset * MICROSECONDS_PER_SECOND),
            capture_anchor_count: self.capture_anchor.anchor_count() as u32,
            capture_rate_offset_ppm: self.capture_anchor.rate_offset_ppm(),
            capture_anchor_residual_us: self.capture_anchor.residual_rms_ns().map(|ns| ns / 1e3),
            capture_gap_count: self.buffer.capture_gap_count(),
            rounds_started: host.map_or(0, |host| host.sync.rounds_started.len() as u32),
            rounds_completed: host.map_or(0, |host| host.sync.rounds_completed),
            rounds_heard: self.counters.rounds_heard,
            emissions_requested: self.counters.emissions_requested,
            emissions_played: self.counters.emissions_played,
            emissions_skipped: self.counters.emissions_skipped,
            muted_node_ids: host.map_or_else(Vec::new, |host| host.sync.muted_nodes.iter().map(|node| node.0).collect()),
            last_local_round: self.last_local_round.clone(),
            peers: self.peer_board.snapshot(),
            transport_unsealed: true,
        }
    }
}
