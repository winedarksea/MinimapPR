//! Carries out node and host actions on one device: frames for other phones
//! become LAN envelopes, frames between the host and its own node loop back
//! in memory (same split as the mesh simulator, `message_routing`), chirp
//! emissions become playback requests tied to the listen round.

use std::collections::VecDeque;

use minimappr_core::mesh::host_state_machine::{HostAction, HostInput};
use minimappr_core::mesh::message_routing::delivery_for;
use minimappr_core::mesh::node_state_machine::{DeviceStatusChange, NodeAction, NodeInput};
use minimappr_core::mesh::{MessageRecipient, OutboundMessage, TrafficClass};
use minimappr_core::protocol::{FieldMessage, NodeId};

use crate::chirp_round_listener::{PendingChirpRound, PendingOwnEmission};
use crate::lan_envelope::LanPayload;
use crate::mesh_device::{DeviceState, FieldDeviceAction, FieldDeviceNotice};

/// Host ↔ own-node exchanges per step; a reply storm beyond this is a bug,
/// and stopping keeps one bad step from hanging the shell's tick loop.
const MAXIMUM_LOOPBACK_DELIVERIES_PER_STEP: usize = 256;
const NANOSECONDS_PER_MILLISECOND: i64 = 1_000_000;

pub(crate) enum LoopbackDelivery {
    ToNode { from: NodeId, message: FieldMessage },
    ToHost { from: NodeId, message: FieldMessage },
}

impl DeviceState {
    fn send_frame(&mut self, outbound: &OutboundMessage, recipient: Option<u32>, actions: &mut Vec<FieldDeviceAction>) {
        let Ok(frame) = outbound.message.encode_frame() else {
            self.counters.encode_failures += 1;
            return;
        };
        let envelope = self.envelope(recipient, LanPayload::FieldFrame(frame));
        actions.push(FieldDeviceAction::SendEnvelope {
            recipient_node_id: recipient,
            envelope,
            is_bulk_traffic: outbound.traffic_class == TrafficClass::Bulk,
        });
    }

    fn route_from_node(&mut self, outbound: OutboundMessage, queue: &mut VecDeque<LoopbackDelivery>, actions: &mut Vec<FieldDeviceAction>) {
        let own = self.node_id;
        match outbound.recipient {
            MessageRecipient::Node(node) if node == own => {
                if self.host.is_some() && delivery_for(&outbound.message).to_host {
                    queue.push_back(LoopbackDelivery::ToHost { from: own, message: outbound.message });
                }
            }
            MessageRecipient::Node(node) => self.send_frame(&outbound, Some(node.0), actions),
            MessageRecipient::AllNodes => self.send_frame(&outbound, None, actions),
        }
    }

    fn route_from_host(&mut self, outbound: OutboundMessage, queue: &mut VecDeque<LoopbackDelivery>, actions: &mut Vec<FieldDeviceAction>) {
        let own = self.node_id;
        let delivery = delivery_for(&outbound.message);
        let loop_to_self = match outbound.recipient {
            MessageRecipient::Node(node) => node == own,
            MessageRecipient::AllNodes => true,
        };
        if loop_to_self {
            if delivery.to_node && self.node.is_some() {
                queue.push_back(LoopbackDelivery::ToNode { from: own, message: outbound.message.clone() });
            }
            if delivery.to_host {
                queue.push_back(LoopbackDelivery::ToHost { from: own, message: outbound.message.clone() });
            }
        }
        match outbound.recipient {
            MessageRecipient::Node(node) if node != own => self.send_frame(&outbound, Some(node.0), actions),
            MessageRecipient::AllNodes => self.send_frame(&outbound, None, actions),
            MessageRecipient::Node(_) => {}
        }
    }

    pub(crate) fn step_node(&mut self, session_ms: u64, input: NodeInput) -> Vec<NodeAction> {
        let Some(node) = self.node.as_mut() else { return Vec::new() };
        let device_sample_index_now = self.buffer.next_device_sample_index();
        node.handle(session_ms, device_sample_index_now, input, &self.buffer)
    }

    pub(crate) fn step_host(&mut self, session_ms: u64, input: HostInput) -> Vec<HostAction> {
        let Some(host) = self.host.as_mut() else { return Vec::new() };
        let actions = host.handle(session_ms, input, &self.localizer);
        let completed = host.sync.rounds_completed;
        if completed != self.absorbed_rounds_completed {
            self.absorbed_rounds_completed = completed;
            let diagnostics = host.sync.last_round_diagnostics.clone();
            self.peer_board.absorb(&diagnostics, self.node_id);
        }
        actions
    }

    /// Apply actions and every loopback delivery they cause.
    pub(crate) fn apply_actions(
        &mut self,
        session_ms: u64,
        node_actions: Vec<NodeAction>,
        host_actions: Vec<HostAction>,
    ) -> Vec<FieldDeviceAction> {
        let mut actions = Vec::new();
        let mut queue = VecDeque::new();
        self.apply_node_actions(node_actions, &mut queue, &mut actions);
        self.apply_host_actions(session_ms, host_actions, &mut queue, &mut actions);
        let mut deliveries = 0;
        while let Some(delivery) = queue.pop_front() {
            deliveries += 1;
            if deliveries > MAXIMUM_LOOPBACK_DELIVERIES_PER_STEP {
                self.counters.loopback_overflows += 1;
                break;
            }
            match delivery {
                LoopbackDelivery::ToNode { from, message } => {
                    let produced = self.step_node(session_ms, NodeInput::MessageReceived { from, message });
                    self.apply_node_actions(produced, &mut queue, &mut actions);
                }
                LoopbackDelivery::ToHost { from, message } => {
                    let produced = self.step_host(session_ms, HostInput::MessageReceived { from, message });
                    self.apply_host_actions(session_ms, produced, &mut queue, &mut actions);
                }
            }
        }
        actions
    }

    fn apply_node_actions(&mut self, node_actions: Vec<NodeAction>, queue: &mut VecDeque<LoopbackDelivery>, actions: &mut Vec<FieldDeviceAction>) {
        // EmitChirp comes before its round's ListenForChirpRound: tie them up after.
        let mut emissions = Vec::new();
        for action in node_actions {
            match action {
                NodeAction::Send(outbound) => self.route_from_node(outbound, queue, actions),
                NodeAction::EmitChirp { round_id, emit_at_ms, .. } => emissions.push((round_id, emit_at_ms)),
                NodeAction::ListenForChirpRound { round_id, listen_until_ms, slots, .. } => {
                    self.listener.pending.insert(
                        round_id,
                        PendingChirpRound { round_id, slots, listen_until_session_ms: listen_until_ms, own_emissions: Vec::new() },
                    );
                }
                NodeAction::BecomeHost { .. } => {
                    // LAN pass: the joiner runs can_host=false; a takeover
                    // request is surfaced, never acted on.
                    actions.push(FieldDeviceAction::UserNotice { notice: FieldDeviceNotice::HostTakeoverRequested });
                }
                NodeAction::UserNotice(notice) => actions.push(FieldDeviceAction::UserNotice { notice: notice.into() }),
            }
        }
        for (round_id, emit_at_ms) in emissions {
            self.request_emission(round_id, emit_at_ms, actions);
        }
    }

    fn request_emission(&mut self, round_id: u32, emit_at_ms: u64, actions: &mut Vec<FieldDeviceAction>) {
        let own = self.node_id;
        let Some(round) = self.listener.pending.get_mut(&round_id) else { return };
        let taken: Vec<usize> = round.own_emissions.iter().map(|emission| emission.slot_index).collect();
        let Some(slot_index) = round
            .slots
            .iter()
            .enumerate()
            .position(|(index, slot)| slot.emitter_node_id == own && slot.slot_start_ms == emit_at_ms && !taken.contains(&index))
        else {
            return;
        };
        let Some(emit_at_local_ns) = self.alignment.local_ns_for_session(emit_at_ms as i64 * NANOSECONDS_PER_MILLISECOND) else {
            return;
        };
        self.next_emission_id += 1;
        let emission_id = self.next_emission_id;
        round.own_emissions.push(PendingOwnEmission { emission_id, slot_index, outcome: None });
        self.counters.emissions_requested += 1;
        actions.push(FieldDeviceAction::EmitChirp {
            emission_id,
            round_id,
            emit_at_local_ns,
            pcm16_stereo_little_endian: self.chirp_playback_pcm.clone(),
        });
    }

    fn apply_host_actions(
        &mut self,
        session_ms: u64,
        host_actions: Vec<HostAction>,
        queue: &mut VecDeque<LoopbackDelivery>,
        actions: &mut Vec<FieldDeviceAction>,
    ) {
        for action in host_actions {
            match action {
                HostAction::Send(outbound) => self.route_from_host(outbound, queue, actions),
                // LAN pass: the WebSocket link is the only transport.
                HostAction::SwitchTransport(_) => {}
                HostAction::ShedLocalClassification => {
                    let produced = self.step_node(session_ms, NodeInput::DeviceStatus(DeviceStatusChange::ShedClassification));
                    self.apply_node_actions(produced, queue, actions);
                }
                HostAction::StepDown { .. } => {
                    self.host = None;
                    actions.push(FieldDeviceAction::HostRoleChanged { is_host: false });
                }
            }
        }
    }
}
