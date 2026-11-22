//! The host role: transport policy, membership, the session event log and
//! its replication, chirp rounds and islands, the call pipeline, entity
//! updates, the classification rota and thermal relief (handover online,
//! shedding on the hotspot). Message handling lives in
//! [`super::host_message_handling`].

use std::collections::BTreeMap;

use crate::protocol::detection_messages::BirdEntityUpdate;
use crate::protocol::mesh_messages::{ClassificationRota, EventLogAppend, HostAnnouncement, HostHandover, TransportNotice};
use crate::protocol::session_messages::NodeDescriptor;
use crate::protocol::wire_codes::{BirdEntityState, SessionChangeReasonCode, SessionTransportCode};
use crate::protocol::{FieldMessage, NodeId};

use super::call_pipeline::{CallContext, CallPipeline, ClipSetLocalizer, LocalizedCallRecord};
use super::classification_rota::clumps_from_layout;
use super::config::{MeshConfig, DEVICE_SAMPLE_RATE_HZ};
use super::event_log::SessionEventLog;
use super::host_membership::SessionMembership;
use super::host_restoration::restore_from_mirror;
use super::host_sync_pipeline::{HostSyncPipeline, SyncRecord};
use super::host_transport_policy::{CellLinkQuality, HostTransportPolicy, SessionTransportPreference, TransportDecision};
use super::{MeshMilliseconds, MessageRecipient, OutboundMessage};
use crate::protocol::mesh_messages::EventLogEntry;

/// Points on the 2σ likelihood ellipse sent with a pinpointed entity.
const CONTOUR_POINT_COUNT: usize = 12;
const CONTOUR_SIGMA_MULTIPLIER: f64 = 2.0;
/// A successor's call ids start at `term << 32`.
const CALL_ID_TERM_SHIFT_BITS: u32 = 32;

/// What a node knows when it becomes host.
#[derive(Debug, Clone, Copy)]
pub struct HostTakeover<'mirror> {
    pub host_node_id: NodeId,
    pub mirror_entries: &'mirror [EventLogEntry],
    pub host_term: u32,
    pub reason: SessionChangeReasonCode,
    pub preference: SessionTransportPreference,
    pub cell: CellLinkQuality,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HostInput {
    Tick,
    MessageReceived { from: NodeId, message: FieldMessage },
    CellLinkChanged(CellLinkQuality),
    RelayRefusedSessionCap,
    OwnThermalHeadroom(f32),
}

#[derive(Debug, Clone, PartialEq)]
pub enum HostAction {
    Send(OutboundMessage),
    SwitchTransport(TransportDecision),
    /// Hot host on the hotspot path: stop classifying and drawing the map.
    ShedLocalClassification,
    /// This device is no longer host (handover sent, or a newer term seen).
    StepDown { successor: Option<NodeId> },
}

#[derive(Debug, Clone)]
pub struct FieldHostStateMachine {
    pub host_node_id: NodeId,
    pub host_term: u32,
    pub config: MeshConfig,
    pub membership: SessionMembership,
    pub event_log: SessionEventLog,
    pub sync: HostSyncPipeline,
    pub calls: CallPipeline,
    pub transport: HostTransportPolicy,
    pub(super) next_bird_entity_id: u64,
    last_announcement_ms: Option<MeshMilliseconds>,
    published_call_records: usize,
    pub stepped_down: bool,
    pub shed_classification: bool,
    rota_epoch: u32,
    published_rotas: BTreeMap<u32, ClassificationRota>,
    pub catch_up_requests_served: u64,
    own_thermal_headroom: f32,
}

impl FieldHostStateMachine {
    fn with_parts(host_node_id: NodeId, host_term: u32, config: MeshConfig, transport: HostTransportPolicy) -> Self {
        Self {
            host_node_id,
            host_term,
            config,
            membership: SessionMembership::default(),
            event_log: SessionEventLog::default(),
            sync: HostSyncPipeline::default(),
            calls: CallPipeline::default(),
            transport,
            next_bird_entity_id: 0,
            last_announcement_ms: None,
            published_call_records: 0,
            stepped_down: false,
            shed_classification: false,
            rota_epoch: 0,
            published_rotas: BTreeMap::new(),
            catch_up_requests_served: 0,
            own_thermal_headroom: 1.0,
        }
    }

    pub fn self_path_length_for(&self, descriptor: &NodeDescriptor) -> f64 {
        let sync = &self.config.island_sync;
        sync.self_path_length_by_platform_m
            .get(&descriptor.platform_name)
            .copied()
            .unwrap_or(sync.default_self_path_length_m)
    }

    /// Start a session on this device.
    pub fn new_session(
        host_node_id: NodeId,
        host_descriptor: NodeDescriptor,
        config: MeshConfig,
        preference: SessionTransportPreference,
        cell: CellLinkQuality,
        now_ms: MeshMilliseconds,
    ) -> (Self, Vec<HostAction>) {
        let (transport, decision) = HostTransportPolicy::new(preference, cell);
        let mut host = Self::with_parts(host_node_id, 1, config, transport);
        let self_path = host.self_path_length_for(&host_descriptor);
        if let Some(record) = host.membership.admit(host_node_id, host_descriptor, self_path) {
            host.append(&FieldMessage::SessionMemberRecord(record));
        }
        host.membership.note_heard(host_node_id, now_ms);
        let mut actions = vec![HostAction::SwitchTransport(decision)];
        actions.extend(host.announce_transport(decision, now_ms));
        (host, actions)
    }

    /// Take over from a mirror (handover or host death).
    pub fn take_over(takeover: HostTakeover<'_>, config: MeshConfig, now_ms: MeshMilliseconds) -> (Self, Vec<HostAction>) {
        let HostTakeover { host_node_id, mirror_entries, host_term, reason, preference, cell } = takeover;
        let restored = restore_from_mirror(mirror_entries);
        let inherited = restored.active_transport.unwrap_or(SessionTransportCode::CloudRelay);
        let transport = HostTransportPolicy::inherit(preference, cell, inherited);
        let mut host = Self::with_parts(host_node_id, host_term.max(restored.highest_host_term + 1), config, transport);
        host.event_log = restored.event_log;
        host.membership = restored.membership;
        host.next_bird_entity_id = restored.highest_bird_entity_id;
        // Call ids stay unique across terms (records outlive their host).
        host.calls.matcher.reserve_call_ids_through(u64::from(host.host_term) << CALL_ID_TERM_SHIFT_BITS);
        for node in host.membership.members.keys().copied().collect::<Vec<_>>() {
            host.membership.note_heard(node, now_ms);
        }
        let mut actions = Vec::new();
        let announcement = host.announcement(reason);
        host.append(&FieldMessage::HostAnnouncement(announcement));
        actions.push(HostAction::SwitchTransport(TransportDecision {
            transport: host.transport.active(),
            reason,
            clip_fetch_enabled: host.transport.clip_fetch_enabled(),
        }));
        actions.extend(host.broadcast_announcement(reason, now_ms));
        (host, actions)
    }

    pub(super) fn append(&mut self, record: &FieldMessage) -> Option<u64> {
        self.event_log.append(self.host_term, record)
    }

    fn announcement(&self, reason: SessionChangeReasonCode) -> HostAnnouncement {
        HostAnnouncement {
            host_node_id: self.host_node_id,
            host_term: self.host_term,
            event_log_head_sequence: self.event_log.head_sequence(),
            transport: self.transport.active(),
            reason,
        }
    }

    fn broadcast_announcement(&mut self, reason: SessionChangeReasonCode, now_ms: MeshMilliseconds) -> Vec<HostAction> {
        self.last_announcement_ms = Some(now_ms);
        vec![HostAction::Send(OutboundMessage::control(
            MessageRecipient::AllNodes,
            FieldMessage::HostAnnouncement(self.announcement(reason)),
        ))]
    }

    pub(super) fn announce_transport(&mut self, decision: TransportDecision, now_ms: MeshMilliseconds) -> Vec<HostAction> {
        let notice = TransportNotice {
            host_node_id: self.host_node_id,
            transport: decision.transport,
            reason: decision.reason,
            clip_fetch_enabled: decision.clip_fetch_enabled,
        };
        self.append(&FieldMessage::TransportNotice(notice));
        self.broadcast_announcement(decision.reason, now_ms)
    }

    pub(super) fn present_nodes(&self, now_ms: MeshMilliseconds) -> Vec<NodeId> {
        self.membership.present_nodes(now_ms, self.config.liveness.node_presence_timeout_ms)
    }

    pub(super) fn call_context<'host>(&'host self, present: &'host [NodeId]) -> CallContext<'host> {
        CallContext {
            engine: &self.sync.engine,
            membership: &self.membership,
            present,
            config: &self.config,
            clip_fetch_enabled: self.transport.clip_fetch_enabled(),
        }
    }

    fn log_sync_records(&mut self, records: Vec<SyncRecord>) {
        let mut layouts_changed = false;
        for record in records {
            let message = match record {
                SyncRecord::Islands(islands) => {
                    for island in islands {
                        self.append(&FieldMessage::IslandMembership(island));
                    }
                    continue;
                }
                SyncRecord::Layout(layout) => {
                    layouts_changed = true;
                    FieldMessage::LayoutModel(layout)
                }
                SyncRecord::Clock(clock) => FieldMessage::ClockModel(clock),
                SyncRecord::NodeMuted(node) => {
                    if let Some(member) = self.membership.members.get_mut(&node) {
                        member.muted = true;
                    }
                    FieldMessage::NodeStatusChange(crate::protocol::mesh_messages::NodeStatusChange {
                        node_id: node,
                        local_sequence: 0,
                        change: crate::protocol::wire_codes::NodeStatusChangeCode::MediaVolumeMuted,
                        device_sample_index: 0,
                    })
                }
            };
            self.append(&message);
        }
        if layouts_changed && self.config.rota.enabled {
            self.refresh_rotas();
        }
    }

    fn refresh_rotas(&mut self) {
        let islands: Vec<_> = self.sync.engine.islands.values().cloned().collect();
        for island in islands {
            let Some(layout) = &island.layout else { continue };
            let classifiers = island
                .members
                .iter()
                .copied()
                .filter(|node| self.membership.members.get(node).is_some_and(|m| !m.mic_only))
                .collect();
            let clumps = clumps_from_layout(&layout.local_positions_m, &classifiers, self.config.rota.clump_radius_m);
            let unchanged = self.published_rotas.get(&island.island_id).is_some_and(|r| r.clumps == clumps);
            if unchanged {
                continue;
            }
            self.rota_epoch += 1;
            let rota = ClassificationRota {
                island_id: island.island_id,
                rota_epoch: self.rota_epoch,
                slot_duration_milliseconds: self.config.rota.slot_duration_ms,
                clumps,
            };
            self.published_rotas.insert(island.island_id, rota.clone());
            self.append(&FieldMessage::ClassificationRota(rota));
        }
    }

    fn entity_update_for(&mut self, record: &LocalizedCallRecord) -> BirdEntityUpdate {
        self.next_bird_entity_id += 1;
        let contour = record.fix.as_ref().map_or_else(Vec::new, |fix| {
            let [[a, b, _], [_, d, _], _] = fix.covariance_m2;
            let half_trace = 0.5 * (a + d);
            let spread = (0.25 * (a - d).powi(2) + b * b).sqrt();
            let (major, minor) = ((half_trace + spread).max(0.0).sqrt(), (half_trace - spread).max(0.0).sqrt());
            let angle = 0.5 * (2.0 * b).atan2(a - d);
            (0..CONTOUR_POINT_COUNT)
                .map(|k| {
                    let t = std::f64::consts::TAU * k as f64 / CONTOUR_POINT_COUNT as f64;
                    let (u, v) = (CONTOUR_SIGMA_MULTIPLIER * major * t.cos(), CONTOUR_SIGMA_MULTIPLIER * minor * t.sin());
                    [
                        (fix.position_m[0] + u * angle.cos() - v * angle.sin()) as f32,
                        (fix.position_m[1] + u * angle.sin() + v * angle.cos()) as f32,
                    ]
                })
                .collect()
        });
        BirdEntityUpdate {
            bird_entity_id: self.next_bird_entity_id,
            event_log_sequence: self.event_log.head_sequence() + 1,
            state: if record.fix.is_some() { BirdEntityState::Pinpointed } else { BirdEntityState::HeardNearby },
            island_id: record.island_id,
            position_east_north_up_m: record.fix.as_ref().map(|fix| fix.position_m.map(|v| v as f32)),
            bearing: None,
            likelihood_contour_east_north_m: contour,
            species_belief: record.species_belief.clone(),
            just_sang: true,
            last_heard_reference_sample_index: (record.centre_reference_s.max(0.0) * DEVICE_SAMPLE_RATE_HZ) as u64,
        }
    }

    fn thermal_relief(&mut self, now_ms: MeshMilliseconds) -> Vec<HostAction> {
        if self.own_thermal_headroom >= self.config.thermal.host_relief_below_headroom {
            return Vec::new();
        }
        if self.transport.active() != SessionTransportCode::CloudRelay {
            if self.shed_classification {
                return Vec::new();
            }
            self.shed_classification = true;
            return vec![HostAction::ShedLocalClassification];
        }
        let present = self.present_nodes(now_ms);
        let successor = self
            .membership
            .succession_order()
            .into_iter()
            .filter(|node| *node != self.host_node_id && present.contains(node))
            .filter_map(|node| Some((node, self.membership.members.get(&node)?.thermal_headroom)))
            .filter(|(_, headroom)| *headroom >= self.config.thermal.successor_minimum_headroom)
            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
            .map(|(node, _)| node);
        let Some(successor) = successor else { return Vec::new() };
        let handover = HostHandover {
            from_host_node_id: self.host_node_id,
            to_node_id: successor,
            host_term: self.host_term + 1,
            reason: SessionChangeReasonCode::HostThermalHeadroomLow,
            event_log_head_sequence: self.event_log.head_sequence() + 1,
        };
        self.append(&FieldMessage::HostHandover(handover.clone()));
        self.stepped_down = true;
        let mut actions = self.flush_log_broadcast();
        actions.push(HostAction::Send(OutboundMessage::control(
            MessageRecipient::AllNodes,
            FieldMessage::HostHandover(handover),
        )));
        actions.push(HostAction::StepDown { successor: Some(successor) });
        actions
    }

    fn flush_log_broadcast(&mut self) -> Vec<HostAction> {
        let entries = self.event_log.take_unbroadcast();
        if entries.is_empty() {
            return Vec::new();
        }
        vec![HostAction::Send(OutboundMessage::control(
            MessageRecipient::AllNodes,
            FieldMessage::EventLogAppend(EventLogAppend {
                host_node_id: self.host_node_id,
                host_term: self.host_term,
                host_head_sequence: self.event_log.head_sequence(),
                entries,
            }),
        ))]
    }

    fn on_tick(&mut self, now_ms: MeshMilliseconds, localizer: &dyn ClipSetLocalizer) -> Vec<HostAction> {
        let mut actions = Vec::new();
        let present = self.present_nodes(now_ms);
        let (messages, records) = self.sync.tick(&present, &self.membership, now_ms, &self.config);
        actions.extend(messages.into_iter().map(HostAction::Send));
        self.log_sync_records(records);
        let ctx_present = present.clone();
        let mut calls = std::mem::take(&mut self.calls);
        let messages = calls.tick(&self.call_context(&ctx_present), localizer, now_ms);
        self.calls = calls;
        actions.extend(messages.into_iter().map(HostAction::Send));
        let fresh: Vec<LocalizedCallRecord> = self.calls.completed[self.published_call_records..].to_vec();
        self.published_call_records = self.calls.completed.len();
        for record in &fresh {
            let update = self.entity_update_for(record);
            self.append(&FieldMessage::BirdEntityUpdate(update));
        }
        actions.extend(self.thermal_relief(now_ms));
        if self.stepped_down {
            return actions;
        }
        actions.extend(self.flush_log_broadcast());
        let announcement_due = self
            .last_announcement_ms
            .is_none_or(|last| now_ms.saturating_sub(last) >= self.config.liveness.host_announcement_interval_ms);
        if announcement_due {
            actions.extend(self.broadcast_announcement(SessionChangeReasonCode::SessionStarted, now_ms));
        }
        actions
    }

    pub fn handle(&mut self, now_ms: MeshMilliseconds, input: HostInput, localizer: &dyn ClipSetLocalizer) -> Vec<HostAction> {
        if self.stepped_down {
            return Vec::new();
        }
        match input {
            HostInput::Tick => self.on_tick(now_ms, localizer),
            HostInput::MessageReceived { from, message } => self.on_message(from, message, now_ms),
            HostInput::CellLinkChanged(quality) => match self.transport.on_cell_link(quality) {
                Some(decision) => {
                    let mut actions = vec![HostAction::SwitchTransport(decision)];
                    actions.extend(self.announce_transport(decision, now_ms));
                    actions
                }
                None => Vec::new(),
            },
            HostInput::RelayRefusedSessionCap => match self.transport.on_relay_cap_reached() {
                Some(decision) => {
                    let mut actions = vec![HostAction::SwitchTransport(decision)];
                    actions.extend(self.announce_transport(decision, now_ms));
                    actions
                }
                None => Vec::new(),
            },
            HostInput::OwnThermalHeadroom(headroom) => {
                self.own_thermal_headroom = headroom;
                Vec::new()
            }
        }
    }
}
