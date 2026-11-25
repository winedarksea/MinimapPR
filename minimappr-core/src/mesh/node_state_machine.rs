//! The node role every phone (and array node) runs: join, heartbeats,
//! detections through the outbox, chirp slots and reports, readiness
//! answers and clips, event-log mirroring with the `since=N` cursor, solo
//! fallback and rejoin, takeover, Mic-only on low thermal headroom and the
//! classification rota. Inbound messages: [`super::node_message_handling`].

use crate::protocol::detection_messages::{ClipOnset, DetectionEvent, SensorFrameSample};
use crate::protocol::mesh_messages::{ClassificationRota, NodeStatusChange};
use crate::protocol::session_messages::{ChirpArrival, ChirpArrivalReport, GnssFix, Heartbeat, JoinRequest, NodeDescriptor};
use crate::protocol::wire_codes::{
    FrequencyBandHz, MotionState, NodeStatusChangeCode, PhonePlacement, SessionChangeReasonCode, SessionId, SessionRole,
    SpeciesScore,
};
use crate::protocol::{FieldMessage, NodeId};

use super::classification_rota;
use super::clock_mapping::NodeClockMapping;
use super::config::{MeshConfig, DEVICE_SAMPLE_RATE_HZ};
use super::detection_outbox::DetectionOutbox;
use super::event_log::MirroredEventLog;
use super::node_host_link::{HostLink, HostLinkEvent};
use super::{MeshMilliseconds, MessageRecipient, OutboundMessage};

/// Re-send a join request this often until accepted.
const JOIN_RETRY_INTERVAL_MS: MeshMilliseconds = 2_000;
/// Battery is not modelled by the state machine; heartbeats report full.
const UNREPORTED_BATTERY_FRACTION: f32 = 1.0;

#[derive(Debug, Clone, PartialEq)]
pub struct LocalClassification {
    pub window_first_device_sample_index: u64,
    pub window_sample_count: u32,
    pub classifier_model_id: u16,
    pub species_scores: Vec<SpeciesScore>,
    pub stage1_contrast_db: f32,
    pub stage1_active_band: Option<FrequencyBandHz>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalChirpRound {
    pub round_id: u32,
    pub own_emission_device_sample_index: Option<u64>,
    pub arrivals: Vec<ChirpArrival>,
    pub own_chirp_inaudible: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeviceStatusChange {
    InputRouteChanged,
    OutputRouteChanged { chirp_speaker_pinned: bool },
    CaptureGapReported,
    AudioSessionInterrupted,
    ThermalHeadroom(f32),
    Motion(MotionState),
    Placement(PhonePlacement),
    GnssFix(GnssFix),
    /// The host role on this device sheds classification (hot hotspot host).
    ShedClassification,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeInput {
    Tick,
    MessageReceived { from: NodeId, message: FieldMessage },
    TransportLinkChanged { link_up: bool },
    ClassifierFired(LocalClassification),
    ChirpRoundHeard(LocalChirpRound),
    DeviceStatus(DeviceStatusChange),
}

/// What the rolling buffer says about a window (shell-provided).
#[derive(Debug, Clone, PartialEq)]
pub struct LocalAudioReadiness {
    pub audio_still_buffered: bool,
    pub speech_free: bool,
    pub handling_noise_free: bool,
    pub in_band_level_over_floor_db: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncodedLocalClip {
    pub clip_container: Vec<u8>,
    pub onsets: Vec<ClipOnset>,
    pub sensor_frames: Vec<SensorFrameSample>,
}

/// The node's rolling buffer, Stage-1 history and clip encoder.
pub trait NodeAudioWindowSource {
    fn readiness(&self, centre_device_sample_index: u64, half_window_samples: u32, band: FrequencyBandHz) -> LocalAudioReadiness;
    fn encode_clip(&self, window_first_device_sample_index: u64, window_sample_count: u32, decimation_filter_id: u8) -> Option<EncodedLocalClip>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeUserNotice {
    MediaVolumeMuted,
    EnteredSolo,
    Rejoined,
    MicOnlyForThermal,
    ListenOnlyChirps,
    ClockInvalidated,
}

/// One emission the listener must search for, on the session clock. The
/// slot list is what lets a phone place a search window per emitter without
/// re-decoding the schedule (plan Stage B: the listener needs emitter, slot
/// start and band, not only the round's end).
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledChirpSlot {
    pub emitter_node_id: NodeId,
    pub slot_start_ms: MeshMilliseconds,
    pub band: FrequencyBandHz,
    pub chirp_code_id: u16,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeAction {
    Send(OutboundMessage),
    EmitChirp { round_id: u32, emit_at_ms: MeshMilliseconds, chirp_low_hz: f32, chirp_high_hz: f32, chirp_code_id: u16 },
    ListenForChirpRound {
        round_id: u32,
        round_start_ms: MeshMilliseconds,
        listen_until_ms: MeshMilliseconds,
        /// Every slot of the round in order, own emissions included (the
        /// opener repeats as the closing chirp).
        slots: Vec<ScheduledChirpSlot>,
    },
    BecomeHost { host_term: u32, reason: SessionChangeReasonCode },
    UserNotice(NodeUserNotice),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NodeCounters {
    pub detections_made: u64,
    pub readiness_answers_sent: u64,
    pub clips_sent: u64,
    pub catch_up_requests_sent: u64,
    pub queued_uploads_sent: u64,
    pub solo_entries: u64,
    pub rejoins: u64,
    pub takeovers: u64,
}

#[derive(Debug, Clone)]
pub struct FieldNodeStateMachine {
    pub node_id: NodeId,
    pub descriptor: NodeDescriptor,
    pub session_id: SessionId,
    pub config: MeshConfig,
    pub link: HostLink,
    pub mirror: MirroredEventLog,
    pub outbox: DetectionOutbox,
    pub counters: NodeCounters,
    pub(super) next_local_sequence: u64,
    pub(super) next_status_sequence: u64,
    pub(super) joined: bool,
    last_join_request_ms: Option<MeshMilliseconds>,
    last_heartbeat_ms: Option<MeshMilliseconds>,
    pub(super) last_catch_up_request_ms: Option<MeshMilliseconds>,
    pub mic_only: bool,
    pub listen_only: bool,
    pub(super) muted_notified: bool,
    thermal_headroom: f32,
    motion_state: MotionState,
    placement: PhonePlacement,
    overrun_count: u32,
    latest_fix: Option<GnssFix>,
    /// Own clock model from the latest logged `ClockModel`.
    pub(super) own_clock: Option<NodeClockMapping>,
    /// Device sample of the last local invalidation; older models are void.
    pub(super) invalidated_after_sample: Option<u64>,
    pub(super) rota: Option<ClassificationRota>,
    /// Host-eligible members in join order (from the mirror).
    pub(super) succession: Vec<NodeId>,
}

impl FieldNodeStateMachine {
    pub fn new(node_id: NodeId, descriptor: NodeDescriptor, session_id: SessionId, initial_host: NodeId, config: MeshConfig, now_ms: MeshMilliseconds) -> Self {
        Self {
            node_id,
            mic_only: !descriptor.can_classify,
            listen_only: !descriptor.can_emit_chirps,
            descriptor,
            session_id,
            outbox: DetectionOutbox::new(config.event_log.outbox_capacity),
            config,
            link: HostLink::new(initial_host, now_ms),
            mirror: MirroredEventLog::default(),
            counters: NodeCounters::default(),
            next_local_sequence: 0,
            next_status_sequence: 0,
            joined: false,
            last_join_request_ms: None,
            last_heartbeat_ms: None,
            last_catch_up_request_ms: None,
            muted_notified: false,
            thermal_headroom: 1.0,
            motion_state: MotionState::Still,
            placement: PhonePlacement::Handheld,
            overrun_count: 0,
            latest_fix: None,
            own_clock: None,
            invalidated_after_sample: None,
            rota: None,
            succession: Vec::new(),
        }
    }

    pub(super) fn to_host(&self, message: FieldMessage) -> NodeAction {
        NodeAction::Send(OutboundMessage::control(MessageRecipient::Node(self.link.host_node_id), message))
    }

    /// Whether the classifier should run on a Stage-1 trigger at this index.
    pub fn has_classification_duty(&self, trigger_device_sample_index: u64) -> bool {
        !self.mic_only
            && classification_rota::has_classification_duty(
                self.rota.as_ref(),
                self.node_id,
                self.own_clock.as_ref(),
                trigger_device_sample_index as f64 / DEVICE_SAMPLE_RATE_HZ,
            )
    }

    /// In sync once a published model postdates the last local invalidation.
    pub fn clock_in_sync(&self) -> bool {
        self.own_clock.is_some()
    }

    /// Own clock model from the latest logged `ClockModel` (status display).
    pub fn own_clock_mapping(&self) -> Option<NodeClockMapping> {
        self.own_clock.clone()
    }

    fn succession_rank(&self) -> Option<usize> {
        self.succession
            .iter()
            .filter(|node| **node != self.link.host_node_id)
            .position(|node| *node == self.node_id)
    }

    fn status_change(&mut self, change: NodeStatusChangeCode, device_sample_index: u64) -> Vec<NodeAction> {
        self.next_status_sequence += 1;
        let message = FieldMessage::NodeStatusChange(NodeStatusChange {
            node_id: self.node_id,
            local_sequence: self.next_status_sequence,
            change,
            device_sample_index,
        });
        if self.link.connected() { vec![self.to_host(message)] } else { Vec::new() }
    }

    fn invalidate_clock(&mut self, change: NodeStatusChangeCode, device_sample_index: u64) -> Vec<NodeAction> {
        self.own_clock = None;
        self.invalidated_after_sample = Some(device_sample_index);
        let mut actions = self.status_change(change, device_sample_index);
        actions.push(NodeAction::UserNotice(NodeUserNotice::ClockInvalidated));
        actions
    }

    fn on_device_status(&mut self, change: DeviceStatusChange, device_sample_index_now: u64) -> Vec<NodeAction> {
        match change {
            DeviceStatusChange::InputRouteChanged => self.invalidate_clock(NodeStatusChangeCode::InputRouteChanged, device_sample_index_now),
            DeviceStatusChange::AudioSessionInterrupted => {
                self.invalidate_clock(NodeStatusChangeCode::AudioSessionInterrupted, device_sample_index_now)
            }
            DeviceStatusChange::CaptureGapReported => {
                self.overrun_count += 1;
                self.invalidate_clock(NodeStatusChangeCode::CaptureGapReported, device_sample_index_now)
            }
            DeviceStatusChange::OutputRouteChanged { chirp_speaker_pinned: true } => {
                self.status_change(NodeStatusChangeCode::OutputRouteChangedSpeakerPinned, device_sample_index_now)
            }
            DeviceStatusChange::OutputRouteChanged { chirp_speaker_pinned: false } => {
                self.listen_only = true;
                let mut actions = self.status_change(NodeStatusChangeCode::OutputRouteChangedListenOnly, device_sample_index_now);
                actions.push(NodeAction::UserNotice(NodeUserNotice::ListenOnlyChirps));
                actions
            }
            DeviceStatusChange::ThermalHeadroom(headroom) => {
                self.thermal_headroom = headroom;
                if headroom < self.config.thermal.node_mic_only_below_headroom && !self.mic_only {
                    self.mic_only = true;
                    let mut actions = self.status_change(NodeStatusChangeCode::ThermalMicOnly, device_sample_index_now);
                    actions.push(NodeAction::UserNotice(NodeUserNotice::MicOnlyForThermal));
                    return actions;
                }
                Vec::new()
            }
            DeviceStatusChange::ShedClassification => {
                self.mic_only = true;
                self.status_change(NodeStatusChangeCode::ThermalMicOnly, device_sample_index_now)
            }
            DeviceStatusChange::Motion(state) => {
                self.motion_state = state;
                Vec::new()
            }
            DeviceStatusChange::Placement(placement) => {
                self.placement = placement;
                Vec::new()
            }
            DeviceStatusChange::GnssFix(fix) => {
                self.latest_fix = Some(fix);
                Vec::new()
            }
        }
    }

    fn on_classification(&mut self, classification: LocalClassification, now_ms: MeshMilliseconds) -> Vec<NodeAction> {
        self.next_local_sequence += 1;
        self.counters.detections_made += 1;
        let detection = DetectionEvent {
            node_id: self.node_id,
            local_sequence: self.next_local_sequence,
            window_first_device_sample_index: classification.window_first_device_sample_index,
            window_sample_count: classification.window_sample_count,
            classifier_model_id: classification.classifier_model_id,
            species_scores: classification.species_scores,
            stage1_contrast_db: classification.stage1_contrast_db,
            stage1_active_band: classification.stage1_active_band,
        };
        self.outbox.push(detection.clone());
        if !self.link.connected() {
            return Vec::new();
        }
        self.outbox.mark_sent((self.node_id, detection.local_sequence), now_ms);
        vec![self.to_host(FieldMessage::DetectionEvent(detection))]
    }

    fn on_chirp_round(&mut self, round: LocalChirpRound) -> Vec<NodeAction> {
        let mut actions = Vec::new();
        if round.own_chirp_inaudible && !self.muted_notified {
            self.muted_notified = true;
            actions.push(NodeAction::UserNotice(NodeUserNotice::MediaVolumeMuted));
        }
        if self.link.connected() {
            actions.push(self.to_host(FieldMessage::ChirpArrivalReport(ChirpArrivalReport {
                round_id: round.round_id,
                reporting_node_id: self.node_id,
                own_emission_device_sample_index: round.own_emission_device_sample_index,
                arrivals: round.arrivals,
                motion_state: self.motion_state,
                own_chirp_inaudible: round.own_chirp_inaudible,
            })));
        }
        actions
    }

    fn on_tick(&mut self, now_ms: MeshMilliseconds, device_sample_index_now: u64) -> Vec<NodeAction> {
        let mut actions = Vec::new();
        match self.link.tick(now_ms, self.succession_rank(), &self.config.liveness) {
            Some(HostLinkEvent::EnteredSolo) => {
                self.counters.solo_entries += 1;
                actions.push(NodeAction::UserNotice(NodeUserNotice::EnteredSolo));
            }
            Some(HostLinkEvent::TakeOver { host_term }) => {
                self.counters.takeovers += 1;
                self.link.adopt_self_as_host(self.node_id, host_term, now_ms);
                actions.push(NodeAction::BecomeHost { host_term, reason: SessionChangeReasonCode::HostSilent });
            }
            None => {}
        }
        if !self.link.connected() {
            return actions;
        }
        if !self.joined {
            if self.last_join_request_ms.is_none_or(|last| now_ms.saturating_sub(last) >= JOIN_RETRY_INTERVAL_MS) {
                self.last_join_request_ms = Some(now_ms);
                actions.push(self.to_host(FieldMessage::JoinRequest(JoinRequest {
                    session_id: self.session_id,
                    join_token: [0; 16],
                    requested_role: SessionRole::Joiner,
                    node_descriptor: self.descriptor.clone(),
                })));
            }
            return actions;
        }
        let liveness = &self.config.liveness;
        if self.last_heartbeat_ms.is_none_or(|last| now_ms.saturating_sub(last) >= liveness.node_heartbeat_interval_ms) {
            self.last_heartbeat_ms = Some(now_ms);
            self.next_status_sequence += 1;
            actions.push(self.to_host(FieldMessage::Heartbeat(Heartbeat {
                node_id: self.node_id,
                local_sequence: self.next_status_sequence,
                device_sample_index_now,
                thermal_headroom_fraction: self.thermal_headroom,
                battery_fraction: UNREPORTED_BATTERY_FRACTION,
                placement: self.placement,
                motion_state: self.motion_state,
                mic_only_mode: self.mic_only,
                capture_overrun_count: self.overrun_count,
                gnss_fix: self.latest_fix.clone(),
            })));
        }
        let log_config = &self.config.event_log;
        let due = self.outbox.take_due(now_ms, log_config.outbox_resend_after_ms, log_config.upload_batch_detections);
        if !due.is_empty() {
            self.counters.queued_uploads_sent += 1;
            actions.push(self.to_host(FieldMessage::QueuedDetectionUpload(
                crate::protocol::detection_messages::QueuedDetectionUpload { node_id: self.node_id, detections: due },
            )));
        }
        if self.link.host_head_sequence > self.mirror.head_sequence() {
            actions.extend(self.request_catch_up(now_ms));
        }
        actions
    }

    pub(super) fn request_catch_up(&mut self, now_ms: MeshMilliseconds) -> Vec<NodeAction> {
        let retry = self.config.event_log.catch_up_retry_interval_ms;
        if self.last_catch_up_request_ms.is_some_and(|last| now_ms.saturating_sub(last) < retry) {
            return Vec::new();
        }
        self.last_catch_up_request_ms = Some(now_ms);
        self.counters.catch_up_requests_sent += 1;
        vec![self.to_host(FieldMessage::EventLogCatchUpRequest(
            crate::protocol::mesh_messages::EventLogCatchUpRequest {
                node_id: self.node_id,
                since_sequence: self.mirror.head_sequence(),
            },
        ))]
    }

    pub fn handle(
        &mut self,
        now_ms: MeshMilliseconds,
        device_sample_index_now: u64,
        input: NodeInput,
        audio: &dyn NodeAudioWindowSource,
    ) -> Vec<NodeAction> {
        match input {
            NodeInput::Tick => self.on_tick(now_ms, device_sample_index_now),
            NodeInput::MessageReceived { from, message } => self.on_message(from, message, now_ms, audio),
            NodeInput::TransportLinkChanged { link_up } => {
                self.link.set_link_up(link_up, now_ms);
                Vec::new()
            }
            NodeInput::ClassifierFired(classification) => self.on_classification(classification, now_ms),
            NodeInput::ChirpRoundHeard(round) => self.on_chirp_round(round),
            NodeInput::DeviceStatus(change) => self.on_device_status(change, device_sample_index_now),
        }
    }
}
