//! One phone in a LAN session: its node, its host role when it started the
//! session, the session clock, the capture anchor, the rolling buffer and
//! the chirp-round listener. Every call takes the phone's **local**
//! monotonic nanoseconds; the device converts to the host's session clock
//! for the state machines (`SessionClockAlignment`) and back to local time
//! for chirp emissions.
//!
//! Joiner lifecycle: link up → Hello (with the expected session id, or zeros
//! to adopt the host's) → host Hello names the host → clock probes until the
//! session clock locks → node state machine created and joins. The joiner
//! never hosts in this pass (`can_host = false`), so a 6 s host silence
//! cannot make it take over.
//!
//! LAN control (Hello, probes): `mesh_device_lan`. Action routing and the
//! host ↔ own-node loopback: `mesh_device_routing`.

use std::collections::BTreeSet;
use std::sync::Mutex;

use minimappr_core::mesh::config::MeshConfig;
use minimappr_core::mesh::host_state_machine::{FieldHostStateMachine, HostInput};
use minimappr_core::mesh::host_transport_policy::{CellLinkQuality, SessionTransportPreference};
use minimappr_core::mesh::node_state_machine::{DeviceStatusChange, FieldNodeStateMachine, NodeInput, NodeUserNotice};
use minimappr_core::mesh::onset_clip_set_localizer::{OnsetClipLocalizerConfig, OnsetClipSetLocalizer};
use minimappr_core::mesh::session_clock_alignment::{SessionClockAlignment, SessionClockAlignmentConfig};
use minimappr_core::protocol::session_messages::NodeDescriptor;
use minimappr_core::protocol::wire_codes::NodeKind;
use minimappr_core::protocol::{NodeId, SessionId};

use crate::capture_clock_anchor::CaptureClockAnchor;
use crate::chirp_detector::FieldChirpDetector;
use crate::chirp_round_listener::{run_searches, ChirpRoundListener, EmissionOutcome, ListenerClocks, ListenerGeometry, PlannedRound};
use crate::chirp_waveforms::{chirp_waveform_from_spec, stereo_pcm16_from_waveform, FieldStereoChannel};
use crate::error::FieldFfiError;
use crate::lan_envelope::{LanEnvelope, LanPayload, SESSION_ID_LENGTH_BYTES};
use crate::mesh_tuning::FieldMeshTuning;
use crate::node_rolling_audio_buffer::NodeRollingAudioBuffer;
use crate::pcm16_bytes::pcm16_samples_from_little_endian_bytes;
use crate::stage1_detector::{FieldStage1Event, FieldStage1Mode};
use crate::sync_status::{FieldDeviceRole, FieldLocalRoundStatus, PeerStatusBoard};

const PHONE_CAPTURE_SAMPLE_RATE_HZ: u32 = 48_000;
const PHONE_MICROPHONE_COUNT: u8 = 1;
/// See `mesh_node` history: caps a bad setup value before Rust allocates.
const MAXIMUM_ROLLING_BUFFER_SECONDS: u32 = 120;
/// Rounds still waiting this long after their listen window are reported
/// with what is known (capture lagging, playback outcome lost).
const ROUND_ABANDON_AFTER_LISTEN_MS: u64 = 3_000;
const NANOSECONDS_PER_MILLISECOND: i64 = 1_000_000;

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FieldDeviceSetup {
    pub node_id: u32,
    /// 16 bytes. A host always has one; a joiner may pass all zeros to adopt
    /// the host's (typed IP:port path) or the id from discovery.
    pub session_id: Vec<u8>,
    pub role: FieldDeviceRole,
    pub platform_name: String,
    pub app_version: String,
    pub can_classify: bool,
    pub can_emit_chirps: bool,
    pub stage1_mode: FieldStage1Mode,
    pub rolling_buffer_seconds: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FieldDeviceNotice {
    MediaVolumeMuted,
    EnteredSolo,
    Rejoined,
    MicOnlyForThermal,
    ListenOnlyChirps,
    ClockInvalidated,
    /// The node asked to become host; this pass never acts on it.
    HostTakeoverRequested,
    /// The host answered with a different session id than expected.
    SessionIdMismatch,
    SessionClockLocked,
    JoinedSession,
}

impl From<NodeUserNotice> for FieldDeviceNotice {
    fn from(notice: NodeUserNotice) -> Self {
        match notice {
            NodeUserNotice::MediaVolumeMuted => Self::MediaVolumeMuted,
            NodeUserNotice::EnteredSolo => Self::EnteredSolo,
            NodeUserNotice::Rejoined => Self::Rejoined,
            NodeUserNotice::MicOnlyForThermal => Self::MicOnlyForThermal,
            NodeUserNotice::ListenOnlyChirps => Self::ListenOnlyChirps,
            NodeUserNotice::ClockInvalidated => Self::ClockInvalidated,
        }
    }
}

#[derive(Debug, Clone, PartialEq, uniffi::Enum)]
pub enum FieldDeviceAction {
    /// `recipient_node_id` `None` = every connected phone.
    SendEnvelope { recipient_node_id: Option<u32>, envelope: Vec<u8>, is_bulk_traffic: bool },
    /// Play `pcm16_stereo_little_endian` so its first frame leaves the
    /// speaker at `emit_at_local_ns`, then call `report_chirp_emission_outcome`.
    EmitChirp { emission_id: u64, round_id: u32, emit_at_local_ns: i64, pcm16_stereo_little_endian: Vec<u8> },
    UserNotice { notice: FieldDeviceNotice },
    HostRoleChanged { is_host: bool },
}

#[derive(Debug, Default, Clone)]
pub(crate) struct DeviceCounters {
    pub emissions_requested: u64,
    pub emissions_played: u64,
    pub emissions_skipped: u64,
    pub rounds_heard: u64,
    pub encode_failures: u64,
    pub loopback_overflows: u64,
    pub rejected_hellos: u64,
    pub dropped_unadmitted_frames: u64,
}

pub(crate) struct DeviceState {
    pub node_id: NodeId,
    pub role: FieldDeviceRole,
    pub session_id: Option<[u8; SESSION_ID_LENGTH_BYTES]>,
    pub descriptor: NodeDescriptor,
    pub config: MeshConfig,
    pub node: Option<FieldNodeStateMachine>,
    pub host: Option<FieldHostStateMachine>,
    pub host_node_id: Option<NodeId>,
    pub localizer: OnsetClipSetLocalizer,
    pub alignment: SessionClockAlignment,
    pub capture_anchor: CaptureClockAnchor,
    pub buffer: NodeRollingAudioBuffer,
    pub listener: ChirpRoundListener,
    pub geometry: ListenerGeometry,
    pub chirp_playback_pcm: Vec<u8>,
    pub link_up: bool,
    pub admitted_peers: BTreeSet<u32>,
    pub next_sender_sequence: u64,
    pub next_probe_id: u32,
    pub last_probe_local_ns: Option<i64>,
    pub last_hello_local_ns: Option<i64>,
    pub next_emission_id: u64,
    pub absorbed_rounds_completed: u32,
    pub peer_board: PeerStatusBoard,
    pub last_local_round: Option<FieldLocalRoundStatus>,
    pub counters: DeviceCounters,
}

impl DeviceState {
    pub(crate) fn envelope(&mut self, recipient: Option<u32>, payload: LanPayload) -> Vec<u8> {
        self.next_sender_sequence += 1;
        LanEnvelope {
            sender_node_id: self.node_id.0,
            recipient_node_id: recipient.unwrap_or(NodeId::BROADCAST.0),
            sender_sequence: self.next_sender_sequence,
            payload,
        }
        .encode()
    }

    pub(crate) fn session_ms(&mut self, local_ns: i64) -> Option<u64> {
        self.alignment.advance(local_ns);
        self.alignment.session_ms_at(local_ns)
    }

    /// Create the joiner's node once the host is known and the clock locked.
    pub(crate) fn ensure_joiner_node(&mut self, local_ns: i64, actions: &mut Vec<FieldDeviceAction>) {
        if self.node.is_some() || self.role != FieldDeviceRole::Joiner {
            return;
        }
        let (Some(host), Some(session_id), Some(session_ms)) = (self.host_node_id, self.session_id, self.session_ms(local_ns)) else {
            return;
        };
        let mut node = FieldNodeStateMachine::new(self.node_id, self.descriptor.clone(), SessionId(session_id), host, self.config.clone(), session_ms);
        node.link.set_link_up(self.link_up, session_ms);
        self.node = Some(node);
        actions.push(FieldDeviceAction::UserNotice { notice: FieldDeviceNotice::SessionClockLocked });
    }

    fn tick_locked(&mut self, local_ns: i64) -> Vec<FieldDeviceAction> {
        let mut actions = self.lan_control_tick(local_ns);
        self.ensure_joiner_node(local_ns, &mut actions);
        let Some(session_ms) = self.session_ms(local_ns) else { return actions };
        let was_joined = self.joined();
        let node_actions = self.step_node(session_ms, NodeInput::Tick);
        let host_actions = self.step_host(session_ms, HostInput::Tick);
        actions.extend(self.apply_actions(session_ms, node_actions, host_actions));
        if !was_joined && self.joined() {
            actions.push(FieldDeviceAction::UserNotice { notice: FieldDeviceNotice::JoinedSession });
        }
        actions
    }

    pub(crate) fn joined(&self) -> bool {
        self.node.as_ref().is_some_and(|node| node.mirror.head_sequence() > 0)
    }

    fn take_ready_rounds(&mut self, local_ns: i64) -> Vec<PlannedRound> {
        let clocks = ListenerClocks { alignment: &self.alignment, capture_anchor: &self.capture_anchor, buffer: &self.buffer, local_now_ns: local_ns };
        self.listener.take_ready_rounds(self.node_id, &self.geometry, &clocks)
    }

    fn report_round(&mut self, local_ns: i64, planned: PlannedRound, results: Vec<crate::chirp_round_listener::SlotSearchResult>) -> Vec<FieldDeviceAction> {
        let round = crate::chirp_round_listener::local_round_from_results(&planned, &results);
        self.counters.rounds_heard += 1;
        self.last_local_round = Some(crate::mesh_device_lan::local_round_status(&planned, &results, round.own_chirp_inaudible));
        let Some(session_ms) = self.session_ms(local_ns) else { return Vec::new() };
        let node_actions = self.step_node(session_ms, NodeInput::ChirpRoundHeard(round));
        self.apply_actions(session_ms, node_actions, Vec::new())
    }
}

#[derive(uniffi::Object)]
pub struct FieldMeshDevice {
    state: Mutex<DeviceState>,
    /// Immutable: detection runs on it outside the state lock.
    detector: FieldChirpDetector,
}

impl FieldMeshDevice {
    pub(crate) fn lock(&self) -> std::sync::MutexGuard<'_, DeviceState> {
        self.state.lock().expect("mesh device mutex poisoned")
    }

    fn parse_session_id(bytes: &[u8]) -> Result<[u8; SESSION_ID_LENGTH_BYTES], FieldFfiError> {
        bytes.try_into().map_err(|_| FieldFfiError::InvalidArgument {
            detail: format!("session id must be {SESSION_ID_LENGTH_BYTES} bytes, got {}", bytes.len()),
        })
    }

    /// Steps that may produce chirp rounds: plan under the lock, detect
    /// outside it, report under it again.
    fn finish_ready_rounds(&self, local_ns: i64, mut actions: Vec<FieldDeviceAction>) -> Vec<FieldDeviceAction> {
        let (planned, self_path_length_m) = {
            let mut state = self.lock();
            let path = state.geometry.self_path_length_m;
            (state.take_ready_rounds(local_ns), path)
        };
        for round in planned {
            let results = run_searches(&round.jobs, &self.detector, self_path_length_m);
            actions.extend(self.lock().report_round(local_ns, round, results));
        }
        actions
    }
}

#[uniffi::export]
impl FieldMeshDevice {
    #[uniffi::constructor]
    pub fn new(setup: FieldDeviceSetup, tuning: FieldMeshTuning, local_now_ns: i64) -> Result<Self, FieldFfiError> {
        let session_id = Self::parse_session_id(&setup.session_id)?;
        let has_session_id = session_id != [0; SESSION_ID_LENGTH_BYTES];
        if setup.role == FieldDeviceRole::Host && !has_session_id {
            return Err(FieldFfiError::InvalidArgument { detail: "a host needs a non-zero session id".into() });
        }
        if setup.rolling_buffer_seconds == 0 || setup.rolling_buffer_seconds > MAXIMUM_ROLLING_BUFFER_SECONDS {
            return Err(FieldFfiError::InvalidArgument {
                detail: format!("rolling buffer must be 1..={MAXIMUM_ROLLING_BUFFER_SECONDS} s, got {}", setup.rolling_buffer_seconds),
            });
        }
        let config = tuning.mesh_config()?;
        let waveform = chirp_waveform_from_spec(&tuning.waveform)?;
        let channel = tuning.chirp_output_channel.unwrap_or_else(|| {
            match config.island_sync.chirp_output_channel_by_platform.get(&setup.platform_name).copied().unwrap_or(config.island_sync.default_chirp_output_channel) {
                minimappr_core::sync::chirp_waveform::StereoOutputChannel::Left => FieldStereoChannel::Left,
                minimappr_core::sync::chirp_waveform::StereoOutputChannel::Right => FieldStereoChannel::Right,
            }
        });
        let chirp_playback_pcm = stereo_pcm16_from_waveform(&waveform, channel, tuning.chirp_peak_dbfs)?;
        let detector = FieldChirpDetector::from_spec(&tuning.waveform, tuning.speed_of_sound_mps())?;
        let is_host = setup.role == FieldDeviceRole::Host;
        let descriptor = NodeDescriptor {
            node_kind: NodeKind::Phone,
            platform_name: setup.platform_name.clone(),
            app_version: setup.app_version,
            capture_sample_rate_hz: PHONE_CAPTURE_SAMPLE_RATE_HZ,
            microphone_count: PHONE_MICROPHONE_COUNT,
            microphone_positions_body_frame_m: Vec::new(),
            can_classify: setup.can_classify,
            can_host: is_host,
            can_emit_chirps: setup.can_emit_chirps,
            has_pps_clock: false,
        };
        let self_path_length_m = config
            .island_sync
            .self_path_length_by_platform_m
            .get(&setup.platform_name)
            .copied()
            .unwrap_or(config.island_sync.default_self_path_length_m);
        let geometry = ListenerGeometry {
            slot_duration_ms: config.scheduler.slot_duration_ms,
            coarse_schedule_uncertainty_ms: config.scheduler.coarse_schedule_uncertainty_ms,
            chirp_sample_count: waveform.samples.len(),
            self_path_length_m,
            abandon_after_listen_ms: ROUND_ABANDON_AFTER_LISTEN_MS,
        };
        let node_id = NodeId(setup.node_id);
        let localizer = OnsetClipSetLocalizer::new(OnsetClipLocalizerConfig {
            sound_speed_mps: config.island_sync.sound_speed_mps,
            ..OnsetClipLocalizerConfig::default()
        });
        let local_ms = (local_now_ns.max(0) / NANOSECONDS_PER_MILLISECOND) as u64;
        let (host, node, host_actions) = if is_host {
            let (host, host_actions) = FieldHostStateMachine::new_session(
                node_id,
                descriptor.clone(),
                config.clone(),
                SessionTransportPreference::LocalOnly,
                CellLinkQuality::NoSignal,
                local_ms,
            );
            let mut node = FieldNodeStateMachine::new(node_id, descriptor.clone(), SessionId(session_id), node_id, config.clone(), local_ms);
            node.link.set_link_up(true, local_ms);
            (Some(host), Some(node), host_actions)
        } else {
            (None, None, Vec::new())
        };
        let mut state = DeviceState {
            node_id,
            role: setup.role,
            session_id: has_session_id.then_some(session_id),
            descriptor,
            node,
            host,
            host_node_id: is_host.then_some(node_id),
            localizer,
            alignment: if is_host { SessionClockAlignment::identity() } else { SessionClockAlignment::new(SessionClockAlignmentConfig::default()) },
            capture_anchor: CaptureClockAnchor::default(),
            buffer: NodeRollingAudioBuffer::new(
                setup.rolling_buffer_seconds as usize * PHONE_CAPTURE_SAMPLE_RATE_HZ as usize,
                setup.stage1_mode.into(),
            ),
            listener: ChirpRoundListener::default(),
            geometry,
            chirp_playback_pcm,
            link_up: is_host,
            admitted_peers: BTreeSet::new(),
            next_sender_sequence: 0,
            next_probe_id: 0,
            last_probe_local_ns: None,
            last_hello_local_ns: None,
            next_emission_id: 0,
            absorbed_rounds_completed: 0,
            peer_board: PeerStatusBoard::default(),
            last_local_round: None,
            counters: DeviceCounters::default(),
            config,
        };
        // The host's opening announcements reach only its own node now;
        // joiners get the next periodic one.
        let _ = state.apply_actions(local_ms, Vec::new(), host_actions);
        Ok(Self { state: Mutex::new(state), detector })
    }

    pub fn push_capture_block(&self, first_device_sample_index: u64, pcm16_little_endian_bytes: Vec<u8>) -> Result<Vec<FieldStage1Event>, FieldFfiError> {
        let samples = pcm16_samples_from_little_endian_bytes(&pcm16_little_endian_bytes)?;
        let mut state = self.lock();
        Ok(state.buffer.push_capture_block(first_device_sample_index, &samples).into_iter().map(FieldStage1Event::from).collect())
    }

    /// `AudioRecord.getTimestamp` result on the same frame timeline as the
    /// device sample index (frames read since `startRecording`).
    pub fn report_capture_timestamp(&self, frame_position: u64, monotonic_ns: i64) {
        self.lock().capture_anchor.report(frame_position, monotonic_ns);
    }

    pub fn tick(&self, local_now_ns: i64) -> Vec<FieldDeviceAction> {
        let actions = self.lock().tick_locked(local_now_ns);
        self.finish_ready_rounds(local_now_ns, actions)
    }

    pub fn receive_envelope(&self, local_now_ns: i64, envelope: Vec<u8>) -> Result<Vec<FieldDeviceAction>, FieldFfiError> {
        let decoded = LanEnvelope::decode(&envelope)?;
        let actions = self.lock().receive_envelope_locked(local_now_ns, decoded)?;
        Ok(self.finish_ready_rounds(local_now_ns, actions))
    }

    pub fn set_transport_link_up(&self, local_now_ns: i64, link_up: bool) -> Vec<FieldDeviceAction> {
        let mut state = self.lock();
        state.link_up = link_up;
        if !link_up {
            state.last_hello_local_ns = None;
        }
        let Some(session_ms) = state.session_ms(local_now_ns) else { return Vec::new() };
        if state.role == FieldDeviceRole::Joiner {
            let node_actions = state.step_node(session_ms, NodeInput::TransportLinkChanged { link_up });
            return state.apply_actions(session_ms, node_actions, Vec::new());
        }
        Vec::new()
    }

    /// `played = false` when the slot could not be scheduled in time: the
    /// round reports "not emitted" and the host never mutes the phone.
    pub fn report_chirp_emission_outcome(&self, emission_id: u64, played: bool, first_frame_local_ns: i64) {
        let mut state = self.lock();
        if state.listener.record_emission_outcome(emission_id, EmissionOutcome { played, first_frame_local_ns }) {
            if played {
                state.counters.emissions_played += 1;
            } else {
                state.counters.emissions_skipped += 1;
            }
        }
    }

    /// Capture overrun or short read: invalidates the clock model.
    pub fn report_capture_gap(&self, local_now_ns: i64) -> Vec<FieldDeviceAction> {
        self.device_status(local_now_ns, DeviceStatusChange::CaptureGapReported)
    }

    pub fn report_thermal_headroom(&self, local_now_ns: i64, headroom_fraction: f32) -> Vec<FieldDeviceAction> {
        self.device_status(local_now_ns, DeviceStatusChange::ThermalHeadroom(headroom_fraction))
    }

    pub fn next_device_sample_index(&self) -> u64 {
        self.lock().buffer.next_device_sample_index()
    }

    pub fn sync_status(&self) -> crate::sync_status::FieldSyncStatus {
        self.lock().sync_status()
    }
}

impl FieldMeshDevice {
    fn device_status(&self, local_now_ns: i64, change: DeviceStatusChange) -> Vec<FieldDeviceAction> {
        let mut state = self.lock();
        let Some(session_ms) = state.session_ms(local_now_ns) else { return Vec::new() };
        let node_actions = state.step_node(session_ms, NodeInput::DeviceStatus(change));
        state.apply_actions(session_ms, node_actions, Vec::new())
    }
}
