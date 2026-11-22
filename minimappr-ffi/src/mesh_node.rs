//! The phone's mesh node: `FieldNodeStateMachine` plus its rolling buffer,
//! stepped by the shell. Every step takes the coarse session clock `now_ms`
//! and returns actions; inbound and outbound messages are encoded frames.

use std::sync::Mutex;

use minimappr_core::mesh::config::MeshConfig;
use minimappr_core::mesh::node_state_machine::{
    DeviceStatusChange, FieldNodeStateMachine, NodeAction, NodeInput, NodeUserNotice,
};
use minimappr_core::mesh::{MessageRecipient, TrafficClass};
use minimappr_core::protocol::session_messages::NodeDescriptor;
use minimappr_core::protocol::wire_codes::NodeKind;
use minimappr_core::protocol::{decode_field_message_frame, DecodedFrame, NodeId, SessionId};

use crate::error::FieldFfiError;
use crate::node_rolling_audio_buffer::NodeRollingAudioBuffer;
use crate::pcm16_bytes::pcm16_samples_from_little_endian_bytes;
use crate::stage1_detector::{FieldStage1Event, FieldStage1Mode};

/// Phones capture mono at the device clock rate the whole core assumes.
const PHONE_CAPTURE_SAMPLE_RATE_HZ: u32 = 48_000;
const PHONE_MICROPHONE_COUNT: u8 = 1;
const SESSION_ID_LENGTH_BYTES: usize = 16;

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FieldNodeSetup {
    pub node_id: u32,
    /// 16-byte session id from the join QR code.
    pub session_id: Vec<u8>,
    pub initial_host_node_id: u32,
    pub platform_name: String,
    pub app_version: String,
    pub can_classify: bool,
    pub can_host: bool,
    pub can_emit_chirps: bool,
    pub stage1_mode: FieldStage1Mode,
    /// Rolling buffer span (plan: about 30 s of PCM).
    pub rolling_buffer_seconds: u32,
    pub now_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FieldNodeUserNotice {
    MediaVolumeMuted,
    EnteredSolo,
    Rejoined,
    MicOnlyForThermal,
    ListenOnlyChirps,
    ClockInvalidated,
}

impl From<NodeUserNotice> for FieldNodeUserNotice {
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
pub enum FieldNodeAction {
    /// `recipient_node_id` is `None` for a broadcast to every node.
    SendFrame { recipient_node_id: Option<u32>, frame: Vec<u8>, is_bulk_traffic: bool },
    EmitChirp { round_id: u32, emit_at_ms: u64, chirp_low_hz: f32, chirp_high_hz: f32, chirp_code_id: u16 },
    ListenForChirpRound { round_id: u32, listen_until_ms: u64 },
    BecomeHost { host_term: u32, reason_code: u8 },
    UserNotice { notice: FieldNodeUserNotice },
}

fn convert_action(action: NodeAction) -> Result<FieldNodeAction, FieldFfiError> {
    Ok(match action {
        NodeAction::Send(outbound) => FieldNodeAction::SendFrame {
            recipient_node_id: match outbound.recipient {
                MessageRecipient::Node(NodeId(node_id)) => Some(node_id),
                MessageRecipient::AllNodes => None,
            },
            frame: outbound
                .message
                .encode_frame()
                .map_err(|error| FieldFfiError::ProtocolFrame { detail: error.to_string() })?,
            is_bulk_traffic: outbound.traffic_class == TrafficClass::Bulk,
        },
        NodeAction::EmitChirp { round_id, emit_at_ms, chirp_low_hz, chirp_high_hz, chirp_code_id } => {
            FieldNodeAction::EmitChirp { round_id, emit_at_ms, chirp_low_hz, chirp_high_hz, chirp_code_id }
        }
        NodeAction::ListenForChirpRound { round_id, listen_until_ms } => {
            FieldNodeAction::ListenForChirpRound { round_id, listen_until_ms }
        }
        NodeAction::BecomeHost { host_term, reason } => FieldNodeAction::BecomeHost { host_term, reason_code: reason.into() },
        NodeAction::UserNotice(notice) => FieldNodeAction::UserNotice { notice: notice.into() },
    })
}

struct FieldMeshNodeState {
    state_machine: FieldNodeStateMachine,
    rolling_audio_buffer: NodeRollingAudioBuffer,
}

impl FieldMeshNodeState {
    fn step(&mut self, now_ms: u64, input: NodeInput) -> Result<Vec<FieldNodeAction>, FieldFfiError> {
        let device_sample_index_now = self.rolling_audio_buffer.next_device_sample_index();
        self.state_machine
            .handle(now_ms, device_sample_index_now, input, &self.rolling_audio_buffer)
            .into_iter()
            .map(convert_action)
            .collect()
    }
}

#[derive(uniffi::Object)]
pub struct FieldMeshNode {
    state: Mutex<FieldMeshNodeState>,
}

impl FieldMeshNode {
    fn step(&self, now_ms: u64, input: NodeInput) -> Result<Vec<FieldNodeAction>, FieldFfiError> {
        self.state.lock().expect("mesh node mutex poisoned").step(now_ms, input)
    }
}

#[uniffi::export]
impl FieldMeshNode {
    #[uniffi::constructor]
    pub fn new(setup: FieldNodeSetup) -> Result<Self, FieldFfiError> {
        let session_id_bytes: [u8; SESSION_ID_LENGTH_BYTES] =
            setup.session_id.as_slice().try_into().map_err(|_| FieldFfiError::InvalidArgument {
                detail: format!("session id must be {SESSION_ID_LENGTH_BYTES} bytes, got {}", setup.session_id.len()),
            })?;
        let descriptor = NodeDescriptor {
            node_kind: NodeKind::Phone,
            platform_name: setup.platform_name,
            app_version: setup.app_version,
            capture_sample_rate_hz: PHONE_CAPTURE_SAMPLE_RATE_HZ,
            microphone_count: PHONE_MICROPHONE_COUNT,
            microphone_positions_body_frame_m: Vec::new(),
            can_classify: setup.can_classify,
            can_host: setup.can_host,
            can_emit_chirps: setup.can_emit_chirps,
            has_pps_clock: false,
        };
        let state_machine = FieldNodeStateMachine::new(
            NodeId(setup.node_id),
            descriptor,
            SessionId(session_id_bytes),
            NodeId(setup.initial_host_node_id),
            MeshConfig::default(),
            setup.now_ms,
        );
        let retained_sample_count = setup.rolling_buffer_seconds as usize * PHONE_CAPTURE_SAMPLE_RATE_HZ as usize;
        Ok(Self {
            state: Mutex::new(FieldMeshNodeState {
                state_machine,
                rolling_audio_buffer: NodeRollingAudioBuffer::new(retained_sample_count, setup.stage1_mode.into()),
            }),
        })
    }

    /// Appends a capture block to the rolling buffer and runs Stage 1 on it.
    pub fn push_capture_block(
        &self,
        first_device_sample_index: u64,
        pcm16_little_endian_bytes: Vec<u8>,
    ) -> Result<Vec<FieldStage1Event>, FieldFfiError> {
        let samples = pcm16_samples_from_little_endian_bytes(&pcm16_little_endian_bytes)?;
        let mut state = self.state.lock().expect("mesh node mutex poisoned");
        Ok(state
            .rolling_audio_buffer
            .push_capture_block(first_device_sample_index, &samples)
            .into_iter()
            .map(FieldStage1Event::from)
            .collect())
    }

    pub fn tick(&self, now_ms: u64) -> Result<Vec<FieldNodeAction>, FieldFfiError> {
        self.step(now_ms, NodeInput::Tick)
    }

    /// Unknown kinds from newer peers are dropped here (no action).
    pub fn receive_frame(&self, now_ms: u64, from_node_id: u32, frame: Vec<u8>) -> Result<Vec<FieldNodeAction>, FieldFfiError> {
        match decode_field_message_frame(&frame).map_err(|error| FieldFfiError::ProtocolFrame { detail: error.to_string() })? {
            DecodedFrame::Known { message, .. } => {
                self.step(now_ms, NodeInput::MessageReceived { from: NodeId(from_node_id), message })
            }
            DecodedFrame::UnknownKind { .. } => Ok(Vec::new()),
        }
    }

    pub fn set_transport_link_up(&self, now_ms: u64, link_up: bool) -> Result<Vec<FieldNodeAction>, FieldFfiError> {
        self.step(now_ms, NodeInput::TransportLinkChanged { link_up })
    }

    /// Capture overrun or short read: invalidates the clock model.
    pub fn report_capture_gap(&self, now_ms: u64) -> Result<Vec<FieldNodeAction>, FieldFfiError> {
        self.step(now_ms, NodeInput::DeviceStatus(DeviceStatusChange::CaptureGapReported))
    }

    /// OS thermal headroom forecast in [0, 1] (1 = cool).
    pub fn report_thermal_headroom(&self, now_ms: u64, headroom_fraction: f32) -> Result<Vec<FieldNodeAction>, FieldFfiError> {
        self.step(now_ms, NodeInput::DeviceStatus(DeviceStatusChange::ThermalHeadroom(headroom_fraction)))
    }

    pub fn next_device_sample_index(&self) -> u64 {
        self.state.lock().expect("mesh node mutex poisoned").rolling_audio_buffer.next_device_sample_index()
    }

    pub fn rolling_buffer_gap_count(&self) -> u64 {
        self.state.lock().expect("mesh node mutex poisoned").rolling_audio_buffer.capture_gap_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol_frames::describe_frame;

    const JOIN_REQUEST_KIND_ID: u16 = 1;
    const HOST_NODE_ID: u32 = 1;

    fn phone_setup(session_id: Vec<u8>) -> FieldNodeSetup {
        FieldNodeSetup {
            node_id: 2,
            session_id,
            initial_host_node_id: HOST_NODE_ID,
            platform_name: "unit-test".into(),
            app_version: "0".into(),
            can_classify: true,
            can_host: true,
            can_emit_chirps: true,
            stage1_mode: FieldStage1Mode::Birds,
            rolling_buffer_seconds: 2,
            now_ms: 0,
        }
    }

    #[test]
    fn first_tick_sends_a_join_request_to_the_host() {
        let node = FieldMeshNode::new(phone_setup(vec![0; SESSION_ID_LENGTH_BYTES])).expect("node");
        let actions = node.tick(0).expect("tick");
        let join = actions.iter().find_map(|action| match action {
            FieldNodeAction::SendFrame { recipient_node_id, frame, .. } => Some((*recipient_node_id, frame.clone())),
            _ => None,
        });
        let (recipient, frame) = join.expect("a frame was sent");
        assert_eq!(recipient, Some(HOST_NODE_ID));
        assert_eq!(describe_frame(frame).expect("describe").message_kind_id, JOIN_REQUEST_KIND_ID);
    }

    #[test]
    fn wrong_length_session_id_is_rejected() {
        assert!(matches!(
            FieldMeshNode::new(phone_setup(vec![0; 3])),
            Err(FieldFfiError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn capture_blocks_advance_the_device_sample_index() {
        let node = FieldMeshNode::new(phone_setup(vec![0; SESSION_ID_LENGTH_BYTES])).expect("node");
        let block_bytes = vec![0_u8; 960 * 2];
        node.push_capture_block(0, block_bytes.clone()).unwrap();
        node.push_capture_block(960, block_bytes).unwrap();
        assert_eq!(node.next_device_sample_index(), 1_920);
        assert_eq!(node.rolling_buffer_gap_count(), 0);
    }
}
