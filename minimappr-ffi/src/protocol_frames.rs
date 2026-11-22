//! Wire-frame helpers. Messages stay encoded on the Kotlin side: the shell
//! moves frames between the transport and the node state machine and only
//! needs to inspect the header (routing, logging, metrics).

use minimappr_core::protocol::{
    decode_field_message_frame, DecodedFrame, FIELD_PROTOCOL_MAJOR_VERSION, FIELD_PROTOCOL_MINOR_VERSION,
    KNOWN_MESSAGE_KIND_IDS,
};

use crate::error::FieldFfiError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Record)]
pub struct FieldFrameSummary {
    pub message_kind_id: u16,
    pub sender_minor_version: u16,
    /// False for a kind from a newer peer (kept, not an error).
    pub kind_is_known_to_this_build: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Record)]
pub struct FieldProtocolVersion {
    pub major_version: u16,
    pub minor_version: u16,
}

#[uniffi::export]
pub fn field_protocol_version() -> FieldProtocolVersion {
    FieldProtocolVersion {
        major_version: FIELD_PROTOCOL_MAJOR_VERSION,
        minor_version: FIELD_PROTOCOL_MINOR_VERSION,
    }
}

#[uniffi::export]
pub fn known_message_kind_ids() -> Vec<u16> {
    KNOWN_MESSAGE_KIND_IDS.to_vec()
}

/// Decodes a frame far enough to validate it and report its header.
#[uniffi::export]
pub fn describe_frame(frame: Vec<u8>) -> Result<FieldFrameSummary, FieldFfiError> {
    let decoded = decode_field_message_frame(&frame)
        .map_err(|error| FieldFfiError::ProtocolFrame { detail: error.to_string() })?;
    Ok(match decoded {
        DecodedFrame::Known { minor_version, message } => FieldFrameSummary {
            message_kind_id: message.message_kind_id(),
            sender_minor_version: minor_version,
            kind_is_known_to_this_build: true,
        },
        DecodedFrame::UnknownKind { minor_version, message_kind_id, .. } => FieldFrameSummary {
            message_kind_id,
            sender_minor_version: minor_version,
            kind_is_known_to_this_build: false,
        },
    })
}

/// Decodes then re-encodes at this build's protocol version, dropping any
/// trailing fields a newer peer appended. Unknown kinds cannot be re-encoded.
#[uniffi::export]
pub fn canonicalize_frame(frame: Vec<u8>) -> Result<Vec<u8>, FieldFfiError> {
    match decode_field_message_frame(&frame).map_err(|error| FieldFfiError::ProtocolFrame { detail: error.to_string() })? {
        DecodedFrame::Known { message, .. } => message
            .encode_frame()
            .map_err(|error| FieldFfiError::ProtocolFrame { detail: error.to_string() }),
        DecodedFrame::UnknownKind { message_kind_id, .. } => Err(FieldFfiError::ProtocolFrame {
            detail: format!("cannot re-encode unknown message kind {message_kind_id}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minimappr_core::protocol::session_messages::Heartbeat;
    use minimappr_core::protocol::wire_codes::{MotionState, PhonePlacement};
    use minimappr_core::protocol::{FieldMessage, NodeId};

    const HEARTBEAT_KIND_ID: u16 = 22;

    fn heartbeat_frame() -> Vec<u8> {
        FieldMessage::Heartbeat(Heartbeat {
            node_id: NodeId(7),
            local_sequence: 3,
            device_sample_index_now: 48_000,
            thermal_headroom_fraction: 1.0,
            battery_fraction: 1.0,
            placement: PhonePlacement::from(1),
            motion_state: MotionState::from(1),
            mic_only_mode: false,
            capture_overrun_count: 0,
            gnss_fix: None,
        })
        .encode_frame()
        .expect("encode heartbeat")
    }

    #[test]
    fn heartbeat_frame_is_described_and_canonicalizes_to_itself() {
        let frame = heartbeat_frame();
        let summary = describe_frame(frame.clone()).expect("describe");
        assert_eq!(summary.message_kind_id, HEARTBEAT_KIND_ID);
        assert!(summary.kind_is_known_to_this_build);
        assert_eq!(canonicalize_frame(frame.clone()).expect("canonicalize"), frame);
    }

    #[test]
    fn empty_frame_is_a_protocol_error() {
        assert!(matches!(describe_frame(Vec::new()), Err(FieldFfiError::ProtocolFrame { .. })));
    }

    #[test]
    fn known_kinds_include_heartbeat() {
        assert!(known_message_kind_ids().contains(&HEARTBEAT_KIND_ID));
    }
}
