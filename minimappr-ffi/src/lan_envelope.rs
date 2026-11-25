//! LAN transport envelope (host-phone WebSocket, unsealed dev transport).
//!
//! Layout: `[ver u8][sender u32][recipient u32][seq u64][kind u8][payload]`,
//! little-endian. The first 17 bytes match `MessageSealer`'s cleartext
//! routing prefix, so sealing can replace the unsealed body later without
//! the shell's routing changing (TODO: join-code SPAKE2 key exchange,
//! `session_crypto::join_code_key_exchange`). The version byte differs from
//! the sealed one so a sealed peer never mistakes this for its own format.

use minimappr_core::protocol::NodeId;

use crate::error::FieldFfiError;

/// Distinct from `SEALED_ENVELOPE_VERSION`: unsealed development transport.
pub const UNSEALED_LAN_ENVELOPE_VERSION: u8 = 0x80;
pub const LAN_PROTOCOL_VERSION: u16 = 1;
const ROUTING_PREFIX_LENGTH_BYTES: usize = 17;
const HEADER_LENGTH_BYTES: usize = ROUTING_PREFIX_LENGTH_BYTES + 1;
pub const SESSION_ID_LENGTH_BYTES: usize = 16;

const KIND_FIELD_FRAME: u8 = 1;
const KIND_HELLO: u8 = 2;
const KIND_CLOCK_PROBE: u8 = 3;
const KIND_CLOCK_ECHO: u8 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanPayload {
    /// One encoded `FieldMessage` frame.
    FieldFrame(Vec<u8>),
    /// Sent by a joiner on connect (session id all-zero = discovering) and
    /// answered by the host with its own.
    Hello { session_id: [u8; SESSION_ID_LENGTH_BYTES], node_id: u32, protocol_version: u16, can_host: bool },
    ClockProbe { probe_id: u32, probe_sent_local_ns: i64 },
    ClockEcho { probe_id: u32, probe_sent_local_ns: i64, probe_received_host_ns: i64, echo_sent_host_ns: i64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanEnvelope {
    pub sender_node_id: u32,
    /// `NodeId::BROADCAST` for every node.
    pub recipient_node_id: u32,
    pub sender_sequence: u64,
    pub payload: LanPayload,
}

fn read_u32(bytes: &[u8], start: usize) -> u32 {
    u32::from_le_bytes(bytes[start..start + 4].try_into().expect("4-byte slice"))
}

fn read_u64(bytes: &[u8], start: usize) -> u64 {
    u64::from_le_bytes(bytes[start..start + 8].try_into().expect("8-byte slice"))
}

fn read_i64(bytes: &[u8], start: usize) -> i64 {
    i64::from_le_bytes(bytes[start..start + 8].try_into().expect("8-byte slice"))
}

fn malformed(detail: &str) -> FieldFfiError {
    FieldFfiError::ProtocolFrame { detail: format!("LAN envelope: {detail}") }
}

impl LanEnvelope {
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER_LENGTH_BYTES + 64);
        bytes.push(UNSEALED_LAN_ENVELOPE_VERSION);
        bytes.extend_from_slice(&self.sender_node_id.to_le_bytes());
        bytes.extend_from_slice(&self.recipient_node_id.to_le_bytes());
        bytes.extend_from_slice(&self.sender_sequence.to_le_bytes());
        match &self.payload {
            LanPayload::FieldFrame(frame) => {
                bytes.push(KIND_FIELD_FRAME);
                bytes.extend_from_slice(frame);
            }
            LanPayload::Hello { session_id, node_id, protocol_version, can_host } => {
                bytes.push(KIND_HELLO);
                bytes.extend_from_slice(session_id);
                bytes.extend_from_slice(&node_id.to_le_bytes());
                bytes.extend_from_slice(&protocol_version.to_le_bytes());
                bytes.push(u8::from(*can_host));
            }
            LanPayload::ClockProbe { probe_id, probe_sent_local_ns } => {
                bytes.push(KIND_CLOCK_PROBE);
                bytes.extend_from_slice(&probe_id.to_le_bytes());
                bytes.extend_from_slice(&probe_sent_local_ns.to_le_bytes());
            }
            LanPayload::ClockEcho { probe_id, probe_sent_local_ns, probe_received_host_ns, echo_sent_host_ns } => {
                bytes.push(KIND_CLOCK_ECHO);
                bytes.extend_from_slice(&probe_id.to_le_bytes());
                bytes.extend_from_slice(&probe_sent_local_ns.to_le_bytes());
                bytes.extend_from_slice(&probe_received_host_ns.to_le_bytes());
                bytes.extend_from_slice(&echo_sent_host_ns.to_le_bytes());
            }
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, FieldFfiError> {
        if bytes.len() < HEADER_LENGTH_BYTES {
            return Err(malformed("shorter than the header"));
        }
        if bytes[0] != UNSEALED_LAN_ENVELOPE_VERSION {
            return Err(malformed(&format!("unsupported version byte {:#04x}", bytes[0])));
        }
        let body = &bytes[HEADER_LENGTH_BYTES..];
        let need = |length: usize| if body.len() == length { Ok(()) } else { Err(malformed("payload length mismatch")) };
        let payload = match bytes[ROUTING_PREFIX_LENGTH_BYTES] {
            KIND_FIELD_FRAME => LanPayload::FieldFrame(body.to_vec()),
            KIND_HELLO => {
                need(SESSION_ID_LENGTH_BYTES + 4 + 2 + 1)?;
                LanPayload::Hello {
                    session_id: body[..SESSION_ID_LENGTH_BYTES].try_into().expect("16-byte slice"),
                    node_id: read_u32(body, SESSION_ID_LENGTH_BYTES),
                    protocol_version: u16::from_le_bytes([body[SESSION_ID_LENGTH_BYTES + 4], body[SESSION_ID_LENGTH_BYTES + 5]]),
                    can_host: body[SESSION_ID_LENGTH_BYTES + 6] != 0,
                }
            }
            KIND_CLOCK_PROBE => {
                need(12)?;
                LanPayload::ClockProbe { probe_id: read_u32(body, 0), probe_sent_local_ns: read_i64(body, 4) }
            }
            KIND_CLOCK_ECHO => {
                need(28)?;
                LanPayload::ClockEcho {
                    probe_id: read_u32(body, 0),
                    probe_sent_local_ns: read_i64(body, 4),
                    probe_received_host_ns: read_i64(body, 12),
                    echo_sent_host_ns: read_i64(body, 20),
                }
            }
            other => return Err(malformed(&format!("unknown kind {other}"))),
        };
        Ok(Self {
            sender_node_id: read_u32(bytes, 1),
            recipient_node_id: read_u32(bytes, 5),
            sender_sequence: read_u64(bytes, 9),
            payload,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Record)]
pub struct FieldLanRouting {
    pub sender_node_id: u32,
    /// `None` = broadcast.
    pub recipient_node_id: Option<u32>,
    pub is_hello: bool,
}

/// Routing fields without decoding the payload: the host's WebSocket server
/// maps a connection to a node id from its first Hello.
#[uniffi::export]
pub fn read_lan_envelope_routing(envelope: Vec<u8>) -> Result<FieldLanRouting, FieldFfiError> {
    if envelope.len() < HEADER_LENGTH_BYTES || envelope[0] != UNSEALED_LAN_ENVELOPE_VERSION {
        return Err(malformed("not an unsealed LAN envelope"));
    }
    let recipient = read_u32(&envelope, 5);
    Ok(FieldLanRouting {
        sender_node_id: read_u32(&envelope, 1),
        recipient_node_id: (recipient != NodeId::BROADCAST.0).then_some(recipient),
        is_hello: envelope[ROUTING_PREFIX_LENGTH_BYTES] == KIND_HELLO,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(payload: LanPayload) {
        let envelope = LanEnvelope { sender_node_id: 7, recipient_node_id: NodeId::BROADCAST.0, sender_sequence: 99, payload };
        assert_eq!(LanEnvelope::decode(&envelope.encode()).unwrap(), envelope);
    }

    #[test]
    fn every_kind_round_trips() {
        round_trip(LanPayload::FieldFrame(vec![1, 2, 3]));
        round_trip(LanPayload::Hello { session_id: [5; 16], node_id: 42, protocol_version: LAN_PROTOCOL_VERSION, can_host: true });
        round_trip(LanPayload::ClockProbe { probe_id: 3, probe_sent_local_ns: -12 });
        round_trip(LanPayload::ClockEcho { probe_id: 3, probe_sent_local_ns: 1, probe_received_host_ns: 2, echo_sent_host_ns: 3 });
    }

    #[test]
    fn routing_prefix_matches_the_sealer_layout() {
        let envelope = LanEnvelope { sender_node_id: 0x0102_0304, recipient_node_id: 9, sender_sequence: 5, payload: LanPayload::FieldFrame(vec![]) };
        let bytes = envelope.encode();
        assert_eq!(&bytes[1..5], &0x0102_0304u32.to_le_bytes());
        let routing = read_lan_envelope_routing(bytes).unwrap();
        assert_eq!(routing.recipient_node_id, Some(9));
        assert!(!routing.is_hello);
    }

    #[test]
    fn truncated_or_foreign_envelopes_are_errors() {
        assert!(LanEnvelope::decode(&[UNSEALED_LAN_ENVELOPE_VERSION; 5]).is_err());
        let mut bytes = LanEnvelope { sender_node_id: 1, recipient_node_id: 2, sender_sequence: 3, payload: LanPayload::ClockProbe { probe_id: 1, probe_sent_local_ns: 1 } }.encode();
        bytes.pop();
        assert!(LanEnvelope::decode(&bytes).is_err());
        bytes[0] = 1;
        assert!(LanEnvelope::decode(&bytes).is_err());
    }
}
