//! Binary relay frame: a fixed cleartext routing header followed by an
//! opaque body the relay never parses.
//!
//! ```text
//! u8 relay_frame_version | u8 relay_frame_kind | u32 sender_peer_id (LE)
//! | u32 recipient_peer_id (LE) | body ...
//! ```
//! The body is either a `minimappr-core` sealed envelope (which binds its own
//! routing as AEAD associated data) or, only during a join-code exchange, an
//! unsealed `JoinCodeKeyExchange` frame whose SPAKE2 messages are safe to
//! expose. The relay routes on this header alone, so it never needs to know
//! the envelope layout or hold any key.

pub const RELAY_FRAME_VERSION: u8 = 1;
pub const RELAY_FRAME_HEADER_LENGTH_BYTES: usize = 1 + 1 + 4 + 4;
/// Recipient meaning "every other peer in the session" (same value as
/// `minimappr_core::protocol::NodeId::BROADCAST`).
pub const BROADCAST_PEER_ID: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelayFrameKind {
    /// Body is a sealed envelope; only holders of the session key can open it.
    SealedEnvelope,
    /// Body is an unsealed join-code key-exchange frame (SPAKE2 flights).
    /// Joiners and viewers may only send these to the host.
    JoinCodeKeyExchange,
}

impl RelayFrameKind {
    pub fn wire_code(self) -> u8 {
        match self {
            Self::SealedEnvelope => 1,
            Self::JoinCodeKeyExchange => 2,
        }
    }

    pub fn from_wire_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::SealedEnvelope),
            2 => Some(Self::JoinCodeKeyExchange),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelayFrameHeader {
    pub kind: RelayFrameKind,
    pub sender_peer_id: u32,
    pub recipient_peer_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelayFrameDecodeError {
    FrameShorterThanHeader,
    UnsupportedRelayFrameVersion(u8),
    UnknownRelayFrameKind(u8),
}

impl std::fmt::Display for RelayFrameDecodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "relay frame decode error: {self:?}")
    }
}

impl std::error::Error for RelayFrameDecodeError {}

impl RelayFrameHeader {
    pub fn encode(&self) -> [u8; RELAY_FRAME_HEADER_LENGTH_BYTES] {
        let mut header_bytes = [0_u8; RELAY_FRAME_HEADER_LENGTH_BYTES];
        header_bytes[0] = RELAY_FRAME_VERSION;
        header_bytes[1] = self.kind.wire_code();
        header_bytes[2..6].copy_from_slice(&self.sender_peer_id.to_le_bytes());
        header_bytes[6..10].copy_from_slice(&self.recipient_peer_id.to_le_bytes());
        header_bytes
    }

    pub fn decode(frame: &[u8]) -> Result<Self, RelayFrameDecodeError> {
        if frame.len() < RELAY_FRAME_HEADER_LENGTH_BYTES {
            return Err(RelayFrameDecodeError::FrameShorterThanHeader);
        }
        if frame[0] != RELAY_FRAME_VERSION {
            return Err(RelayFrameDecodeError::UnsupportedRelayFrameVersion(frame[0]));
        }
        let kind = RelayFrameKind::from_wire_code(frame[1])
            .ok_or(RelayFrameDecodeError::UnknownRelayFrameKind(frame[1]))?;
        let read_u32 = |start: usize| {
            u32::from_le_bytes(frame[start..start + 4].try_into().expect("4-byte slice"))
        };
        Ok(Self {
            kind,
            sender_peer_id: read_u32(2),
            recipient_peer_id: read_u32(6),
        })
    }
}

/// Header ++ body as one WebSocket binary message.
pub fn encode_relay_frame(header: &RelayFrameHeader, body: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(RELAY_FRAME_HEADER_LENGTH_BYTES + body.len());
    frame.extend_from_slice(&header.encode());
    frame.extend_from_slice(body);
    frame
}

/// Splits a relay frame into its header and opaque body.
pub fn decode_relay_frame(frame: &[u8]) -> Result<(RelayFrameHeader, &[u8]), RelayFrameDecodeError> {
    let header = RelayFrameHeader::decode(frame)?;
    Ok((header, &frame[RELAY_FRAME_HEADER_LENGTH_BYTES..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trips_and_body_is_untouched() {
        let header = RelayFrameHeader {
            kind: RelayFrameKind::JoinCodeKeyExchange,
            sender_peer_id: 0x8000_0001,
            recipient_peer_id: BROADCAST_PEER_ID,
        };
        let frame = encode_relay_frame(&header, b"opaque");
        let (decoded, body) = decode_relay_frame(&frame).unwrap();
        assert_eq!(decoded, header);
        assert_eq!(body, b"opaque");
    }

    #[test]
    fn rejects_short_wrong_version_and_unknown_kind() {
        assert_eq!(
            RelayFrameHeader::decode(&[1, 1, 0]),
            Err(RelayFrameDecodeError::FrameShorterThanHeader)
        );
        let mut frame = encode_relay_frame(
            &RelayFrameHeader {
                kind: RelayFrameKind::SealedEnvelope,
                sender_peer_id: 1,
                recipient_peer_id: 2,
            },
            &[],
        );
        frame[1] = 9;
        assert_eq!(
            RelayFrameHeader::decode(&frame),
            Err(RelayFrameDecodeError::UnknownRelayFrameKind(9))
        );
        frame[0] = 7;
        assert_eq!(
            RelayFrameHeader::decode(&frame),
            Err(RelayFrameDecodeError::UnsupportedRelayFrameVersion(7))
        );
    }
}
