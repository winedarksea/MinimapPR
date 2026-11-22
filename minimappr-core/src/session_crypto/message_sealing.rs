//! XChaCha20-Poly1305 sealing of encoded `FieldMessage` frames.
//!
//! Envelope (version 1):
//! ```text
//! u8 version | u32 sender_node_id | u32 recipient_node_id | u64 sender_sequence
//! | 24-byte nonce | ciphertext || 16-byte Poly1305 tag
//! ```
//! The 17-byte routing prefix is cleartext so the untrusted relay can route
//! by node id, and it is bound as AEAD associated data so the relay cannot
//! re-address or renumber a message (receivers still check the recipient:
//! `open_addressed_to_local_node`). The nonce is random per message
//! (XChaCha's 192-bit nonce makes random nonces collision-safe). The cipher
//! key is HKDF-derived from the QR session key with the session id as salt,
//! so a message from one session never opens in another.

use std::collections::HashMap;

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use hkdf::Hkdf;
use rand_core::{CryptoRng, RngCore};
use sha2::Sha256;

use crate::protocol::{NodeId, SessionId};

pub const SEALED_ENVELOPE_VERSION: u8 = 1;
pub const SESSION_KEY_LENGTH_BYTES: usize = 32;
pub const XCHACHA_NONCE_LENGTH_BYTES: usize = 24;
pub const POLY1305_TAG_LENGTH_BYTES: usize = 16;
pub const ENVELOPE_ROUTING_PREFIX_LENGTH_BYTES: usize = 1 + 4 + 4 + 8;
const MESSAGE_KEY_DERIVATION_LABEL: &[u8] = b"minimappr-field/v1/message-seal";
/// Out-of-order tolerance of the replay window (one bit per sequence).
pub const REPLAY_WINDOW_SEQUENCE_SPAN: u64 = u64::BITS as u64;

/// 256-bit session key carried in the QR code; never reaches the relay.
#[derive(Clone, PartialEq, Eq)]
pub struct SessionKey(pub [u8; SESSION_KEY_LENGTH_BYTES]);

impl std::fmt::Debug for SessionKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SessionKey(<redacted>)")
    }
}

impl SessionKey {
    pub fn generate(random_source: &mut (impl RngCore + CryptoRng)) -> Self {
        let mut key_bytes = [0_u8; SESSION_KEY_LENGTH_BYTES];
        random_source.fill_bytes(&mut key_bytes);
        Self(key_bytes)
    }
}

/// Cleartext routing fields of an envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvelopeRouting {
    pub sender_node_id: NodeId,
    pub recipient_node_id: NodeId,
    /// Strictly increasing per sender for the session's lifetime; checked by
    /// `ReplayWindow`. A sender that restarts must resume above its last
    /// value (persist it), or peers drop it as a replay until it catches up.
    pub sender_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SealingError {
    EnvelopeTooShort,
    UnsupportedEnvelopeVersion(u8),
    /// Wrong key, wrong session, or the envelope was modified.
    AuthenticationFailed,
    EncryptionFailed,
    /// Authentic, but addressed to another node (relay mis-delivery).
    NotAddressedToLocalNode,
    /// Authentic, but claims the local node as sender (relay reflection).
    ReflectedOwnMessage,
}

impl std::fmt::Display for SealingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "sealing error: {self:?}")
    }
}

impl std::error::Error for SealingError {}

/// Seals and opens envelopes under one session's message key.
pub struct MessageSealer {
    cipher: XChaCha20Poly1305,
}

impl MessageSealer {
    pub fn new(session_key: &SessionKey, session_id: &SessionId) -> Self {
        let key_derivation = Hkdf::<Sha256>::new(Some(&session_id.0), &session_key.0);
        let mut message_key = [0_u8; SESSION_KEY_LENGTH_BYTES];
        key_derivation
            .expand(MESSAGE_KEY_DERIVATION_LABEL, &mut message_key)
            .expect("32 bytes is a valid HKDF-SHA256 output length");
        Self {
            cipher: XChaCha20Poly1305::new((&message_key).into()),
        }
    }

    pub fn seal(
        &self,
        routing: EnvelopeRouting,
        plaintext_frame: &[u8],
        random_source: &mut (impl RngCore + CryptoRng),
    ) -> Result<Vec<u8>, SealingError> {
        let routing_prefix = encode_routing_prefix(&routing);
        let mut nonce_bytes = [0_u8; XCHACHA_NONCE_LENGTH_BYTES];
        random_source.fill_bytes(&mut nonce_bytes);
        let ciphertext = self
            .cipher
            .encrypt(
                XNonce::from_slice(&nonce_bytes),
                Payload {
                    msg: plaintext_frame,
                    aad: &routing_prefix,
                },
            )
            .map_err(|_| SealingError::EncryptionFailed)?;
        let mut envelope =
            Vec::with_capacity(routing_prefix.len() + nonce_bytes.len() + ciphertext.len());
        envelope.extend_from_slice(&routing_prefix);
        envelope.extend_from_slice(&nonce_bytes);
        envelope.extend_from_slice(&ciphertext);
        Ok(envelope)
    }

    /// Authenticates and decrypts; returns routing and the plaintext frame.
    /// Replay filtering is the caller's `ReplayWindow`.
    pub fn open(&self, envelope: &[u8]) -> Result<(EnvelopeRouting, Vec<u8>), SealingError> {
        let routing = read_envelope_routing(envelope)?;
        let minimum_length = ENVELOPE_ROUTING_PREFIX_LENGTH_BYTES
            + XCHACHA_NONCE_LENGTH_BYTES
            + POLY1305_TAG_LENGTH_BYTES;
        if envelope.len() < minimum_length {
            return Err(SealingError::EnvelopeTooShort);
        }
        let (routing_prefix, rest) = envelope.split_at(ENVELOPE_ROUTING_PREFIX_LENGTH_BYTES);
        let (nonce_bytes, ciphertext) = rest.split_at(XCHACHA_NONCE_LENGTH_BYTES);
        let plaintext = self
            .cipher
            .decrypt(
                XNonce::from_slice(nonce_bytes),
                Payload {
                    msg: ciphertext,
                    aad: routing_prefix,
                },
            )
            .map_err(|_| SealingError::AuthenticationFailed)?;
        Ok((routing, plaintext))
    }

    /// `open` plus the receiver-side routing checks every node must make.
    /// The AAD only stops the relay *editing* routing; it can still deliver
    /// node B's sealed message to node C, or reflect a node's own message
    /// back to it, and both open under the shared session key.
    pub fn open_addressed_to_local_node(
        &self,
        envelope: &[u8],
        local_node_id: NodeId,
    ) -> Result<(EnvelopeRouting, Vec<u8>), SealingError> {
        let (routing, plaintext) = self.open(envelope)?;
        if routing.recipient_node_id != local_node_id
            && routing.recipient_node_id != NodeId::BROADCAST
        {
            return Err(SealingError::NotAddressedToLocalNode);
        }
        if routing.sender_node_id == local_node_id {
            return Err(SealingError::ReflectedOwnMessage);
        }
        Ok((routing, plaintext))
    }
}

fn encode_routing_prefix(routing: &EnvelopeRouting) -> [u8; ENVELOPE_ROUTING_PREFIX_LENGTH_BYTES] {
    let mut prefix = [0_u8; ENVELOPE_ROUTING_PREFIX_LENGTH_BYTES];
    prefix[0] = SEALED_ENVELOPE_VERSION;
    prefix[1..5].copy_from_slice(&routing.sender_node_id.0.to_le_bytes());
    prefix[5..9].copy_from_slice(&routing.recipient_node_id.0.to_le_bytes());
    prefix[9..17].copy_from_slice(&routing.sender_sequence.to_le_bytes());
    prefix
}

/// Reads routing without a key: what the relay uses to forward.
pub fn read_envelope_routing(envelope: &[u8]) -> Result<EnvelopeRouting, SealingError> {
    if envelope.len() < ENVELOPE_ROUTING_PREFIX_LENGTH_BYTES {
        return Err(SealingError::EnvelopeTooShort);
    }
    if envelope[0] != SEALED_ENVELOPE_VERSION {
        return Err(SealingError::UnsupportedEnvelopeVersion(envelope[0]));
    }
    let read_u32 = |start: usize| {
        u32::from_le_bytes(envelope[start..start + 4].try_into().expect("4-byte slice"))
    };
    Ok(EnvelopeRouting {
        sender_node_id: NodeId(read_u32(1)),
        recipient_node_id: NodeId(read_u32(5)),
        sender_sequence: u64::from_le_bytes(envelope[9..17].try_into().expect("8-byte slice")),
    })
}

/// Per-sender sliding replay window (highest sequence + 64-bit bitmap),
/// the IPsec/DTLS pattern: tolerates reordering, rejects duplicates.
#[derive(Debug, Default, Clone)]
pub struct ReplayWindow {
    per_sender: HashMap<NodeId, (u64, u64)>,
}

impl ReplayWindow {
    /// True (and records it) if `sequence` from `sender` is new. Call only
    /// after `open` succeeded, so forged envelopes cannot advance the window.
    pub fn accept(&mut self, sender: NodeId, sequence: u64) -> bool {
        let Some((highest, seen_bitmap)) = self.per_sender.get_mut(&sender) else {
            self.per_sender.insert(sender, (sequence, 1));
            return true;
        };
        if sequence > *highest {
            let advance = sequence - *highest;
            *seen_bitmap = if advance >= REPLAY_WINDOW_SEQUENCE_SPAN {
                0
            } else {
                *seen_bitmap << advance
            };
            *seen_bitmap |= 1;
            *highest = sequence;
            return true;
        }
        let age = *highest - sequence;
        if age >= REPLAY_WINDOW_SEQUENCE_SPAN {
            return false;
        }
        let bit = 1_u64 << age;
        if *seen_bitmap & bit != 0 {
            return false;
        }
        *seen_bitmap |= bit;
        true
    }
}
