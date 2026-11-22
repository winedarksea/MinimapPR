//! Short-join-code key exchange: SPAKE2 (Ed25519 group) with explicit key
//! confirmation, then the host seals the session key to the joiner.
//!
//! Flights (relay carries them as `JoinCodeKeyExchange` messages):
//! 1. joiner -> host: SPAKE2 message A
//! 2. host -> joiner: SPAKE2 message B (uniformly random-looking; reveals
//!    nothing about the code)
//! 3. joiner -> host: joiner confirmation tag
//! 4. host -> joiner: session key sealed under the exchanged key
//!
//! The joiner proves knowledge first, so an attacker posing as a joiner can
//! test exactly one code guess per exchange and the host sees and counts
//! every wrong one (`maximum_failed_confirmations`, then the code is dead).
//! Only the *secret* part of the join code is the SPAKE2 password; the
//! relay-visible lookup part must never be fed in (see notes doc).
//! Both identities embed the session id, so a transcript from one session is
//! useless in another.

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use rand_core::{CryptoRng, RngCore};
use sha2::Sha256;
use spake2::{Ed25519Group, Identity, Password, Spake2};

use super::message_sealing::{SessionKey, SESSION_KEY_LENGTH_BYTES, XCHACHA_NONCE_LENGTH_BYTES};
use crate::protocol::SessionId;

const JOINER_IDENTITY_PREFIX: &[u8] = b"minimappr-field/v1/joiner/";
const HOST_IDENTITY_PREFIX: &[u8] = b"minimappr-field/v1/host/";
const JOINER_CONFIRMATION_LABEL: &[u8] = b"minimappr-field/v1/join/joiner-confirm";
const SESSION_KEY_WRAP_LABEL: &[u8] = b"minimappr-field/v1/join/session-key-wrap";
const DERIVED_KEY_LENGTH_BYTES: usize = 32;
/// Wrong guesses a join code survives before the host retires it.
pub const DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS: u32 = 5;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinCodeExchangeError {
    MalformedSpakeMessage,
    /// Wrong join code (or an impostor host/joiner).
    ConfirmationFailed,
    /// The code has absorbed its wrong-guess budget; issue a new one.
    JoinCodeAttemptsExhausted,
    SealedSessionKeyInvalid,
}

impl std::fmt::Display for JoinCodeExchangeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "join code exchange error: {self:?}")
    }
}

impl std::error::Error for JoinCodeExchangeError {}

fn identities(session_id: &SessionId) -> (Identity, Identity) {
    let joiner = [JOINER_IDENTITY_PREFIX, &session_id.0].concat();
    let host = [HOST_IDENTITY_PREFIX, &session_id.0].concat();
    (Identity::new(&joiner), Identity::new(&host))
}

/// Keys both sides derive from the SPAKE2 output (which already hashes the
/// password, both identities and both messages).
struct ExchangedKeys {
    joiner_confirmation_key: [u8; DERIVED_KEY_LENGTH_BYTES],
    session_key_wrap_key: [u8; DERIVED_KEY_LENGTH_BYTES],
}

impl ExchangedKeys {
    fn derive(spake_output: &[u8], session_id: &SessionId) -> Self {
        let key_derivation = Hkdf::<Sha256>::new(Some(&session_id.0), spake_output);
        let mut joiner_confirmation_key = [0_u8; DERIVED_KEY_LENGTH_BYTES];
        let mut session_key_wrap_key = [0_u8; DERIVED_KEY_LENGTH_BYTES];
        key_derivation
            .expand(JOINER_CONFIRMATION_LABEL, &mut joiner_confirmation_key)
            .expect("valid HKDF length");
        key_derivation
            .expand(SESSION_KEY_WRAP_LABEL, &mut session_key_wrap_key)
            .expect("valid HKDF length");
        Self {
            joiner_confirmation_key,
            session_key_wrap_key,
        }
    }

    fn confirmation_mac(&self, transcript: &[u8]) -> HmacSha256 {
        let mut mac = <HmacSha256 as Mac>::new_from_slice(&self.joiner_confirmation_key)
            .expect("HMAC accepts any key length");
        mac.update(transcript);
        mac
    }
}

/// Joiner, after sending flight 1.
pub struct JoinerAwaitingHostMessage {
    spake_state: Spake2<Ed25519Group>,
    joiner_spake_message: Vec<u8>,
    session_id: SessionId,
}

/// Joiner, after sending flight 3.
pub struct JoinerAwaitingSessionKey {
    keys: ExchangedKeys,
    session_id: SessionId,
}

impl JoinerAwaitingHostMessage {
    /// Starts the exchange; returns the state and flight 1.
    pub fn start(
        join_code_secret: &[u8],
        session_id: SessionId,
        random_source: &mut (impl RngCore + CryptoRng),
    ) -> (Self, Vec<u8>) {
        let (joiner_identity, host_identity) = identities(&session_id);
        let (spake_state, joiner_spake_message) = Spake2::<Ed25519Group>::start_a_with_rng(
            &Password::new(join_code_secret),
            &joiner_identity,
            &host_identity,
            random_source,
        );
        (
            Self {
                spake_state,
                joiner_spake_message: joiner_spake_message.clone(),
                session_id,
            },
            joiner_spake_message,
        )
    }

    /// Consumes flight 2; returns the next state and flight 3.
    pub fn on_host_spake_message(
        self,
        host_spake_message: &[u8],
    ) -> Result<(JoinerAwaitingSessionKey, Vec<u8>), JoinCodeExchangeError> {
        let spake_output = self
            .spake_state
            .finish(host_spake_message)
            .map_err(|_| JoinCodeExchangeError::MalformedSpakeMessage)?;
        let keys = ExchangedKeys::derive(&spake_output, &self.session_id);
        let transcript = [self.joiner_spake_message.as_slice(), host_spake_message].concat();
        let confirmation_tag = keys
            .confirmation_mac(&transcript)
            .finalize()
            .into_bytes()
            .to_vec();
        Ok((
            JoinerAwaitingSessionKey {
                keys,
                session_id: self.session_id,
            },
            confirmation_tag,
        ))
    }
}

impl JoinerAwaitingSessionKey {
    /// Consumes flight 4. Success also proves the host knew the code.
    pub fn on_sealed_session_key(
        self,
        sealed_session_key: &[u8],
    ) -> Result<SessionKey, JoinCodeExchangeError> {
        if sealed_session_key.len() < XCHACHA_NONCE_LENGTH_BYTES {
            return Err(JoinCodeExchangeError::SealedSessionKeyInvalid);
        }
        let (nonce_bytes, ciphertext) = sealed_session_key.split_at(XCHACHA_NONCE_LENGTH_BYTES);
        let cipher = XChaCha20Poly1305::new((&self.keys.session_key_wrap_key).into());
        let key_bytes = cipher
            .decrypt(
                XNonce::from_slice(nonce_bytes),
                Payload {
                    msg: ciphertext,
                    aad: &self.session_id.0,
                },
            )
            .map_err(|_| JoinCodeExchangeError::SealedSessionKeyInvalid)?;
        let key_array: [u8; SESSION_KEY_LENGTH_BYTES] = key_bytes
            .try_into()
            .map_err(|_| JoinCodeExchangeError::SealedSessionKeyInvalid)?;
        Ok(SessionKey(key_array))
    }
}

/// Host-side responder for one active join code.
pub struct HostJoinCodeResponder {
    join_code_secret: Vec<u8>,
    session_id: SessionId,
    failed_confirmations: u32,
    maximum_failed_confirmations: u32,
}

/// Host state for one in-flight joiner attempt (keyed by attempt id).
pub struct HostPendingJoin {
    keys: ExchangedKeys,
    transcript: Vec<u8>,
}

impl HostJoinCodeResponder {
    pub fn new(
        join_code_secret: &[u8],
        session_id: SessionId,
        maximum_failed_confirmations: u32,
    ) -> Self {
        Self {
            join_code_secret: join_code_secret.to_vec(),
            session_id,
            failed_confirmations: 0,
            maximum_failed_confirmations,
        }
    }

    pub fn failed_confirmations(&self) -> u32 {
        self.failed_confirmations
    }

    pub fn is_exhausted(&self) -> bool {
        self.failed_confirmations >= self.maximum_failed_confirmations
    }

    /// Consumes flight 1; returns the pending state and flight 2.
    pub fn respond_to_joiner(
        &self,
        joiner_spake_message: &[u8],
        random_source: &mut (impl RngCore + CryptoRng),
    ) -> Result<(HostPendingJoin, Vec<u8>), JoinCodeExchangeError> {
        if self.is_exhausted() {
            return Err(JoinCodeExchangeError::JoinCodeAttemptsExhausted);
        }
        let (joiner_identity, host_identity) = identities(&self.session_id);
        let (spake_state, host_spake_message) = Spake2::<Ed25519Group>::start_b_with_rng(
            &Password::new(&self.join_code_secret),
            &joiner_identity,
            &host_identity,
            random_source,
        );
        let spake_output = spake_state
            .finish(joiner_spake_message)
            .map_err(|_| JoinCodeExchangeError::MalformedSpakeMessage)?;
        Ok((
            HostPendingJoin {
                keys: ExchangedKeys::derive(&spake_output, &self.session_id),
                transcript: [joiner_spake_message, host_spake_message.as_slice()].concat(),
            },
            host_spake_message,
        ))
    }

    /// Consumes flight 3. On a valid tag returns flight 4 (the sealed session
    /// key); on a wrong tag counts a failed guess.
    pub fn on_joiner_confirmation(
        &mut self,
        pending: HostPendingJoin,
        joiner_confirmation_tag: &[u8],
        session_key: &SessionKey,
        random_source: &mut (impl RngCore + CryptoRng),
    ) -> Result<Vec<u8>, JoinCodeExchangeError> {
        if self.is_exhausted() {
            return Err(JoinCodeExchangeError::JoinCodeAttemptsExhausted);
        }
        if pending
            .keys
            .confirmation_mac(&pending.transcript)
            .verify_slice(joiner_confirmation_tag)
            .is_err()
        {
            self.failed_confirmations += 1;
            return Err(JoinCodeExchangeError::ConfirmationFailed);
        }
        let mut nonce_bytes = [0_u8; XCHACHA_NONCE_LENGTH_BYTES];
        random_source.fill_bytes(&mut nonce_bytes);
        let cipher = XChaCha20Poly1305::new((&pending.keys.session_key_wrap_key).into());
        let ciphertext = cipher
            .encrypt(
                XNonce::from_slice(&nonce_bytes),
                Payload {
                    msg: &session_key.0,
                    aad: &self.session_id.0,
                },
            )
            .map_err(|_| JoinCodeExchangeError::SealedSessionKeyInvalid)?;
        Ok([nonce_bytes.as_slice(), &ciphertext].concat())
    }
}
