//! Drives the core SPAKE2 join-code exchange through relay
//! `JoinCodeKeyExchange` flights (unsealed: they establish the key).
//!
//! Peer ids: the host assigns node ids below `PROVISIONAL_JOINER_PEER_ID_FLOOR`;
//! a code joiner, which has no node id yet, binds a random provisional peer
//! id at or above it for the exchange, then reconnects with the node id the
//! host gives it in the sealed `JoinAccepted`.

use std::collections::HashMap;

use minimappr_core::protocol::session_messages::JoinCodeKeyExchange;
use minimappr_core::protocol::wire_codes::KeyExchangeFlight;
use minimappr_core::protocol::SessionId;
use minimappr_core::session_crypto::{
    HostJoinCodeResponder, HostPendingJoin, JoinCodeExchangeError, JoinerAwaitingHostMessage,
    JoinerAwaitingSessionKey, SessionKey,
};
use minimappr_relay_wire::BROADCAST_PEER_ID;
use rand_core::{CryptoRng, RngCore};

pub const PROVISIONAL_JOINER_PEER_ID_FLOOR: u32 = 0x8000_0000;

/// A random provisional peer id in `[floor, BROADCAST_PEER_ID)`.
pub fn random_provisional_joiner_peer_id(random_source: &mut (impl RngCore + CryptoRng)) -> u32 {
    let span = BROADCAST_PEER_ID - PROVISIONAL_JOINER_PEER_ID_FLOOR;
    PROVISIONAL_JOINER_PEER_ID_FLOOR + random_source.next_u32() % span
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelayJoinExchangeError {
    Core(JoinCodeExchangeError),
    WrongSession,
    WrongAttempt,
    UnexpectedFlight(KeyExchangeFlight),
    /// The host refused: wrong code, or the code's guess budget is spent.
    RejectedByHost,
    TooManyPendingExchanges,
    ExchangeAlreadyFinished,
}

impl From<JoinCodeExchangeError> for RelayJoinExchangeError {
    fn from(error: JoinCodeExchangeError) -> Self {
        Self::Core(error)
    }
}

enum JoinerPhase {
    AwaitingHostMessage(JoinerAwaitingHostMessage),
    AwaitingSessionKey(JoinerAwaitingSessionKey),
    Finished,
}

/// Joiner side of one attempt.
pub struct JoinerRelayKeyExchange {
    phase: JoinerPhase,
    session_id: SessionId,
    joiner_attempt_id: u32,
}

/// What the joiner does after a host flight.
pub enum JoinerExchangeStep {
    SendToHost(JoinCodeKeyExchange),
    SessionKeyObtained(SessionKey),
}

impl JoinerRelayKeyExchange {
    /// Starts an attempt; returns the state and flight 1 for the host.
    pub fn start(
        join_code_secret: &[u8],
        session_id: SessionId,
        random_source: &mut (impl RngCore + CryptoRng),
    ) -> (Self, JoinCodeKeyExchange) {
        let joiner_attempt_id = random_source.next_u32();
        let (awaiting, joiner_spake_message) =
            JoinerAwaitingHostMessage::start(join_code_secret, session_id, random_source);
        let first_flight = JoinCodeKeyExchange {
            session_id,
            joiner_attempt_id,
            flight: KeyExchangeFlight::JoinerSpakeMessage,
            payload: joiner_spake_message,
        };
        (
            Self {
                phase: JoinerPhase::AwaitingHostMessage(awaiting),
                session_id,
                joiner_attempt_id,
            },
            first_flight,
        )
    }

    pub fn on_host_flight(
        &mut self,
        host_flight: &JoinCodeKeyExchange,
    ) -> Result<JoinerExchangeStep, RelayJoinExchangeError> {
        if host_flight.session_id != self.session_id {
            return Err(RelayJoinExchangeError::WrongSession);
        }
        if host_flight.joiner_attempt_id != self.joiner_attempt_id {
            return Err(RelayJoinExchangeError::WrongAttempt);
        }
        let phase = std::mem::replace(&mut self.phase, JoinerPhase::Finished);
        match (phase, host_flight.flight) {
            (_, KeyExchangeFlight::Rejected) => Err(RelayJoinExchangeError::RejectedByHost),
            (JoinerPhase::AwaitingHostMessage(awaiting), KeyExchangeFlight::HostSpakeMessage) => {
                let (next, confirmation_tag) = awaiting.on_host_spake_message(&host_flight.payload)?;
                self.phase = JoinerPhase::AwaitingSessionKey(next);
                Ok(JoinerExchangeStep::SendToHost(JoinCodeKeyExchange {
                    session_id: self.session_id,
                    joiner_attempt_id: self.joiner_attempt_id,
                    flight: KeyExchangeFlight::JoinerConfirmation,
                    payload: confirmation_tag,
                }))
            }
            (JoinerPhase::AwaitingSessionKey(awaiting), KeyExchangeFlight::SealedSessionKey) => {
                let session_key = awaiting.on_sealed_session_key(&host_flight.payload)?;
                Ok(JoinerExchangeStep::SessionKeyObtained(session_key))
            }
            (JoinerPhase::Finished, _) => Err(RelayJoinExchangeError::ExchangeAlreadyFinished),
            (_, unexpected) => Err(RelayJoinExchangeError::UnexpectedFlight(unexpected)),
        }
    }
}

/// Host side: one responder per live join code, many concurrent attempts
/// keyed by (relay sender peer id, joiner attempt id).
pub struct HostRelayJoinCodeResponder {
    responder: HostJoinCodeResponder,
    session_id: SessionId,
    pending_exchanges: HashMap<(u32, u32), HostPendingJoin>,
    maximum_pending_exchanges: usize,
}

/// Host's answer to one joiner flight: always a flight back to that joiner,
/// plus whether the join completed (so the host can expect a `JoinRequest`).
pub struct HostExchangeReply {
    pub reply_flight: JoinCodeKeyExchange,
    pub joiner_received_session_key: bool,
    /// Set when this flight failed; `JoinCodeAttemptsExhausted` means the
    /// host should revoke the code and issue a new one.
    pub failure: Option<RelayJoinExchangeError>,
}

impl HostRelayJoinCodeResponder {
    pub fn new(
        join_code_secret: &[u8],
        session_id: SessionId,
        maximum_failed_confirmations: u32,
        maximum_pending_exchanges: usize,
    ) -> Self {
        Self {
            responder: HostJoinCodeResponder::new(
                join_code_secret,
                session_id,
                maximum_failed_confirmations,
            ),
            session_id,
            pending_exchanges: HashMap::new(),
            maximum_pending_exchanges,
        }
    }

    pub fn failed_confirmations(&self) -> u32 {
        self.responder.failed_confirmations()
    }

    pub fn is_exhausted(&self) -> bool {
        self.responder.is_exhausted()
    }

    pub fn on_joiner_flight(
        &mut self,
        sender_peer_id: u32,
        joiner_flight: &JoinCodeKeyExchange,
        session_key: &SessionKey,
        random_source: &mut (impl RngCore + CryptoRng),
    ) -> HostExchangeReply {
        let attempt_key = (sender_peer_id, joiner_flight.joiner_attempt_id);
        let outcome = self.advance(attempt_key, joiner_flight, session_key, random_source);
        let (flight, payload, joiner_received_session_key, failure) = match outcome {
            Ok((flight, payload)) => {
                let done = flight == KeyExchangeFlight::SealedSessionKey;
                (flight, payload, done, None)
            }
            Err(error) => {
                self.pending_exchanges.remove(&attempt_key);
                (KeyExchangeFlight::Rejected, Vec::new(), false, Some(error))
            }
        };
        HostExchangeReply {
            reply_flight: JoinCodeKeyExchange {
                session_id: self.session_id,
                joiner_attempt_id: joiner_flight.joiner_attempt_id,
                flight,
                payload,
            },
            joiner_received_session_key,
            failure,
        }
    }

    fn advance(
        &mut self,
        attempt_key: (u32, u32),
        joiner_flight: &JoinCodeKeyExchange,
        session_key: &SessionKey,
        random_source: &mut (impl RngCore + CryptoRng),
    ) -> Result<(KeyExchangeFlight, Vec<u8>), RelayJoinExchangeError> {
        if joiner_flight.session_id != self.session_id {
            return Err(RelayJoinExchangeError::WrongSession);
        }
        match joiner_flight.flight {
            KeyExchangeFlight::JoinerSpakeMessage => {
                if self.pending_exchanges.len() >= self.maximum_pending_exchanges
                    && !self.pending_exchanges.contains_key(&attempt_key)
                {
                    return Err(RelayJoinExchangeError::TooManyPendingExchanges);
                }
                let (pending, host_spake_message) = self
                    .responder
                    .respond_to_joiner(&joiner_flight.payload, random_source)?;
                self.pending_exchanges.insert(attempt_key, pending);
                Ok((KeyExchangeFlight::HostSpakeMessage, host_spake_message))
            }
            KeyExchangeFlight::JoinerConfirmation => {
                let pending = self
                    .pending_exchanges
                    .remove(&attempt_key)
                    .ok_or(RelayJoinExchangeError::WrongAttempt)?;
                let sealed_session_key = self.responder.on_joiner_confirmation(
                    pending,
                    &joiner_flight.payload,
                    session_key,
                    random_source,
                )?;
                Ok((KeyExchangeFlight::SealedSessionKey, sealed_session_key))
            }
            other => Err(RelayJoinExchangeError::UnexpectedFlight(other)),
        }
    }
}
