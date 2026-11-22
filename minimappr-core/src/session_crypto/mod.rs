//! Session confidentiality against an untrusted relay.
//!
//! - `message_sealing`: XChaCha20-Poly1305 envelopes under the QR session key.
//! - `join_code_key_exchange`: SPAKE2 so a late joiner with a short spoken
//!   join code gets the session key from the host; the code is never the key.
//!
//! Sans-IO: every function that needs randomness takes a caller-supplied
//! `rand_core` CSPRNG (the shell passes the OS RNG), so no getrandom/JS glue
//! is linked and wasm32 builds stay plain.

pub mod join_code_key_exchange;
pub mod message_sealing;

pub use join_code_key_exchange::{
    HostJoinCodeResponder, HostPendingJoin, JoinCodeExchangeError, JoinerAwaitingHostMessage,
    JoinerAwaitingSessionKey, DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS,
};
pub use message_sealing::{
    read_envelope_routing, EnvelopeRouting, MessageSealer, ReplayWindow, SealingError, SessionKey,
};
pub use rand_core::{CryptoRng, RngCore};
