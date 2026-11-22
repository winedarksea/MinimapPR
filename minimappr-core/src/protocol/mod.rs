//! Field Mode wire protocol: one versioned message enum, postcard-encoded,
//! sealed by `session_crypto` before it reaches an untrusted relay.

pub mod detection_messages;
pub mod field_message;
pub mod mesh_messages;
pub mod session_messages;
pub mod wire_codes;

pub use field_message::{
    decode_field_message_frame, DecodedFrame, FieldMessage, FieldMessageDecodeError,
    FieldMessageEncodeError, WireHeader, FIELD_PROTOCOL_MAJOR_VERSION,
    FIELD_PROTOCOL_MINOR_VERSION, KNOWN_MESSAGE_KIND_IDS,
};
pub use wire_codes::{NodeId, SessionId};
