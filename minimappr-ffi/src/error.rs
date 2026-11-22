//! One error type for every fallible FFI call. Variants carry the core
//! error's `Display` text: the shell logs and surfaces it, it never branches
//! on the inner detail.

#[derive(Debug, Clone, PartialEq, Eq, uniffi::Error)]
pub enum FieldFfiError {
    /// An argument the core would reject (wrong length, out of range).
    InvalidArgument { detail: String },
    /// Clip decimation, FLAC or container failure.
    ClipCodec { detail: String },
    /// Wire frame could not be encoded or decoded.
    ProtocolFrame { detail: String },
}

impl std::fmt::Display for FieldFfiError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument { detail } => write!(formatter, "invalid argument: {detail}"),
            Self::ClipCodec { detail } => write!(formatter, "clip codec: {detail}"),
            Self::ProtocolFrame { detail } => write!(formatter, "protocol frame: {detail}"),
        }
    }
}

impl std::error::Error for FieldFfiError {}
