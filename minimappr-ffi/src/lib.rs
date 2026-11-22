//! `minimappr-ffi`: the UniFFI surface the Field Mode KMP app (`field-app/`)
//! calls through Gobley-generated Kotlin bindings.
//!
//! Contract: a thin adapter, no logic of its own beyond type conversion and
//! the node's PCM rolling buffer. Four capability groups:
//! - [`stage1_detector`]: the always-on Stage-1 trigger.
//! - [`clip_codec`]: 48 -> 24 kHz decimation + FLAC/MMCL clip encode/decode.
//! - [`protocol_frames`]: wire-frame inspection and canonical re-encoding.
//! - [`mesh_node`]: the sans-IO `FieldNodeStateMachine`, stepped by the shell.
//!
//! Field messages cross the boundary only as encoded frames (`Vec<u8>`), so
//! the 33-kind message enum never has to be mirrored in UniFFI types.

pub mod clip_codec;
pub mod error;
pub mod mesh_node;
pub mod node_rolling_audio_buffer;
pub mod pcm16_bytes;
pub mod protocol_frames;
pub mod stage1_detector;

pub use error::FieldFfiError;

/// Version of this FFI surface; bump on any breaking signature change so the
/// shell can refuse a mismatched native library at startup.
pub const FIELD_FFI_SURFACE_VERSION: u32 = 1;

/// Lets the shell verify the native library loaded and matches its bindings.
#[uniffi::export]
pub fn field_ffi_surface_version() -> u32 {
    FIELD_FFI_SURFACE_VERSION
}

uniffi::setup_scaffolding!();
