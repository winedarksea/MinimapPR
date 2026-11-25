//! `minimappr-ffi`: the UniFFI surface the Field Mode KMP app (`field-app/`)
//! calls through Gobley-generated Kotlin bindings.
//!
//! Contract: a thin adapter, no logic of its own beyond type conversion and
//! the node's PCM rolling buffer. Capability groups:
//! - [`stage1_detector`]: the always-on Stage-1 trigger.
//! - [`clip_codec`]: 48 -> 24 kHz decimation + FLAC/MMCL clip encode/decode.
//! - [`protocol_frames`]: wire-frame inspection and canonical re-encoding.
//! - [`chirp_waveforms`] / [`chirp_detector`]: ranging chirp playback buffers and picks.
//! - [`mesh_device`]: one phone in a LAN session (node, optional host, session
//!   clock, chirp listener), stepped by the shell with local monotonic time.
//!
//! Field messages cross the boundary only as encoded frames (`Vec<u8>`), so
//! the 33-kind message enum never has to be mirrored in UniFFI types.

pub mod chirp_detector;
pub mod chirp_waveforms;
pub mod clip_codec;
pub mod error;
pub mod capture_clock_anchor;
pub mod chirp_round_listener;
pub mod lan_envelope;
pub mod mesh_device;
mod mesh_device_lan;
mod mesh_device_routing;
pub mod mesh_tuning;
pub mod sync_status;
pub mod node_rolling_audio_buffer;
pub mod pcm16_bytes;
pub mod protocol_frames;
pub mod stage1_detector;

pub use error::FieldFfiError;

/// Version of this FFI surface; bump on any breaking signature change so the
/// shell can refuse a mismatched native library at startup.
pub const FIELD_FFI_SURFACE_VERSION: u32 = 3;

/// Lets the shell verify the native library loaded and matches its bindings.
#[uniffi::export]
pub fn field_ffi_surface_version() -> u32 {
    FIELD_FFI_SURFACE_VERSION
}

uniffi::setup_scaffolding!();
