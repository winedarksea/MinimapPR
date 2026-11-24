//! `minimappr-core`: platform-neutral DSP and localization building blocks.
//!
//! Contract: std + serde + rustfft + num-complex, plus base64/serde_json for
//! the store-forward payload parser. No async runtime, filesystem or
//! wall-clock access, so the same code runs in the native sidecar, on phones
//! and in the WASM PWA. Logging is opt-in via the `tracing` feature.

pub mod audio_payload;
pub mod butterworth_biquad;
pub mod dsp_math;
pub mod gcc_phat;
// Localization overhaul R0: Rust <-> Python solve-provenance + geometry-push wire types.
pub mod localization_sensor_entry;
pub mod range_projection;
// Localization overhaul R2: Python-identical site ENU projection + sensor rotation.
pub mod sensor_world_position;
pub mod site_frame_projection;
pub mod sensor_stream_buffer;
pub mod sirith_array_geometry;
pub mod srp_layout;
pub mod srp_phat;
// W2-core-codec-proto
pub mod clip_codec;
pub mod protocol;
#[cfg(feature = "session-crypto")]
pub mod session_crypto;
pub mod stage1;
// W2-core-solver
pub mod tdoa_solver;
// W2-core-sync: chirp sync, clock model, acoustic islands, layout + GPS anchor.
pub mod layout;
pub mod sync;
// W3-mesh-sim: sans-IO node/host state machines.
pub mod mesh;
// W4-port-server-track (S2): onset-frame SRP clustering + echo checks.
pub mod onset_clustering;
// W4-port-server-track: perched tracker + species log-odds (S3), bird entities (S6).
pub mod wildlife_tracking;
pub mod bird_entity;
