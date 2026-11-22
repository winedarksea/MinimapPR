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
pub mod range_projection;
pub mod sensor_stream_buffer;
pub mod sirith_array_geometry;
pub mod srp_phat;
