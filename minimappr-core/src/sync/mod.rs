//! Field Mode acoustic self-survey and clock sync (plan: "Acoustic
//! self-survey and clock sync", "Group splits and acoustic islands").
//!
//! Pipeline per round: play `chirp_waveform`/`chirp_codes` buffers in
//! round-robin slots → `first_arrival` picks each arrival on every phone's
//! own sample clock → `chirp_round` turns the arrival matrix into BeepBeep
//! pairs (`two_way_ranging`) → distances feed `crate::layout`, offsets feed
//! `clock_graph` and then each phone's `clock_filter`. `acoustic_islands`
//! decides which phones share a clock domain at all.
//!
//! Ranging, clock and island code consume arrival *times*, never waveforms,
//! so they work unchanged with clap or audible fallback arrivals.

pub mod acoustic_islands;
pub mod chirp_codes;
pub mod chirp_round;
pub mod chirp_waveform;
pub mod clock_filter;
pub mod clock_graph;
pub mod first_arrival;
pub mod matched_filter;
pub mod split_part_detector;
pub mod two_way_ranging;
