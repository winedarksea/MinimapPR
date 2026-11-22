//! Field Mode clip codec: exact-timing 48 -> 24 kHz decimation, lossless
//! FLAC, and an MMB3-style binary clip container.
//!
//! Pipeline on the phone: rolling buffer -> `FieldClipDecimator` ->
//! `encode_field_clip` (FLAC + container). On the host: `decode_field_clip`
//! gives PCM16 at 24 kHz plus the first sample's device-clock index, so TDOA
//! timing is exact to the 48 kHz sample.

pub mod clip_container;
pub mod decimation_filter_design;
#[cfg(feature = "flac-codec")]
pub mod field_clip;
#[cfg(feature = "flac-codec")]
pub mod flac_clip_codec;
pub mod linear_phase_decimator;

pub use clip_container::{ClipContainerError, FieldClipContainer, FieldClipHeader};
pub use linear_phase_decimator::{DecimatedClipWindow, DecimationError, FieldClipDecimator};

/// Wire-stable id of the decimation filter recorded in every clip header.
/// Never renumber; add new ids for new responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DecimationFilterId {
    /// Anti-alias lowpass (10.5 kHz pass, 12.5 kHz stop, 90 dB) plus the
    /// linear-phase rumble cut (~25 Hz corner). Default for birds.
    BirdBandWithRumbleCut = 1,
    /// Anti-alias lowpass only, flat to DC: drone and gunshot modes keep
    /// sub-100 Hz energy.
    FlatToDirectCurrentForDroneAndGunshot = 2,
}

impl DecimationFilterId {
    pub fn from_wire_byte(wire_byte: u8) -> Option<Self> {
        match wire_byte {
            1 => Some(Self::BirdBandWithRumbleCut),
            2 => Some(Self::FlatToDirectCurrentForDroneAndGunshot),
            _ => None,
        }
    }

    pub fn wire_byte(self) -> u8 {
        self as u8
    }
}

/// Wire-stable id of the audio codec inside a clip container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ClipCodecId {
    /// Interleaved little-endian PCM16 (debug / fallback).
    Pcm16LittleEndian = 0,
    /// Native FLAC stream (`fLaC` magic), 16-bit.
    Flac = 1,
}

impl ClipCodecId {
    pub fn from_wire_byte(wire_byte: u8) -> Option<Self> {
        match wire_byte {
            0 => Some(Self::Pcm16LittleEndian),
            1 => Some(Self::Flac),
            _ => None,
        }
    }

    pub fn wire_byte(self) -> u8 {
        self as u8
    }
}
