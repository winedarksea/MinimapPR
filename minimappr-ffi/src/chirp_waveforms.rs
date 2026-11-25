//! Ranging chirp playback buffers from the core generators. Kotlin plays
//! exactly these bytes and the detector correlates against the same
//! `ChirpWaveform`, so playback and template can never disagree.

use minimappr_core::sync::chirp_waveform::{
    generate_plain_chirp, generate_up_down_chirp, stereo_single_channel_buffer, ChirpBand, ChirpWaveform,
    StereoOutputChannel, SweepLaw,
};

use crate::error::FieldFfiError;
use crate::pcm16_bytes::pcm16_little_endian_bytes;

pub const CHIRP_SAMPLE_RATE_HZ: f64 = 48_000.0;
/// Chirps shorter than 5 ms have too little time-bandwidth for a clean pick;
/// longer than 200 ms no longer fit a 250 ms slot with any flight time.
const MINIMUM_CHIRP_DURATION_S: f64 = 0.005;
const MAXIMUM_CHIRP_DURATION_S: f64 = 0.2;
const NYQUIST_HZ: f64 = CHIRP_SAMPLE_RATE_HZ / 2.0;
/// Peaks above full scale clip in the PCM16 conversion.
const MAXIMUM_PEAK_DBFS: f64 = 0.0;
const PCM16_FULL_SCALE: f64 = i16::MAX as f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FieldChirpWaveformKind {
    /// Up then down hyperbolic sweep: Doppler bias cancels in the part mean.
    UpDownHyperbolic,
    HyperbolicFm,
    LinearFm,
}

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct FieldChirpWaveformSpec {
    pub kind: FieldChirpWaveformKind,
    pub band_low_hz: f64,
    pub band_high_hz: f64,
    pub duration_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FieldStereoChannel {
    Left,
    Right,
    /// Both speakers: only for the Phase 1 channel-map survey (a mono chirp
    /// leaves the phone twice, ~15 cm apart, and spoils d_ii).
    Both,
}

pub fn validate_waveform_spec(spec: &FieldChirpWaveformSpec) -> Result<(), FieldFfiError> {
    let band_is_valid = spec.band_low_hz.is_finite()
        && spec.band_high_hz.is_finite()
        && spec.band_low_hz > 0.0
        && spec.band_high_hz > spec.band_low_hz
        && spec.band_high_hz < NYQUIST_HZ;
    if !band_is_valid {
        return Err(FieldFfiError::InvalidArgument {
            detail: format!("chirp band {}..{} Hz must sit inside (0, {NYQUIST_HZ}) Hz", spec.band_low_hz, spec.band_high_hz),
        });
    }
    if !(MINIMUM_CHIRP_DURATION_S..=MAXIMUM_CHIRP_DURATION_S).contains(&spec.duration_s) {
        return Err(FieldFfiError::InvalidArgument {
            detail: format!(
                "chirp duration {} s outside {MINIMUM_CHIRP_DURATION_S}..={MAXIMUM_CHIRP_DURATION_S} s",
                spec.duration_s
            ),
        });
    }
    Ok(())
}

/// Unit-peak waveform with its detection parts.
pub fn chirp_waveform_from_spec(spec: &FieldChirpWaveformSpec) -> Result<ChirpWaveform, FieldFfiError> {
    validate_waveform_spec(spec)?;
    let band = ChirpBand { low_hz: spec.band_low_hz, high_hz: spec.band_high_hz };
    Ok(match spec.kind {
        FieldChirpWaveformKind::UpDownHyperbolic => {
            generate_up_down_chirp(band, spec.duration_s, SweepLaw::HyperbolicFm, CHIRP_SAMPLE_RATE_HZ)
        }
        FieldChirpWaveformKind::HyperbolicFm => ChirpWaveform::single_part(generate_plain_chirp(
            band,
            spec.duration_s,
            SweepLaw::HyperbolicFm,
            CHIRP_SAMPLE_RATE_HZ,
        )),
        FieldChirpWaveformKind::LinearFm => ChirpWaveform::single_part(generate_plain_chirp(
            band,
            spec.duration_s,
            SweepLaw::LinearFm,
            CHIRP_SAMPLE_RATE_HZ,
        )),
    })
}

/// Interleaved stereo PCM16 little-endian (`[L0, R0, L1, R1, …]`).
pub fn stereo_pcm16_from_waveform(
    waveform: &ChirpWaveform,
    channel: FieldStereoChannel,
    peak_dbfs: f64,
) -> Result<Vec<u8>, FieldFfiError> {
    if !peak_dbfs.is_finite() || peak_dbfs > MAXIMUM_PEAK_DBFS {
        return Err(FieldFfiError::InvalidArgument { detail: format!("peak {peak_dbfs} dBFS must be finite and <= 0") });
    }
    let gain = 10f64.powf(peak_dbfs / 20.0) * PCM16_FULL_SCALE;
    let interleaved: Vec<f32> = match channel {
        FieldStereoChannel::Left => stereo_single_channel_buffer(&waveform.samples, StereoOutputChannel::Left),
        FieldStereoChannel::Right => stereo_single_channel_buffer(&waveform.samples, StereoOutputChannel::Right),
        FieldStereoChannel::Both => waveform.samples.iter().flat_map(|sample| [*sample, *sample]).collect(),
    };
    let samples: Vec<i16> = interleaved
        .iter()
        .map(|sample| (*sample as f64 * gain).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16)
        .collect();
    Ok(pcm16_little_endian_bytes(&samples))
}

/// Playback buffer for one chirp on the chosen speaker channel.
#[uniffi::export]
pub fn chirp_playback_pcm16(
    spec: FieldChirpWaveformSpec,
    channel: FieldStereoChannel,
    peak_dbfs: f64,
) -> Result<Vec<u8>, FieldFfiError> {
    stereo_pcm16_from_waveform(&chirp_waveform_from_spec(&spec)?, channel, peak_dbfs)
}

/// Length of one chirp in frames (stereo frames = mono samples).
#[uniffi::export]
pub fn chirp_frame_count(spec: FieldChirpWaveformSpec) -> Result<u32, FieldFfiError> {
    Ok(chirp_waveform_from_spec(&spec)?.samples.len() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcm16_bytes::pcm16_samples_from_little_endian_bytes;

    pub(crate) fn default_spec() -> FieldChirpWaveformSpec {
        FieldChirpWaveformSpec {
            kind: FieldChirpWaveformKind::UpDownHyperbolic,
            band_low_hz: 13_000.0,
            band_high_hz: 16_000.0,
            duration_s: 0.05,
        }
    }

    fn channel_samples(bytes: &[u8], channel_offset: usize) -> Vec<i16> {
        pcm16_samples_from_little_endian_bytes(bytes).unwrap().into_iter().skip(channel_offset).step_by(2).collect()
    }

    #[test]
    fn buffer_has_one_stereo_frame_per_waveform_sample() {
        let bytes = chirp_playback_pcm16(default_spec(), FieldStereoChannel::Right, -6.0).unwrap();
        assert_eq!(bytes.len(), 2_400 * 2 * 2);
        assert_eq!(chirp_frame_count(default_spec()).unwrap(), 2_400);
    }

    #[test]
    fn unused_channel_is_silent_and_peak_follows_dbfs() {
        let bytes = chirp_playback_pcm16(default_spec(), FieldStereoChannel::Left, -6.0).unwrap();
        assert!(channel_samples(&bytes, 1).iter().all(|sample| *sample == 0));
        let peak = channel_samples(&bytes, 0).iter().map(|s| s.unsigned_abs()).max().unwrap() as f64;
        let expected = 10f64.powf(-6.0 / 20.0) * PCM16_FULL_SCALE;
        assert!((peak - expected).abs() / expected < 0.01, "peak {peak} vs {expected}");
    }

    #[test]
    fn both_channels_carry_identical_samples() {
        let bytes = chirp_playback_pcm16(default_spec(), FieldStereoChannel::Both, -3.0).unwrap();
        assert_eq!(channel_samples(&bytes, 0), channel_samples(&bytes, 1));
    }

    #[test]
    fn invalid_specs_are_rejected() {
        let bad_band = FieldChirpWaveformSpec { band_high_hz: 30_000.0, ..default_spec() };
        assert!(chirp_playback_pcm16(bad_band, FieldStereoChannel::Left, -6.0).is_err());
        let bad_duration = FieldChirpWaveformSpec { duration_s: 1.0, ..default_spec() };
        assert!(chirp_playback_pcm16(bad_duration, FieldStereoChannel::Left, -6.0).is_err());
        assert!(chirp_playback_pcm16(default_spec(), FieldStereoChannel::Left, 3.0).is_err());
    }
}
