//! Exact-timing 48 -> 24 kHz decimation of a clip window cut from a phone's
//! rolling buffer.
//!
//! Timing contract: output sample `n` is the filtered signal *at* device-clock
//! sample `window_first_sample_index + 2 n`. Both filter stages are symmetric
//! FIRs with integer group delay, and the delay is removed by centring the
//! convolution on that index rather than by shifting afterwards, so an impulse
//! at device sample `window_first + 2 k` peaks at output `k` exactly.
//!
//! The rolling buffer supplies real neighbouring audio as filter context;
//! only context that falls outside the buffer is zero-padded (and flagged).

use super::decimation_filter_design::{
    design_kaiser_windowed_sinc_lowpass, FIELD_CLIP_ANTI_ALIAS_LOWPASS_SPECIFICATION,
    FIELD_CLIP_DECIMATION_FACTOR,
};
use super::DecimationFilterId;

/// Bird-mode rumble cut: two cascaded centred moving averages of this odd
/// length at 24 kHz form a triangle kernel `T`; the cut is `delta - T`.
/// Length 481 puts -4.4 dB near 25 Hz and -17 dB near 10 Hz: it removes wind
/// and handling DC drift but keeps grouse drumming (~40 Hz) mostly intact.
pub const BIRD_MODE_RUMBLE_CUT_BOXCAR_LENGTH_OUTPUT_SAMPLES: usize = 481;

const PCM16_MAXIMUM: f64 = i16::MAX as f64;
const PCM16_MINIMUM: f64 = i16::MIN as f64;

/// Decimated clip window plus the exact device-clock mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecimatedClipWindow {
    pub decimation_filter_id: DecimationFilterId,
    /// Device-clock (48 kHz) index of output sample 0.
    pub first_sample_index_device_clock: u64,
    /// Output sample `n` sits at device index `first + n * decimation_factor`.
    pub decimation_factor: u32,
    pub samples_output_rate: Vec<i16>,
    /// Output samples that saturated PCM16 after filtering.
    pub clipped_output_sample_count: u32,
    /// True when some filter context fell outside the rolling buffer and was
    /// zero-padded, i.e. the first/last few ms carry an edge transient.
    pub filter_context_was_zero_padded: bool,
}

/// Errors from window extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecimationError {
    EmptyWindow,
    WindowOutsideRollingBuffer,
}

impl std::fmt::Display for DecimationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyWindow => write!(formatter, "clip window has zero samples"),
            Self::WindowOutsideRollingBuffer => {
                write!(formatter, "clip window does not overlap the rolling buffer")
            }
        }
    }
}

impl std::error::Error for DecimationError {}

/// Two-stage linear-phase decimator for one `DecimationFilterId`.
#[derive(Debug, Clone)]
pub struct FieldClipDecimator {
    decimation_filter_id: DecimationFilterId,
    anti_alias_lowpass_taps: Vec<f64>,
    rumble_cut_boxcar_length_output_samples: Option<usize>,
}

impl FieldClipDecimator {
    pub fn for_filter(decimation_filter_id: DecimationFilterId) -> Self {
        let rumble_cut_boxcar_length_output_samples = match decimation_filter_id {
            DecimationFilterId::BirdBandWithRumbleCut => {
                Some(BIRD_MODE_RUMBLE_CUT_BOXCAR_LENGTH_OUTPUT_SAMPLES)
            }
            DecimationFilterId::FlatToDirectCurrentForDroneAndGunshot => None,
        };
        Self {
            decimation_filter_id,
            anti_alias_lowpass_taps: design_kaiser_windowed_sinc_lowpass(
                &FIELD_CLIP_ANTI_ALIAS_LOWPASS_SPECIFICATION,
            ),
            rumble_cut_boxcar_length_output_samples,
        }
    }

    pub fn decimation_filter_id(&self) -> DecimationFilterId {
        self.decimation_filter_id
    }

    pub fn anti_alias_lowpass_taps(&self) -> &[f64] {
        &self.anti_alias_lowpass_taps
    }

    /// Group delay of the anti-alias stage in 48 kHz samples (removed).
    pub fn anti_alias_group_delay_input_samples(&self) -> usize {
        (self.anti_alias_lowpass_taps.len() - 1) / 2
    }

    /// Group delay of the rumble cut in 24 kHz samples (removed); 0 if none.
    pub fn rumble_cut_group_delay_output_samples(&self) -> usize {
        self.rumble_cut_boxcar_length_output_samples
            .map_or(0, |boxcar_length| boxcar_length - 1)
    }

    /// Device-clock samples of context needed on each side of a window for a
    /// transient-free result. Callers cut clips with at least this margin
    /// still inside the rolling buffer.
    pub fn required_context_input_samples_each_side(&self) -> usize {
        self.anti_alias_group_delay_input_samples()
            + self.rumble_cut_group_delay_output_samples() * FIELD_CLIP_DECIMATION_FACTOR
    }

    /// Decimates `[window_first, window_first + window_input_sample_count)` of
    /// the device clock, reading filter context from `rolling_buffer`, whose
    /// first element is device sample `rolling_buffer_first_sample_index`.
    pub fn decimate_window_from_rolling_buffer(
        &self,
        rolling_buffer: &[i16],
        rolling_buffer_first_sample_index: u64,
        window_first_sample_index: u64,
        window_input_sample_count: usize,
    ) -> Result<DecimatedClipWindow, DecimationError> {
        if window_input_sample_count == 0 {
            return Err(DecimationError::EmptyWindow);
        }
        let buffer_start = i128::from(rolling_buffer_first_sample_index);
        let buffer_end = buffer_start + rolling_buffer.len() as i128;
        let window_start = i128::from(window_first_sample_index);
        let window_end = window_start + window_input_sample_count as i128;
        if window_end <= buffer_start || window_start >= buffer_end {
            return Err(DecimationError::WindowOutsideRollingBuffer);
        }
        let context = self.required_context_input_samples_each_side() as i128;
        let filter_context_was_zero_padded =
            window_start - context < buffer_start || window_end + context > buffer_end;

        let output_sample_count = window_input_sample_count.div_ceil(FIELD_CLIP_DECIMATION_FACTOR);
        let rumble_margin = self.rumble_cut_group_delay_output_samples();
        let extended_output_count = output_sample_count + 2 * rumble_margin;
        let read_device_sample = |device_index: i128| -> f64 {
            if device_index < buffer_start || device_index >= buffer_end {
                0.0
            } else {
                f64::from(rolling_buffer[(device_index - buffer_start) as usize])
            }
        };
        let anti_alias_delay = self.anti_alias_group_delay_input_samples() as i128;
        let lowpassed_extended: Vec<f64> = (0..extended_output_count)
            .map(|extended_index| {
                let centre_device_index = window_start
                    + (extended_index as i128 - rumble_margin as i128)
                        * FIELD_CLIP_DECIMATION_FACTOR as i128;
                // y(p) = sum_k h[k] x(p + D - k): centred, so zero net delay.
                self.anti_alias_lowpass_taps
                    .iter()
                    .enumerate()
                    .map(|(tap_index, tap)| {
                        tap * read_device_sample(
                            centre_device_index + anti_alias_delay - tap_index as i128,
                        )
                    })
                    .sum()
            })
            .collect();

        let filtered_output: Vec<f64> = match self.rumble_cut_boxcar_length_output_samples {
            Some(boxcar_length) => {
                apply_centred_triangle_rumble_cut(&lowpassed_extended, boxcar_length)
            }
            None => lowpassed_extended,
        };
        debug_assert_eq!(filtered_output.len(), output_sample_count);

        let mut clipped_output_sample_count = 0_u32;
        let samples_output_rate = filtered_output
            .iter()
            .map(|value| {
                let rounded = value.round();
                if !(PCM16_MINIMUM..=PCM16_MAXIMUM).contains(&rounded) {
                    clipped_output_sample_count += 1;
                }
                rounded.clamp(PCM16_MINIMUM, PCM16_MAXIMUM) as i16
            })
            .collect();
        Ok(DecimatedClipWindow {
            decimation_filter_id: self.decimation_filter_id,
            first_sample_index_device_clock: window_first_sample_index,
            decimation_factor: FIELD_CLIP_DECIMATION_FACTOR as u32,
            samples_output_rate,
            clipped_output_sample_count,
            filter_context_was_zero_padded,
        })
    }
}

/// `x - T * x` with `T` two cascaded centred moving averages of odd length
/// `boxcar_length`. Input carries `boxcar_length - 1` margin samples on each
/// side; output is the margin-free centre, so the kernel's delay is removed.
fn apply_centred_triangle_rumble_cut(extended_signal: &[f64], boxcar_length: usize) -> Vec<f64> {
    let boxcar_half_width = (boxcar_length - 1) / 2;
    let first_average = centred_moving_average_valid(extended_signal, boxcar_half_width);
    let second_average = centred_moving_average_valid(&first_average, boxcar_half_width);
    let total_margin = 2 * boxcar_half_width;
    second_average
        .iter()
        .enumerate()
        .map(|(output_index, triangle_smoothed)| {
            extended_signal[output_index + total_margin] - triangle_smoothed
        })
        .collect()
}

/// Centred moving average over `2 * half_width + 1` samples, "valid" part
/// only (output shorter by `2 * half_width`). Prefix sums keep it O(n).
fn centred_moving_average_valid(signal: &[f64], half_width: usize) -> Vec<f64> {
    let window_length = 2 * half_width + 1;
    if signal.len() < window_length {
        return Vec::new();
    }
    let mut prefix_sums = Vec::with_capacity(signal.len() + 1);
    prefix_sums.push(0.0);
    for value in signal {
        prefix_sums.push(prefix_sums.last().copied().unwrap_or(0.0) + value);
    }
    (0..=signal.len() - window_length)
        .map(|start| {
            (prefix_sums[start + window_length] - prefix_sums[start]) / window_length as f64
        })
        .collect()
}
