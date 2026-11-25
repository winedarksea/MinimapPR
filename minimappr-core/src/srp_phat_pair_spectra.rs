//! GCC-PHAT for every admitted pair of one SRP solve, sharing each channel's
//! forward spectrum across the pairs it joins.

use num_complex::Complex32;

use crate::gcc_phat::{
    empty_phat_correlation, phat_correlation_from_spectra, phat_fft_len, phat_forward_spectrum,
    GccPhatCorrelation,
};
use crate::localization_band_policy::{apply_tukey_taper, PairFrontEnd};

/// Each channel's forward spectrum over its whole window, computed the first
/// time a pair needs it. With N channels a solve correlates up to N(N-1)/2
/// pairs, and the per-pair forward FFTs were the bulk of its cost. A channel
/// in both intra-array and cross-node pairs needs two spectra: rectangular
/// and tapered (one taper fraction per solve, from its band policy).
pub(super) struct FullWindowSpectra {
    rectangular_by_channel: Vec<Option<Vec<Complex32>>>,
    tapered_by_channel: Vec<Option<Vec<Complex32>>>,
}

/// The forward spectrum of `samples`, Tukey-tapered first when asked.
fn window_spectrum(samples: &[f32], fft_len: usize, taper_fraction: Option<f32>) -> Vec<Complex32> {
    match taper_fraction {
        None => phat_forward_spectrum(samples, fft_len),
        Some(fraction) => {
            let mut tapered = samples.to_vec();
            apply_tukey_taper(&mut tapered, fraction);
            phat_forward_spectrum(&tapered, fft_len)
        }
    }
}

impl FullWindowSpectra {
    pub(super) fn new(channel_count: usize) -> Self {
        Self {
            rectangular_by_channel: vec![None; channel_count],
            tapered_by_channel: vec![None; channel_count],
        }
    }

    fn cache(&mut self, taper_fraction: Option<f32>) -> &mut Vec<Option<Vec<Complex32>>> {
        if taper_fraction.is_some() {
            &mut self.tapered_by_channel
        } else {
            &mut self.rectangular_by_channel
        }
    }

    fn compute_once(&mut self, windows: &[Vec<f32>], channel: usize, taper_fraction: Option<f32>) {
        let window = &windows[channel];
        self.cache(taper_fraction)[channel].get_or_insert_with(|| {
            window_spectrum(window, phat_fft_len(window.len()), taper_fraction)
        });
    }

    fn computed(&self, channel: usize, taper_fraction: Option<f32>) -> &[Complex32] {
        let cache = if taper_fraction.is_some() {
            &self.tapered_by_channel
        } else {
            &self.rectangular_by_channel
        };
        cache[channel]
            .as_deref()
            .expect("compute_once runs before computed")
    }
}

/// GCC-PHAT of one pair, bit-identical to `gcc_phat::phat_correlation` on the
/// same (tapered) windows: both channels are truncated to the shorter window
/// before any taper, so a channel's cached full-window spectrum stands in
/// only when its window is that length.
#[allow(clippy::too_many_arguments)]
pub(super) fn pair_phat_correlation(
    windows: &[Vec<f32>],
    full_window_spectra: &mut FullWindowSpectra,
    ch_a: usize,
    ch_b: usize,
    sample_rate_hz: u32,
    max_tau_s: f32,
    front_end: PairFrontEnd,
    interp_factor: usize,
) -> GccPhatCorrelation {
    let n = windows[ch_a].len().min(windows[ch_b].len());
    if n == 0 || sample_rate_hz == 0 {
        return empty_phat_correlation();
    }
    let taper_fraction = front_end.taper_fraction;
    let fft_len = phat_fft_len(n);
    let truncated_spectrum =
        |channel: usize| window_spectrum(&windows[channel][..n], fft_len, taper_fraction);
    let spectrum_a_owned = (windows[ch_a].len() != n).then(|| truncated_spectrum(ch_a));
    let spectrum_b_owned = (windows[ch_b].len() != n).then(|| truncated_spectrum(ch_b));
    if spectrum_a_owned.is_none() {
        full_window_spectra.compute_once(windows, ch_a, taper_fraction);
    }
    if spectrum_b_owned.is_none() {
        full_window_spectra.compute_once(windows, ch_b, taper_fraction);
    }
    let spectrum_a = spectrum_a_owned
        .as_deref()
        .unwrap_or_else(|| full_window_spectra.computed(ch_a, taper_fraction));
    let spectrum_b = spectrum_b_owned
        .as_deref()
        .unwrap_or_else(|| full_window_spectra.computed(ch_b, taper_fraction));
    phat_correlation_from_spectra(
        spectrum_a,
        spectrum_b,
        fft_len,
        sample_rate_hz,
        max_tau_s,
        Some(front_end.band_hz),
        interp_factor,
    )
}

#[cfg(test)]
#[path = "srp_phat_pair_spectra_tests.rs"]
mod srp_phat_pair_spectra_tests;
