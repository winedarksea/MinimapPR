//! GCC-PHAT for every admitted pair of one SRP solve, sharing each channel's
//! forward spectrum across the pairs it joins.

use num_complex::Complex32;

use crate::gcc_phat::{
    empty_phat_correlation, phat_correlation_from_spectra, phat_fft_len, phat_forward_spectrum,
    GccPhatCorrelation,
};

/// Each channel's forward spectrum over its whole window, computed the first
/// time a pair needs it. With N channels a solve correlates up to N(N-1)/2
/// pairs, and the per-pair forward FFTs were the bulk of its cost.
pub(super) struct FullWindowSpectra {
    by_channel: Vec<Option<Vec<Complex32>>>,
}

impl FullWindowSpectra {
    pub(super) fn new(channel_count: usize) -> Self {
        Self {
            by_channel: vec![None; channel_count],
        }
    }

    fn compute_once(&mut self, windows: &[Vec<f32>], channel: usize) {
        let window = &windows[channel];
        self.by_channel[channel]
            .get_or_insert_with(|| phat_forward_spectrum(window, phat_fft_len(window.len())));
    }

    fn computed(&self, channel: usize) -> &[Complex32] {
        self.by_channel[channel]
            .as_deref()
            .expect("compute_once runs before computed")
    }
}

/// GCC-PHAT of one pair, bit-identical to `gcc_phat::phat_correlation`: both
/// channels are truncated to the shorter window, so a channel's cached
/// full-window spectrum stands in only when its window is that length.
#[allow(clippy::too_many_arguments)]
pub(super) fn pair_phat_correlation(
    windows: &[Vec<f32>],
    full_window_spectra: &mut FullWindowSpectra,
    ch_a: usize,
    ch_b: usize,
    sample_rate_hz: u32,
    max_tau_s: f32,
    band_hz: Option<[f32; 2]>,
    interp_factor: usize,
) -> GccPhatCorrelation {
    let n = windows[ch_a].len().min(windows[ch_b].len());
    if n == 0 || sample_rate_hz == 0 {
        return empty_phat_correlation();
    }
    let fft_len = phat_fft_len(n);
    let truncated_spectrum =
        |channel: usize| phat_forward_spectrum(&windows[channel][..n], fft_len);
    let spectrum_a_owned = (windows[ch_a].len() != n).then(|| truncated_spectrum(ch_a));
    let spectrum_b_owned = (windows[ch_b].len() != n).then(|| truncated_spectrum(ch_b));
    if spectrum_a_owned.is_none() {
        full_window_spectra.compute_once(windows, ch_a);
    }
    if spectrum_b_owned.is_none() {
        full_window_spectra.compute_once(windows, ch_b);
    }
    let spectrum_a = spectrum_a_owned
        .as_deref()
        .unwrap_or_else(|| full_window_spectra.computed(ch_a));
    let spectrum_b = spectrum_b_owned
        .as_deref()
        .unwrap_or_else(|| full_window_spectra.computed(ch_b));
    phat_correlation_from_spectra(
        spectrum_a,
        spectrum_b,
        fft_len,
        sample_rate_hz,
        max_tau_s,
        band_hz,
        interp_factor,
    )
}

#[cfg(test)]
#[path = "srp_phat_pair_spectra_tests.rs"]
mod srp_phat_pair_spectra_tests;
