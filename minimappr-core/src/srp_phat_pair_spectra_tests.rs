use super::*;
use crate::gcc_phat::phat_correlation;
use crate::localization_band_policy::{apply_tukey_taper, PairFrontEnd};

fn pseudo_random_window(mut seed: u32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

fn assert_bit_identical(left: &GccPhatCorrelation, right: &GccPhatCorrelation) {
    let bits = |values: &[f32]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&left.lags_seconds), bits(&right.lags_seconds));
    assert_eq!(bits(&left.magnitudes), bits(&right.magnitudes));
    assert_eq!(
        left.tdoa.delay_samples.to_bits(),
        right.tdoa.delay_samples.to_bits()
    );
    assert_eq!(
        left.tdoa.lag_seconds.to_bits(),
        right.tdoa.lag_seconds.to_bits()
    );
    assert_eq!(
        left.tdoa.confidence.to_bits(),
        right.tdoa.confidence.to_bits()
    );
    assert_eq!(
        left.tdoa.peak_value.to_bits(),
        right.tdoa.peak_value.to_bits()
    );
}

#[test]
fn shared_spectra_reproduce_per_pair_correlation_bit_for_bit() {
    // Unequal lengths (a late member's short window) and an empty channel
    // exercise the truncated and empty paths beside the cached one.
    let windows = vec![
        pseudo_random_window(1, 1536),
        pseudo_random_window(2, 1536),
        pseudo_random_window(3, 1200),
        Vec::new(),
        pseudo_random_window(5, 1536),
    ];
    let mut spectra = FullWindowSpectra::new(windows.len());
    let front_ends = [
        (1.0e-4, PairFrontEnd { band_hz: [50.0, 3430.0], taper_fraction: None, phat_beta: None }),
        (0.012, PairFrontEnd { band_hz: [0.0, 0.0], taper_fraction: None, phat_beta: None }),
        (
            0.012,
            PairFrontEnd { band_hz: [50.0, 11_000.0], taper_fraction: Some(0.2), phat_beta: Some(0.8) },
        ),
    ];
    let windowed = |channel: usize, n: usize, taper_fraction: Option<f32>| {
        let mut window = windows[channel][..n].to_vec();
        if let Some(fraction) = taper_fraction {
            apply_tukey_taper(&mut window, fraction);
        }
        window
    };
    for ch_a in 0..windows.len() {
        for ch_b in ch_a + 1..windows.len() {
            // The cache holds rectangular and tapered spectra side by side, so
            // a channel's earlier pairs must not leak into a later front end.
            for (max_tau_s, front_end) in front_ends {
                let shared = pair_phat_correlation(
                    &windows,
                    &mut spectra,
                    ch_a,
                    ch_b,
                    48_000,
                    max_tau_s,
                    front_end,
                    4,
                );
                let n = windows[ch_a].len().min(windows[ch_b].len());
                let per_pair = if n == 0 {
                    phat_correlation(&[], &[], 48_000, max_tau_s, None, 4)
                } else {
                    let fft_len = phat_fft_len(n);
                    phat_correlation_from_spectra(
                        &phat_forward_spectrum(&windowed(ch_a, n, front_end.taper_fraction), fft_len),
                        &phat_forward_spectrum(&windowed(ch_b, n, front_end.taper_fraction), fft_len),
                        fft_len,
                        48_000,
                        max_tau_s,
                        Some(front_end.band_hz),
                        front_end.phat_beta,
                        4,
                    )
                };
                assert_bit_identical(&shared, &per_pair);
            }
        }
    }
}
