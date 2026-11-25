use super::*;
use crate::gcc_phat::phat_correlation;

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
    for ch_a in 0..windows.len() {
        for ch_b in ch_a + 1..windows.len() {
            for (max_tau_s, band_hz) in [(1.0e-4, Some([50.0, 3430.0])), (0.012, None)] {
                let shared = pair_phat_correlation(
                    &windows,
                    &mut spectra,
                    ch_a,
                    ch_b,
                    48_000,
                    max_tau_s,
                    band_hz,
                    4,
                );
                let per_pair = phat_correlation(
                    &windows[ch_a],
                    &windows[ch_b],
                    48_000,
                    max_tau_s,
                    band_hz,
                    4,
                );
                assert_bit_identical(&shared, &per_pair);
            }
        }
    }
}
