//! Which GCC-PHAT band, and which event-window taper, each microphone pair of
//! a solve uses. Mirrors `minimappr/core/localization_band_policy.py`.
//!
//! PHAT whitens every bin to unit magnitude, so a bin with no coherent signal
//! adds random phase; the band ceiling is what keeps a compact array's bearing
//! from reversing (a 1500 Hz tone on the 50 mm tetra flipped with a Nyquist
//! band). That ceiling is a property of the array's own spacing, so it is
//! derived per array: c / (2 · d_max) over the array's solving mics (owner
//! decision 2026-11-25). The SIRITH tetra's widest pair (MK1/MK3 to MK4,
//! 52.5 mm) gives ≈ 3266 Hz, the planar's 50 mm diagonal ≈ 3432 Hz.
//!
//! Cross-node pairs have metre-to-kilometre baselines where c / 2d is
//! meaningless for a broadband TDOA, so they use a wide band and a Tukey
//! taper. The taper removes the broadband step a rectangular cut through a
//! loud call leaves at the same sample on every channel, which PHAT turns
//! into a zero-lag peak (E8 point-network bake-off). It is not applied to
//! intra-array pairs: there the true lag is already near zero, and on tonal
//! calls a taper measurably worsened tetra and planar bearings (less
//! rectangular-window leakage leaves fewer coherent bins after whitening).

/// Below this widest-pair spacing an array is treated as coincident: no
/// spatial aliasing, so its intra-array ceiling is Nyquist.
const COINCIDENT_ARRAY_SPACING_M: f32 = 1.0e-6;
/// Whitening exponent of plain PHAT (every bin to unit magnitude).
const PLAIN_PHAT_BETA: f32 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalizationBandPolicy {
    /// Low edge of every pair's band (rumble rejection), Hz.
    pub band_min_hz: f32,
    /// Intra-array ceiling override, Hz; `0` derives c / (2 · d_max) from
    /// the array's own solving mics.
    pub intra_array_band_max_hz: f32,
    /// Cross-node pair ceiling, Hz; `0` means Nyquist.
    pub cross_node_band_max_hz: f32,
    /// Fraction of a cross-node pair's window that is cosine-ramped (half at
    /// each end); `0` keeps the rectangular cut.
    pub cross_node_window_taper_fraction: f32,
    /// Whitening exponent of cross-node pairs (PHAT-beta): `1` is plain PHAT.
    /// Below 1 a bin keeps some of its magnitude, so the empty bins a
    /// band-limited source leaves in the wide band count less than its signal
    /// bins (plain PHAT raised them to equal weight and biased the lag).
    pub cross_node_phat_beta: f32,
}

impl Default for LocalizationBandPolicy {
    fn default() -> Self {
        Self {
            band_min_hz: 50.0,
            intra_array_band_max_hz: 0.0,
            cross_node_band_max_hz: 11_000.0,
            cross_node_window_taper_fraction: 0.2,
            cross_node_phat_beta: 0.7,
        }
    }
}

impl LocalizationBandPolicy {
    /// One fixed band for intra-array pairs (parity fixtures captured at a
    /// fixed band); cross-node pairs keep the defaults.
    pub fn with_intra_array_band(band_hz: [f32; 2]) -> Self {
        Self {
            band_min_hz: band_hz[0],
            intra_array_band_max_hz: band_hz[1],
            ..Self::default()
        }
    }
}

/// Spatial-aliasing ceiling c / (2 · d_max) of one rigid array; `0` (Nyquist)
/// for fewer than two mics or coincident mics, which cannot alias.
pub fn array_alias_ceiling_hz<'a>(
    mic_positions_m: impl IntoIterator<Item = &'a [f32; 3]>,
    sound_speed_mps: f32,
) -> f32 {
    let positions: Vec<&[f32; 3]> = mic_positions_m.into_iter().collect();
    let mut widest_pair_m = 0.0_f32;
    for (index, a) in positions.iter().enumerate() {
        for b in &positions[index + 1..] {
            let separation_m =
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            widest_pair_m = widest_pair_m.max(separation_m);
        }
    }
    if widest_pair_m < COINCIDENT_ARRAY_SPACING_M {
        0.0
    } else {
        sound_speed_mps / (2.0 * widest_pair_m)
    }
}

/// What one pair correlates over: its band (a `0` ceiling means Nyquist; the
/// GCC-PHAT front end clamps it) and its window taper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairFrontEnd {
    pub band_hz: [f32; 2],
    /// `None` keeps the rectangular window.
    pub taper_fraction: Option<f32>,
    /// PHAT-beta whitening exponent; `None` is plain PHAT.
    pub phat_beta: Option<f32>,
}

/// Per-array ceilings of one solve, resolved once and looked up per pair.
#[derive(Clone, Debug)]
pub struct ChannelPairFrontEnds {
    policy: LocalizationBandPolicy,
    /// Indexed by buffer channel: the rigid array (node) it belongs to.
    array_index_by_channel: Vec<usize>,
    /// Indexed by array: its intra-array ceiling, Hz (`0` = Nyquist).
    intra_array_ceiling_hz: Vec<f32>,
}

impl ChannelPairFrontEnds {
    /// `array_index_by_channel` (indexed by buffer channel) says which rigid
    /// array each channel is on; `None` puts every channel on one array (a
    /// single-node solve). Only `active_channels` set an array's geometry:
    /// the ceiling is that of the pairs actually correlated.
    pub fn resolve(
        policy: LocalizationBandPolicy,
        active_channels: &[usize],
        mic_positions_m: &[[f32; 3]],
        array_index_by_channel: Option<&[usize]>,
        sound_speed_mps: f32,
    ) -> Self {
        let array_index_by_channel: Vec<usize> = match array_index_by_channel {
            Some(indices) => indices.to_vec(),
            None => vec![0; mic_positions_m.len()],
        };
        let array_count = array_index_by_channel.iter().copied().max().map_or(0, |max| max + 1);
        let intra_array_ceiling_hz = (0..array_count)
            .map(|array| {
                if policy.intra_array_band_max_hz > 0.0 {
                    return policy.intra_array_band_max_hz;
                }
                array_alias_ceiling_hz(
                    active_channels
                        .iter()
                        .filter(|channel| array_index_by_channel.get(**channel) == Some(&array))
                        .filter_map(|channel| mic_positions_m.get(*channel)),
                    sound_speed_mps,
                )
            })
            .collect();
        Self {
            policy,
            array_index_by_channel,
            intra_array_ceiling_hz,
        }
    }

    pub fn is_cross_node_pair(&self, ch_a: usize, ch_b: usize) -> bool {
        self.array_index_by_channel.get(ch_a) != self.array_index_by_channel.get(ch_b)
    }

    pub fn pair_front_end(&self, ch_a: usize, ch_b: usize) -> PairFrontEnd {
        if self.is_cross_node_pair(ch_a, ch_b) {
            let taper = self.policy.cross_node_window_taper_fraction;
            let beta = self.policy.cross_node_phat_beta;
            return PairFrontEnd {
                band_hz: [self.policy.band_min_hz, self.policy.cross_node_band_max_hz],
                taper_fraction: (taper > 0.0).then_some(taper),
                phat_beta: (beta != PLAIN_PHAT_BETA).then_some(beta),
            };
        }
        let ceiling_hz = self
            .array_index_by_channel
            .get(ch_a)
            .and_then(|array| self.intra_array_ceiling_hz.get(*array))
            .copied()
            .unwrap_or(0.0);
        PairFrontEnd {
            band_hz: [self.policy.band_min_hz, ceiling_hz],
            taper_fraction: None,
            phat_beta: None,
        }
    }
}

/// Tukey taper in place: `taper_fraction` of the window is a raised-cosine
/// ramp, half at each end. Mirrors the Python `tukey_taper`.
pub fn apply_tukey_taper(window: &mut [f32], taper_fraction: f32) {
    let length = window.len();
    let ramp_samples =
        ((0.5 * f64::from(taper_fraction.clamp(0.0, 1.0)) * length as f64) as usize).min(length / 2);
    for index in 0..ramp_samples {
        let gain = 0.5 * (1.0 - (std::f64::consts::PI * index as f64 / ramp_samples as f64).cos());
        window[index] *= gain as f32;
        window[length - 1 - index] *= gain as f32;
    }
}

#[cfg(test)]
#[path = "localization_band_policy_tests.rs"]
mod localization_band_policy_tests;
