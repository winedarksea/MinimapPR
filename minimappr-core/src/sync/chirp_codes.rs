//! Session-seeded per-phone chirp codes.
//!
//! A code is an up half and a down half, each the chirp band split into
//! `segment_count` sub-bands played in a per-phone hop order (all segments
//! of the up half sweep up, all of the down half sweep down). Each half
//! spans the whole band, so timing resolution is that of the full band,
//! but two different codes only line up where the same sub-band sits at
//! the same time, so their cross-correlation stays well below either
//! autocorrelation peak. That lets a receiver tell overlapping arrivals
//! apart (two islands drifting back into earshot).
//!
//! The halves are picked separately: every up segment shifts its peak the
//! same way under Doppler and every down segment the other way, so the
//! mean of the two half arrivals cancels the walking-Doppler bias (see
//! `split_part_detector`).
//!
//! Codes are drawn from the session seed, so two groups meeting on a trail
//! do not share them.

use super::chirp_waveform::{
    generate_segmented_sweep, ChirpBand, ChirpWaveform, SweepDirection, SweepLaw,
    SweepSpecification,
};

/// Default hop count: 4 sub-bands → 24 × 24 = 576 distinct codes per session.
pub const DEFAULT_CODE_SEGMENT_COUNT: usize = 4;
/// Taper per segment: short ramps keep hops click-free without eating much
/// of each segment's energy.
pub const DEFAULT_CODE_SEGMENT_TAPER_FRACTION: f64 = 0.08;

/// SplitMix64: tiny, deterministic, platform-independent PRNG.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..bound` (bound > 0).
    pub fn next_below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound.max(1)
    }

    /// Uniform in [0, 1).
    pub fn next_unit(&mut self) -> f64 {
        const MANTISSA_BITS: u32 = 53;
        (self.next_u64() >> (64 - MANTISSA_BITS)) as f64 / (1u64 << MANTISSA_BITS) as f64
    }
}

/// FNV-1a 64-bit hash of a session identifier (join code, UUID text).
pub fn session_seed_from_identifier(session_identifier: &str) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    session_identifier
        .bytes()
        .fold(FNV_OFFSET_BASIS, |hash, byte| {
            (hash ^ byte as u64).wrapping_mul(FNV_PRIME)
        })
}

/// One phone's code: sub-band hop order of the up half and of the down half.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneChirpCode {
    pub up_sub_band_order: Vec<usize>,
    pub down_sub_band_order: Vec<usize>,
}

fn all_permutations(count: usize) -> Vec<Vec<usize>> {
    fn recurse(prefix: &mut Vec<usize>, remaining: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if remaining.is_empty() {
            out.push(prefix.clone());
            return;
        }
        for index in 0..remaining.len() {
            let value = remaining.remove(index);
            prefix.push(value);
            recurse(prefix, remaining, out);
            prefix.pop();
            remaining.insert(index, value);
        }
    }
    let mut out = Vec::new();
    recurse(&mut Vec::new(), &mut (0..count).collect(), &mut out);
    out
}

/// Deterministic per-session list of distinct phone codes.
#[derive(Clone, Debug)]
pub struct SessionChirpCodebook {
    pub band: ChirpBand,
    pub total_duration_s: f64,
    pub law: SweepLaw,
    codes: Vec<PhoneChirpCode>,
}

impl SessionChirpCodebook {
    /// Shuffle every (up order, down order) pair with the session seed;
    /// slot index k gets the k-th code.
    pub fn new(
        session_seed: u64,
        band: ChirpBand,
        total_duration_s: f64,
        segment_count: usize,
        law: SweepLaw,
    ) -> Self {
        let orders = all_permutations(segment_count.max(2));
        let mut codes: Vec<PhoneChirpCode> = orders
            .iter()
            .flat_map(|up| {
                orders.iter().map(move |down| PhoneChirpCode {
                    up_sub_band_order: up.clone(),
                    down_sub_band_order: down.clone(),
                })
            })
            .collect();
        let mut rng = SplitMix64::new(session_seed);
        for index in (1..codes.len()).rev() {
            let swap_with = rng.next_below(index as u64 + 1) as usize;
            codes.swap(index, swap_with);
        }
        Self {
            band,
            total_duration_s,
            law,
            codes,
        }
    }

    pub fn distinct_code_count(&self) -> usize {
        self.codes.len()
    }

    pub fn code_for_slot(&self, slot_index: usize) -> &PhoneChirpCode {
        &self.codes[slot_index % self.codes.len()]
    }

    fn half_segments(&self, order: &[usize], direction: SweepDirection) -> Vec<SweepSpecification> {
        let sub_bands = self.band.split_into_sub_bands(order.len());
        let segment_duration_s = 0.5 * self.total_duration_s / order.len() as f64;
        order
            .iter()
            .map(|&sub_band_index| SweepSpecification {
                band: sub_bands[sub_band_index],
                duration_s: segment_duration_s,
                law: self.law,
                direction,
            })
            .collect()
    }

    /// Two-part waveform (up half, down half) for a slot; also the templates.
    pub fn waveform_for_slot(&self, slot_index: usize, sample_rate_hz: f64) -> ChirpWaveform {
        let code = self.code_for_slot(slot_index);
        let render = |order: &[usize], direction| {
            generate_segmented_sweep(
                &self.half_segments(order, direction),
                DEFAULT_CODE_SEGMENT_TAPER_FRACTION,
                sample_rate_hz,
            )
        };
        ChirpWaveform::from_parts(vec![
            render(&code.up_sub_band_order, SweepDirection::Up),
            render(&code.down_sub_band_order, SweepDirection::Down),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::chirp_waveform::CHIRP_BAND_15_TO_18_KHZ;
    use crate::sync::matched_filter::MatchedFilter;

    const SAMPLE_RATE_HZ: f64 = 48_000.0;

    fn codebook(seed: u64) -> SessionChirpCodebook {
        SessionChirpCodebook::new(
            seed,
            CHIRP_BAND_15_TO_18_KHZ,
            0.06,
            DEFAULT_CODE_SEGMENT_COUNT,
            SweepLaw::HyperbolicFm,
        )
    }

    #[test]
    fn codebook_is_deterministic_and_distinct() {
        let first = codebook(42);
        let again = codebook(42);
        assert_eq!(first.distinct_code_count(), 576);
        for slot in 0..8 {
            assert_eq!(first.code_for_slot(slot), again.code_for_slot(slot));
            for other in 0..slot {
                assert_ne!(first.code_for_slot(slot), first.code_for_slot(other));
            }
        }
    }

    #[test]
    fn different_sessions_draw_different_codes() {
        let first = codebook(session_seed_from_identifier("GROUP-A"));
        let second = codebook(session_seed_from_identifier("GROUP-B"));
        let differing = (0..6)
            .filter(|slot| first.code_for_slot(*slot) != second.code_for_slot(*slot))
            .count();
        assert!(differing >= 5);
    }

    #[test]
    fn waveform_has_up_and_down_halves() {
        let waveform = codebook(7).waveform_for_slot(3, SAMPLE_RATE_HZ);
        assert_eq!(waveform.parts.len(), 2);
        assert_eq!(waveform.samples.len(), 2880);
        assert_eq!(
            waveform.parts[1].start_sample,
            waveform.parts[0].samples.len()
        );
    }

    #[test]
    fn cross_correlation_is_well_below_autocorrelation() {
        let book = codebook(1234);
        let first = book.waveform_for_slot(0, SAMPLE_RATE_HZ).samples;
        let second = book.waveform_for_slot(1, SAMPLE_RATE_HZ).samples;
        let filter = MatchedFilter::new(&first);
        let padding = vec![0.0f32; first.len()];
        let pad = |signal: &[f32]| [padding.as_slice(), signal, padding.as_slice()].concat();
        let peak = |signal: &[f32]| {
            filter
                .correlation_envelope(&pad(signal))
                .iter()
                .fold(0.0f32, |m, v| m.max(*v))
        };
        let (auto_peak, cross_peak) = (peak(&first), peak(&second));
        assert!(
            cross_peak < 0.5 * auto_peak,
            "cross {cross_peak} vs auto {auto_peak}"
        );
    }
}
