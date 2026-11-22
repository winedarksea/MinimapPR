//! Session-seeded per-phone chirp codes.
//!
//! A code is the chirp band split into `segment_count` sub-bands, played in
//! a per-phone hop order with a per-segment sweep direction. The total
//! bandwidth (and so the matched-filter timing resolution) is the whole
//! band, but two different codes only line up where the same sub-band sits
//! at the same time with the same direction, so their cross-correlation
//! stays well below either autocorrelation peak. That lets a receiver tell
//! overlapping arrivals apart (two islands drifting back into earshot).
//!
//! Direction patterns are balanced (as many down-sweeps as up-sweeps): the
//! range–Doppler coupling of an up-sweep and a down-sweep have opposite
//! signs, so a balanced code largely cancels the Doppler timing bias.
//!
//! Codes are drawn from the session seed, so two groups meeting on a trail
//! do not share them.

use super::chirp_waveform::{
    generate_segmented_sweep, ChirpBand, SweepDirection, SweepLaw, SweepSpecification,
};

/// Default hop count: 4 sub-bands → 24 hop orders × 6 balanced direction
/// patterns = 144 distinct codes per session.
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

/// One phone's code: hop order over sub-bands and per-segment directions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneChirpCode {
    pub sub_band_order: Vec<usize>,
    pub segment_directions: Vec<SweepDirection>,
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

fn balanced_direction_patterns(count: usize) -> Vec<Vec<SweepDirection>> {
    (0u32..(1u32 << count))
        .filter(|mask| mask.count_ones() as usize == count / 2)
        .map(|mask| {
            (0..count)
                .map(|bit| {
                    if mask & (1 << bit) != 0 {
                        SweepDirection::Down
                    } else {
                        SweepDirection::Up
                    }
                })
                .collect()
        })
        .collect()
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
    /// Shuffle every (hop order, balanced direction pattern) with the
    /// session seed; slot index k gets the k-th code.
    pub fn new(
        session_seed: u64,
        band: ChirpBand,
        total_duration_s: f64,
        segment_count: usize,
        law: SweepLaw,
    ) -> Self {
        let segment_count = segment_count.max(2);
        let mut codes: Vec<PhoneChirpCode> = Vec::new();
        for order in all_permutations(segment_count) {
            for directions in balanced_direction_patterns(segment_count) {
                codes.push(PhoneChirpCode {
                    sub_band_order: order.clone(),
                    segment_directions: directions,
                });
            }
        }
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

    /// Segment list for a slot's code.
    pub fn segments_for_slot(&self, slot_index: usize) -> Vec<SweepSpecification> {
        let code = self.code_for_slot(slot_index);
        let sub_bands = self.band.split_into_sub_bands(code.sub_band_order.len());
        let segment_duration_s = self.total_duration_s / code.sub_band_order.len() as f64;
        code.sub_band_order
            .iter()
            .zip(&code.segment_directions)
            .map(|(&sub_band_index, &direction)| SweepSpecification {
                band: sub_bands[sub_band_index],
                duration_s: segment_duration_s,
                law: self.law,
                direction,
            })
            .collect()
    }

    /// Unit-peak waveform (and matched-filter template) for a slot.
    pub fn waveform_for_slot(&self, slot_index: usize, sample_rate_hz: f64) -> Vec<f32> {
        generate_segmented_sweep(
            &self.segments_for_slot(slot_index),
            DEFAULT_CODE_SEGMENT_TAPER_FRACTION,
            sample_rate_hz,
        )
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
        assert_eq!(first.distinct_code_count(), 144);
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
    fn direction_patterns_are_balanced() {
        let book = codebook(7);
        for slot in 0..10 {
            let downs = book
                .code_for_slot(slot)
                .segment_directions
                .iter()
                .filter(|d| **d == SweepDirection::Down)
                .count();
            assert_eq!(downs, DEFAULT_CODE_SEGMENT_COUNT / 2);
        }
    }

    #[test]
    fn cross_correlation_is_well_below_autocorrelation() {
        let book = codebook(1234);
        let first = book.waveform_for_slot(0, SAMPLE_RATE_HZ);
        let second = book.waveform_for_slot(1, SAMPLE_RATE_HZ);
        let filter = MatchedFilter::new(&first);
        let padding = vec![0.0f32; first.len()];
        let pad = |signal: &[f32]| [padding.as_slice(), signal, padding.as_slice()].concat();
        let auto_peak = filter
            .correlation_envelope(&pad(&first))
            .iter()
            .fold(0.0f32, |m, v| m.max(*v));
        let cross_peak = filter
            .correlation_envelope(&pad(&second))
            .iter()
            .fold(0.0f32, |m, v| m.max(*v));
        assert!(
            cross_peak < 0.5 * auto_peak,
            "cross {cross_peak} vs auto {auto_peak}"
        );
    }
}
