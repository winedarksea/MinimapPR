//! Cheap seeded noise. Background noise covers every sample of every
//! channel (tens of millions of draws per render), so it uses one `u64`
//! per sample: four 16-bit uniforms summed (Irwin–Hall, n = 4) give a
//! near-Gaussian unit-variance value. Exact Gaussian tails don't matter
//! for ambience; speed does.

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

const IRWIN_HALL_TERMS: f64 = 4.0;
const UNIFORM_16_BIT_SCALE: f64 = 1.0 / 65536.0;
/// Variance of a sum of n U(0,1) is n/12.
const IRWIN_HALL_STANDARD_DEVIATION: f64 = 0.577_350_269_189_625_8; // sqrt(4/12)
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Approximately N(0, 1) from a single 64-bit draw.
#[inline]
pub fn standard_normal_approx<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    irwin_hall_from_bits(rng.next_u64())
}

/// Independent, reproducible RNG stream for (scenario seed, purpose, index).
/// Keyed by name so enabling one chaos knob never perturbs another's draws.
pub fn derive_stream_rng(scenario_seed: u64, purpose: &str, index: u64) -> ChaCha8Rng {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in purpose.bytes().chain(index.to_le_bytes()) {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    ChaCha8Rng::seed_from_u64(scenario_seed ^ hash)
}

/// xoshiro256++ — ~1 ns per draw, used only for whole-stream background
/// noise (seeded from a ChaCha stream, so still fully reproducible).
pub struct FastNoiseRng {
    state: [u64; 4],
}

impl FastNoiseRng {
    pub fn from_seed_stream(seed_stream: &mut ChaCha8Rng) -> Self {
        let mut state = [0u64; 4];
        for word in state.iter_mut() {
            *word = seed_stream.next_u64();
        }
        if state == [0; 4] {
            state[0] = FNV_OFFSET_BASIS;
        }
        Self { state }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    #[inline]
    pub fn standard_normal_approx(&mut self) -> f32 {
        irwin_hall_from_bits(self.next_u64())
    }
}

#[inline]
fn irwin_hall_from_bits(bits: u64) -> f32 {
    let sum = (bits & 0xFFFF) as f64
        + ((bits >> 16) & 0xFFFF) as f64
        + ((bits >> 32) & 0xFFFF) as f64
        + (bits >> 48) as f64;
    ((sum * UNIFORM_16_BIT_SCALE - IRWIN_HALL_TERMS / 2.0) / IRWIN_HALL_STANDARD_DEVIATION) as f32
}

/// Streaming background noise: white noise of RMS `white_rms` (mic self
/// noise) plus independent low-pass-tilted noise of RMS `tilted_rms`
/// (one-pole at `tilt_corner_hz`: forest/wind ambience is bass-heavy).
pub struct BackgroundNoiseSource {
    rng: FastNoiseRng,
    white_rms: f32,
    tilt_input_gain: f32,
    pole: f32,
    tilted_state: f32,
}

impl BackgroundNoiseSource {
    pub fn new(
        seed_stream: &mut ChaCha8Rng,
        white_rms: f32,
        tilted_rms: f32,
        tilt_corner_hz: f64,
        sample_rate_hz: f64,
    ) -> Self {
        let pole = (-std::f64::consts::TAU * tilt_corner_hz / sample_rate_hz).exp() as f32;
        // One-pole output variance = (1 - p) / (1 + p) for unit-variance input.
        let tilt_gain = tilted_rms * ((1.0 + pole) / (1.0 - pole)).sqrt();
        Self {
            rng: FastNoiseRng::from_seed_stream(seed_stream),
            white_rms,
            tilt_input_gain: tilt_gain * (1.0 - pole),
            pole,
            tilted_state: 0.0,
        }
    }

    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        let white = self.rng.standard_normal_approx();
        let for_tilt = self.rng.standard_normal_approx();
        self.tilted_state = self.pole * self.tilted_state + self.tilt_input_gain * for_tilt;
        self.white_rms * white + self.tilted_state
    }
}

/// Uniform draw in [low, high] (returns `low` for an empty range).
pub fn uniform_in_range<R: RngCore + ?Sized>(rng: &mut R, low: f64, high: f64) -> f64 {
    if high <= low {
        return low;
    }
    low + (high - low) * (rng.next_u64() as f64 / u64::MAX as f64)
}
