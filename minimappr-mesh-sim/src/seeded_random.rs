//! Seeds and deterministic randomness.
//!
//! One scenario seed drives everything: geometry jitter, clocks, GPS error,
//! network latency/loss, timing noise. Each purpose draws from its own
//! ChaCha stream, and per-(device, event) noise is a pure hash of the seed,
//! so the order in which state machines happen to ask never changes a draw.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub const SEED_ENVIRONMENT_VARIABLE: &str = "MINIMAPPR_SIM_SEED";

/// The fixed PR list (test strategy: 10 seeds per PR, regression seeds
/// from failed nightly runs are appended here after the fix).
pub const PULL_REQUEST_SEEDS: [u64; 10] = [
    1, 7, 42, 1_234, 9_001, 31_337, 271_828, 314_159, 8_675_309, 20_261_122,
];

/// Seeds a test should run: the env override alone, else the PR list.
pub fn seeds_for_test() -> Vec<u64> {
    match std::env::var(SEED_ENVIRONMENT_VARIABLE) {
        Ok(value) => vec![value.trim().parse().expect("MINIMAPPR_SIM_SEED must be an unsigned integer")],
        Err(_) => PULL_REQUEST_SEEDS.to_vec(),
    }
}

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;
const SPLITMIX_INCREMENT: u64 = 0x9e37_79b9_7f4a_7c15;
const SPLITMIX_MULTIPLIER_A: u64 = 0xbf58_476d_1ce4_e5b9;
const SPLITMIX_MULTIPLIER_B: u64 = 0x94d0_49bb_1331_11eb;
const UNIT_INTERVAL_MANTISSA_BITS: u32 = 53;

fn fnv1a(text: &str) -> u64 {
    text.bytes()
        .fold(FNV_OFFSET_BASIS, |hash, byte| (hash ^ byte as u64).wrapping_mul(FNV_PRIME))
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(SPLITMIX_INCREMENT);
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_MULTIPLIER_A);
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_MULTIPLIER_B);
    value ^ (value >> 31)
}

/// Independent stream for one purpose ("network", "clocks", ...).
pub fn purpose_stream(seed: u64, purpose: &str) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(splitmix(seed ^ fnv1a(purpose)))
}

fn hashed_bits(seed: u64, purpose: &str, a: u64, b: u64) -> u64 {
    splitmix(splitmix(splitmix(seed ^ fnv1a(purpose)) ^ a) ^ b.wrapping_mul(SPLITMIX_INCREMENT))
}

/// Uniform in (0, 1), a pure function of its arguments.
pub fn hashed_uniform(seed: u64, purpose: &str, a: u64, b: u64) -> f64 {
    let bits = hashed_bits(seed, purpose, a, b) >> (64 - UNIT_INTERVAL_MANTISSA_BITS);
    (bits as f64 + 0.5) / (1u64 << UNIT_INTERVAL_MANTISSA_BITS) as f64
}

/// Standard normal (Box–Muller), a pure function of its arguments.
pub fn hashed_gaussian(seed: u64, purpose: &str, a: u64, b: u64) -> f64 {
    let first = hashed_uniform(seed, purpose, a, b);
    let second = hashed_uniform(seed, purpose, a ^ SPLITMIX_INCREMENT, b.wrapping_add(1));
    (-2.0 * first.ln()).sqrt() * (std::f64::consts::TAU * second).cos()
}
