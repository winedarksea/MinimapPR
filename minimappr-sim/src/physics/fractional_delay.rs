//! Kaiser-windowed-sinc fractional-delay interpolator.
//!
//! Whole-sample delays hide interpolation bugs and the in-clip stretch from
//! clock-rate offsets, so every propagation path evaluates the source at an
//! arbitrary real-valued sample position. The kernel is tabulated at
//! `phase_count` sub-sample phases (nearest phase is used): with 4096 phases
//! the fractional position is quantised to ≤ 1/8192 sample, far below the
//! 0.01-sample accuracy the physics tests require.

/// Kernel design parameters (exposed so tests can tighten or relax them).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FractionalDelayKernelConfig {
    /// Taps on each side of the interpolation point (total taps = 2×).
    pub half_width_taps: usize,
    /// Number of tabulated sub-sample phases.
    pub phase_count: usize,
    /// Kaiser window shape β; higher = lower sidelobes, wider transition.
    pub kaiser_beta: f64,
    /// Low-pass cutoff as a fraction of Nyquist (1.0 = full band).
    pub cutoff_fraction_of_nyquist: f64,
}

impl Default for FractionalDelayKernelConfig {
    fn default() -> Self {
        Self {
            half_width_taps: 16,
            phase_count: 4096,
            kaiser_beta: 8.6,
            cutoff_fraction_of_nyquist: 0.97,
        }
    }
}

/// Precomputed polyphase table: `table[phase * taps + k]`.
#[derive(Clone, Debug)]
pub struct FractionalDelayKernel {
    half_width_taps: usize,
    taps_per_phase: usize,
    phase_count: usize,
    table: Vec<f32>,
}

const BESSEL_SERIES_TERMS: usize = 64;
const BESSEL_SERIES_CONVERGENCE: f64 = 1e-16;

/// Modified Bessel function of the first kind, order zero (power series).
fn bessel_i0(x: f64) -> f64 {
    let half_x = x / 2.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..BESSEL_SERIES_TERMS {
        term *= (half_x / k as f64) * (half_x / k as f64);
        sum += term;
        if term < sum * BESSEL_SERIES_CONVERGENCE {
            break;
        }
    }
    sum
}

/// Kaiser-windowed sinc evaluated at offset `x` samples from the centre.
pub fn windowed_sinc(x: f64, half_width: f64, cutoff: f64, kaiser_beta: f64) -> f64 {
    if x.abs() >= half_width {
        return 0.0;
    }
    let argument = std::f64::consts::PI * cutoff * x;
    let sinc = if argument.abs() < 1e-12 {
        1.0
    } else {
        argument.sin() / argument
    };
    let normalized = x / half_width;
    let window =
        bessel_i0(kaiser_beta * (1.0 - normalized * normalized).sqrt()) / bessel_i0(kaiser_beta);
    cutoff * sinc * window
}

impl FractionalDelayKernel {
    pub fn new(config: FractionalDelayKernelConfig) -> Self {
        let taps_per_phase = 2 * config.half_width_taps;
        let half_width = config.half_width_taps as f64;
        let mut table = vec![0f32; taps_per_phase * config.phase_count];
        for phase in 0..config.phase_count {
            let fraction = phase as f64 / config.phase_count as f64;
            let row = &mut table[phase * taps_per_phase..(phase + 1) * taps_per_phase];
            // Tap k sits at integer sample (floor + k - half_width + 1); its
            // distance to the interpolation point is fraction - (k - half_width + 1).
            let mut weight_sum = 0.0;
            let weights: Vec<f64> = (0..taps_per_phase)
                .map(|k| {
                    let distance = fraction - (k as f64 - half_width + 1.0);
                    let weight = windowed_sinc(
                        distance,
                        half_width,
                        config.cutoff_fraction_of_nyquist,
                        config.kaiser_beta,
                    );
                    weight_sum += weight;
                    weight
                })
                .collect();
            // Normalise DC gain per phase so a constant signal stays constant
            // regardless of fraction (removes phase-dependent amplitude ripple).
            for (slot, weight) in row.iter_mut().zip(weights) {
                *slot = (weight / weight_sum) as f32;
            }
        }
        Self {
            half_width_taps: config.half_width_taps,
            taps_per_phase,
            phase_count: config.phase_count,
            table,
        }
    }

    pub fn half_width_taps(&self) -> usize {
        self.half_width_taps
    }

    /// Band-limited value of `buffer` at real-valued sample `position`.
    /// Returns 0 when the kernel support leaves the buffer (callers pad).
    #[inline]
    pub fn evaluate(&self, buffer: &[f32], position: f64) -> f32 {
        let floor = position.floor();
        let mut base = floor as i64;
        let mut phase = ((position - floor) * self.phase_count as f64).round() as usize;
        if phase == self.phase_count {
            phase = 0;
            base += 1;
        }
        let first_tap = base - self.half_width_taps as i64 + 1;
        if first_tap < 0 || first_tap as usize + self.taps_per_phase > buffer.len() {
            return 0.0;
        }
        let start = first_tap as usize;
        let samples = &buffer[start..start + self.taps_per_phase];
        let weights = &self.table[phase * self.taps_per_phase..(phase + 1) * self.taps_per_phase];
        dot_product(samples, weights)
    }
}

/// Lanes of independent accumulators: lets the compiler vectorise the dot
/// product (a single serial f32 sum cannot be reordered, so it won't).
const DOT_PRODUCT_LANES: usize = 8;

#[inline]
fn dot_product(samples: &[f32], weights: &[f32]) -> f32 {
    let mut lanes = [0f32; DOT_PRODUCT_LANES];
    let sample_chunks = samples.chunks_exact(DOT_PRODUCT_LANES);
    let weight_chunks = weights.chunks_exact(DOT_PRODUCT_LANES);
    let tail: f32 = sample_chunks
        .remainder()
        .iter()
        .zip(weight_chunks.remainder())
        .map(|(s, w)| s * w)
        .sum();
    for (sample_chunk, weight_chunk) in sample_chunks.zip(weight_chunks) {
        for lane in 0..DOT_PRODUCT_LANES {
            lanes[lane] += sample_chunk[lane] * weight_chunk[lane];
        }
    }
    lanes.iter().sum::<f32>() + tail
}

impl Default for FractionalDelayKernel {
    fn default() -> Self {
        Self::new(FractionalDelayKernelConfig::default())
    }
}
