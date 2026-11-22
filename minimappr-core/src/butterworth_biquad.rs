//! Butterworth biquad cascades shared by the sidecar preprocessing chain and
//! Field Mode nodes. Pure math with no IO, so it builds for wasm32 and mobile.

// ---------------------------------------------------------------------------
// Biquad cascade — Butterworth design via direct bilinear transform with
// per-section Q. Direct Form II Transposed for numerical stability.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BiquadCoefficients {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

#[derive(Clone, Debug, Default)]
pub struct BiquadState {
    z1: f32,
    z2: f32,
}

impl BiquadCoefficients {
    #[inline]
    pub fn process(&self, x: f32, state: &mut BiquadState) -> f32 {
        let y = self.b0 * x + state.z1;
        state.z1 = self.b1 * x - self.a1 * y + state.z2;
        state.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Even-order Butterworth lowpass as a cascade of `order/2` biquads.
/// Odd orders are rounded up to the next even by callers — the biquad
/// cascade can't realize an odd-order pole structure on its own.
pub fn butter_lowpass_sos(
    cutoff_hz: f64,
    sample_rate_hz: u32,
    order: u8,
) -> Vec<BiquadCoefficients> {
    let order = effective_filter_order(order);
    let n_sections = (order / 2) as usize;
    let fs = f64::from(sample_rate_hz.max(1));
    let nyquist = fs * 0.5;
    let fc = cutoff_hz.clamp(1.0, nyquist * 0.999);
    let k = (std::f64::consts::PI * fc / fs).tan();
    let k2 = k * k;
    let n = f64::from(order);
    (0..n_sections)
        .map(|s| {
            let theta = std::f64::consts::PI * (2.0 * s as f64 + 1.0) / (2.0 * n);
            let q = 1.0 / (2.0 * theta.cos());
            let norm = 1.0 / (1.0 + k / q + k2);
            BiquadCoefficients {
                b0: (k2 * norm) as f32,
                b1: (2.0 * k2 * norm) as f32,
                b2: (k2 * norm) as f32,
                a1: (2.0 * (k2 - 1.0) * norm) as f32,
                a2: ((1.0 - k / q + k2) * norm) as f32,
            }
        })
        .collect()
}

/// Even-order Butterworth highpass as a cascade of `order/2` biquads.
pub fn butter_highpass_sos(
    cutoff_hz: f64,
    sample_rate_hz: u32,
    order: u8,
) -> Vec<BiquadCoefficients> {
    let order = effective_filter_order(order);
    let n_sections = (order / 2) as usize;
    let fs = f64::from(sample_rate_hz.max(1));
    let nyquist = fs * 0.5;
    let fc = cutoff_hz.clamp(1.0, nyquist * 0.999);
    let k = (std::f64::consts::PI * fc / fs).tan();
    let k2 = k * k;
    let n = f64::from(order);
    (0..n_sections)
        .map(|s| {
            let theta = std::f64::consts::PI * (2.0 * s as f64 + 1.0) / (2.0 * n);
            let q = 1.0 / (2.0 * theta.cos());
            let norm = 1.0 / (1.0 + k / q + k2);
            BiquadCoefficients {
                b0: norm as f32,
                b1: (-2.0 * norm) as f32,
                b2: norm as f32,
                a1: (2.0 * (k2 - 1.0) * norm) as f32,
                a2: ((1.0 - k / q + k2) * norm) as f32,
            }
        })
        .collect()
}

fn effective_filter_order(order: u8) -> u8 {
    // Coerce to an even order ≥ 2; the cascade has no realizable odd-order form.
    let bounded = order.max(2);
    if bounded.is_multiple_of(2) {
        bounded
    } else {
        bounded + 1
    }
}
