//! RBJ "Audio EQ Cookbook" biquads for receiver-side colouring (mic
//! response, pocket muffling, Bluetooth band-limit, body shadow). IIR keeps
//! the whole-stream cost at ~5 MACs/sample, which matters for 60 s renders.

use std::f64::consts::PI;

pub const BUTTERWORTH_SECOND_ORDER_Q: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// Shelf slope S = 1: steepest shelf without overshoot (RBJ cookbook).
const SHELF_SLOPE: f64 = 1.0;

#[derive(Clone, Copy, Debug)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl Biquad {
    fn normalized(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    pub fn lowpass(cutoff_hz: f64, quality_factor: f64, sample_rate_hz: f64) -> Self {
        let omega = 2.0 * PI * cutoff_hz / sample_rate_hz;
        let alpha = omega.sin() / (2.0 * quality_factor);
        let cos_omega = omega.cos();
        Self::normalized(
            (1.0 - cos_omega) / 2.0,
            1.0 - cos_omega,
            (1.0 - cos_omega) / 2.0,
            1.0 + alpha,
            -2.0 * cos_omega,
            1.0 - alpha,
        )
    }

    pub fn highpass(cutoff_hz: f64, quality_factor: f64, sample_rate_hz: f64) -> Self {
        let omega = 2.0 * PI * cutoff_hz / sample_rate_hz;
        let alpha = omega.sin() / (2.0 * quality_factor);
        let cos_omega = omega.cos();
        Self::normalized(
            (1.0 + cos_omega) / 2.0,
            -(1.0 + cos_omega),
            (1.0 + cos_omega) / 2.0,
            1.0 + alpha,
            -2.0 * cos_omega,
            1.0 - alpha,
        )
    }

    /// High shelf: gain `gain_db` above `corner_hz`, unity below.
    pub fn high_shelf(corner_hz: f64, gain_db: f64, sample_rate_hz: f64) -> Self {
        let amplitude = 10f64.powf(gain_db / 40.0);
        let omega = 2.0 * PI * corner_hz / sample_rate_hz;
        let cos_omega = omega.cos();
        let alpha = omega.sin() / 2.0
            * ((amplitude + 1.0 / amplitude) * (1.0 / SHELF_SLOPE - 1.0) + 2.0).sqrt();
        let two_sqrt_a_alpha = 2.0 * amplitude.sqrt() * alpha;
        Self::normalized(
            amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_omega + two_sqrt_a_alpha),
            -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cos_omega),
            amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_omega - two_sqrt_a_alpha),
            (amplitude + 1.0) - (amplitude - 1.0) * cos_omega + two_sqrt_a_alpha,
            2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cos_omega),
            (amplitude + 1.0) - (amplitude - 1.0) * cos_omega - two_sqrt_a_alpha,
        )
    }

    /// Band-pass (constant 0 dB peak gain), used for speech formants.
    pub fn bandpass(center_hz: f64, quality_factor: f64, sample_rate_hz: f64) -> Self {
        let omega = 2.0 * PI * center_hz / sample_rate_hz;
        let alpha = omega.sin() / (2.0 * quality_factor);
        let cos_omega = omega.cos();
        Self::normalized(
            alpha,
            0.0,
            -alpha,
            1.0 + alpha,
            -2.0 * cos_omega,
            1.0 - alpha,
        )
    }

    /// Magnitude response at `frequency_hz` (used by tests and diagnostics).
    pub fn magnitude_at(&self, frequency_hz: f64, sample_rate_hz: f64) -> f64 {
        let omega = 2.0 * PI * frequency_hz / sample_rate_hz;
        let (cos1, sin1, cos2, sin2) = (
            omega.cos(),
            omega.sin(),
            (2.0 * omega).cos(),
            (2.0 * omega).sin(),
        );
        let numerator_re = self.b0 + self.b1 * cos1 + self.b2 * cos2;
        let numerator_im = -(self.b1 * sin1 + self.b2 * sin2);
        let denominator_re = 1.0 + self.a1 * cos1 + self.a2 * cos2;
        let denominator_im = -(self.a1 * sin1 + self.a2 * sin2);
        (numerator_re.hypot(numerator_im)) / (denominator_re.hypot(denominator_im))
    }
}

/// Q values for an even-order Butterworth cascade of second-order sections.
pub fn butterworth_section_quality_factors(order: usize) -> Vec<f64> {
    let section_count = order.max(2) / 2;
    (0..section_count)
        .map(|k| {
            let pole_angle = PI * (2 * k + 1) as f64 / (2 * section_count * 2) as f64;
            1.0 / (2.0 * pole_angle.cos())
        })
        .collect()
}

/// Cascade of biquads with Direct Form II transposed state.
#[derive(Clone, Debug, Default)]
pub struct BiquadCascade {
    sections: Vec<Biquad>,
}

impl BiquadCascade {
    pub fn new(sections: Vec<Biquad>) -> Self {
        Self { sections }
    }

    pub fn butterworth_lowpass(order: usize, cutoff_hz: f64, sample_rate_hz: f64) -> Self {
        Self::new(
            butterworth_section_quality_factors(order)
                .into_iter()
                .map(|q| Biquad::lowpass(cutoff_hz, q, sample_rate_hz))
                .collect(),
        )
    }

    pub fn butterworth_highpass(order: usize, cutoff_hz: f64, sample_rate_hz: f64) -> Self {
        Self::new(
            butterworth_section_quality_factors(order)
                .into_iter()
                .map(|q| Biquad::highpass(cutoff_hz, q, sample_rate_hz))
                .collect(),
        )
    }

    pub fn push(&mut self, section: Biquad) {
        self.sections.push(section);
    }

    pub fn extend(&mut self, other: BiquadCascade) {
        self.sections.extend(other.sections);
    }

    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    pub fn magnitude_at(&self, frequency_hz: f64, sample_rate_hz: f64) -> f64 {
        self.sections
            .iter()
            .map(|s| s.magnitude_at(frequency_hz, sample_rate_hz))
            .product()
    }

    /// Filter `samples` in place (fresh zero state).
    pub fn process_in_place(&self, samples: &mut [f32]) {
        let mut state = self.streaming_state();
        for sample in samples.iter_mut() {
            *sample = state.process_sample(*sample as f64) as f32;
        }
    }

    /// Per-sample filter state, for fusing several stages into one pass
    /// over a long stream (one memory sweep instead of one per section).
    pub fn streaming_state(&self) -> StreamingCascade<'_> {
        StreamingCascade {
            sections: &self.sections,
            states: vec![[0.0; 2]; self.sections.len()],
        }
    }
}

pub struct StreamingCascade<'a> {
    sections: &'a [Biquad],
    states: Vec<[f64; 2]>,
}

impl StreamingCascade<'_> {
    /// Direct Form II transposed, all sections in series.
    #[inline]
    pub fn process_sample(&mut self, mut value: f64) -> f64 {
        for (section, state) in self.sections.iter().zip(self.states.iter_mut()) {
            let output = section.b0 * value + state[0];
            state[0] = section.b1 * value - section.a1 * output + state[1];
            state[1] = section.b2 * value - section.a2 * output;
            value = output;
        }
        value
    }
}
