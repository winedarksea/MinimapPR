//! ISO 9613-1:1993 pure-tone atmospheric absorption.
//!
//! Absorption is what makes a 13–16 kHz vs 18–20 kHz chirp comparison (and a
//! 100 m faint-bird scenario) meaningful: at 20 °C / 50 % RH air removes
//! ~0.03 dB/m at 4 kHz but ~0.3 dB/m at 20 kHz.

use super::constants::{
    CELSIUS_TO_KELVIN_OFFSET, ISO9613_REFERENCE_TEMPERATURE_K, ISO9613_TRIPLE_POINT_ISOTHERM_K,
    PERCENT_PER_FRACTION, STANDARD_ATMOSPHERE_PRESSURE_KPA,
};

// ISO 9613-1 equation coefficients (Eq. 3–5 and B.1). Named so the formula
// below reads like the standard.
const SATURATION_EXPONENT_SLOPE: f64 = -6.8346;
const SATURATION_EXPONENT_POWER: f64 = 1.261;
const SATURATION_EXPONENT_OFFSET: f64 = 4.6151;
const OXYGEN_RELAXATION_BASE: f64 = 24.0;
const OXYGEN_RELAXATION_HUMIDITY_SCALE: f64 = 4.04e4;
const OXYGEN_RELAXATION_HUMIDITY_OFFSET_A: f64 = 0.02;
const OXYGEN_RELAXATION_HUMIDITY_OFFSET_B: f64 = 0.391;
const NITROGEN_RELAXATION_BASE: f64 = 9.0;
const NITROGEN_RELAXATION_HUMIDITY_SCALE: f64 = 280.0;
const NITROGEN_RELAXATION_EXPONENT_SCALE: f64 = -4.170;
const CLASSICAL_ABSORPTION_COEFFICIENT: f64 = 1.84e-11;
const OXYGEN_VIBRATIONAL_COEFFICIENT: f64 = 0.01275;
const OXYGEN_VIBRATIONAL_TEMPERATURE_K: f64 = 2239.1;
const NITROGEN_VIBRATIONAL_COEFFICIENT: f64 = 0.1068;
const NITROGEN_VIBRATIONAL_TEMPERATURE_K: f64 = 3352.0;
/// The standard's leading 8.686 factor (nepers → dB with its α definition).
const ISO9613_DECIBEL_FACTOR: f64 = 8.686;

/// Air state that drives absorption.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtmosphericConditions {
    pub temperature_c: f64,
    pub relative_humidity_fraction: f64,
    pub pressure_kpa: f64,
}

impl Default for AtmosphericConditions {
    fn default() -> Self {
        Self {
            temperature_c: 20.0,
            relative_humidity_fraction: 0.5,
            pressure_kpa: STANDARD_ATMOSPHERE_PRESSURE_KPA,
        }
    }
}

impl AtmosphericConditions {
    /// Pure-tone absorption coefficient α in dB per metre at `frequency_hz`.
    pub fn absorption_db_per_m(&self, frequency_hz: f64) -> f64 {
        let temperature_k = self.temperature_c + CELSIUS_TO_KELVIN_OFFSET;
        let relative_pressure = self.pressure_kpa / STANDARD_ATMOSPHERE_PRESSURE_KPA;
        let temperature_ratio = temperature_k / ISO9613_REFERENCE_TEMPERATURE_K;

        let saturation_exponent = SATURATION_EXPONENT_SLOPE
            * (ISO9613_TRIPLE_POINT_ISOTHERM_K / temperature_k).powf(SATURATION_EXPONENT_POWER)
            + SATURATION_EXPONENT_OFFSET;
        let saturation_pressure_ratio = 10f64.powf(saturation_exponent);
        let molar_water_vapour_percent =
            self.relative_humidity_fraction * PERCENT_PER_FRACTION * saturation_pressure_ratio
                / relative_pressure;

        let h = molar_water_vapour_percent;
        let oxygen_relaxation_hz = relative_pressure
            * (OXYGEN_RELAXATION_BASE
                + OXYGEN_RELAXATION_HUMIDITY_SCALE * h * (OXYGEN_RELAXATION_HUMIDITY_OFFSET_A + h)
                    / (OXYGEN_RELAXATION_HUMIDITY_OFFSET_B + h));
        let nitrogen_relaxation_hz = relative_pressure
            * temperature_ratio.powf(-0.5)
            * (NITROGEN_RELAXATION_BASE
                + NITROGEN_RELAXATION_HUMIDITY_SCALE
                    * h
                    * (NITROGEN_RELAXATION_EXPONENT_SCALE
                        * (temperature_ratio.powf(-1.0 / 3.0) - 1.0))
                        .exp());

        let f2 = frequency_hz * frequency_hz;
        let classical =
            CLASSICAL_ABSORPTION_COEFFICIENT / relative_pressure * temperature_ratio.sqrt();
        let oxygen = OXYGEN_VIBRATIONAL_COEFFICIENT
            * (-OXYGEN_VIBRATIONAL_TEMPERATURE_K / temperature_k).exp()
            / (oxygen_relaxation_hz + f2 / oxygen_relaxation_hz);
        let nitrogen = NITROGEN_VIBRATIONAL_COEFFICIENT
            * (-NITROGEN_VIBRATIONAL_TEMPERATURE_K / temperature_k).exp()
            / (nitrogen_relaxation_hz + f2 / nitrogen_relaxation_hz);
        ISO9613_DECIBEL_FACTOR
            * f2
            * (classical + temperature_ratio.powf(-2.5) * (oxygen + nitrogen))
    }
}

/// α per real-FFT bin, precomputed once per (conditions, FFT length) so the
/// per-path attenuation response is one `powf` per bin.
#[derive(Clone, Debug)]
pub struct AbsorptionSpectrum {
    pub fft_length: usize,
    absorption_db_per_m_by_bin: Vec<f64>,
}

impl AbsorptionSpectrum {
    pub fn new(conditions: &AtmosphericConditions, fft_length: usize, sample_rate_hz: f64) -> Self {
        let absorption_db_per_m_by_bin = (0..=fft_length / 2)
            .map(|bin| {
                conditions.absorption_db_per_m(bin as f64 * sample_rate_hz / fft_length as f64)
            })
            .collect();
        Self {
            fft_length,
            absorption_db_per_m_by_bin,
        }
    }

    /// Worst-case (Nyquist) attenuation over `path_length_m`, used to skip
    /// filtering entirely for centimetre-scale self paths.
    pub fn max_attenuation_db(&self, path_length_m: f64) -> f64 {
        self.absorption_db_per_m_by_bin
            .last()
            .copied()
            .unwrap_or(0.0)
            * path_length_m
    }

    /// Linear amplitude response for `path_length_m`, bin k for k in 0..=N/2.
    pub fn amplitude_response(&self, path_length_m: f64) -> Vec<f32> {
        self.absorption_db_per_m_by_bin
            .iter()
            .map(|alpha| 10f64.powf(-alpha * path_length_m / 20.0) as f32)
            .collect()
    }
}
