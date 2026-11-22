//! Ranging/sync chirp waveforms played from phone speakers (LFM / HFM),
//! Tukey-tapered so the self-recording limiter sees realistic edges.

use std::f64::consts::{PI, TAU};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ChirpWaveform {
    #[default]
    LinearFm,
    /// Doppler-tolerant hyperbolic FM (instantaneous period linear in time).
    HyperbolicFm,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChirpWaveformParams {
    pub band_low_hz: f64,
    pub band_high_hz: f64,
    pub duration_s: f64,
    pub waveform: ChirpWaveform,
    /// Tukey taper: fraction of the chirp spent ramping (split over both ends).
    pub taper_fraction: f64,
}

impl Default for ChirpWaveformParams {
    fn default() -> Self {
        Self {
            band_low_hz: 15_000.0,
            band_high_hz: 18_000.0,
            duration_s: 0.05,
            waveform: ChirpWaveform::LinearFm,
            taper_fraction: 0.1,
        }
    }
}

fn tukey_gain(index: usize, length: usize, taper_fraction: f64) -> f64 {
    let taper_samples = ((length as f64) * taper_fraction / 2.0) as usize;
    if taper_samples == 0 {
        return 1.0;
    }
    let edge_distance = index.min(length - 1 - index);
    if edge_distance >= taper_samples {
        1.0
    } else {
        0.5 - 0.5 * (PI * edge_distance as f64 / taper_samples as f64).cos()
    }
}

/// Unit-peak chirp at `sample_rate_hz` (sweeps low → high).
pub fn generate_chirp(params: &ChirpWaveformParams, sample_rate_hz: f64) -> Vec<f32> {
    let length = (params.duration_s * sample_rate_hz).round() as usize;
    let duration = params.duration_s.max(f64::EPSILON);
    let (f0, f1) = (params.band_low_hz, params.band_high_hz);
    (0..length)
        .map(|index| {
            let t = index as f64 / sample_rate_hz;
            let phase = match params.waveform {
                ChirpWaveform::LinearFm => TAU * (f0 * t + (f1 - f0) * t * t / (2.0 * duration)),
                ChirpWaveform::HyperbolicFm => {
                    // f(t) = f0 f1 T / (f1 T - (f1 - f0) t); phase is its integral.
                    let k = (f1 - f0) / (f1 * duration);
                    -TAU * f0 / k * (1.0 - k * t).ln()
                }
            };
            (phase.sin() * tukey_gain(index, length, params.taper_fraction)) as f32
        })
        .collect()
}
