//! Per-path air-absorption filtering of an emission, done in the frequency
//! domain: one forward FFT per emission, then one spectral multiply +
//! inverse FFT per distinct path-length bucket. Zero-phase, so no group
//! delay needs compensating and the propagation delay stays exact.
//!
//! Sharing: every (device, path) of one emission whose length falls in the
//! same bucket reuses one filtered buffer. The bucket width is chosen so
//! the attenuation error at the emission's band top (99.9 % energy
//! frequency) is at most ±tolerance/2 — the representative length is the
//! bucket centre. Tolerance 0 disables sharing (exact per-path filters).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::physics::atmosphere::{AbsorptionSpectrum, AtmosphericConditions};
use crate::world::PlannedEmission;

/// Below this worst-case (Nyquist) attenuation the path is left unfiltered.
const NEGLIGIBLE_ABSORPTION_DB: f64 = 0.01;
/// Spectral energy fraction defining an emission's band top.
const BAND_TOP_ENERGY_FRACTION: f64 = 0.999;

type SharedBuffer = Arc<Vec<f32>>;

/// An emission zero-padded to a power-of-two FFT length, its spectrum and
/// its cache of filtered versions keyed by path-length bucket.
pub struct PreparedEmission {
    /// Index of emission sample 0 inside `padded`.
    pub offset: usize,
    pub padded: SharedBuffer,
    spectrum: Vec<Complex32>,
    /// Bucket width in metres; 0 = no sharing.
    bucket_width_m: f64,
    filtered_by_bucket: Mutex<HashMap<i64, Arc<OnceLock<SharedBuffer>>>>,
}

impl PreparedEmission {
    /// Drop cached filtered buffers (call once the emission is rendered).
    pub fn release_filters(&self) {
        self.filtered_by_bucket
            .lock()
            .expect("filter cache poisoned")
            .clear();
    }
}

/// Thread-safe FFT plan + absorption-table cache keyed by FFT length.
pub struct AbsorptionFilterBank {
    conditions: AtmosphericConditions,
    sample_rate_hz: f64,
    enabled: bool,
    sharing_tolerance_db: f64,
    planner: Mutex<FftPlanner<f32>>,
    spectra: Mutex<HashMap<usize, Arc<AbsorptionSpectrum>>>,
}

fn band_top_frequency_hz(spectrum: &[Complex32], sample_rate_hz: f64) -> f64 {
    let half = spectrum.len() / 2;
    let energies: Vec<f64> = spectrum[..=half]
        .iter()
        .map(|c| c.norm_sqr() as f64)
        .collect();
    let total: f64 = energies.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut cumulative = 0.0;
    for (bin, energy) in energies.iter().enumerate() {
        cumulative += energy;
        if cumulative >= BAND_TOP_ENERGY_FRACTION * total {
            return bin as f64 * sample_rate_hz / spectrum.len() as f64;
        }
    }
    sample_rate_hz / 2.0
}

impl AbsorptionFilterBank {
    pub fn new(
        conditions: AtmosphericConditions,
        sample_rate_hz: f64,
        enabled: bool,
        sharing_tolerance_db: f64,
    ) -> Self {
        Self {
            conditions,
            sample_rate_hz,
            enabled,
            sharing_tolerance_db,
            planner: Mutex::new(FftPlanner::new()),
            spectra: Mutex::new(HashMap::new()),
        }
    }

    fn plan(&self, length: usize, inverse: bool) -> Arc<dyn Fft<f32>> {
        let mut planner = self.planner.lock().expect("fft planner poisoned");
        if inverse {
            planner.plan_fft_inverse(length)
        } else {
            planner.plan_fft_forward(length)
        }
    }

    fn absorption_spectrum(&self, fft_length: usize) -> Arc<AbsorptionSpectrum> {
        let mut cache = self.spectra.lock().expect("absorption cache poisoned");
        cache
            .entry(fft_length)
            .or_insert_with(|| {
                Arc::new(AbsorptionSpectrum::new(
                    &self.conditions,
                    fft_length,
                    self.sample_rate_hz,
                ))
            })
            .clone()
    }

    pub fn prepare(&self, emission: &PlannedEmission, guard_samples: usize) -> PreparedEmission {
        let length = emission.samples_pa_at_1m.len();
        let fft_length = (length + 2 * guard_samples).next_power_of_two();
        let mut padded = vec![0f32; fft_length];
        padded[guard_samples..guard_samples + length].copy_from_slice(&emission.samples_pa_at_1m);
        let (spectrum, bucket_width_m) = if self.enabled {
            let mut buffer: Vec<Complex32> =
                padded.iter().map(|s| Complex32::new(*s, 0.0)).collect();
            self.plan(fft_length, false).process(&mut buffer);
            let alpha_at_band_top = self
                .conditions
                .absorption_db_per_m(band_top_frequency_hz(&buffer, self.sample_rate_hz));
            let width = if self.sharing_tolerance_db > 0.0 && alpha_at_band_top > 0.0 {
                self.sharing_tolerance_db / alpha_at_band_top
            } else {
                0.0
            };
            (buffer, width)
        } else {
            (Vec::new(), 0.0)
        };
        PreparedEmission {
            offset: guard_samples,
            padded: Arc::new(padded),
            spectrum,
            bucket_width_m,
            filtered_by_bucket: Mutex::new(HashMap::new()),
        }
    }

    fn filter_exact(
        &self,
        prepared: &PreparedEmission,
        table: &AbsorptionSpectrum,
        path_length_m: f64,
    ) -> SharedBuffer {
        let fft_length = prepared.spectrum.len();
        let response = table.amplitude_response(path_length_m);
        let mut buffer: Vec<Complex32> = prepared
            .spectrum
            .iter()
            .enumerate()
            .map(|(bin, value)| value * response[bin.min(fft_length - bin)])
            .collect();
        self.plan(fft_length, true).process(&mut buffer);
        let normalization = 1.0 / fft_length as f32;
        Arc::new(buffer.iter().map(|c| c.re * normalization).collect())
    }

    /// Emission as heard after `path_length_m` of air (same layout as `padded`).
    pub fn filtered_for_path(
        &self,
        prepared: &PreparedEmission,
        path_length_m: f64,
    ) -> SharedBuffer {
        if !self.enabled || prepared.spectrum.is_empty() {
            return prepared.padded.clone();
        }
        let table = self.absorption_spectrum(prepared.spectrum.len());
        if table.max_attenuation_db(path_length_m) < NEGLIGIBLE_ABSORPTION_DB {
            return prepared.padded.clone();
        }
        if prepared.bucket_width_m <= 0.0 {
            return self.filter_exact(prepared, &table, path_length_m);
        }
        let bucket = (path_length_m / prepared.bucket_width_m).floor() as i64;
        let representative_length_m = (bucket as f64 + 0.5) * prepared.bucket_width_m;
        let slot = prepared
            .filtered_by_bucket
            .lock()
            .expect("filter cache poisoned")
            .entry(bucket)
            .or_insert_with(|| Arc::new(OnceLock::new()))
            .clone();
        slot.get_or_init(|| self.filter_exact(prepared, &table, representative_length_m))
            .clone()
    }
}
