//! Running noise-floor estimate in linear energy, one value per band (plus a
//! broadband mean-square floor used for noise fill).
//!
//! Gated EMA: symmetric tracking while a value sits within +/- gate of its
//! floor (unbiased for stationary noise), a very slow rise above the gate (a
//! call does not lift its own floor) and a faster fall below it (the floor
//! follows the background down quickly when wind drops).

use super::configuration::Stage1Configuration;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseFloorSmoothing {
    pub warmup_frames: u64,
    pub warmup_smoothing: f64,
    pub tracking_smoothing: f64,
    pub rise_while_active_smoothing: f64,
    pub fall_smoothing: f64,
    pub update_gate_db: f64,
}

impl NoiseFloorSmoothing {
    pub fn from_configuration(configuration: &Stage1Configuration) -> Self {
        Self {
            warmup_frames: configuration.noise_floor_warmup_frames,
            warmup_smoothing: configuration.noise_floor_warmup_smoothing,
            tracking_smoothing: configuration.noise_floor_tracking_smoothing,
            rise_while_active_smoothing: configuration.noise_floor_rise_while_active_smoothing,
            fall_smoothing: configuration.noise_floor_fall_smoothing,
            update_gate_db: configuration.noise_floor_update_gate_db,
        }
    }

    fn smoothing_for(&self, observed_energy: f64, floor_energy: f64, frames_observed: u64) -> f64 {
        if frames_observed < self.warmup_frames {
            return self.warmup_smoothing;
        }
        let excess_db = energy_ratio_db(observed_energy, floor_energy);
        if excess_db > self.update_gate_db {
            self.rise_while_active_smoothing
        } else if excess_db < -self.update_gate_db {
            self.fall_smoothing
        } else {
            self.tracking_smoothing
        }
    }
}

/// `10 log10(numerator / denominator)`, safe for zeros.
pub fn energy_ratio_db(numerator_energy: f64, denominator_energy: f64) -> f64 {
    10.0 * (numerator_energy.max(f64::MIN_POSITIVE) / denominator_energy.max(f64::MIN_POSITIVE))
        .log10()
}

#[derive(Debug, Clone)]
pub struct GatedNoiseFloorTracker {
    smoothing: NoiseFloorSmoothing,
    band_floor_energies: Vec<f64>,
    broadband_floor_mean_square: f64,
    frames_observed: u64,
}

impl GatedNoiseFloorTracker {
    pub fn new(band_count: usize, smoothing: NoiseFloorSmoothing) -> Self {
        Self {
            smoothing,
            band_floor_energies: vec![0.0; band_count],
            broadband_floor_mean_square: 0.0,
            frames_observed: 0,
        }
    }

    pub fn band_floor_energies(&self) -> &[f64] {
        &self.band_floor_energies
    }

    pub fn broadband_floor_rms(&self) -> f64 {
        self.broadband_floor_mean_square.sqrt()
    }

    pub fn frames_observed(&self) -> u64 {
        self.frames_observed
    }

    pub fn update(&mut self, band_energies: &[f64], broadband_mean_square: f64) {
        if self.frames_observed == 0 {
            self.band_floor_energies.copy_from_slice(band_energies);
            self.broadband_floor_mean_square = broadband_mean_square;
        } else {
            for (floor, &observed) in self.band_floor_energies.iter_mut().zip(band_energies) {
                let smoothing =
                    self.smoothing
                        .smoothing_for(observed, *floor, self.frames_observed);
                *floor += smoothing * (observed - *floor);
            }
            let smoothing = self.smoothing.smoothing_for(
                broadband_mean_square,
                self.broadband_floor_mean_square,
                self.frames_observed,
            );
            self.broadband_floor_mean_square +=
                smoothing * (broadband_mean_square - self.broadband_floor_mean_square);
        }
        self.frames_observed += 1;
    }
}
