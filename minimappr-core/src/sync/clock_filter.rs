//! Per-phone clock model: offset plus rate relative to the island's
//! reference clock, in a two-state Kalman filter fed by chirp rounds.
//!
//! State at reference time t: θ(t) = local − reference (s) and ε = rate
//! offset (local ticks per reference tick − 1). Crystal rate changes slowly
//! (thermal time constants of minutes), so ε follows a slow random walk and
//! the rate comes from the spaced heartbeat rounds; one post-detection
//! round is then fused against a rate that is already known (two rounds
//! 3 s apart with ~20 µs noise would give the rate only to ~9 ppm).
//!
//! Any reported gap, route change or interruption invalidates the model.
//! A *silent* gap shows up as an offset innovation far beyond the noise
//! (a dropped buffer is ≥ 5 ms against ~20 µs pick noise); the filter then
//! re-seeds the offset but keeps the rate (the crystal did not change).

/// Tunables for `PhoneClockFilter`.
#[derive(Clone, Debug, PartialEq)]
pub struct ClockFilterConfig {
    /// Prior σ of the rate offset before two spaced rounds (±60 ppm crystals).
    pub initial_rate_sigma: f64,
    /// Rate random-walk density: σ_ε grows by this · √(seconds).
    pub rate_random_walk_per_sqrt_s: f64,
    /// Offset white jitter density (s per √s), e.g. audio-clock phase noise.
    pub offset_random_walk_s_per_sqrt_s: f64,
    /// Updates must span at least this long before the rate counts as valid.
    pub minimum_rate_baseline_s: f64,
    /// Innovation (in σ) above which a single update is treated as an outlier.
    pub outlier_gate_sigma: f64,
    /// Absolute innovation above which the stream is assumed to have lost or
    /// gained samples silently.
    pub silent_gap_jump_threshold_s: f64,
}

impl Default for ClockFilterConfig {
    fn default() -> Self {
        Self {
            initial_rate_sigma: 100e-6,
            rate_random_walk_per_sqrt_s: 2e-8,
            offset_random_walk_s_per_sqrt_s: 1e-6,
            minimum_rate_baseline_s: 20.0,
            outlier_gate_sigma: 6.0,
            silent_gap_jump_threshold_s: 1e-3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClockUpdateOutcome {
    Initialised,
    Accepted {
        normalized_innovation: f64,
    },
    RejectedOutlier {
        normalized_innovation: f64,
    },
    /// Offset jumped by more than the silent-gap threshold; offset re-seeded.
    SilentGapDetected {
        jump_s: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClockPrediction {
    pub offset_s: f64,
    pub offset_sigma_s: f64,
    pub rate_offset: f64,
    pub rate_sigma: f64,
}

#[derive(Clone, Debug)]
struct ClockState {
    reference_time_s: f64,
    offset_s: f64,
    rate_offset: f64,
    /// Covariance [[θθ, θε], [θε, εε]].
    covariance: [[f64; 2]; 2],
    first_update_time_s: f64,
}

/// Offset + rate filter for one phone.
#[derive(Clone, Debug)]
pub struct PhoneClockFilter {
    pub config: ClockFilterConfig,
    state: Option<ClockState>,
}

impl PhoneClockFilter {
    pub fn new(config: ClockFilterConfig) -> Self {
        Self {
            config,
            state: None,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.state.is_some()
    }

    /// True once updates span the minimum baseline.
    pub fn has_valid_rate(&self) -> bool {
        self.state.as_ref().is_some_and(|s| {
            s.reference_time_s - s.first_update_time_s >= self.config.minimum_rate_baseline_s
        })
    }

    /// Reported gap, input route change or session interruption.
    pub fn invalidate(&mut self) {
        self.state = None;
    }

    fn predicted_state(&self, state: &ClockState, reference_time_s: f64) -> ClockState {
        let dt = reference_time_s - state.reference_time_s;
        let [[p00, p01], [_, p11]] = state.covariance;
        let elapsed = dt.abs();
        let rate_noise = self.config.rate_random_walk_per_sqrt_s.powi(2) * elapsed;
        let offset_noise = self.config.offset_random_walk_s_per_sqrt_s.powi(2) * elapsed;
        // F = [[1, dt], [0, 1]]; P' = F P Fᵀ + Q (rate walk integrates into offset).
        let new_p00 =
            p00 + 2.0 * dt * p01 + dt * dt * p11 + offset_noise + rate_noise * dt * dt / 3.0;
        let new_p01 = p01 + dt * p11 + rate_noise * dt / 2.0;
        let new_p11 = p11 + rate_noise;
        ClockState {
            reference_time_s,
            offset_s: state.offset_s + state.rate_offset * dt,
            rate_offset: state.rate_offset,
            covariance: [[new_p00, new_p01], [new_p01, new_p11]],
            first_update_time_s: state.first_update_time_s,
        }
    }

    /// Offset/rate prediction at a reference time (works backward too: a
    /// post-detection round's model applies to the call's buffered audio).
    pub fn predict(&self, reference_time_s: f64) -> Option<ClockPrediction> {
        let state = self.predicted_state(self.state.as_ref()?, reference_time_s);
        Some(ClockPrediction {
            offset_s: state.offset_s,
            offset_sigma_s: state.covariance[0][0].max(0.0).sqrt(),
            rate_offset: state.rate_offset,
            rate_sigma: state.covariance[1][1].max(0.0).sqrt(),
        })
    }

    /// Fuse one offset observation (local − reference at `reference_time_s`).
    pub fn update(
        &mut self,
        reference_time_s: f64,
        observed_offset_s: f64,
        observation_sigma_s: f64,
    ) -> ClockUpdateOutcome {
        let measurement_variance = observation_sigma_s.powi(2).max(f64::MIN_POSITIVE);
        let Some(previous) = self.state.as_ref() else {
            self.state = Some(ClockState {
                reference_time_s,
                offset_s: observed_offset_s,
                rate_offset: 0.0,
                covariance: [
                    [measurement_variance, 0.0],
                    [0.0, self.config.initial_rate_sigma.powi(2)],
                ],
                first_update_time_s: reference_time_s,
            });
            return ClockUpdateOutcome::Initialised;
        };
        let predicted = self.predicted_state(previous, reference_time_s);
        let innovation = observed_offset_s - predicted.offset_s;
        let innovation_variance = predicted.covariance[0][0] + measurement_variance;
        let normalized_innovation = innovation / innovation_variance.sqrt();

        if innovation.abs() > self.config.silent_gap_jump_threshold_s
            && normalized_innovation.abs() > self.config.outlier_gate_sigma
        {
            // Keep the rate only once it rests on the minimum baseline: an
            // unvalidated rate from two noisy rounds a few seconds apart can
            // be hundreds of ppm off, and re-seeding the offset while keeping
            // it makes every later prediction miss by milliseconds, so each
            // round is flagged as a gap again and the rate never recovers
            // (seen live: 281 ppm locked in on a two-phone session).
            let rate_is_validated = self.has_valid_rate();
            self.state = Some(if rate_is_validated {
                ClockState {
                    offset_s: observed_offset_s,
                    covariance: [[measurement_variance, 0.0], [0.0, predicted.covariance[1][1]]],
                    ..predicted
                }
            } else {
                ClockState {
                    reference_time_s,
                    offset_s: observed_offset_s,
                    rate_offset: 0.0,
                    covariance: [[measurement_variance, 0.0], [0.0, self.config.initial_rate_sigma.powi(2)]],
                    first_update_time_s: reference_time_s,
                }
            });
            return ClockUpdateOutcome::SilentGapDetected { jump_s: innovation };
        }
        if normalized_innovation.abs() > self.config.outlier_gate_sigma {
            self.state = Some(predicted);
            return ClockUpdateOutcome::RejectedOutlier {
                normalized_innovation,
            };
        }
        let [[p00, p01], [_, p11]] = predicted.covariance;
        let gain_offset = p00 / innovation_variance;
        let gain_rate = p01 / innovation_variance;
        self.state = Some(ClockState {
            offset_s: predicted.offset_s + gain_offset * innovation,
            rate_offset: predicted.rate_offset + gain_rate * innovation,
            covariance: [
                [(1.0 - gain_offset) * p00, (1.0 - gain_offset) * p01],
                [(1.0 - gain_offset) * p01, p11 - gain_rate * p01],
            ],
            ..predicted
        });
        ClockUpdateOutcome::Accepted {
            normalized_innovation,
        }
    }

    /// Ticks per reference tick (1 + ε), for in-exchange rate compensation
    /// and in-clip resampling.
    pub fn ticks_per_reference_tick(&self) -> Option<f64> {
        self.state.as_ref().map(|s| 1.0 + s.rate_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::chirp_codes::SplitMix64;

    fn gaussian(rng: &mut SplitMix64) -> f64 {
        // Box–Muller from two uniforms.
        let u1 = ((rng.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    const TRUE_OFFSET_S: f64 = 0.0123;
    const TRUE_RATE_OFFSET: f64 = 37e-6;
    const PICK_SIGMA_S: f64 = 20e-6;
    const HEARTBEAT_PERIOD_S: f64 = 90.0;

    fn true_offset(t: f64) -> f64 {
        TRUE_OFFSET_S + TRUE_RATE_OFFSET * t
    }

    fn fed_filter(heartbeats: usize, seed: u64) -> PhoneClockFilter {
        let mut rng = SplitMix64::new(seed);
        let mut filter = PhoneClockFilter::new(ClockFilterConfig::default());
        for beat in 0..heartbeats {
            let t = beat as f64 * HEARTBEAT_PERIOD_S;
            filter.update(
                t,
                true_offset(t) + PICK_SIGMA_S * gaussian(&mut rng),
                PICK_SIGMA_S,
            );
        }
        filter
    }

    #[test]
    fn heartbeats_recover_rate_and_offset() {
        let filter = fed_filter(6, 11);
        assert!(filter.has_valid_rate());
        let t = 5.0 * HEARTBEAT_PERIOD_S + 30.0;
        let prediction = filter.predict(t).unwrap();
        assert!(
            (prediction.rate_offset - TRUE_RATE_OFFSET).abs() < 0.5e-6,
            "{prediction:?}"
        );
        assert!((prediction.offset_s - true_offset(t)).abs() < 25e-6);
        assert!(prediction.offset_sigma_s < 40e-6);
    }

    #[test]
    fn single_update_has_no_valid_rate_and_wide_prediction() {
        let filter = fed_filter(1, 3);
        assert!(filter.is_valid());
        assert!(!filter.has_valid_rate());
        let prediction = filter.predict(60.0).unwrap();
        assert!(prediction.offset_sigma_s > 1e-3, "60 s at ±100 ppm prior");
    }

    #[test]
    fn backward_prediction_applies_to_buffered_audio() {
        let filter = fed_filter(5, 5);
        let call_time = 4.0 * HEARTBEAT_PERIOD_S - 4.5;
        let prediction = filter.predict(call_time).unwrap();
        assert!((prediction.offset_s - true_offset(call_time)).abs() < 25e-6);
    }

    #[test]
    fn silent_gap_is_detected_and_rate_kept() {
        let mut filter = fed_filter(5, 7);
        let t = 5.0 * HEARTBEAT_PERIOD_S;
        let dropped_samples_s = 1024.0 / 48_000.0;
        let outcome = filter.update(t, true_offset(t) - dropped_samples_s, PICK_SIGMA_S);
        assert!(matches!(
            outcome,
            ClockUpdateOutcome::SilentGapDetected { .. }
        ));
        let prediction = filter.predict(t + 60.0).unwrap();
        assert!((prediction.rate_offset - TRUE_RATE_OFFSET).abs() < 1e-6);
        assert!((prediction.offset_s - (true_offset(t + 60.0) - dropped_samples_s)).abs() < 100e-6);
    }

    #[test]
    fn moderate_outlier_is_rejected_without_reseeding() {
        let mut filter = fed_filter(5, 9);
        let t = 5.0 * HEARTBEAT_PERIOD_S;
        let outcome = filter.update(t, true_offset(t) + 400e-6, PICK_SIGMA_S);
        assert!(matches!(
            outcome,
            ClockUpdateOutcome::RejectedOutlier { .. }
        ));
        assert!((filter.predict(t).unwrap().offset_s - true_offset(t)).abs() < 30e-6);
    }

    #[test]
    fn invalidation_clears_the_model() {
        let mut filter = fed_filter(3, 1);
        filter.invalidate();
        assert!(!filter.is_valid());
        assert!(filter.predict(0.0).is_none());
        assert!(filter.ticks_per_reference_tick().is_none());
    }

    #[test]
    fn silent_gap_before_the_rate_baseline_resets_an_unvalidated_rate() {
        let mut filter = PhoneClockFilter::new(ClockFilterConfig::default());
        filter.update(0.0, 0.0, 15e-6);
        // A 300 µs pick error two seconds later drags the young rate to ~150 ppm.
        filter.update(2.0, 300e-6, 15e-6);
        assert!(filter.predict(2.0).unwrap().rate_offset > 100e-6);
        assert!(matches!(filter.update(7.0, 0.0, 15e-6), ClockUpdateOutcome::SilentGapDetected { .. }));
        assert_eq!(filter.predict(7.0).unwrap().rate_offset, 0.0);
        assert!(!filter.has_valid_rate());
    }
}
