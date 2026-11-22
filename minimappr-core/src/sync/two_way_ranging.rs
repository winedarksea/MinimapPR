//! BeepBeep two-way acoustic ranging between two phones.
//!
//! A chirps, then B. Each phone counts, on its own audio clock, the time
//! between the two arrivals it hears. With d_AA and d_BB the
//! speaker-to-mic distances of the mic each phone uses:
//!
//! d_AB = c/2 · [(a_hears_b − a_hears_a) − (b_hears_b − b_hears_a)] + (d_AA + d_BB)/2
//!
//! and, because both flights take the same time, the same four arrivals
//! give B's clock offset relative to A:
//!
//! θ_BA = ½[(b_hears_a − a_hears_a) + (b_hears_b − a_hears_b)] + (d_AA − d_BB)/(2c)
//!
//! The distance subtracts two long intervals counted on different crystals,
//! so rate offsets matter (50 ppm over 2 s ≈ 1.7 cm). Two compensations:
//! scale B's interval by the tracked rate ratio, or symmetric double-sided
//! ranging where A chirps again after B and the rate cancels in the
//! UWB-style formula ToF = (Ra·Rb − Da·Db)/(Ra + Rb + Da + Db).
//!
//! Times are seconds on each phone's local sample clock (sample index /
//! nominal rate), so the module is waveform-agnostic: chirp, clap and
//! audible fallback arrivals all go through the same code.

/// Arrival times, each on the clock of the phone that heard it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairArrivalTimes {
    pub a_hears_a_s: f64,
    pub a_hears_b_s: f64,
    pub b_hears_a_s: f64,
    pub b_hears_b_s: f64,
}

/// Speaker → mic distance of the mic each phone ranges with (d_ii).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfPathLengths {
    pub a_m: f64,
    pub b_m: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RateCompensation {
    /// Assume both clocks tick at the nominal rate.
    None,
    /// B's clock ticks this many times per A tick ((1+α_B)/(1+α_A)), e.g.
    /// from per-phone clock filters fed by heartbeat rounds.
    TrackedRateRatio { b_ticks_per_a_tick: f64 },
    /// A chirped again after B: symmetric double-sided ranging.
    DoubleSided {
        a_hears_a_repeat_s: f64,
        b_hears_a_repeat_s: f64,
    },
}

impl RateCompensation {
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::TrackedRateRatio { .. } => "tracked_rate_ratio",
            Self::DoubleSided { .. } => "double_sided",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoWayRangingResult {
    pub distance_m: f64,
    /// local_B − local_A at the same instant, around the exchange midpoint.
    pub clock_offset_b_minus_a_s: f64,
    /// A-clock time the offset refers to: the emission midpoint plus half a
    /// flight time (the four arrivals' centre). Under a rate offset Δk the
    /// offset at the bare emission midpoint differs by Δk · d / 2c.
    pub offset_reference_time_on_a_s: f64,
    /// Rate ratio implied by the double-sided exchange, when available.
    pub measured_b_ticks_per_a_tick: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RangingError {
    /// The arrivals imply a distance below `-tolerance` (swapped or bogus picks).
    NegativeDistance {
        distance_m: f64,
    },
    NonFiniteInput,
    /// Double-sided intervals sum to zero or are out of order.
    DegenerateDoubleSidedIntervals,
}

/// A negative distance down to this is clamped to 0 (two phones touching,
/// with pick noise); anything more negative is an error.
pub const NEGATIVE_DISTANCE_TOLERANCE_M: f64 = 0.05;

fn double_sided_core_flight_s(
    times: &PairArrivalTimes,
    a_repeat: f64,
    b_repeat: f64,
) -> Option<(f64, f64)> {
    let round_a = times.a_hears_b_s - times.a_hears_a_s;
    let delay_a = a_repeat - times.a_hears_b_s;
    let delay_b = times.b_hears_b_s - times.b_hears_a_s;
    let round_b = b_repeat - times.b_hears_b_s;
    let denominator = round_a + round_b + delay_a + delay_b;
    if denominator.abs() < f64::EPSILON || delay_a <= 0.0 || round_b <= 0.0 {
        return None;
    }
    let core_flight = (round_a * round_b - delay_a * delay_b) / denominator;
    let a_span = a_repeat - times.a_hears_a_s;
    let b_span = b_repeat - times.b_hears_a_s;
    Some((core_flight, b_span / a_span))
}

/// Range one pair from one exchange.
pub fn range_pair(
    times: &PairArrivalTimes,
    self_paths: SelfPathLengths,
    speed_of_sound_mps: f64,
    compensation: RateCompensation,
) -> Result<TwoWayRangingResult, RangingError> {
    let inputs = [
        times.a_hears_a_s,
        times.a_hears_b_s,
        times.b_hears_a_s,
        times.b_hears_b_s,
        self_paths.a_m,
        self_paths.b_m,
        speed_of_sound_mps,
    ];
    if inputs.iter().any(|v| !v.is_finite()) {
        return Err(RangingError::NonFiniteInput);
    }
    let self_path_mean_m = 0.5 * (self_paths.a_m + self_paths.b_m);
    let interval_on_a = times.a_hears_b_s - times.a_hears_a_s;
    let interval_on_b = times.b_hears_b_s - times.b_hears_a_s;

    let (distance_m, measured_ratio) = match compensation {
        RateCompensation::None => (
            0.5 * speed_of_sound_mps * (interval_on_a - interval_on_b) + self_path_mean_m,
            None,
        ),
        RateCompensation::TrackedRateRatio { b_ticks_per_a_tick } => {
            let interval_on_b_in_a_ticks = interval_on_b / b_ticks_per_a_tick;
            (
                0.5 * speed_of_sound_mps * (interval_on_a - interval_on_b_in_a_ticks)
                    + self_path_mean_m,
                None,
            )
        }
        RateCompensation::DoubleSided {
            a_hears_a_repeat_s,
            b_hears_a_repeat_s,
        } => {
            let (core_flight, ratio) =
                double_sided_core_flight_s(times, a_hears_a_repeat_s, b_hears_a_repeat_s)
                    .ok_or(RangingError::DegenerateDoubleSidedIntervals)?;
            (
                speed_of_sound_mps * core_flight + self_path_mean_m,
                Some(ratio),
            )
        }
    };
    if distance_m < -NEGATIVE_DISTANCE_TOLERANCE_M {
        return Err(RangingError::NegativeDistance { distance_m });
    }
    let distance_m = distance_m.max(0.0);

    let clock_offset_b_minus_a_s = 0.5
        * ((times.b_hears_a_s - times.a_hears_a_s) + (times.b_hears_b_s - times.a_hears_b_s))
        + (self_paths.a_m - self_paths.b_m) / (2.0 * speed_of_sound_mps);
    let a_emits_on_a = times.a_hears_a_s - self_paths.a_m / speed_of_sound_mps;
    let b_emits_on_a = times.a_hears_b_s - distance_m / speed_of_sound_mps;
    let half_flight_s = 0.5 * distance_m / speed_of_sound_mps;
    Ok(TwoWayRangingResult {
        distance_m,
        clock_offset_b_minus_a_s,
        offset_reference_time_on_a_s: 0.5 * (a_emits_on_a + b_emits_on_a) + half_flight_s,
        measured_b_ticks_per_a_tick: measured_ratio,
    })
}

fn vector_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Remove the speaker–mic geometry bias from one exchange.
///
/// The exchange measures the flights speaker_A → mic_B and speaker_B →
/// mic_A. They are equal only if each speaker sits on its mic: with
/// speaker offsets v_A, v_B the clock offset is biased by
/// (L_AB − L_BA)/(2c) ≈ −(v_A + v_B)·û/(2c) — up to d_ii/c (≈ 440 µs for an
/// earpiece 15 cm from the mic, ≈ 30–60 µs for a bottom speaker) — and the
/// distance is the mean flight, not the mic-to-mic distance. Given the
/// layout's mic positions and each phone's speaker offset in the world
/// frame (per-model speaker vector rotated by the phone's attitude), this
/// returns the result referred to the mics. Exact, not linearised.
pub fn correct_for_speaker_mic_offsets(
    result: &TwoWayRangingResult,
    mic_a_m: [f64; 3],
    mic_b_m: [f64; 3],
    speaker_offset_a_world_m: [f64; 3],
    speaker_offset_b_world_m: [f64; 3],
    speed_of_sound_mps: f64,
) -> TwoWayRangingResult {
    let speaker_a = [
        mic_a_m[0] + speaker_offset_a_world_m[0],
        mic_a_m[1] + speaker_offset_a_world_m[1],
        mic_a_m[2] + speaker_offset_a_world_m[2],
    ];
    let speaker_b = [
        mic_b_m[0] + speaker_offset_b_world_m[0],
        mic_b_m[1] + speaker_offset_b_world_m[1],
        mic_b_m[2] + speaker_offset_b_world_m[2],
    ];
    let flight_a_to_b = vector_distance(speaker_a, mic_b_m);
    let flight_b_to_a = vector_distance(speaker_b, mic_a_m);
    let mic_to_mic = vector_distance(mic_a_m, mic_b_m);
    TwoWayRangingResult {
        distance_m: result.distance_m - (0.5 * (flight_a_to_b + flight_b_to_a) - mic_to_mic),
        clock_offset_b_minus_a_s: result.clock_offset_b_minus_a_s
            - (flight_a_to_b - flight_b_to_a) / (2.0 * speed_of_sound_mps),
        ..*result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_offsets_bias_offset_and_correction_removes_it() {
        // Both phones face north with the speaker 12 cm south of the mic;
        // B is 10 m north of A: flights 10.12 m and 9.88 m.
        let c = SPEED_OF_SOUND_MPS;
        let (mic_a, mic_b) = ([0.0, 0.0, 1.0], [0.0, 10.0, 1.0]);
        let speaker = [0.0, -0.12, 0.0];
        let (flight_ab, flight_ba) = (10.12, 9.88);
        let true_offset = 0.0042;
        let self_path = 0.12;
        let times = PairArrivalTimes {
            a_hears_a_s: 1.0 + self_path / c,
            a_hears_b_s: 1.3 + flight_ba / c,
            b_hears_a_s: true_offset + 1.0 + flight_ab / c,
            b_hears_b_s: true_offset + 1.3 + self_path / c,
        };
        let paths = SelfPathLengths {
            a_m: self_path,
            b_m: self_path,
        };
        let raw = range_pair(&times, paths, c, RateCompensation::None).unwrap();
        assert!((raw.clock_offset_b_minus_a_s - true_offset - 0.24 / (2.0 * c)).abs() < 1e-9);
        let corrected = correct_for_speaker_mic_offsets(&raw, mic_a, mic_b, speaker, speaker, c);
        assert!((corrected.clock_offset_b_minus_a_s - true_offset).abs() < 1e-9);
        assert!((corrected.distance_m - 10.0).abs() < 1e-9);
    }

    const SPEED_OF_SOUND_MPS: f64 = 343.0;

    /// Exact arrivals for clocks local = offset + (1 + rate) · true_time.
    struct SyntheticPair {
        distance_m: f64,
        self_a_m: f64,
        self_b_m: f64,
        a_offset_s: f64,
        a_rate: f64,
        b_offset_s: f64,
        b_rate: f64,
    }

    impl SyntheticPair {
        fn local_a(&self, t: f64) -> f64 {
            self.a_offset_s + (1.0 + self.a_rate) * t
        }
        fn local_b(&self, t: f64) -> f64 {
            self.b_offset_s + (1.0 + self.b_rate) * t
        }
        fn arrivals(&self, a_emits: f64, b_emits: f64) -> PairArrivalTimes {
            let c = SPEED_OF_SOUND_MPS;
            PairArrivalTimes {
                a_hears_a_s: self.local_a(a_emits + self.self_a_m / c),
                a_hears_b_s: self.local_a(b_emits + self.distance_m / c),
                b_hears_a_s: self.local_b(a_emits + self.distance_m / c),
                b_hears_b_s: self.local_b(b_emits + self.self_b_m / c),
            }
        }
        fn true_offset_at(&self, t: f64) -> f64 {
            self.local_b(t) - self.local_a(t)
        }
    }

    fn pair_with_drift(rate_ppm_a: f64, rate_ppm_b: f64) -> SyntheticPair {
        SyntheticPair {
            distance_m: 17.3,
            self_a_m: 0.02,
            self_b_m: 0.13,
            a_offset_s: 0.004,
            a_rate: rate_ppm_a * 1e-6,
            b_offset_s: -0.011,
            b_rate: rate_ppm_b * 1e-6,
        }
    }

    fn self_paths(pair: &SyntheticPair) -> SelfPathLengths {
        SelfPathLengths {
            a_m: pair.self_a_m,
            b_m: pair.self_b_m,
        }
    }

    #[test]
    fn exact_distance_and_offset_without_drift() {
        let pair = pair_with_drift(0.0, 0.0);
        let result = range_pair(
            &pair.arrivals(1.0, 1.25),
            self_paths(&pair),
            SPEED_OF_SOUND_MPS,
            RateCompensation::None,
        )
        .unwrap();
        assert!((result.distance_m - pair.distance_m).abs() < 1e-9);
        assert!((result.clock_offset_b_minus_a_s - pair.true_offset_at(1.125)).abs() < 1e-9);
    }

    #[test]
    fn omitting_self_path_term_biases_by_half_their_sum() {
        let pair = pair_with_drift(0.0, 0.0);
        let result = range_pair(
            &pair.arrivals(1.0, 1.25),
            SelfPathLengths { a_m: 0.0, b_m: 0.0 },
            SPEED_OF_SOUND_MPS,
            RateCompensation::None,
        )
        .unwrap();
        let expected_bias = 0.5 * (pair.self_a_m + pair.self_b_m);
        assert!((pair.distance_m - result.distance_m - expected_bias).abs() < 1e-9);
    }

    #[test]
    fn drift_biases_uncompensated_distance_and_compensation_removes_it() {
        let pair = pair_with_drift(-30.0, 30.0);
        let gap_s = 2.0;
        let arrivals = pair.arrivals(1.0, 1.0 + gap_s);
        let uncompensated = range_pair(
            &arrivals,
            self_paths(&pair),
            SPEED_OF_SOUND_MPS,
            RateCompensation::None,
        )
        .unwrap();
        // 60 ppm relative over 2 s → 120 µs → c/2 · 120 µs ≈ 2 cm.
        assert!((uncompensated.distance_m - pair.distance_m).abs() > 0.015);

        let ratio = (1.0 + pair.b_rate) / (1.0 + pair.a_rate);
        let tracked = range_pair(
            &arrivals,
            self_paths(&pair),
            SPEED_OF_SOUND_MPS,
            RateCompensation::TrackedRateRatio {
                b_ticks_per_a_tick: ratio,
            },
        )
        .unwrap();
        assert!(
            (tracked.distance_m - pair.distance_m).abs() < 1e-3,
            "{tracked:?}"
        );

        let repeat_true_time = 1.0 + 2.0 * gap_s;
        let c = SPEED_OF_SOUND_MPS;
        let double_sided = range_pair(
            &arrivals,
            self_paths(&pair),
            SPEED_OF_SOUND_MPS,
            RateCompensation::DoubleSided {
                a_hears_a_repeat_s: pair.local_a(repeat_true_time + pair.self_a_m / c),
                b_hears_a_repeat_s: pair.local_b(repeat_true_time + pair.distance_m / c),
            },
        )
        .unwrap();
        assert!(
            (double_sided.distance_m - pair.distance_m).abs() < 1e-3,
            "{double_sided:?}"
        );
        let measured = double_sided.measured_b_ticks_per_a_tick.unwrap();
        assert!((measured - ratio).abs() < 1e-9);
    }

    #[test]
    fn double_sided_handles_asymmetric_turnarounds() {
        let pair = pair_with_drift(45.0, -15.0);
        let arrivals = pair.arrivals(0.5, 0.7);
        let c = SPEED_OF_SOUND_MPS;
        let repeat = 2.1;
        let result = range_pair(
            &arrivals,
            self_paths(&pair),
            c,
            RateCompensation::DoubleSided {
                a_hears_a_repeat_s: pair.local_a(repeat + pair.self_a_m / c),
                b_hears_a_repeat_s: pair.local_b(repeat + pair.distance_m / c),
            },
        )
        .unwrap();
        // First-order scale residual d · rate (17 m × 45 ppm ≈ 0.8 mm).
        assert!((result.distance_m - pair.distance_m).abs() < 1e-3);
    }

    #[test]
    fn swapped_arrivals_are_rejected() {
        let pair = pair_with_drift(0.0, 0.0);
        let mut arrivals = pair.arrivals(1.0, 1.25);
        std::mem::swap(&mut arrivals.a_hears_a_s, &mut arrivals.a_hears_b_s);
        let result = range_pair(
            &arrivals,
            self_paths(&pair),
            SPEED_OF_SOUND_MPS,
            RateCompensation::None,
        );
        assert!(matches!(result, Err(RangingError::NegativeDistance { .. })));
    }

    #[test]
    fn non_finite_input_is_rejected() {
        let pair = pair_with_drift(0.0, 0.0);
        let mut arrivals = pair.arrivals(1.0, 1.25);
        arrivals.b_hears_a_s = f64::NAN;
        let result = range_pair(
            &arrivals,
            self_paths(&pair),
            SPEED_OF_SOUND_MPS,
            RateCompensation::None,
        );
        assert_eq!(result, Err(RangingError::NonFiniteInput));
    }
}
