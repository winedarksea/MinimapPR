//! One chirp round: who heard which emission when, turned into pairwise
//! two-way ranging measurements.
//!
//! The host assigns slots round-robin (one phone per slot, never two at
//! once: a phone's own speaker 1 cm from its mic drowns a distant chirp).
//! A round may end with the opening phone chirping again (the closing
//! chirp), which gives every phone's rate relative to the opener from the
//! same round — the double-sided alternative to tracked rates.

use super::clock_graph::ClockOffsetConstraint;
use super::two_way_ranging::{
    range_pair, PairArrivalTimes, RangingError, RateCompensation, SelfPathLengths,
    TwoWayRangingResult,
};

/// Arrival matrix for one round, on each receiver's local clock.
#[derive(Clone, Debug, PartialEq)]
pub struct ChirpRoundObservation {
    /// Emitter (participant index) of each emission in slot order. With a
    /// closing chirp the last entry repeats the first emitter.
    pub emission_emitters: Vec<usize>,
    /// `arrival_local_s[receiver][emission]`; `None` = not detected.
    pub arrival_local_s: Vec<Vec<Option<f64>>>,
    /// d_ii of the mic each participant ranges with.
    pub self_path_lengths_m: Vec<f64>,
}

/// How a round compensates clock-rate offsets between participants.
#[derive(Clone, Debug, PartialEq)]
pub enum RoundRateCompensation {
    None,
    /// Each participant's ticks per reference tick (1 + α_i), from the
    /// per-phone clock filters.
    TrackedRates(Vec<f64>),
    /// Rates from the closing chirp (opener emits first and last); pairs
    /// that include the opener use the full double-sided formula.
    ClosingChirp,
}

impl RoundRateCompensation {
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::TrackedRates(_) => "tracked_rates",
            Self::ClosingChirp => "closing_chirp",
        }
    }
}

/// One ranged pair from a round (i < j by emission order).
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseRangingMeasurement {
    pub first_participant: usize,
    pub second_participant: usize,
    pub result: TwoWayRangingResult,
    /// Time between the two phones' slots on the first phone's clock; the
    /// clock graph down-weights long gaps (motion without steps).
    pub slot_gap_s: f64,
    pub compensation_used: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseRangingFailure {
    pub first_participant: usize,
    pub second_participant: usize,
    pub error: RangingError,
}

impl ChirpRoundObservation {
    pub fn participant_count(&self) -> usize {
        self.arrival_local_s.len()
    }

    /// First emission index of each participant (`None` = did not chirp).
    pub fn first_emission_of(&self, participant: usize) -> Option<usize> {
        self.emission_emitters
            .iter()
            .position(|e| *e == participant)
    }

    /// Emission index of the closing chirp, if the opener chirped twice.
    pub fn closing_emission_index(&self) -> Option<usize> {
        let opener = *self.emission_emitters.first()?;
        let last = self.emission_emitters.len() - 1;
        (last > 0 && self.emission_emitters[last] == opener).then_some(last)
    }

    fn arrival(&self, receiver: usize, emission: usize) -> Option<f64> {
        self.arrival_local_s
            .get(receiver)?
            .get(emission)
            .copied()
            .flatten()
    }

    /// Each participant's ticks per opener tick, measured between the
    /// opener's first and closing chirp. Assumes nobody moved in between;
    /// 1 cm of relative motion over a 1.4 s round is already ~20 ppm.
    pub fn closing_chirp_rate_ratios(&self) -> Option<Vec<Option<f64>>> {
        let closing = self.closing_emission_index()?;
        let opener = self.emission_emitters[0];
        let opener_span = self.arrival(opener, closing)? - self.arrival(opener, 0)?;
        if opener_span <= 0.0 {
            return None;
        }
        Some(
            (0..self.participant_count())
                .map(|receiver| {
                    let span = self.arrival(receiver, closing)? - self.arrival(receiver, 0)?;
                    Some(span / opener_span)
                })
                .collect(),
        )
    }

    fn pair_compensation(
        &self,
        first: usize,
        second: usize,
        compensation: &RoundRateCompensation,
        closing_ratios: &Option<Vec<Option<f64>>>,
    ) -> Option<RateCompensation> {
        match compensation {
            RoundRateCompensation::None => Some(RateCompensation::None),
            RoundRateCompensation::TrackedRates(rates) => {
                Some(RateCompensation::TrackedRateRatio {
                    b_ticks_per_a_tick: rates.get(second)? / rates.get(first)?,
                })
            }
            RoundRateCompensation::ClosingChirp => {
                let closing = self.closing_emission_index()?;
                if first == self.emission_emitters[0] {
                    return Some(RateCompensation::DoubleSided {
                        a_hears_a_repeat_s: self.arrival(first, closing)?,
                        b_hears_a_repeat_s: self.arrival(second, closing)?,
                    });
                }
                let ratios = closing_ratios.as_ref()?;
                Some(RateCompensation::TrackedRateRatio {
                    b_ticks_per_a_tick: (*ratios.get(second)?)? / (*ratios.get(first)?)?,
                })
            }
        }
    }

    /// Range every pair that heard each other in both directions.
    pub fn pairwise_measurements(
        &self,
        speed_of_sound_mps: f64,
        compensation: &RoundRateCompensation,
    ) -> (Vec<PairwiseRangingMeasurement>, Vec<PairwiseRangingFailure>) {
        let closing_ratios = match compensation {
            RoundRateCompensation::ClosingChirp => self.closing_chirp_rate_ratios(),
            _ => None,
        };
        let mut emitters: Vec<(usize, usize)> = (0..self.participant_count())
            .filter_map(|p| self.first_emission_of(p).map(|e| (e, p)))
            .collect();
        emitters.sort_unstable();
        let mut measurements = Vec::new();
        let mut failures = Vec::new();
        for (index, &(first_emission, first)) in emitters.iter().enumerate() {
            for &(second_emission, second) in &emitters[index + 1..] {
                let arrivals = (|| {
                    Some(PairArrivalTimes {
                        a_hears_a_s: self.arrival(first, first_emission)?,
                        a_hears_b_s: self.arrival(first, second_emission)?,
                        b_hears_a_s: self.arrival(second, first_emission)?,
                        b_hears_b_s: self.arrival(second, second_emission)?,
                    })
                })();
                let Some(arrivals) = arrivals else { continue };
                let Some(pair_compensation) =
                    self.pair_compensation(first, second, compensation, &closing_ratios)
                else {
                    continue;
                };
                // A malformed observation (fewer d_ii than receivers) skips the
                // pair instead of panicking.
                let (Some(&a_m), Some(&b_m)) = (
                    self.self_path_lengths_m.get(first),
                    self.self_path_lengths_m.get(second),
                ) else {
                    continue;
                };
                let self_paths = SelfPathLengths { a_m, b_m };
                match range_pair(&arrivals, self_paths, speed_of_sound_mps, pair_compensation) {
                    Ok(result) => measurements.push(PairwiseRangingMeasurement {
                        first_participant: first,
                        second_participant: second,
                        result,
                        slot_gap_s: arrivals.a_hears_b_s - arrivals.a_hears_a_s,
                        compensation_used: pair_compensation.name(),
                    }),
                    Err(error) => failures.push(PairwiseRangingFailure {
                        first_participant: first,
                        second_participant: second,
                        error,
                    }),
                }
            }
        }
        (measurements, failures)
    }
}

/// Turn pairwise offsets into clock-graph constraints at one common
/// reference time. Each pair's offset refers to its own exchange midpoint;
/// with ±60 ppm crystals the offset moves ~60 µs across a 0.5 s spread, so
/// each is carried to `reference_time_s` with the participants' rates
/// (ticks per reference tick). Without rates the offsets are used as-is.
/// Pair times are on the first phone's clock; `coarse_offsets_s[p]`
/// (local − reference; millisecond accuracy is plenty, 1 ms × 100 ppm =
/// 0.1 µs) moves them onto the reference clock — from the clock filters,
/// or from a first uncorrected graph solve. Missing entries count as 0.
/// `motion_energy[p]` is participant p's linear-acceleration energy over
/// the round; a constraint carries the sum of its two phones'.
pub fn clock_offset_constraints(
    measurements: &[PairwiseRangingMeasurement],
    reference_time_s: f64,
    ticks_per_reference_tick: Option<&[f64]>,
    coarse_offsets_s: &[f64],
    motion_energy: &[f64],
) -> Vec<ClockOffsetConstraint> {
    measurements
        .iter()
        .map(|m| {
            let (i, j) = (m.first_participant, m.second_participant);
            let pair_time_on_reference = m.result.offset_reference_time_on_a_s
                - coarse_offsets_s.get(i).copied().unwrap_or(0.0);
            let drift_correction = ticks_per_reference_tick
                .and_then(|rates| Some((rates.get(i)?, rates.get(j)?)))
                .map_or(0.0, |(rate_i, rate_j)| {
                    (rate_j - rate_i) * (reference_time_s - pair_time_on_reference)
                });
            ClockOffsetConstraint {
                node_i: i,
                node_j: j,
                offset_j_minus_i_s: m.result.clock_offset_b_minus_a_s + drift_correction,
                slot_gap_s: m.slot_gap_s,
                motion_energy: motion_energy.get(i).copied().unwrap_or(0.0)
                    + motion_energy.get(j).copied().unwrap_or(0.0),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEED_OF_SOUND_MPS: f64 = 343.0;

    /// Three static phones (a triangle), drifting clocks, optional closing chirp.
    fn synthetic_round(with_closing_chirp: bool) -> ChirpRoundObservation {
        let positions: [[f64; 2]; 3] = [[0.0, 0.0], [10.0, 0.0], [4.0, 7.0]];
        let offsets = [0.0, 0.013, -0.007];
        let rates_ppm = [0.0, 50.0, -40.0];
        let self_paths = vec![0.02, 0.05, 0.11];
        let slot_s = 0.25;
        let mut emitters = vec![0, 1, 2];
        if with_closing_chirp {
            emitters.push(0);
        }
        let distance = |a: usize, b: usize| {
            let dx = positions[a][0] - positions[b][0];
            let dy = positions[a][1] - positions[b][1];
            (dx * dx + dy * dy).sqrt()
        };
        let arrival_local_s = (0..3)
            .map(|receiver| {
                emitters
                    .iter()
                    .enumerate()
                    .map(|(slot, &emitter)| {
                        let path = if receiver == emitter {
                            self_paths[receiver]
                        } else {
                            distance(receiver, emitter)
                        };
                        let true_time = 1.0 + slot as f64 * slot_s + path / SPEED_OF_SOUND_MPS;
                        Some(offsets[receiver] + (1.0 + rates_ppm[receiver] * 1e-6) * true_time)
                    })
                    .collect()
            })
            .collect();
        ChirpRoundObservation {
            emission_emitters: emitters,
            arrival_local_s,
            self_path_lengths_m: self_paths,
        }
    }

    fn true_distance(first: usize, second: usize) -> f64 {
        match (first, second) {
            (0, 1) => 10.0,
            (0, 2) => (16.0f64 + 49.0).sqrt(),
            (1, 2) => (36.0f64 + 49.0).sqrt(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn all_pairs_ranged_and_closing_chirp_cancels_drift() {
        let round = synthetic_round(true);
        let (none, _) =
            round.pairwise_measurements(SPEED_OF_SOUND_MPS, &RoundRateCompensation::None);
        let (closing, failures) =
            round.pairwise_measurements(SPEED_OF_SOUND_MPS, &RoundRateCompensation::ClosingChirp);
        assert!(failures.is_empty());
        assert_eq!(closing.len(), 3);
        let worst = |list: &[PairwiseRangingMeasurement]| {
            list.iter()
                .map(|m| {
                    (m.result.distance_m - true_distance(m.first_participant, m.second_participant))
                        .abs()
                })
                .fold(0.0, f64::max)
        };
        assert!(
            worst(&none) > 2e-3,
            "drift should show without compensation"
        );
        // Residual ≈ d · rate offset (local seconds are not true seconds): < 1 mm.
        assert!(
            worst(&closing) < 1e-3,
            "closing chirp worst {}",
            worst(&closing)
        );
        assert_eq!(closing[0].compensation_used, "double_sided");
        assert_eq!(closing[2].compensation_used, "tracked_rate_ratio");
    }

    #[test]
    fn tracked_rates_cancel_drift() {
        let round = synthetic_round(false);
        let rates = vec![1.0, 1.0 + 50e-6, 1.0 - 40e-6];
        let (measurements, _) = round.pairwise_measurements(
            SPEED_OF_SOUND_MPS,
            &RoundRateCompensation::TrackedRates(rates),
        );
        for m in &measurements {
            let error =
                m.result.distance_m - true_distance(m.first_participant, m.second_participant);
            assert!(error.abs() < 1e-3, "{m:?}");
        }
    }

    #[test]
    fn closing_chirp_rate_ratios_match_truth() {
        let round = synthetic_round(true);
        let ratios = round.closing_chirp_rate_ratios().unwrap();
        assert!((ratios[1].unwrap() - (1.0 + 50e-6)).abs() < 1e-9);
        assert!((ratios[2].unwrap() - (1.0 - 40e-6)).abs() < 1e-9);
        let no_closing = synthetic_round(false);
        assert!(no_closing.closing_chirp_rate_ratios().is_none());
    }

    #[test]
    fn missing_arrival_skips_only_that_pair() {
        let mut round = synthetic_round(false);
        round.arrival_local_s[2][1] = None;
        let (measurements, _) =
            round.pairwise_measurements(SPEED_OF_SOUND_MPS, &RoundRateCompensation::None);
        assert_eq!(measurements.len(), 2);
        assert!(measurements
            .iter()
            .all(|m| !(m.first_participant == 1 && m.second_participant == 2)));
    }

    #[test]
    fn missing_self_path_lengths_skip_pairs_without_panicking() {
        let mut round = synthetic_round(false);
        round.self_path_lengths_m.truncate(2);
        let (measurements, _) =
            round.pairwise_measurements(SPEED_OF_SOUND_MPS, &RoundRateCompensation::None);
        assert_eq!(measurements.len(), 1);
        assert_eq!(
            (
                measurements[0].first_participant,
                measurements[0].second_participant
            ),
            (0, 1)
        );
    }

    #[test]
    fn offset_constraints_are_carried_to_the_reference_time() {
        let round = synthetic_round(false);
        let (measurements, _) =
            round.pairwise_measurements(SPEED_OF_SOUND_MPS, &RoundRateCompensation::None);
        let rates = [1.0, 1.0 + 50e-6, 1.0 - 40e-6];
        let offsets = [0.0, 0.013, -0.007];
        let reference_time_s = 1.25;
        let constraints = clock_offset_constraints(
            &measurements,
            reference_time_s,
            Some(&rates),
            &offsets,
            &[0.0, 0.5, 0.0],
        );
        for constraint in &constraints {
            let local = |p: usize| offsets[p] + rates[p] * reference_time_s;
            let truth = local(constraint.node_j) - local(constraint.node_i);
            assert!(
                (constraint.offset_j_minus_i_s - truth).abs() < 1e-7,
                "{constraint:?}"
            );
        }
        assert_eq!(constraints[0].motion_energy, 0.5);
    }
}
