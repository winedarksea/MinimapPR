//! Scores a `RenderedRound` against simulator truth: pick errors, detection
//! failures, BeepBeep ranging and clock-offset errors per rate
//! compensation, clock-graph offsets and layout error.
#![allow(dead_code)]

use minimappr_core::layout::gps_anchor_fit::weighted_kabsch_2d;
use minimappr_core::layout::robust_layout_refine::{
    solve_layout, LayoutRefineConfig, RangeMeasurement,
};
use minimappr_core::sync::chirp_codes::SplitMix64;
use minimappr_core::sync::chirp_round::{
    clock_offset_constraints, ChirpRoundObservation, RoundRateCompensation,
};
use minimappr_core::sync::clock_graph::{solve_clock_graph, ClockGraphConfig};
use minimappr_core::sync::two_way_ranging::correct_for_speaker_mic_offsets;

use super::sim_sync_trial::RenderedRound;

/// Rate error of a heartbeat-fed clock filter (plan: ~0.5 ppm after a few heartbeats).
pub const TRACKED_RATE_ERROR_PPM: f64 = 0.5;
/// Range σ handed to the layout solve.
pub const LAYOUT_RANGE_SIGMA_M: f64 = 0.03;
/// A pick further than this from truth counts as a wrong lock (reflection, precursor).
pub const WRONG_LOCK_THRESHOLD_US: f64 = 200.0;
const PARTS_PER_MILLION: f64 = 1e6;

pub const COMPENSATION_NAMES: [&str; 3] = ["none", "tracked", "closing"];

#[derive(Clone, Debug, Default)]
pub struct CompensationScores {
    pub ranging_errors_m: Vec<f64>,
    /// Raw BeepBeep offsets (speaker–mic asymmetry not removed).
    pub raw_pair_offset_errors_us: Vec<f64>,
    /// After `correct_for_speaker_mic_offsets` (when the trial asks for it).
    pub pair_offset_errors_us: Vec<f64>,
    pub graph_offset_errors_us: Vec<f64>,
    pub layout_errors_m: Vec<f64>,
    pub layout_failures: usize,
}

#[derive(Clone, Debug, Default)]
pub struct TrialScores {
    pub remote_pick_errors_us: Vec<f64>,
    pub self_pick_errors_us: Vec<f64>,
    pub self_raw_onset_errors_us: Vec<f64>,
    pub remote_expected: usize,
    pub remote_failures: usize,
    pub self_expected: usize,
    pub self_failures: usize,
    /// Indexed like `COMPENSATION_NAMES`.
    pub by_compensation: [CompensationScores; 3],
}

impl TrialScores {
    pub fn merge(&mut self, other: TrialScores) {
        self.remote_pick_errors_us
            .extend(other.remote_pick_errors_us);
        self.self_pick_errors_us.extend(other.self_pick_errors_us);
        self.self_raw_onset_errors_us
            .extend(other.self_raw_onset_errors_us);
        self.remote_expected += other.remote_expected;
        self.remote_failures += other.remote_failures;
        self.self_expected += other.self_expected;
        self.self_failures += other.self_failures;
        for (mine, theirs) in self.by_compensation.iter_mut().zip(other.by_compensation) {
            mine.ranging_errors_m.extend(theirs.ranging_errors_m);
            mine.raw_pair_offset_errors_us
                .extend(theirs.raw_pair_offset_errors_us);
            mine.pair_offset_errors_us
                .extend(theirs.pair_offset_errors_us);
            mine.graph_offset_errors_us
                .extend(theirs.graph_offset_errors_us);
            mine.layout_errors_m.extend(theirs.layout_errors_m);
            mine.layout_failures += theirs.layout_failures;
        }
    }
}

fn gaussian(rng: &mut SplitMix64) -> f64 {
    let u1 = ((rng.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

pub fn observation_of(round: &RenderedRound) -> ChirpRoundObservation {
    let fs = round.plan.sample_rate_hz;
    let count = round.config.phone_count();
    let mut arrival_local_s = vec![vec![None; round.emission_emitters.len()]; count];
    for pick in &round.picks {
        arrival_local_s[pick.receiver][pick.emission] = pick.detected_sample.map(|s| s / fs);
    }
    ChirpRoundObservation {
        emission_emitters: round.emission_emitters.clone(),
        arrival_local_s,
        self_path_lengths_m: round.self_path_lengths_m.clone(),
    }
}

fn local_seconds(round: &RenderedRound, phone: usize, true_time_s: f64) -> f64 {
    let fs = round.plan.sample_rate_hz;
    round.plan.devices[phone]
        .clock
        .sample_of_true_time(true_time_s, fs)
        / fs
}

fn true_time_of_local(round: &RenderedRound, phone: usize, local_s: f64) -> f64 {
    let fs = round.plan.sample_rate_hz;
    round.plan.devices[phone]
        .clock
        .true_time_of_sample(local_s * fs, fs)
}

/// Ticks per reference (phone 0) tick for each compensation.
fn rates_for(
    round: &RenderedRound,
    compensation: usize,
    observation: &ChirpRoundObservation,
) -> Option<Vec<f64>> {
    let drift = |phone: usize| 1.0 + round.plan.devices[phone].clock.drift_ppm / PARTS_PER_MILLION;
    match COMPENSATION_NAMES[compensation] {
        "tracked" => {
            let mut rng = SplitMix64::new(round.config.seed ^ 0x7AC);
            Some(
                (0..round.config.phone_count())
                    .map(|p| {
                        drift(p) / drift(0)
                            + if p == 0 {
                                0.0
                            } else {
                                TRACKED_RATE_ERROR_PPM / PARTS_PER_MILLION * gaussian(&mut rng)
                            }
                    })
                    .collect(),
            )
        }
        "closing" => observation
            .closing_chirp_rate_ratios()
            .map(|ratios| ratios.iter().map(|r| r.unwrap_or(1.0)).collect()),
        _ => None,
    }
}

fn aligned_layout_errors(estimated: &[[f64; 2]], truth: &[[f64; 2]]) -> Vec<f64> {
    let weights = vec![1.0; truth.len()];
    [false, true]
        .iter()
        .map(|&mirrored| {
            let transform = weighted_kabsch_2d(estimated, truth, &weights, mirrored);
            estimated
                .iter()
                .zip(truth)
                .map(|(e, t)| {
                    let p = transform.apply(*e);
                    ((p[0] - t[0]).powi(2) + (p[1] - t[1]).powi(2)).sqrt()
                })
                .collect::<Vec<f64>>()
        })
        .min_by(|a, b| a.iter().sum::<f64>().total_cmp(&b.iter().sum::<f64>()))
        .unwrap_or_default()
}

/// `correct_speaker_geometry`: refer offsets/distances to the mics using
/// the true speaker vectors and mic positions (stands in for per-model
/// speaker vectors + phone attitude + the solved layout).
pub fn score_round(round: &RenderedRound, correct_speaker_geometry: bool) -> TrialScores {
    let fs = round.plan.sample_rate_hz;
    let c = round.plan.speed_of_sound_mps;
    let count = round.config.phone_count();
    let mut scores = TrialScores::default();
    for pick in &round.picks {
        let error = pick.error_us(fs);
        if pick.is_self {
            scores.self_expected += 1;
            match error {
                Some(e) => scores.self_pick_errors_us.push(e),
                None => scores.self_failures += 1,
            }
            if let Some(raw) = pick.raw_onset_sample {
                scores
                    .self_raw_onset_errors_us
                    .push((raw - pick.true_sample) / fs * 1e6);
            }
        } else {
            scores.remote_expected += 1;
            match error {
                Some(e) => scores.remote_pick_errors_us.push(e),
                None => scores.remote_failures += 1,
            }
        }
    }
    let observation = observation_of(round);
    let reference_true_time = round.config.round_mid_time_s();
    let reference_local = local_seconds(round, 0, reference_true_time);
    let first_emission = |p: usize| {
        round
            .emission_emitters
            .iter()
            .position(|e| *e == p)
            .unwrap_or(0)
    };
    let mid_positions: Vec<[f64; 3]> = (0..count)
        .map(|p| {
            let device = &round.plan.devices[p];
            let mic = device.mic_offsets_m[usize::from(round.config.use_far_mic)];
            let base = device.trajectory.position_at(reference_true_time);
            [base[0] + mic[0], base[1] + mic[1], base[2] + mic[2]]
        })
        .collect();

    for (compensation_index, name) in COMPENSATION_NAMES.iter().enumerate() {
        let rates = rates_for(round, compensation_index, &observation);
        let round_compensation = match *name {
            "none" => RoundRateCompensation::None,
            "tracked" => RoundRateCompensation::TrackedRates(rates.clone().unwrap_or_default()),
            _ if observation.closing_emission_index().is_some() => {
                RoundRateCompensation::ClosingChirp
            }
            _ => continue,
        };
        let target = &mut scores.by_compensation[compensation_index];
        let (mut measurements, _failures) =
            observation.pairwise_measurements(c, &round_compensation);
        for m in measurements.iter_mut() {
            let (a, b) = (m.first_participant, m.second_participant);
            let truth = 0.5
                * (round.path_lengths_m[b][first_emission(a)]
                    + round.path_lengths_m[a][first_emission(b)]);
            target.ranging_errors_m.push(m.result.distance_m - truth);
            let t = true_time_of_local(round, a, m.result.offset_reference_time_on_a_s);
            let truth_offset = local_seconds(round, b, t) - local_seconds(round, a, t);
            target
                .raw_pair_offset_errors_us
                .push((m.result.clock_offset_b_minus_a_s - truth_offset) * 1e6);
            if correct_speaker_geometry {
                m.result = correct_for_speaker_mic_offsets(
                    &m.result,
                    mid_positions[a],
                    mid_positions[b],
                    round.speaker_offsets_from_mic_m[a],
                    round.speaker_offsets_from_mic_m[b],
                    c,
                );
            }
            target
                .pair_offset_errors_us
                .push((m.result.clock_offset_b_minus_a_s - truth_offset) * 1e6);
        }
        // Two-pass graph: a first solve gives the coarse offsets that put pair times on the reference clock.
        let graph_config = ClockGraphConfig {
            speed_of_sound_mps: c,
            ..ClockGraphConfig::default()
        };
        let no_motion = vec![0.0; count];
        let first = clock_offset_constraints(
            &measurements,
            reference_local,
            rates.as_deref(),
            &[],
            &no_motion,
        );
        if let Ok(coarse) = solve_clock_graph(count, 0, &first, &graph_config) {
            let coarse_offsets: Vec<f64> =
                coarse.offsets_s.iter().map(|o| o.unwrap_or(0.0)).collect();
            let constraints = clock_offset_constraints(
                &measurements,
                reference_local,
                rates.as_deref(),
                &coarse_offsets,
                &no_motion,
            );
            if let Ok(solution) = solve_clock_graph(count, 0, &constraints, &graph_config) {
                for p in 1..count {
                    if let Some(offset) = solution.offsets_s[p] {
                        let truth = local_seconds(round, p, reference_true_time) - reference_local;
                        target.graph_offset_errors_us.push((offset - truth) * 1e6);
                    }
                }
            }
        }
        let ranges: Vec<RangeMeasurement> = measurements
            .iter()
            .map(|m| RangeMeasurement {
                node_i: m.first_participant,
                node_j: m.second_participant,
                distance_m: m.result.distance_m,
                sigma_m: LAYOUT_RANGE_SIGMA_M,
            })
            .collect();
        let heights: Vec<f64> = mid_positions.iter().map(|p| p[2]).collect();
        match solve_layout(count, &heights, &ranges, &LayoutRefineConfig::default()) {
            Ok(layout) => {
                let estimated: Vec<[f64; 2]> =
                    layout.positions_m.iter().map(|p| [p[0], p[1]]).collect();
                let truth: Vec<[f64; 2]> = mid_positions.iter().map(|p| [p[0], p[1]]).collect();
                target
                    .layout_errors_m
                    .extend(aligned_layout_errors(&estimated, &truth));
            }
            Err(_) => target.layout_failures += 1,
        }
    }
    scores
}

/// Sorted-copy percentile of absolute values (p in 0..=100).
pub fn absolute_percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted: Vec<f64> = values.iter().map(|v| v.abs()).collect();
    sorted.sort_by(f64::total_cmp);
    let rank = (percentile / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn fraction_beyond(values: &[f64], threshold: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().filter(|v| v.abs() > threshold).count() as f64 / values.len() as f64
}
