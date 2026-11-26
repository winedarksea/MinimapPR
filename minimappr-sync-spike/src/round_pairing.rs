//! Picks every chirp on both phones and pairs them into rounds
//! (A_k, B_j, A_{k+1}), where B_j is the chirp A heard between its own k-th
//! and (k+1)-th chirp.
//!
//! Self picks: each emission's predicted capture sample (emission
//! `target_ns` through the capture timestamp anchor) ± a few tens of ms.
//! Remote picks: the first few matches use a wide window around the
//! emitter's self pick mapped across phones by wall clock (NTP-level, or
//! "both captures started together" without it), a quarter period wide so
//! the listener's own chirps (half a period away) stay out. The median
//! offset of those matches then narrows every later window. The two
//! directions get separate medians: their flight times add with opposite
//! signs, ~117 ms apart at 20 m.

use minimappr_core::sync::first_arrival::ArrivalContext;
use minimappr_core::sync::two_way_ranging::PairArrivalTimes;

use crate::analysis_config::AnalysisConfig;
use crate::arrival_scan::{ArrivalScanner, ScannedArrival};
use crate::bundle_loader::PhoneBundle;

const MILLISECONDS_PER_SECOND: f64 = 1_000.0;
/// Blind self alignment: emissions within this of a blind pick count as matched.
const BLIND_ALIGNMENT_TOLERANCE_S: f64 = 0.005;

#[derive(Debug, Clone, PartialEq)]
pub struct SelfPick {
    pub emission_index: u64,
    pub output_frame_index: u64,
    pub emission_t_ns: i64,
    pub predicted_sample: Option<f64>,
    pub arrival: Option<ScannedArrival>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhonePicks {
    pub self_picks: Vec<SelfPick>,
    /// Same order as the *other* phone's `self_picks`.
    pub remote_picks: Vec<Option<ScannedArrival>>,
    pub blind_scanned: bool,
    /// Median remote − mapped emitter-self sample offset used for refinement.
    pub remote_offset_samples: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedRound {
    pub a_emission_index: u64,
    pub b_emission_index: u64,
    pub times: PairArrivalTimes,
    /// A's next chirp as heard by A and by B (double-sided ranging).
    pub repeat: Option<(f64, f64)>,
    pub snr_db: [f64; 4],
    /// A's audio time of the round (its first self pick), seconds.
    pub a_time_s: f64,
    /// Continuous capture stretch the round lies in (breaks on A and B
    /// before it); θ steps between stretches. Set by `continuous_rounds`.
    pub capture_segment: usize,
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

/// Self picks on one phone.
pub fn pick_self_arrivals(bundle: &PhoneBundle, scanner: &ArrivalScanner, self_path_m: f64, config: &AnalysisConfig) -> (Vec<SelfPick>, bool) {
    let fs = config.sample_rate_hz;
    let context = ArrivalContext::SelfRecording { self_path_length_m: Some(self_path_m) };
    let emissions: Vec<_> = bundle.emissions().into_iter().filter(|e| e.3).collect();
    let predicted: Vec<Option<f64>> = emissions
        .iter()
        .map(|(_, _, target_ns, _, _)| Some(bundle.capture_timeline.as_ref()?.frame_at_ns((*target_ns)? as f64)))
        .collect();
    let have_predictions = predicted.iter().any(Option::is_some);
    let blind_offset = if have_predictions { None } else { blind_output_to_capture_offset(bundle, scanner, &emissions, config) };
    let picks = emissions
        .iter()
        .zip(predicted)
        .map(|((emission_index, output_frame_index, _, _, t_ns), predicted_sample)| {
            let predicted_sample = predicted_sample.or_else(|| blind_offset.map(|offset| *output_frame_index as f64 + offset));
            let arrival = predicted_sample
                .and_then(|centre| scanner.detect_near(&bundle.audio.samples, centre, config.self_search_half_window_s * fs, context));
            SelfPick { emission_index: *emission_index, output_frame_index: *output_frame_index, emission_t_ns: *t_ns, predicted_sample, arrival }
        })
        .collect();
    (picks, !have_predictions)
}

/// No timestamps: find the capture − output frame offset that lines the
/// emission schedule up with the most blind picks.
fn blind_output_to_capture_offset(
    bundle: &PhoneBundle,
    scanner: &ArrivalScanner,
    emissions: &[(u64, u64, Option<i64>, bool, i64)],
    config: &AnalysisConfig,
) -> Option<f64> {
    let fs = config.sample_rate_hz;
    let period_samples = bundle.session.schedule.period_ms as f64 / MILLISECONDS_PER_SECOND * fs;
    let blind = scanner.blind_scan(
        &bundle.audio.samples,
        (config.blind_scan_chunk_s * fs) as usize,
        config.blind_scan_minimum_snr_db,
        (0.25 * period_samples).max(scanner.waveform_length() as f64) as usize,
    );
    let tolerance = BLIND_ALIGNMENT_TOLERANCE_S * fs;
    let first_output = emissions.first()?.1 as f64;
    blind
        .iter()
        .map(|pick| pick.sample_index - first_output)
        .max_by_key(|offset| {
            emissions
                .iter()
                .filter(|e| blind.iter().any(|pick| (pick.sample_index - (e.1 as f64 + offset)).abs() < tolerance))
                .count()
        })
}

/// Remote picks on `listener` of `emitter`'s chirps.
pub fn pick_remote_arrivals(
    listener: &PhoneBundle,
    emitter: &PhoneBundle,
    emitter_self_picks: &[SelfPick],
    scanner: &ArrivalScanner,
    config: &AnalysisConfig,
) -> (Vec<Option<ScannedArrival>>, Option<f64>) {
    let fs = config.sample_rate_hz;
    let period_s = emitter.session.schedule.period_ms.max(1) as f64 / MILLISECONDS_PER_SECOND;
    let wall_clocks = emitter.wall_clock_mapping().zip(listener.wall_clock_mapping());
    let map_across = |emitter_sample: f64| -> f64 {
        match &wall_clocks {
            Some((from, to)) => to.frame_at_wall_ns(from.wall_ns_at_frame(emitter_sample)),
            None => emitter_sample,
        }
    };
    // With wall clocks the mapping is good to NTP accuracy, so a narrow
    // window keeps the listener's own (much louder) chirps out even when
    // the two schedules drifted close; without them the start skew is
    // unknown and only a quarter period is safe.
    let wide_half_window = if wall_clocks.is_some() { config.wall_clock_match_half_window_s } else { 0.25 * period_s } * fs;
    let flight_allowance = config.maximum_flight_s * fs;
    let audio = &listener.audio.samples;
    // Offsets are taken against the mapped sample, not the raw one: the
    // mapping goes through each phone's segmented capture timeline, so a
    // silent drop on either phone (which moves the raw cross-phone sample
    // offset by the whole drop) leaves the mapped offset unchanged.
    let mut offsets = Vec::new();
    for pick in emitter_self_picks.iter().filter_map(|p| p.arrival) {
        if offsets.len() >= config.coarse_match_count {
            break;
        }
        let mapped = map_across(pick.sample_index);
        let centre = mapped + 0.5 * flight_allowance;
        if let Some(remote) = scanner.detect_near(audio, centre, wide_half_window + 0.5 * flight_allowance, ArrivalContext::RemoteEmitter) {
            offsets.push(remote.sample_index - mapped);
        }
    }
    let offset = median(&mut offsets);
    let picks = emitter_self_picks
        .iter()
        .map(|pick| {
            let (Some(arrival), Some(offset)) = (pick.arrival, offset) else { return None };
            scanner.detect_near(audio, map_across(arrival.sample_index) + offset, config.remote_refined_half_window_s * fs, ArrivalContext::RemoteEmitter)
        })
        .collect();
    (picks, offset)
}

/// Rounds (A_k, B_j, A_{k+1}) from both phones' picks.
pub fn pair_rounds(a: &PhonePicks, b: &PhonePicks, sample_rate_hz: f64) -> Vec<PairedRound> {
    let seconds = |sample: f64| sample / sample_rate_hz;
    let mut rounds = Vec::new();
    for k in 0..a.self_picks.len().saturating_sub(1) {
        let (Some(a_k), Some(a_next)) = (a.self_picks[k].arrival, a.self_picks[k + 1].arrival) else { continue };
        let Some(b_hears_a_k) = b.remote_picks.get(k).copied().flatten() else { continue };
        let candidate = a.remote_picks.iter().enumerate().find_map(|(j, heard)| {
            let heard = (*heard)?;
            (heard.sample_index > a_k.sample_index && heard.sample_index < a_next.sample_index).then_some((j, heard))
        });
        let Some((j, a_hears_b)) = candidate else { continue };
        let Some(b_self) = b.self_picks.get(j).and_then(|pick| pick.arrival) else { continue };
        let repeat = b.remote_picks.get(k + 1).copied().flatten().map(|b_next| (seconds(a_next.sample_index), seconds(b_next.sample_index)));
        rounds.push(PairedRound {
            a_emission_index: a.self_picks[k].emission_index,
            b_emission_index: b.self_picks[j].emission_index,
            times: PairArrivalTimes {
                a_hears_a_s: seconds(a_k.sample_index),
                a_hears_b_s: seconds(a_hears_b.sample_index),
                b_hears_a_s: seconds(b_hears_a_k.sample_index),
                b_hears_b_s: seconds(b_self.sample_index),
            },
            repeat,
            snr_db: [a_k.snr_db, a_hears_b.snr_db, b_hears_a_k.snr_db, b_self.snr_db],
            a_time_s: seconds(a_k.sample_index),
            capture_segment: 0,
        });
    }
    rounds
}
