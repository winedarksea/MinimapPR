//! Matched-filter first-arrival picking.
//!
//! Remote chirps: reflections off trunks and the ground can be stronger
//! than the direct path, so the pick is the *earliest* envelope peak that
//! clears the noise floor and a level relative to the strongest peak — and
//! that cannot be explained as a sidelobe of the strongest peak (the
//! template's own autocorrelation sidelobes are tabulated once). Real
//! transducers raise the first sidelobe above the template's: the Pixel 7a
//! earpiece lifts it by up to 4.7 dB (13–16 kHz remote picks), which the old
//! 1.6× (+4.1 dB) margin let through on one sweep half every few rounds —
//! an 11-sample (40 mm) remote jump (2026-11-26 field runs r3_dii20_*).
//!
//! Self recordings (default `SelfRecordingPick::StrongestPeak`): the pick
//! is the strongest peak. The earliest-peak search is wrong here: at 1–30
//! cm there is no weaker earlier path worth finding, while the near-field
//! speaker→mic channel lifts sidelobes up to +10 dB over the template's
//! (Pixel 7a earpiece, 12–18 kHz), far past any usable margin. The older
//! body-path precursor policy (`EarliestOutsideBodyPrecursor`: earlier
//! peaks within the body/air lead of d_ii rejected, a limiter-boosted
//! strongest precursor skipped for a later "airborne" peak) is kept for
//! comparison; in the field it failed: on the 7a earpiece the peak it took
//! for a body precursor is the placement-stable arrival — it became the
//! strongest peak after a re-placement and the policy skipped it for a copy
//! 24 samples later (−87 mm), and the window coupled d_ii to the pick.
//! Self picks are timed at the envelope peak.
//!
//! Everything here works on any template, including a short click template
//! for clap-based or audible fallback sync.

use std::ops::Range;

use super::matched_filter::{parabolic_peak_offset, MatchedFilter};

/// Structural sound speed through glass/aluminium phone frames.
pub const SOUND_SPEED_IN_PHONE_BODY_MPS: f64 = 5_000.0;
/// Longest body path assumed when d_ii is unknown (earpiece-to-bottom-mic class).
pub const DEFAULT_MAXIMUM_SELF_PATH_LENGTH_M: f64 = 0.30;

#[derive(Clone, Debug, PartialEq)]
pub struct FirstArrivalConfig {
    /// Minimum peak envelope over the median envelope (noise floor), dB.
    pub minimum_snr_db: f64,
    /// Earliest-peak acceptance relative to the strongest peak, dB (negative).
    pub first_arrival_relative_level_db: f64,
    /// How far before the strongest peak the earliest-peak search may go, s.
    pub maximum_lead_before_strongest_s: f64,
    /// A candidate must exceed the strongest peak's own sidelobe at that lag
    /// by this factor to count as a separate arrival.
    pub sidelobe_rejection_margin: f64,
    /// Slack added to the computed body/air lead window, s.
    pub precursor_lead_tolerance_s: f64,
    pub air_sound_speed_mps: f64,
    /// Second comparable peak this soon after the pick flags a double
    /// emission (bottom speaker + earpiece ~15 cm apart ≈ 440 µs).
    pub double_arrival_maximum_separation_s: f64,
    pub double_arrival_relative_level_db: f64,
    pub self_recording_pick: SelfRecordingPick,
    /// Two copies closer than the main-lobe width merge into one broad
    /// peak; a half-amplitude width this many times the template's flags it.
    pub double_arrival_width_ratio: f64,
    pub timing: ArrivalTiming,
}

/// Which peak a self recording (a phone hearing its own chirp) is timed at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelfRecordingPick {
    /// The strongest envelope peak; d_ii only enters the range equation.
    StrongestPeak,
    /// Earliest comparable peak outside the body/air precursor window of
    /// d_ii, skipping a limiter-boosted strongest precursor (pre-field
    /// policy; see the module notes for why it is not the default).
    EarliestOutsideBodyPrecursor,
}

/// How the arrival time is read off the picked envelope peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArrivalTiming {
    /// Parabolic vertex of the envelope peak.
    EnvelopePeak,
    /// Where the rising edge crosses `fraction` of the peak, plus the
    /// template's own rise time to that fraction. A reflection arriving
    /// inside the main lobe (a ground bounce ~0.3–0.5 ms late) drags the
    /// peak later but barely moves the leading edge.
    LeadingEdge { fraction: f32 },
}

/// Default leading-edge fraction (−6 dB).
pub const DEFAULT_LEADING_EDGE_FRACTION: f32 = 0.5;

impl Default for FirstArrivalConfig {
    fn default() -> Self {
        Self {
            timing: ArrivalTiming::LeadingEdge {
                fraction: DEFAULT_LEADING_EDGE_FRACTION,
            },
            minimum_snr_db: 14.0,
            first_arrival_relative_level_db: -10.0,
            maximum_lead_before_strongest_s: 0.03,
            // +6 dB over the template sidelobe: clears the 7a earpiece's
            // +4.7 dB (worst seen on remote picks, 2026-11-26) while a true
            // earlier copy above −10 dB still passes beyond ~2 main lobes.
            sidelobe_rejection_margin: 2.0,
            precursor_lead_tolerance_s: 60e-6,
            air_sound_speed_mps: 343.0,
            double_arrival_maximum_separation_s: 0.8e-3,
            double_arrival_relative_level_db: -6.0,
            double_arrival_width_ratio: 1.5,
            self_recording_pick: SelfRecordingPick::StrongestPeak,
        }
    }
}

/// Where the recording's chirp came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArrivalContext {
    RemoteEmitter,
    /// The phone's own chirp; `self_path_length_m` is d_ii for the mic used.
    SelfRecording {
        self_path_length_m: Option<f64>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChirpArrival {
    /// Fractional lag (template start) in the recording slice's samples.
    pub sample_index: f64,
    pub picked_amplitude: f32,
    pub strongest_amplitude: f32,
    pub snr_db: f64,
    /// Strongest peak minus pick, samples (negative when the strongest peak
    /// was a limiter-boosted self-recording precursor before the pick).
    pub lead_before_strongest_samples: f64,
    /// Self recording: an earlier peak cleared the acceptance and sidelobe
    /// tests but was not picked (under `StrongestPeak`, any such peak; under
    /// `EarliestOutsideBodyPrecursor`, one taken as the body precursor).
    pub precursor_rejected: bool,
    /// Lag (samples) between that earlier peak and the pick: diagnostics
    /// for self channels with two comparable arrivals (the 7a earpiece's
    /// sit 24 samples apart at 13–16 kHz, their level ratio set by placement).
    pub rejected_precursor_lead_samples: Option<f64>,
    pub double_arrival_suspected: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ArrivalDetectionFailure {
    EmptySearchWindow,
    BelowNoiseFloor { best_snr_db: f64 },
}

fn amplitude_ratio(level_db: f64) -> f64 {
    10f64.powf(level_db / 20.0)
}

fn decibels(ratio: f64) -> f64 {
    20.0 * ratio.max(f64::MIN_POSITIVE).log10()
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    let middle = sorted.len() / 2;
    let (_, median, _) = sorted.select_nth_unstable_by(middle, f32::total_cmp);
    *median
}

fn is_local_peak(envelope: &[f32], index: usize) -> bool {
    let left = if index == 0 {
        f32::MIN
    } else {
        envelope[index - 1]
    };
    let right = envelope.get(index + 1).copied().unwrap_or(f32::MIN);
    envelope[index] >= left && envelope[index] > right
}

#[derive(Clone, Debug)]
pub struct FirstArrivalDetector {
    filter: MatchedFilter,
    /// Normalised autocorrelation envelope of the template at lags 0..N.
    autocorrelation_envelope: Vec<f32>,
    pub config: FirstArrivalConfig,
    pub sample_rate_hz: f64,
}

impl FirstArrivalDetector {
    pub fn new(template: &[f32], config: FirstArrivalConfig, sample_rate_hz: f64) -> Self {
        let filter = MatchedFilter::new(template);
        let mut padded = template.to_vec();
        padded.resize(template.len() * 2, 0.0);
        let autocorrelation = filter.correlation_envelope(&padded);
        let zero_lag = autocorrelation
            .first()
            .copied()
            .unwrap_or(1.0)
            .max(f32::EPSILON);
        let autocorrelation_envelope = autocorrelation.iter().map(|v| v / zero_lag).collect();
        Self {
            filter,
            autocorrelation_envelope,
            config,
            sample_rate_hz,
        }
    }

    pub fn template_length(&self) -> usize {
        self.filter.template_length()
    }

    fn sidelobe_of_strongest(&self, lag_samples: usize, strongest: f32) -> f32 {
        self.autocorrelation_envelope
            .get(lag_samples)
            .copied()
            .unwrap_or(0.0)
            * strongest
    }

    /// Body-path lead of the precursor over the airborne self path, samples.
    fn body_air_lead_samples(&self, self_path_length_m: Option<f64>) -> f64 {
        let length = self_path_length_m.unwrap_or(DEFAULT_MAXIMUM_SELF_PATH_LENGTH_M);
        (length / self.config.air_sound_speed_mps - length / SOUND_SPEED_IN_PHONE_BODY_MPS)
            * self.sample_rate_hz
    }

    /// Earlier peaks within this lag of the strongest are treated as the
    /// body-path precursor for a self recording with this d_ii.
    pub fn precursor_lead_window_samples(&self, self_path_length_m: Option<f64>) -> f64 {
        // Interference with the airborne main lobe can move the apparent
        // precursor peak up to a main-lobe width earlier than its true lead.
        self.body_air_lead_samples(self_path_length_m)
            + self.config.precursor_lead_tolerance_s * self.sample_rate_hz
            + self.main_lobe_width_samples() as f64
    }

    /// Lags after the strongest peak where the airborne path must sit if the
    /// strongest peak is a limiter-boosted precursor. `None` when that cannot
    /// happen: with a known d_ii whose lead is under half a main lobe the
    /// precursor merges into the airborne peak instead of forming its own.
    /// Starting at the known lead keeps close hand/table reflections (which
    /// follow the airborne peak by an arbitrary delay) out of the window.
    fn airborne_after_precursor_lags(
        &self,
        self_path_length_m: Option<f64>,
    ) -> Option<Range<usize>> {
        let main_lobe = self.main_lobe_width_samples() as f64;
        let tolerance = self.config.precursor_lead_tolerance_s * self.sample_rate_hz;
        let window_end = self
            .precursor_lead_window_samples(self_path_length_m)
            .ceil() as usize
            + 1;
        if self_path_length_m.is_none() {
            return Some(1..window_end);
        }
        let lead = self.body_air_lead_samples(self_path_length_m);
        if lead < 0.5 * main_lobe {
            return None;
        }
        let window_start = ((lead - tolerance).floor().max(1.0)) as usize;
        Some(window_start..window_end)
    }

    /// Lag of the template autocorrelation's first minimum (≈ 1 / bandwidth).
    fn main_lobe_width_samples(&self) -> usize {
        self.autocorrelation_envelope
            .windows(2)
            .position(|pair| pair[1] > pair[0])
            .unwrap_or(0)
    }

    /// Pick the first arrival among template-start lags in `search_lags`
    /// (clamped to the recording).
    pub fn detect(
        &self,
        recording: &[f32],
        search_lags: Range<usize>,
        context: ArrivalContext,
    ) -> Result<ChirpArrival, ArrivalDetectionFailure> {
        let envelope = self.filter.correlation_envelope(recording);
        self.detect_in_envelope(&envelope, search_lags, context)
    }

    /// As `detect`, on a precomputed correlation envelope.
    pub fn detect_in_envelope(
        &self,
        envelope: &[f32],
        search_lags: Range<usize>,
        context: ArrivalContext,
    ) -> Result<ChirpArrival, ArrivalDetectionFailure> {
        let end = search_lags.end.min(envelope.len());
        let start = search_lags.start.min(end);
        if start >= end {
            return Err(ArrivalDetectionFailure::EmptySearchWindow);
        }
        let noise_floor = median(envelope).max(f32::MIN_POSITIVE);
        let (strongest_index, strongest) =
            (start..end)
                .map(|i| (i, envelope[i]))
                .fold(
                    (start, f32::MIN),
                    |best, item| if item.1 > best.1 { item } else { best },
                );
        let snr_db = decibels(strongest as f64 / noise_floor as f64);
        if snr_db < self.config.minimum_snr_db {
            return Err(ArrivalDetectionFailure::BelowNoiseFloor {
                best_snr_db: snr_db,
            });
        }

        let relative_floor =
            strongest as f64 * amplitude_ratio(self.config.first_arrival_relative_level_db);
        let noise_threshold = noise_floor as f64 * amplitude_ratio(self.config.minimum_snr_db);
        let acceptance = relative_floor.max(noise_threshold) as f32;
        let maximum_lead =
            (self.config.maximum_lead_before_strongest_s * self.sample_rate_hz) as usize;
        let search_start = strongest_index.saturating_sub(maximum_lead).max(start);
        let self_policy = match context {
            ArrivalContext::SelfRecording { .. } => Some(self.config.self_recording_pick),
            ArrivalContext::RemoteEmitter => None,
        };
        let precursor_window = match context {
            ArrivalContext::SelfRecording { self_path_length_m }
                if self_policy == Some(SelfRecordingPick::EarliestOutsideBodyPrecursor) =>
            {
                Some(self.precursor_lead_window_samples(self_path_length_m))
            }
            _ => None,
        };
        let strongest_only = self_policy == Some(SelfRecordingPick::StrongestPeak);

        let mut picked_index = strongest_index;
        let mut precursor_rejected = false;
        let mut rejected_precursor_lead_samples = None;
        for index in search_start..strongest_index {
            let value = envelope[index];
            if value < acceptance || !is_local_peak(envelope, index) {
                continue;
            }
            let lag = strongest_index - index;
            let sidelobe = self.sidelobe_of_strongest(lag, strongest);
            if (value as f64) < sidelobe as f64 * self.config.sidelobe_rejection_margin {
                continue;
            }
            if strongest_only {
                // Diagnostics only: the earliest comparable earlier peak.
                precursor_rejected = true;
                rejected_precursor_lead_samples = Some(lag as f64);
                break;
            }
            if let Some(window) = precursor_window {
                if (lag as f64) <= window {
                    precursor_rejected = true;
                    rejected_precursor_lead_samples.get_or_insert(lag as f64);
                    continue;
                }
            }
            picked_index = index;
            break;
        }
        // EarliestOutsideBodyPrecursor only: if limiting made the precursor the strongest peak,
        // the airborne path is the first comparable peak one body/air lead
        // after it. Skipped when an earlier peak was already rejected as the
        // precursor: then the strongest peak *is* the airborne path, and a
        // later peak is a reflection.
        let later_lags = match context {
            ArrivalContext::SelfRecording { self_path_length_m }
                if precursor_window.is_some() && !precursor_rejected =>
            {
                self.airborne_after_precursor_lags(self_path_length_m)
            }
            _ => None,
        };
        if let Some(lags) = later_lags {
            let window_start = (strongest_index + lags.start).min(end);
            let window_end = (strongest_index + lags.end).min(end);
            let later_airborne = (window_start..window_end).find(|&index| {
                let value = envelope[index];
                value >= acceptance
                    && is_local_peak(envelope, index)
                    && value as f64
                        >= self.sidelobe_of_strongest(index - strongest_index, strongest) as f64
                            * self.config.sidelobe_rejection_margin
            });
            if let Some(index) = later_airborne {
                picked_index = index;
                precursor_rejected = true;
                rejected_precursor_lead_samples = Some((index - strongest_index) as f64);
            }
        }

        let double_arrival_suspected =
            self.second_comparable_peak_follows(envelope, picked_index, end);
        let peak_vertex = picked_index as f64 + parabolic_peak_offset(envelope, picked_index);
        let timing = match context {
            ArrivalContext::SelfRecording { .. } => ArrivalTiming::EnvelopePeak,
            ArrivalContext::RemoteEmitter => self.config.timing,
        };
        let sample_index = match timing {
            ArrivalTiming::EnvelopePeak => peak_vertex,
            ArrivalTiming::LeadingEdge { fraction } => self
                .leading_edge_arrival(envelope, picked_index, fraction)
                .unwrap_or(peak_vertex),
        };
        Ok(ChirpArrival {
            sample_index,
            picked_amplitude: envelope[picked_index],
            strongest_amplitude: strongest,
            snr_db,
            // Negative when the pick follows a limiter-boosted precursor.
            lead_before_strongest_samples: strongest_index as f64 - picked_index as f64,
            precursor_rejected,
            rejected_precursor_lead_samples,
            double_arrival_suspected,
        })
    }

    /// Lag (fractional samples) at which the template's own normalised
    /// autocorrelation envelope falls to `fraction`.
    fn template_rise_samples(&self, fraction: f32) -> Option<f64> {
        let lag = self
            .autocorrelation_envelope
            .iter()
            .position(|v| *v < fraction)
            .filter(|lag| *lag > 0)?; // fraction > 1 would put the crossing before lag 0
        let (inside, outside) = (
            self.autocorrelation_envelope[lag - 1],
            self.autocorrelation_envelope[lag],
        );
        Some(lag as f64 - 1.0 + ((inside - fraction) / (inside - outside)) as f64)
    }

    /// Rising-edge crossing of `fraction` × peak before `picked`, moved
    /// forward by the template rise time. `None` if the edge is not found.
    fn leading_edge_arrival(&self, envelope: &[f32], picked: usize, fraction: f32) -> Option<f64> {
        let level = envelope[picked] * fraction;
        let below = (0..picked).rev().find(|i| envelope[*i] < level)?;
        let (low, high) = (envelope[below], envelope[below + 1]);
        let crossing = below as f64 + ((level - low) / (high - low).max(f32::MIN_POSITIVE)) as f64;
        Some(crossing + self.template_rise_samples(fraction)?)
    }

    fn second_comparable_peak_follows(&self, envelope: &[f32], picked: usize, end: usize) -> bool {
        let window =
            (self.config.double_arrival_maximum_separation_s * self.sample_rate_hz) as usize;
        let level =
            envelope[picked] as f64 * amplitude_ratio(self.config.double_arrival_relative_level_db);
        let template_half_width = self
            .autocorrelation_envelope
            .iter()
            .position(|v| *v < 0.5)
            .unwrap_or(1);
        let half_level = envelope[picked] * 0.5;
        let right_edge = (picked..end)
            .find(|i| envelope[*i] < half_level)
            .unwrap_or(end);
        let left_edge = (0..picked)
            .rev()
            .find(|i| envelope[*i] < half_level)
            .unwrap_or(0);
        let merged_width_ratio = (right_edge - left_edge) as f64 / (2 * template_half_width) as f64;
        if merged_width_ratio > self.config.double_arrival_width_ratio {
            return true;
        }
        // Resolved second copy after the pick's own main lobe.
        let main_lobe = 2 * template_half_width;
        ((picked + main_lobe)..(picked + window).min(end)).any(|index| {
            let value = envelope[index];
            is_local_peak(envelope, index)
                && value as f64 >= level
                && value as f64
                    >= self.sidelobe_of_strongest(index - picked, envelope[picked]) as f64
                        * self.config.sidelobe_rejection_margin
        })
    }
}

/// Baseline for the pre-study: first sample whose |x| exceeds a fraction of
/// the window peak (what a naive onset detector does). Returns the index.
pub fn raw_threshold_onset(
    recording: &[f32],
    search: Range<usize>,
    fraction_of_peak: f32,
) -> Option<usize> {
    let end = search.end.min(recording.len());
    let window = recording.get(search.start..end)?;
    let peak = window.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if peak <= 0.0 {
        return None;
    }
    window
        .iter()
        .position(|s| s.abs() >= peak * fraction_of_peak)
        .map(|offset| search.start + offset)
}

#[cfg(test)]
#[path = "first_arrival_tests.rs"]
mod tests;
