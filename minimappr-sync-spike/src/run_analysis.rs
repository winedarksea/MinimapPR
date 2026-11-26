//! One run end to end: load both bundles, pick, pair, range, clock
//! residuals, onset stability, leakage and channel-map evidence, into a
//! serialisable [`RunReport`]. `analyse_single` does the one-phone subset
//! (self loopback) used for the single-phone milestone.

use std::path::Path;

use minimappr_core::sync::two_way_ranging::SelfPathLengths;

use crate::analysis_config::AnalysisConfig;
use crate::arrival_scan::{waveform_from_description, ArrivalScanner};
use crate::bundle_loader::PhoneBundle;
use crate::bundle_schema::{RunManifest, DEFAULT_AIR_TEMPERATURE_C};
use crate::capture_continuity::{capture_continuity, continuous_rounds, CaptureContinuity};
use crate::channel_map::{dominant_orientation, ChannelMapObservation};
use crate::clock_residuals::{clock_residual_report, ClockResidualReport};
use crate::onset_stability::{onset_stability, OnsetStabilityReport};
use crate::pair_ranging_series::{distance_statistics, implied_mean_self_path_m, median_of, range_rounds, CompensationMode, DistanceStatistics, RangedRound, SpeakerGeometry};
use crate::round_pairing::{pair_rounds, pick_remote_arrivals, pick_self_arrivals, PhonePicks};
use crate::spectral_leakage::{leakage_report, LeakageReport};

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PhoneSummary {
    pub role: String,
    pub platform_name: String,
    pub sdk_int: u32,
    pub capture_source: serde_json::Value,
    pub duration_s: f64,
    pub emissions_played: usize,
    pub capture_gaps: usize,
    pub writer_overflows: usize,
    pub capture_anchor_residual_us: Option<f64>,
    pub capture_rate_offset_ppm: Option<f64>,
    /// Silent drops (timestamp steps) and self-interval breaks.
    pub capture_continuity: CaptureContinuity,
    pub blind_scanned: bool,
    pub onset: OnsetStabilityReport,
    pub self_leakage: LeakageReport,
    pub remote_leakage: Option<LeakageReport>,
    pub remote_detected: usize,
    pub median_remote_snr_db: Option<f64>,
    pub screen_off_events: usize,
    pub volume_zero_seen: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RunReport {
    pub run_label: String,
    pub matrix_run: String,
    pub waveform: crate::bundle_schema::WaveformDescription,
    pub period_ms: u64,
    pub tape_distance_m: Option<f64>,
    pub temperature_c: f64,
    pub temperature_assumed: bool,
    pub speed_of_sound_mps: f64,
    pub self_paths_m: [f64; 2],
    pub speaker_correction_applied: bool,
    pub phones: Vec<PhoneSummary>,
    pub round_count: usize,
    /// Paired rounds left out because a capture/playback break overlaps them.
    pub rounds_excluded_for_capture_breaks: usize,
    pub distance: Vec<DistanceStatistics>,
    pub implied_mean_self_path_m: Option<f64>,
    pub clock: Option<ClockResidualReport>,
    pub channel_map: Vec<ChannelMapObservation>,
    #[serde(skip)]
    pub ranged_rounds: Vec<RangedRound>,
}

fn count_events(bundle: &PhoneBundle, predicate: impl Fn(&crate::bundle_schema::BundleEvent) -> bool) -> usize {
    bundle.events.iter().filter(|e| predicate(e)).count()
}

fn continuity_of(bundle: &PhoneBundle, picks: &PhonePicks, config: &AnalysisConfig) -> CaptureContinuity {
    let drops = bundle.capture_timeline.as_ref().map_or(&[][..], |timeline| &timeline.silent_drops[..]);
    capture_continuity(&picks.self_picks, drops, config.self_interval_break_tolerance_samples, config.sample_rate_hz)
}

fn phone_summary(role: &str, bundle: &PhoneBundle, picks: &PhonePicks, scanner: &ArrivalScanner, config: &AnalysisConfig) -> PhoneSummary {
    use crate::bundle_schema::BundleEvent as E;
    let self_starts: Vec<f64> = picks.self_picks.iter().filter_map(|p| p.arrival.map(|a| a.sample_index)).collect();
    let remote: Vec<_> = picks.remote_picks.iter().flatten().collect();
    let remote_starts: Vec<f64> = remote.iter().map(|a| a.sample_index).collect();
    let remote_snrs: Vec<f64> = remote.iter().map(|a| a.snr_db).collect();
    let fs = config.sample_rate_hz;
    let continuity = continuity_of(bundle, picks, config);
    PhoneSummary {
        role: role.into(),
        platform_name: bundle.session.platform_name.clone(),
        sdk_int: bundle.session.device.sdk_int,
        capture_source: bundle.session.capture.get("actual_source").cloned().unwrap_or_default(),
        duration_s: bundle.duration_s(),
        emissions_played: picks.self_picks.len(),
        capture_gaps: count_events(bundle, |e| matches!(e, E::CaptureGap { .. })),
        writer_overflows: count_events(bundle, |e| matches!(e, E::WriterOverflow { .. })),
        capture_anchor_residual_us: bundle.capture_timeline.as_ref().map(|t| t.residual_rms_ns() / 1e3),
        capture_rate_offset_ppm: bundle.capture_timeline.as_ref().and_then(|t| t.rate_offset_ppm()),
        onset: onset_stability(&picks.self_picks, &bundle.markers(), &continuity, fs),
        capture_continuity: continuity,
        blind_scanned: picks.blind_scanned,
        self_leakage: leakage_report(&bundle.audio.samples, &self_starts, scanner.waveform_length(), config.leakage_band_hz, fs),
        remote_leakage: (!remote_starts.is_empty()).then(|| leakage_report(&bundle.audio.samples, &remote_starts, scanner.waveform_length(), config.leakage_band_hz, fs)),
        remote_detected: remote.len(),
        median_remote_snr_db: median_of(&remote_snrs),
        screen_off_events: count_events(bundle, |e| matches!(e, E::Screen { on: false, .. })),
        volume_zero_seen: bundle.events.iter().any(|e| matches!(e, E::Volume { music_index: 0, .. })),
    }
}

fn scanner_for(bundle: &PhoneBundle, speed_of_sound_mps: f64, config: &AnalysisConfig) -> Result<ArrivalScanner, String> {
    let waveform = waveform_from_description(&bundle.session.schedule.waveform, config.sample_rate_hz)?;
    Ok(ArrivalScanner::new(&waveform, speed_of_sound_mps, config.sample_rate_hz))
}

pub fn load_manifest(run_directory: &Path) -> Result<RunManifest, String> {
    let text = std::fs::read_to_string(run_directory.join("run_manifest.json")).map_err(|e| format!("run_manifest.json: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("run_manifest.json: {e}"))
}

pub fn analyse_run(run_directory: &Path, config: &AnalysisConfig) -> Result<RunReport, String> {
    let manifest = load_manifest(run_directory)?;
    let role = |name: &str| manifest.roles.get(name).cloned().ok_or(format!("manifest has no role {name}"));
    let (role_a, role_b) = (role("A")?, role("B")?);
    let a = PhoneBundle::load(&run_directory.join(&role_a.dir))?;
    let b = PhoneBundle::load(&run_directory.join(&role_b.dir))?;
    if a.session.schedule.waveform != b.session.schedule.waveform {
        return Err("the two phones played different waveforms".into());
    }
    let speed_of_sound_mps = manifest.speed_of_sound_mps();
    let scanner = scanner_for(&a, speed_of_sound_mps, config)?;
    let self_paths = SelfPathLengths {
        a_m: role_a.self_path_m.unwrap_or(config.default_self_path_length_m),
        b_m: role_b.self_path_m.unwrap_or(config.default_self_path_length_m),
    };
    let (a_self, a_blind) = pick_self_arrivals(&a, &scanner, self_paths.a_m, config);
    let (b_self, b_blind) = pick_self_arrivals(&b, &scanner, self_paths.b_m, config);
    let (a_remote, a_offset) = pick_remote_arrivals(&a, &b, &b_self, &scanner, config);
    let (b_remote, b_offset) = pick_remote_arrivals(&b, &a, &a_self, &scanner, config);
    let a_picks = PhonePicks { self_picks: a_self, remote_picks: a_remote, blind_scanned: a_blind, remote_offset_samples: a_offset };
    let b_picks = PhonePicks { self_picks: b_self, remote_picks: b_remote, blind_scanned: b_blind, remote_offset_samples: b_offset };
    let paired = pair_rounds(&a_picks, &b_picks, config.sample_rate_hz);
    let (a_continuity, b_continuity) = (continuity_of(&a, &a_picks, config), continuity_of(&b, &b_picks, config));
    let (rounds, rounds_excluded_for_capture_breaks) = continuous_rounds(paired, &a_continuity, &b_continuity, config.sample_rate_hz);
    let speaker_geometry = match (manifest.tape_distance_m, role_a.speaker_offset_world_m, role_b.speaker_offset_world_m) {
        (Some(distance), Some(speaker_a), Some(speaker_b)) => Some(SpeakerGeometry {
            mic_a_m: [0.0, 0.0, 0.0],
            mic_b_m: [distance, 0.0, 0.0],
            speaker_offset_a_world_m: speaker_a,
            speaker_offset_b_world_m: speaker_b,
        }),
        _ => None,
    };
    let ranged = range_rounds(&rounds, self_paths, speed_of_sound_mps, speaker_geometry);
    let distance = CompensationMode::ALL
        .iter()
        .filter_map(|mode| distance_statistics(&ranged, *mode, manifest.tape_distance_m, config.range_outlier_threshold_m))
        .collect();
    let clock_mode = [CompensationMode::DoubleSided, CompensationMode::TrackedRateRatio, CompensationMode::None]
        .into_iter()
        .find(|mode| ranged.iter().filter(|r| r.result(*mode).is_some()).count() >= 3);
    let clock = clock_mode.map(|mode| {
        let (points, segment_ids): (Vec<(f64, f64)>, Vec<usize>) = ranged
            .iter()
            .filter_map(|r| r.result(mode).map(|res| ((res.offset_reference_time_on_a_s, res.clock_offset_b_minus_a_s), r.round.capture_segment)))
            .unzip();
        clock_residual_report(&points, &segment_ids, mode.name(), config.heartbeat_cadence_s)
    });
    let phones = vec![phone_summary("A", &a, &a_picks, &scanner, config), phone_summary("B", &b, &b_picks, &scanner, config)];
    let channel_map = [(&a, &b, &phones[1]), (&b, &a, &phones[0])]
        .iter()
        .map(|(emitter, listener, listener_summary)| ChannelMapObservation {
            emitter_platform: emitter.session.platform_name.clone(),
            emitter_channel: emitter.session.schedule.channel.clone(),
            emitter_orientation: dominant_orientation(&emitter.events),
            listener_platform: listener.session.platform_name.clone(),
            median_remote_snr_db: listener_summary.median_remote_snr_db,
            remote_detected: listener_summary.remote_detected,
            remote_expected: emitter.emissions().iter().filter(|e| e.3).count(),
        })
        .collect();
    Ok(RunReport {
        run_label: manifest.label.clone(),
        matrix_run: manifest.matrix_run.clone(),
        waveform: a.session.schedule.waveform.clone(),
        period_ms: a.session.schedule.period_ms,
        tape_distance_m: manifest.tape_distance_m,
        temperature_c: manifest.temperature_c.unwrap_or(DEFAULT_AIR_TEMPERATURE_C),
        temperature_assumed: manifest.temperature_c.is_none(),
        speed_of_sound_mps,
        self_paths_m: [self_paths.a_m, self_paths.b_m],
        speaker_correction_applied: speaker_geometry.is_some(),
        phones,
        round_count: ranged.len(),
        rounds_excluded_for_capture_breaks,
        distance,
        implied_mean_self_path_m: manifest.tape_distance_m.and_then(|tape| implied_mean_self_path_m(&ranged, tape)),
        clock,
        channel_map,
        ranged_rounds: ranged,
    })
}

/// One phone alone: self loopback only (no ranging).
pub fn analyse_single(phone_directory: &Path, config: &AnalysisConfig) -> Result<PhoneSummary, String> {
    let bundle = PhoneBundle::load(phone_directory)?;
    let scanner = scanner_for(&bundle, RunManifest::default().speed_of_sound_mps(), config)?;
    let (self_picks, blind) = pick_self_arrivals(&bundle, &scanner, config.default_self_path_length_m, config);
    let picks = PhonePicks { self_picks, remote_picks: Vec::new(), blind_scanned: blind, remote_offset_samples: None };
    Ok(phone_summary(&bundle.session.role, &bundle, &picks, &scanner, config))
}
